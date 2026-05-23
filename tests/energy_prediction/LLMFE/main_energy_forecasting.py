"""Run LLM-FE on the ASHRAE-style energy forecasting task.

Place this file at the root of the LLM-FE repository and run it from that root.
It keeps upstream LLM-FE components unchanged and supplies a task-specific
specification plus temporal evaluator data.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict

from llmfe import config as config_lib
from llmfe import evaluator
from llmfe import pipeline
from llmfe import sampler

from energy_forecasting_adapter import (
    CandidatePolicyConfig,
    EnergyForecastingConfig,
    LightGBMSearchConfig,
    TemporalSplitConfig,
    load_development_data,
    metadata_for_energy_features,
)


class _WallClockTimeout(Exception):
    pass


def _timeout_handler(signum, frame):  # pragma: no cover - OS/signal behavior.
    raise _WallClockTimeout("LLM-FE wall-clock budget reached.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("LLM-FE energy forecasting runner")
    p.add_argument("--data_parquet", required=True, help="Preprocessed parquet dataset directory/file.")
    p.add_argument("--out_dir", required=True, help="Output/log directory for the LLM-FE search.")
    p.add_argument(
        "--spec_path",
        default="specs/specification_energy_forecasting.txt",
        help="LLM-FE task specification file.",
    )
    p.add_argument("--metadata_json", default=None, help="Optional feature-metadata JSON file.")

    # Temporal split: Idea-1 default.
    p.add_argument("--train_start_day", type=int, default=0)
    p.add_argument("--train_end_day_exclusive", type=int, default=280)
    p.add_argument("--inner_eval_start_day", type=int, default=280)
    p.add_argument("--inner_eval_end_day_exclusive", type=int, default=320)
    p.add_argument("--final_test_start_day", type=int, default=320)
    p.add_argument("--final_test_end_day_exclusive", type=int, default=366)
    p.add_argument("--early_stopping_tail_days", type=int, default=14)

    # Budget controls.
    p.add_argument("--wall_clock_hours", type=float, default=12.0)
    p.add_argument(
        "--max_candidates",
        type=int,
        default=60,
        help=(
            "Approximate maximum number of sampled candidates. Upstream sampler currently "
            "divides max_sample_nums by 5 internally, so this runner passes max_candidates*5."
        ),
    )
    p.add_argument("--evaluate_timeout_seconds", type=int, default=3600)
    p.add_argument("--samples_per_prompt", type=int, default=3)
    p.add_argument("--functions_per_prompt", type=int, default=2)
    p.add_argument("--num_islands", type=int, default=3)
    p.add_argument("--use_api", action="store_true", default=False)
    p.add_argument("--api_model", type=str, default="gpt-3.5-turbo")

    # LightGBM settings matching the supplied debugpy configuration by default.
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--num_leaves", type=int, default=63)
    p.add_argument("--max_depth", type=int, default=-1)
    p.add_argument("--min_data_in_leaf", type=int, default=500)
    p.add_argument("--feature_fraction", type=float, default=0.8)
    p.add_argument("--bagging_fraction", type=float, default=0.8)
    p.add_argument("--bagging_freq", type=int, default=1)
    p.add_argument("--lambda_l2", type=float, default=1.0)
    p.add_argument("--max_bin", type=int, default=255)
    p.add_argument("--n_estimators", type=int, default=12000)
    p.add_argument("--early_stopping_rounds", type=int, default=200)
    p.add_argument("--early_stopping_min_delta", type=float, default=10e-4)
    p.add_argument("--log_period", type=int, default=400)
    p.add_argument("--seed", type=int, default=155)
    p.add_argument("--num_threads", type=int, default=0)
    p.add_argument("--device_type", type=str, default="cpu", choices=["cpu", "gpu", "cuda"])

    # Candidate guard policy.
    p.add_argument("--max_new_columns", type=int, default=32)
    p.add_argument("--rowwise_probe_rows", type=int, default=12)
    return p.parse_args()


def build_energy_config(args: argparse.Namespace) -> EnergyForecastingConfig:
    return EnergyForecastingConfig(
        split=TemporalSplitConfig(
            train_start_day=args.train_start_day,
            train_end_day_exclusive=args.train_end_day_exclusive,
            inner_eval_start_day=args.inner_eval_start_day,
            inner_eval_end_day_exclusive=args.inner_eval_end_day_exclusive,
            final_test_start_day=args.final_test_start_day,
            final_test_end_day_exclusive=args.final_test_end_day_exclusive,
            early_stopping_tail_days=args.early_stopping_tail_days,
        ),
        lgbm=LightGBMSearchConfig(
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            min_data_in_leaf=args.min_data_in_leaf,
            feature_fraction=args.feature_fraction,
            bagging_fraction=args.bagging_fraction,
            bagging_freq=args.bagging_freq,
            lambda_l2=args.lambda_l2,
            max_bin=args.max_bin,
            n_estimators=args.n_estimators,
            early_stopping_rounds=args.early_stopping_rounds,
            early_stopping_min_delta=args.early_stopping_min_delta,
            log_period=args.log_period,
            seed=args.seed,
            num_threads=args.num_threads,
            device_type=args.device_type,
        ),
        candidate_policy=CandidatePolicyConfig(
            max_new_columns=args.max_new_columns,
            rowwise_probe_rows=args.rowwise_probe_rows,
            rowwise_probe_seed=args.seed,
        ),
        out_dir=args.out_dir,
    )


def load_metadata(path: str | None) -> Dict[str, str]:
    meta = metadata_for_energy_features()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user_meta = json.load(f)
        meta.update({str(k): str(v) for k, v in user_meta.items()})
    return meta


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_path = Path(args.spec_path)
    if not spec_path.exists():
        raise FileNotFoundError(f"Specification file not found: {spec_path}")
    specification = spec_path.read_text(encoding="utf-8")

    energy_config = build_energy_config(args)
    dataset = {"data": load_development_data(args.data_parquet, energy_config, physically_exclude_final_test=True)}
    meta_data = load_metadata(args.metadata_json)

    (out_dir / "energy_llmfe_config.json").write_text(
        json.dumps(energy_config.to_jsonable(), indent=2, default=str), encoding="utf-8"
    )

    exp_buffer_config = config_lib.ExperienceBufferConfig(
        functions_per_prompt=args.functions_per_prompt,
        num_islands=args.num_islands,
    )
    llmfe_config = config_lib.Config(
        experience_buffer=exp_buffer_config,
        num_samplers=1,
        num_evaluators=1,
        samples_per_prompt=args.samples_per_prompt,
        evaluate_timeout_seconds=args.evaluate_timeout_seconds,
        use_api=args.use_api,
        api_model=args.api_model,
    )
    class_config = config_lib.ClassConfig(
        llm_class=sampler.LocalLLM,
        sandbox_class=evaluator.LocalSandbox,
    )

    budget_seconds = int(args.wall_clock_hours * 3600)
    if hasattr(signal, "SIGALRM") and budget_seconds > 0:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(budget_seconds)

    try:
        # Upstream Sampler checks max_sample_nums//5, so pass max_candidates*5 to
        # preserve the user-facing approximate candidate count.
        pipeline.main(
            specification=specification,
            inputs=dataset,
            config=llmfe_config,
            meta_data=meta_data,
            max_sample_nums=args.max_candidates * 5,
            class_config=class_config,
            log_dir=str(out_dir),
        )
    except _WallClockTimeout:
        print("[Stop] LLM-FE wall-clock budget reached; search stopped.", file=sys.stderr)
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)


if __name__ == "__main__":
    main()
