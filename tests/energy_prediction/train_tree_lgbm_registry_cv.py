from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
import re

import lightgbm as lgb
import numpy as np
import pandas as pd

import utility
from supplementary_feature_registry import (
    get_supplementary_extractor,
    list_supplementary_extractors,
    normalize_supplementary_encoder_name,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("LightGBM CV trainer for ASHRAE GEPIII parquet dataset")

    # IO
    p.add_argument(
        "--data_parquet", type=str, required=True, help="Path to engineered parquet dataset dir/file from previous step"
    )
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for models + logs")

    # Objective
    p.add_argument("--label_col", type=str, default="meter_reading", help="Target column")
    p.add_argument(
        "--use_log1p_target",
        action="store_true",
        default=True,
        help="Train on log1p(y); RMSE on log1p equals RMSLE on original scale",
    )

    # Expanding-window CV controls
    p.add_argument(
        "--cv_num_folds",
        type=int,
        default=4,
        help="Number of outer expanding-window folds",
    )
    p.add_argument(
        "--cv_first_train_days",
        type=int,
        default=180,
        help="Length in days of the first outer-train window from dataset start",
    )
    p.add_argument(
        "--cv_eval_days",
        type=int,
        default=45,
        help="Outer evaluation horizon in days for each fold",
    )
    p.add_argument(
        "--cv_fold_step_days",
        type=int,
        default=45,
        help="Step in days between successive outer-train end points",
    )
    p.add_argument(
        "--cv_tail_days",
        type=int,
        default=14,
        help="Inner tail window in days, carved from outer-train, used only for early stopping",
    )

    # Feature selection / identity removal
    p.add_argument(
        "--keep_site_id", action="store_true", default=False, help="Include site_id as a categorical feature"
    )
    p.add_argument(
        "--keep_building_id", action="store_true", default=False, help="Include building_id as a categorical feature"
    )

    # Supplementary feature encoders
    p.add_argument(
        "--supplementary_encoder",
        type=str,
        default="none",
        choices=list_supplementary_extractors(include_none=True),
        help="Optional supplementary feature encoder block",
    )
    p.add_argument(
        "--supplementary_exclude_building_id",
        action="store_true",
        default=False,
        help="Prevent supplementary encoders from using building_id directly or via building-keyed derived features.",
    )
    p.add_argument(
        "--use_expert_features",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )

    # LightGBM knobs
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--num_leaves", type=int, default=63)
    p.add_argument("--max_depth", type=int, default=-1)
    p.add_argument("--min_data_in_leaf", type=int, default=200)
    p.add_argument("--feature_fraction", type=float, default=0.8)
    p.add_argument("--bagging_fraction", type=float, default=0.8)
    p.add_argument("--bagging_freq", type=int, default=1)
    p.add_argument("--lambda_l2", type=float, default=1.0)
    p.add_argument("--max_bin", type=int, default=255)

    p.add_argument("--n_estimators", type=int, default=2000)
    p.add_argument("--early_stopping_rounds", type=int, default=100)
    p.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=1e-4,
        help="Minimum validation improvement required to reset early stopping patience.",
    )
    p.add_argument("--log_period", type=int, default=50)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_threads", type=int, default=0, help="0 = use all threads")

    # device (GPU support)
    p.add_argument(
        "--device_type",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "cuda"],
        help="LightGBM tree learner device. gpu=OpenCL, cuda=CUDA build (usually from source).",
    )
    p.add_argument("--gpu_platform_id", type=int, default=-1, help="OpenCL platform id (gpu only). -1 = default")
    p.add_argument(
        "--gpu_device_id", type=int, default=-1, help="OpenCL device id within platform (gpu only). -1 = default"
    )
    p.add_argument(
        "--gpu_use_dp",
        action="store_true",
        default=False,
        help="Use double precision on GPU (slower, sometimes more accurate).",
    )
    p.add_argument(
        "--log_tag", type=str, default="lgbm_cv", help="TensorBoard run subdir name (used under out_dir/tb_lgbm/)"
    )

    args = p.parse_args()
    if args.use_expert_features and args.supplementary_encoder == "none":
        args.supplementary_encoder = "expert"
    args.supplementary_encoder = normalize_supplementary_encoder_name(args.supplementary_encoder)

    if args.cv_num_folds < 1:
        raise ValueError("--cv_num_folds must be >= 1")
    if args.cv_first_train_days <= 0:
        raise ValueError("--cv_first_train_days must be > 0")
    if args.cv_eval_days <= 0:
        raise ValueError("--cv_eval_days must be > 0")
    if args.cv_fold_step_days <= 0:
        raise ValueError("--cv_fold_step_days must be > 0")
    if args.cv_tail_days <= 0:
        raise ValueError("--cv_tail_days must be > 0")
    if args.cv_tail_days >= args.cv_first_train_days:
        raise ValueError("--cv_tail_days must be smaller than --cv_first_train_days")
    return args


def print_design_sanity(args: argparse.Namespace) -> None:
    use_site_id = bool(getattr(args, "keep_site_id", False)) and not bool(getattr(args, "drop_site_id", False))
    use_building_id = bool(getattr(args, "keep_building_id", False)) and not bool(
        getattr(args, "drop_building_id", False)
    )

    print("\n" + "=" * 80)
    print("[Sanity] Training design summary")
    print(f"  supplementary_encoder               : {args.supplementary_encoder}")
    print(f"  supplementary_exclude_building_id   : {bool(args.supplementary_exclude_building_id)}")
    print(f"  keep_site_id                        : {use_site_id}")
    print(f"  keep_building_id                    : {use_building_id}")
    print(f"  use_log1p_target                    : {bool(args.use_log1p_target)}")
    print(f"  cv_num_folds                        : {args.cv_num_folds}")
    print(f"  cv_first_train_days                 : {args.cv_first_train_days}")
    print(f"  cv_eval_days                        : {args.cv_eval_days}")
    print(f"  cv_fold_step_days                   : {args.cv_fold_step_days}")
    print(f"  cv_tail_days                        : {args.cv_tail_days}")
    print(f"  n_estimators                        : {args.n_estimators}")
    print(f"  early_stopping_rounds               : {args.early_stopping_rounds}")
    print(f"  early_stopping_min_delta            : {args.early_stopping_min_delta}")
    print(f"  seed                                : {args.seed}")

    warnings = []
    errors = []

    if args.cv_num_folds <= 0:
        errors.append("cv_num_folds must be > 0")
    if args.cv_first_train_days <= 0:
        errors.append("cv_first_train_days must be > 0")
    if args.cv_eval_days <= 0:
        errors.append("cv_eval_days must be > 0")
    if args.cv_fold_step_days <= 0:
        errors.append("cv_fold_step_days must be > 0")
    if args.cv_tail_days <= 0:
        errors.append("cv_tail_days must be > 0")

    if args.cv_tail_days >= args.cv_first_train_days:
        errors.append("cv_tail_days must be smaller than cv_first_train_days")
    if args.cv_tail_days >= args.cv_eval_days:
        warnings.append("cv_tail_days >= cv_eval_days; this is unusual and may make early stopping noisy")
    if args.early_stopping_rounds >= args.n_estimators:
        warnings.append("early_stopping_rounds >= n_estimators; early stopping may never trigger")
    if use_building_id and args.supplementary_exclude_building_id:
        warnings.append(
            "keep_building_id=True but supplementary_exclude_building_id=True; "
            "baseline model can still use building_id, supplementary encoders cannot"
        )
    if use_building_id and args.supplementary_encoder != "none":
        warnings.append(
            "building_id is enabled in the main model; this may dominate semantic comparisons via memorization"
        )
    if args.supplementary_encoder == "none" and args.supplementary_exclude_building_id:
        warnings.append("supplementary_exclude_building_id=True has no effect when supplementary_encoder=none")
    if args.cv_first_train_days < 120:
        warnings.append(
            "cv_first_train_days is quite small; target-derived and lagged supplementary features may be unstable"
        )

    if warnings:
        print("[Sanity] Warnings:")
        for msg in warnings:
            print(f"  - {msg}")

    if errors:
        print("[Sanity] Errors:")
        for msg in errors:
            print(f"  - {msg}")
        raise ValueError("Invalid CV / design configuration; see sanity-check errors above.")

    print("=" * 80 + "\n")


def _shape_str(x) -> str:
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return str(x.shape)
    if hasattr(x, "shape"):
        return str(tuple(x.shape))
    try:
        return f"({len(x)},)"
    except Exception:
        return "<unknown>"


def print_fold_shapes(
    fold_idx: int,
    *,
    df_full: pd.DataFrame | None = None,
    df_outer_train: pd.DataFrame | None = None,
    df_outer_eval: pd.DataFrame | None = None,
    df_inner_fit: pd.DataFrame | None = None,
    df_inner_tail: pd.DataFrame | None = None,
    sup_fit_in: pd.DataFrame | None = None,
    sup_tail_in: pd.DataFrame | None = None,
    sup_outer_train_in: pd.DataFrame | None = None,
    sup_outer_eval_in: pd.DataFrame | None = None,
    sup_fit_out: pd.DataFrame | None = None,
    sup_tail_out: pd.DataFrame | None = None,
    sup_outer_train_out: pd.DataFrame | None = None,
    sup_outer_eval_out: pd.DataFrame | None = None,
    X_inner_fit=None,
    X_inner_tail=None,
    X_outer_train=None,
    X_outer_eval=None,
) -> None:
    print(f"[Fold {fold_idx}] shape summary")
    if df_full is not None:
        print(f"  df_full             : {_shape_str(df_full)}")
    if df_outer_train is not None:
        print(f"  df_outer_train      : {_shape_str(df_outer_train)}")
    if df_outer_eval is not None:
        print(f"  df_outer_eval       : {_shape_str(df_outer_eval)}")
    if df_inner_fit is not None:
        print(f"  df_inner_fit        : {_shape_str(df_inner_fit)}")
    if df_inner_tail is not None:
        print(f"  df_inner_tail       : {_shape_str(df_inner_tail)}")

    if sup_fit_in is not None:
        print(f"  sup_fit_in          : {_shape_str(sup_fit_in)}")
    if sup_tail_in is not None:
        print(f"  sup_tail_in         : {_shape_str(sup_tail_in)}")
    if sup_outer_train_in is not None:
        print(f"  sup_outer_train_in  : {_shape_str(sup_outer_train_in)}")
    if sup_outer_eval_in is not None:
        print(f"  sup_outer_eval_in   : {_shape_str(sup_outer_eval_in)}")

    if sup_fit_out is not None:
        print(f"  sup_fit_out         : {_shape_str(sup_fit_out)}")
    if sup_tail_out is not None:
        print(f"  sup_tail_out        : {_shape_str(sup_tail_out)}")
    if sup_outer_train_out is not None:
        print(f"  sup_outer_train_out : {_shape_str(sup_outer_train_out)}")
    if sup_outer_eval_out is not None:
        print(f"  sup_outer_eval_out  : {_shape_str(sup_outer_eval_out)}")

    if X_inner_fit is not None:
        print(f"  X_inner_fit         : {_shape_str(X_inner_fit)}")
    if X_inner_tail is not None:
        print(f"  X_inner_tail        : {_shape_str(X_inner_tail)}")
    if X_outer_train is not None:
        print(f"  X_outer_train       : {_shape_str(X_outer_train)}")
    if X_outer_eval is not None:
        print(f"  X_outer_eval        : {_shape_str(X_outer_eval)}")


def _sanitize_run_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:180]


def build_default_run_name(args: argparse.Namespace) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [
        "lgbmcv",
        stamp,
        f"lr{args.learning_rate:g}",
        f"L{args.num_leaves}",
        f"minleaf{args.min_data_in_leaf}",
        args.supplementary_encoder,
        f"f{args.cv_num_folds}",
        f"first{args.cv_first_train_days}",
        f"eval{args.cv_eval_days}",
        f"step{args.cv_fold_step_days}",
        f"tail{args.cv_tail_days}",
    ]
    if getattr(args, "keep_site_id", False):
        parts.append("sid")
    if getattr(args, "keep_building_id", False):
        parts.append("bid")
    if getattr(args, "supplementary_exclude_building_id", False):
        parts.append("supNoBid")

    base = "_".join(parts)
    if args.log_tag:
        base = f"{base}__{args.log_tag}"
    return _sanitize_run_name(base)


def choose_feature_columns(all_cols: List[str]) -> List[str]:
    engineered = [
        "hour_sin",
        "hour_cos",
        "dayofweek",
        "doy_sin",
        "doy_cos",
        "is_weekend",
        "is_na_holiday",
        "is_eu_holiday",
        "is_holiday_any",
        "is_business_hours",
        "CDH_18C",
        "HDH_18C",
        "is_hot_24C",
        "is_cold_10C",
        "dewpoint_depression",
        "log_sqft",
        "year_built_clipped",
    ]
    raw = [
        "meter",
        "primary_use",
        "square_feet",
        "floor_count",
        "air_temperature",
        "cloud_coverage",
        "dew_temperature",
        "precip_depth_1_hr",
        "sea_level_pressure",
        "wind_direction",
        "wind_speed",
    ]
    return [c for c in (raw + engineered) if c in all_cols]


def _coerce_supplementary_df(
    feature_df,
    *,
    expected_rows: int,
    label_col: str,
) -> pd.DataFrame:
    if feature_df is None:
        raise ValueError("Supplementary feature extractor returned None; expected a DataFrame.")
    if not isinstance(feature_df, pd.DataFrame):
        feature_df = pd.DataFrame(feature_df)
    if len(feature_df) != expected_rows:
        raise ValueError(f"Supplementary feature rows mismatch: {len(feature_df)} vs {expected_rows}")

    forbidden = {label_col, "timestamp", "building_id", "site_id"}
    cols_to_drop = [c for c in feature_df.columns if c in forbidden]
    if cols_to_drop:
        feature_df = feature_df.drop(columns=cols_to_drop)

    return feature_df.reset_index(drop=True)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _rmsle_from_log_targets(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    y_true_orig = np.expm1(y_true_log)
    y_pred_orig = np.expm1(y_pred_log)
    return float(np.sqrt(np.mean((np.log1p(y_pred_orig) - np.log1p(y_true_orig)) ** 2)))


def _build_day_index(df: pd.DataFrame, timestamp_col: str) -> pd.Series:
    ts = pd.to_datetime(df[timestamp_col], utc=False)
    if ts.isna().any():
        raise ValueError(f"{timestamp_col} contains NaT values; cannot build CV folds.")
    day_floor = ts.dt.floor("D")
    base_day = day_floor.min()
    return (day_floor - base_day).dt.days.astype(int)


def _make_expanding_cv_folds(
    day_index: pd.Series,
    *,
    num_folds: int,
    first_train_days: int,
    eval_days: int,
    fold_step_days: int,
    tail_days: int,
) -> List[Dict[str, Any]]:
    max_day_exclusive = int(day_index.max()) + 1
    folds: List[Dict[str, Any]] = []

    for fold_no in range(num_folds):
        outer_train_end = first_train_days + fold_no * fold_step_days
        outer_eval_start = outer_train_end
        outer_eval_end = outer_eval_start + eval_days
        inner_tail_start = outer_train_end - tail_days

        if inner_tail_start <= 0:
            raise ValueError(
                f"Fold {fold_no + 1}: inner tail start <= 0. Increase --cv_first_train_days or reduce --cv_tail_days."
            )
        if outer_eval_end > max_day_exclusive:
            raise ValueError(
                f"Fold {fold_no + 1}: outer eval end day {outer_eval_end} exceeds dataset length {max_day_exclusive} days. "
                "Reduce --cv_num_folds / --cv_eval_days / --cv_fold_step_days or --cv_first_train_days."
            )

        inner_fit_mask = day_index < inner_tail_start
        inner_tail_mask = (day_index >= inner_tail_start) & (day_index < outer_train_end)
        outer_train_mask = day_index < outer_train_end
        outer_eval_mask = (day_index >= outer_eval_start) & (day_index < outer_eval_end)

        if int(inner_fit_mask.sum()) == 0 or int(inner_tail_mask.sum()) == 0 or int(outer_eval_mask.sum()) == 0:
            raise ValueError(f"Fold {fold_no + 1}: one of the train/tail/eval splits is empty.")

        folds.append(
            {
                "fold_no": fold_no + 1,
                "inner_fit_mask": inner_fit_mask.to_numpy(),
                "inner_tail_mask": inner_tail_mask.to_numpy(),
                "outer_train_mask": outer_train_mask.to_numpy(),
                "outer_eval_mask": outer_eval_mask.to_numpy(),
                "boundaries": {
                    "inner_fit": [0, inner_tail_start],
                    "inner_tail": [inner_tail_start, outer_train_end],
                    "outer_train": [0, outer_train_end],
                    "outer_eval": [outer_eval_start, outer_eval_end],
                },
                "row_counts": {
                    "inner_fit": int(inner_fit_mask.sum()),
                    "inner_tail": int(inner_tail_mask.sum()),
                    "outer_train": int(outer_train_mask.sum()),
                    "outer_eval": int(outer_eval_mask.sum()),
                },
            }
        )
    return folds


def _lightgbm_params(args: argparse.Namespace) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l2": args.lambda_l2,
        "max_bin": args.max_bin,
        "seed": args.seed,
        "num_threads": args.num_threads,
        "verbosity": -1,
    }
    if args.device_type == "gpu":
        params.update(
            {
                "device_type": args.device_type,
                "gpu_platform_id": args.gpu_platform_id,
                "gpu_device_id": args.gpu_device_id,
                "gpu_use_dp": args.gpu_use_dp,
            }
        )
    return params


def _build_stage_matrices(
    *,
    base_df: pd.DataFrame,
    keep_cols: List[str],
    drop_cols: List[str],
    categorical_cols: List[str],
    fit_mask: np.ndarray,
    eval_mask: np.ndarray,
    args: argparse.Namespace,
    cache_path: Path | None = None,
) -> Dict[str, Any]:
    fit_index = base_df.index[fit_mask]
    eval_index = base_df.index[eval_mask]

    stage_index = list(fit_index) + list(eval_index)
    stage_df = base_df.loc[stage_index].copy()

    stage_categorical_cols = list(categorical_cols)
    supplementary_cols: List[str] = []
    encoder_cache: Dict[str, Any] | None = None

    if args.supplementary_encoder != "none":
        extractor = get_supplementary_extractor(args.supplementary_encoder)
        encoder_cache = {
            "encoder_name": args.supplementary_encoder,
            "label_col": args.label_col,
            "exclude_building_id": bool(args.supplementary_exclude_building_id),
        }

        sup_fit_input = base_df.loc[fit_index].copy()
        if args.supplementary_exclude_building_id and "building_id" in sup_fit_input.columns:
            sup_fit_input = sup_fit_input.drop(columns=["building_id"])

        sup_eval_input = base_df.loc[eval_index].copy()
        if args.label_col in sup_eval_input.columns:
            sup_eval_input = sup_eval_input.drop(columns=[args.label_col])
        if args.supplementary_exclude_building_id and "building_id" in sup_eval_input.columns:
            sup_eval_input = sup_eval_input.drop(columns=["building_id"])

        sup_fit_df = _coerce_supplementary_df(
            extractor(
                sup_fit_input,
                cache=encoder_cache,
                mode="fit_transform",
                label_col=args.label_col,
                exclude_building_id=args.supplementary_exclude_building_id,
            ),
            expected_rows=len(fit_index),
            label_col=args.label_col,
        )
        sup_eval_df = _coerce_supplementary_df(
            extractor(
                sup_eval_input,
                cache=encoder_cache,
                mode="transform",
                label_col=args.label_col,
                exclude_building_id=args.supplementary_exclude_building_id,
            ),
            expected_rows=len(eval_index),
            label_col=args.label_col,
        )

        if list(sup_fit_df.columns) != list(sup_eval_df.columns):
            missing_in_eval = sorted(set(sup_fit_df.columns) - set(sup_eval_df.columns))
            missing_in_fit = sorted(set(sup_eval_df.columns) - set(sup_fit_df.columns))
            raise ValueError(
                "Supplementary feature columns differ between fit-transform and transform. "
                f"missing_in_eval={missing_in_eval[:20]} "
                f"missing_in_fit={missing_in_fit[:20]}"
            )

        existing_cols = set(stage_df.columns)
        rename_map: Dict[str, str] = {}
        used_new = set()
        for c in sup_fit_df.columns:
            new_c = c
            if new_c in existing_cols or new_c in used_new:
                base = f"supp__{c}"
                new_c = base
                k = 2
                while new_c in existing_cols or new_c in used_new:
                    new_c = f"{base}__{k}"
                    k += 1
                rename_map[c] = new_c
            used_new.add(new_c)

        if rename_map:
            sup_fit_df = sup_fit_df.rename(columns=rename_map)
            sup_eval_df = sup_eval_df.rename(columns=rename_map)

        sup_fit_df.index = fit_index
        sup_eval_df.index = eval_index
        supplementary_df = pd.concat([sup_fit_df, sup_eval_df], axis=0).reindex(stage_df.index)

        obj_cols = [c for c in supplementary_df.columns if supplementary_df[c].dtype == object]
        if obj_cols:
            supplementary_df[obj_cols] = supplementary_df[obj_cols].astype("category")
            stage_categorical_cols = list(dict.fromkeys(stage_categorical_cols + obj_cols))

        supplementary_cols = list(supplementary_df.columns)
        stage_df = pd.concat([stage_df, supplementary_df], axis=1)

        if cache_path is not None:
            with open(cache_path, "wb") as f:
                pickle.dump(encoder_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        pass

    feature_cols = [c for c in keep_cols if c in stage_df.columns] + supplementary_cols
    stage_drop_cols = [c for c in drop_cols if c in stage_df.columns]
    df_model = stage_df[feature_cols + [args.label_col] + stage_drop_cols].copy()

    y_stage = df_model[args.label_col].to_numpy()
    if args.use_log1p_target:
        y_stage = np.log1p(y_stage).astype(np.float32)
    df_model[args.label_col] = y_stage

    X_all, y_all, cat_present = utility.tree_prepare_tabular_matrices(
        df_model,
        label_col=args.label_col,
        drop_cols=stage_drop_cols,
        categorical_cols=stage_categorical_cols,
    )

    fit_len = len(fit_index)
    X_fit = X_all.iloc[:fit_len].copy()
    y_fit = y_all[:fit_len]
    X_eval = X_all.iloc[fit_len:].copy()
    y_eval = y_all[fit_len:]

    return {
        "X_fit": X_fit,
        "y_fit": y_fit,
        "X_eval": X_eval,
        "y_eval": y_eval,
        "cat_present": cat_present,
        "feature_columns": list(X_fit.columns),
        "supplementary_columns": supplementary_cols,
        "encoder_cache": encoder_cache,
    }


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def main() -> None:
    args = parse_args()
    print_design_sanity(args)
    utility.common_set_seed(args.seed)

    out_dir = Path(args.out_dir)
    data_path = Path(args.data_parquet)
    run_name = build_default_run_name(args)
    model_out_dir = out_dir / "models" / run_name
    utility.common_ensure_dir(model_out_dir)

    tb_root = out_dir / "tb_lgbm"
    tb_dir = tb_root / run_name
    utility.common_ensure_dir(tb_dir)

    base_needed = [args.label_col, "timestamp", "building_id", "site_id"]
    df_min = utility.common_load_parquet_dataset(data_path, columns=base_needed, engine="pyarrow")

    keep_cols = choose_feature_columns(
        all_cols=(
            base_needed
            + [
                "meter",
                "primary_use",
                "square_feet",
                "floor_count",
                "air_temperature",
                "cloud_coverage",
                "dew_temperature",
                "precip_depth_1_hr",
                "sea_level_pressure",
                "wind_direction",
                "wind_speed",
                "hour_sin",
                "hour_cos",
                "dayofweek",
                "doy_sin",
                "doy_cos",
                "is_weekend",
                "is_na_holiday",
                "is_eu_holiday",
                "is_holiday_any",
                "is_business_hours",
                "CDH_18C",
                "HDH_18C",
                "is_hot_24C",
                "is_cold_10C",
                "dewpoint_depression",
                "log_sqft",
                "year_built_clipped",
            ]
        )
    )

    use_site_id = bool(getattr(args, "keep_site_id", False)) and not bool(getattr(args, "drop_site_id", False))
    use_building_id = bool(getattr(args, "keep_building_id", False)) and not bool(
        getattr(args, "drop_building_id", False)
    )

    if use_site_id and "site_id" in df_min.columns and "site_id" not in keep_cols:
        keep_cols.append("site_id")
    if use_building_id and "building_id" in df_min.columns and "building_id" not in keep_cols:
        keep_cols.append("building_id")

    load_cols = sorted(set(base_needed + ["timestamp"] + keep_cols))
    if args.supplementary_encoder != "none":
        df = utility.common_load_parquet_dataset(data_path, engine="pyarrow")
    else:
        df = utility.common_load_parquet_dataset(data_path, columns=load_cols, engine="pyarrow")

    if "timestamp" not in df.columns:
        raise ValueError("Dataset must contain a timestamp column for CV splitting.")

    df = df.sort_values(["timestamp"], kind="mergesort").reset_index(drop=True)
    keep_cols = [c for c in keep_cols if c in df.columns]

    drop_cols = ["timestamp"]
    if (not use_building_id) and "building_id" in df.columns:
        drop_cols.append("building_id")
    if (not use_site_id) and "site_id" in df.columns:
        drop_cols.append("site_id")

    categorical_cols = ["primary_use", "meter", "dayofweek"]
    if use_site_id and "site_id" in df.columns:
        categorical_cols.append("site_id")
    if use_building_id and "building_id" in df.columns:
        categorical_cols.append("building_id")
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    if args.device_type == "gpu" and args.max_bin > 63:
        print("[Warn] GPU mode usually benefits from smaller --max_bin (e.g. 63).")

    day_index = _build_day_index(df, "timestamp")
    folds = _make_expanding_cv_folds(
        day_index,
        num_folds=args.cv_num_folds,
        first_train_days=args.cv_first_train_days,
        eval_days=args.cv_eval_days,
        fold_step_days=args.cv_fold_step_days,
        tail_days=args.cv_tail_days,
    )

    params = _lightgbm_params(args)
    _save_json(model_out_dir / "train_args_and_params.json", {"params": params, **vars(args)})

    cv_fold_summaries: List[Dict[str, Any]] = []

    for fold in folds:
        fold_no = fold["fold_no"]
        fold_name = f"fold_{fold_no:02d}"
        fold_dir = model_out_dir / fold_name
        utility.common_ensure_dir(fold_dir)

        print(
            f"[Fold {fold_no}] outer_train={fold['boundaries']['outer_train']} "
            f"inner_tail={fold['boundaries']['inner_tail']} "
            f"outer_eval={fold['boundaries']['outer_eval']}"
        )

        ts = time.time()
        # Stage 1: inner tail is used only for early stopping
        inner_stage = _build_stage_matrices(
            base_df=df,
            keep_cols=keep_cols,
            drop_cols=drop_cols,
            categorical_cols=categorical_cols,
            fit_mask=fold["inner_fit_mask"],
            eval_mask=fold["inner_tail_mask"],
            args=args,
            cache_path=None,
        )
        print(f"[Fold {fold_no}] Stage 1: built inner train/tail matrices in {time.time() - ts:.1f}s")

        dtrain_inner = lgb.Dataset(
            inner_stage["X_fit"],
            label=inner_stage["y_fit"],
            categorical_feature=inner_stage["cat_present"],
            free_raw_data=False,
        )
        dval_inner = lgb.Dataset(
            inner_stage["X_eval"],
            label=inner_stage["y_eval"],
            categorical_feature=inner_stage["cat_present"],
            reference=dtrain_inner,
            free_raw_data=False,
        )

        print_fold_shapes(
            fold_no,
            df_full=df,
            X_inner_fit=inner_stage["X_fit"],
            X_inner_tail=inner_stage["X_eval"],
        )

        inner_evals_result: Dict[str, Any] = {}
        print(f"[Fold {fold_no}] Stage 1: early-stopping fit on inner train/tail...")
        t0 = time.time()
        booster_inner = lgb.train(
            params=params,
            train_set=dtrain_inner,
            num_boost_round=args.n_estimators,
            valid_sets=[dtrain_inner, dval_inner],
            valid_names=["train", "tail"],
            callbacks=[
                lgb.record_evaluation(inner_evals_result),
                lgb.log_evaluation(period=args.log_period),
                lgb.early_stopping(
                    stopping_rounds=args.early_stopping_rounds,
                    first_metric_only=True,
                    verbose=True,
                    min_delta=args.early_stopping_min_delta,
                ),
            ],
        )
        dt_inner = time.time() - t0
        best_iter = int(booster_inner.best_iteration or args.n_estimators)
        print(f"[Fold {fold_no}] Stage 1 done in {dt_inner:.1f}s  best_iter={best_iter}")

        _save_json(fold_dir / "inner_evals_result.json", inner_evals_result)
        utility.tree_write_tensorboard_evals(inner_evals_result, tb_dir / fold_name / "inner_tail")

        # Stage 2: refit on full outer train; outer eval is for feature-quality measurement only
        cache_path = None
        if args.supplementary_encoder != "none":
            cache_path = fold_dir / f"{args.supplementary_encoder}_feature_cache.pkl"

        ts = time.time()
        outer_stage = _build_stage_matrices(
            base_df=df,
            keep_cols=keep_cols,
            drop_cols=drop_cols,
            categorical_cols=categorical_cols,
            fit_mask=fold["outer_train_mask"],
            eval_mask=fold["outer_eval_mask"],
            args=args,
            cache_path=cache_path,
        )
        print(f"[Fold {fold_no}] Stage 2: built outer train/eval matrices in {time.time() - ts:.1f}s")

        dtrain_outer = lgb.Dataset(
            outer_stage["X_fit"],
            label=outer_stage["y_fit"],
            categorical_feature=outer_stage["cat_present"],
            free_raw_data=False,
        )

        print_fold_shapes(
            fold_no,
            X_outer_train=outer_stage["X_fit"],
            X_outer_eval=outer_stage["X_eval"],
        )

        print(f"[Fold {fold_no}] Stage 2: refit on outer train with fixed best_iter={best_iter}...")
        outer_evals_result = {}
        t1 = time.time()
        booster_outer = lgb.train(
            params=params,
            train_set=dtrain_outer,
            num_boost_round=best_iter,
            valid_sets=[dtrain_outer],
            valid_names=["train"],
            callbacks=[lgb.record_evaluation(outer_evals_result), lgb.log_evaluation(period=args.log_period * 4)],
        )
        dt_outer = time.time() - t1
        print(f"[Fold {fold_no}] Stage 2 done in {dt_outer:.1f}s")

        # pred_train = booster_outer.predict(outer_stage["X_fit"], num_iteration=best_iter)
        pred_eval = booster_outer.predict(outer_stage["X_eval"], num_iteration=best_iter)

        train_rmse_log = float(outer_evals_result["train"]["rmse"][-1])
        # train_rmse_log = _rmse(pred_train, outer_stage["y_fit"])
        eval_rmse_log = _rmse(pred_eval, outer_stage["y_eval"])

        fold_metrics: Dict[str, Any] = {
            "fold_no": fold_no,
            "best_iteration": best_iter,
            "train_time_inner_sec": dt_inner,
            "train_time_outer_sec": dt_outer,
            "row_counts": fold["row_counts"],
            "boundaries": fold["boundaries"],
            "metrics": {
                "train_rmse_log1p": train_rmse_log,
                "eval_rmse_log1p": eval_rmse_log,
                "generalization_gap_rmse_log1p": eval_rmse_log - train_rmse_log,
            },
        }

        if args.use_log1p_target:
            # RMSE on log1p(y) is numerically the same quantity as RMSLE on original scale.
            fold_metrics["metrics"]["train_rmsle_original"] = fold_metrics["metrics"]["train_rmse_log1p"]
            fold_metrics["metrics"]["eval_rmsle_original"] = _rmsle_from_log_targets(outer_stage["y_eval"], pred_eval)
            fold_metrics["metrics"]["generalization_gap_rmsle_original"] = (
                fold_metrics["metrics"]["eval_rmsle_original"] - fold_metrics["metrics"]["train_rmsle_original"]
            )

        model_path = fold_dir / "lgbm_model.txt"
        booster_outer.save_model(str(model_path))

        manifest = {
            "feature_columns": outer_stage["feature_columns"],
            "categorical_columns": outer_stage["cat_present"],
            "supplementary_encoder": args.supplementary_encoder,
            "supplementary_exclude_building_id": bool(args.supplementary_exclude_building_id),
            "cv_fold_no": fold_no,
            "cv_boundaries": fold["boundaries"],
            "best_iteration_from_inner_tail": best_iter,
        }
        if args.supplementary_encoder != "none":
            manifest["supplementary_cache_file"] = f"{args.supplementary_encoder}_feature_cache.pkl"

        _save_json(fold_dir / "fold_metrics.json", fold_metrics)
        _save_json(fold_dir / "feature_manifest.json", manifest)

        print(
            f"[Fold {fold_no}] eval RMSE(log1p)={fold_metrics['metrics']['eval_rmse_log1p']:.6f}"
            + (
                f"  eval RMSLE(original)={fold_metrics['metrics']['eval_rmsle_original']:.6f}"
                if args.use_log1p_target
                else ""
            )
        )

        cv_fold_summaries.append(fold_metrics)

    # Aggregate summary
    eval_rmse_logs = [f["metrics"]["eval_rmse_log1p"] for f in cv_fold_summaries]
    train_rmse_logs = [f["metrics"]["train_rmse_log1p"] for f in cv_fold_summaries]
    best_iters = [f["best_iteration"] for f in cv_fold_summaries]

    cv_summary: Dict[str, Any] = {
        "cv_plan": {
            "num_folds": args.cv_num_folds,
            "first_train_days": args.cv_first_train_days,
            "eval_days": args.cv_eval_days,
            "fold_step_days": args.cv_fold_step_days,
            "tail_days": args.cv_tail_days,
        },
        "supplementary_encoder": args.supplementary_encoder,
        "supplementary_exclude_building_id": bool(args.supplementary_exclude_building_id),
        "folds": cv_fold_summaries,
        "aggregate": {
            "mean_train_rmse_log1p": float(np.mean(train_rmse_logs)),
            "std_train_rmse_log1p": float(np.std(train_rmse_logs)),
            "mean_eval_rmse_log1p": float(np.mean(eval_rmse_logs)),
            "std_eval_rmse_log1p": float(np.std(eval_rmse_logs)),
            "mean_best_iteration": float(np.mean(best_iters)),
            "median_best_iteration": int(np.median(best_iters)),
        },
    }

    if args.use_log1p_target:
        eval_rmsle = [f["metrics"]["eval_rmsle_original"] for f in cv_fold_summaries]
        train_rmsle = [f["metrics"]["train_rmsle_original"] for f in cv_fold_summaries]
        cv_summary["aggregate"].update(
            {
                "mean_train_rmsle_original": float(np.mean(train_rmsle)),
                "std_train_rmsle_original": float(np.std(train_rmsle)),
                "mean_eval_rmsle_original": float(np.mean(eval_rmsle)),
                "std_eval_rmsle_original": float(np.std(eval_rmsle)),
            }
        )

    _save_json(model_out_dir / "cv_summary.json", cv_summary)
    pd.DataFrame(
        [
            {
                "fold_no": f["fold_no"],
                "best_iteration": f["best_iteration"],
                "train_rmse_log1p": f["metrics"]["train_rmse_log1p"],
                "eval_rmse_log1p": f["metrics"]["eval_rmse_log1p"],
                **(
                    {
                        "train_rmsle_original": f["metrics"]["train_rmsle_original"],
                        "eval_rmsle_original": f["metrics"]["eval_rmsle_original"],
                    }
                    if args.use_log1p_target
                    else {}
                ),
            }
            for f in cv_fold_summaries
        ]
    ).to_csv(model_out_dir / "cv_fold_metrics.csv", index=False)

    print("[Save] CV summary:", model_out_dir / "cv_summary.json")
    print(
        f"[CV] mean eval RMSE(log1p)={cv_summary['aggregate']['mean_eval_rmse_log1p']:.6f} "
        f"+/- {cv_summary['aggregate']['std_eval_rmse_log1p']:.6f}"
    )
    if args.use_log1p_target:
        print(
            f"[CV] mean eval RMSLE(original)={cv_summary['aggregate']['mean_eval_rmsle_original']:.6f} "
            f"+/- {cv_summary['aggregate']['std_eval_rmsle_original']:.6f}"
        )


if __name__ == "__main__":
    main()
