from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import re
import numpy as np
import pandas as pd
import lightgbm as lgb

import utility
from expert_feature_extractor import extract_expert_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("LightGBM baseline (tree-only) for ASHRAE GEPIII parquet dataset")

    # IO
    p.add_argument(
        "--data_parquet", type=str, required=True, help="Path to engineered parquet dataset dir/file from previous step"
    )
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for model + logs")

    # Split / objective
    p.add_argument("--val_days", type=int, default=60, help="Validation window size (days) as last N days")
    p.add_argument("--label_col", type=str, default="meter_reading", help="Target column")
    p.add_argument(
        "--use_log1p_target",
        action="store_true",
        default=True,
        help="Train on log1p(y); RMSE on log1p equals RMSLE on original scale",
    )

    # Feature selection / identity removal
    p.add_argument("--drop_site_id", action="store_true", default=True)
    p.add_argument("--drop_building_id", action="store_true", default=True)

    # Expert features
    p.add_argument(
        "--use_expert_features",
        action="store_true",
        default=False,
        help="If set, calls expert_features.extract_expert_features(df) and concatenates",
    )

    # LightGBM knobs (configurable)
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
        "--log_tag", type=str, default="lgbm_run", help="TensorBoard run subdir name (used under out_dir/tb_lgbm/)"
    )

    return p.parse_args()


def _sanitize_run_name(s: str) -> str:
    # Keep it filesystem + TB friendly
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:180]  # avoid ridiculously long paths


def build_default_run_name(args: argparse.Namespace) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [
        "lgbm",
        stamp,
        f"lr{args.learning_rate:g}",
        f"L{args.num_leaves}",
        f"minleaf{args.min_data_in_leaf}",
        "exp" if args.use_expert_features else "base",
    ]
    base = "_".join(parts)
    if args.log_tag:
        base = f"{base}__{args.log_tag}"
    return _sanitize_run_name(base)


def choose_feature_columns(all_cols: List[str]) -> List[str]:
    """
    Minimal, explainable baseline: use raw weather/building meta + your engineered features.
    Exclude label, identifiers, and timestamps later via drop_cols.

    Adjust here if you want stricter control; this keeps it simple.
    """
    # Preferred baseline engineered feature set (from your approved list)
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

    # Raw columns to include (not “engineered”; just the measured/context fields)
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

    keep = [c for c in (raw + engineered) if c in all_cols]
    return keep


def main() -> None:
    import pickle
    from pathlib import Path
    from typing import List
    import time
    import numpy as np
    import pandas as pd
    import lightgbm as lgb

    args = parse_args()
    utility.common_set_seed(args.seed)

    out_dir = Path(args.out_dir)
    data_path = Path(args.data_parquet)
    model_out_dir = Path(args.out_dir + f"/models/{build_default_run_name(args)}")
    utility.common_ensure_dir(model_out_dir)

    run_name = build_default_run_name(args)
    tb_root = out_dir / "tb_lgbm"
    tb_dir = tb_root / run_name
    utility.common_ensure_dir(tb_dir)

    # Columns needed for training + split + categoricals + label
    base_needed = [
        args.label_col,
        "timestamp",
        "building_id",
        "site_id",
    ]
    df_min = utility.common_load_parquet_dataset(data_path, columns=base_needed, engine="pyarrow")
    all_cols = list(df_min.columns)

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

    load_cols = sorted(set(base_needed + ["timestamp"] + keep_cols))
    if args.use_expert_features:
        df = utility.common_load_parquet_dataset(data_path, engine="pyarrow")
    else:
        df = utility.common_load_parquet_dataset(data_path, columns=load_cols, engine="pyarrow")

    keep_cols = [c for c in keep_cols if c in df.columns]

    drop_cols = ["timestamp"]
    if args.drop_building_id and "building_id" in df.columns:
        drop_cols.append("building_id")
    if args.drop_site_id and "site_id" in df.columns:
        drop_cols.append("site_id")

    categorical_cols = ["primary_use", "meter", "dayofweek"]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    # -------------------------------------------------------------------------
    # Expert feature extraction (BEFORE split): append engineered features to df.
    # Also: persist the expert cache alongside the trained model.
    # -------------------------------------------------------------------------
    expert_cols: List[str] = []
    expert_cache: dict = {}  # <- will be filled by extract_expert_features(...)
    expert_cache_path: Path | None = None

    if args.use_expert_features:
        print("[Expert] extracting expert features (full dataset -> engineered dataframe) ...")
        expert_df = extract_expert_features(
            df, cache=expert_cache
        )  # <- key change vs current pipeline :contentReference[oaicite:0]{index=0}

        if expert_df is None:
            print("[Expert] extractor returned None")
            raise ValueError("Expert feature extractor returned None; expected a DataFrame with expert features.")

        if not isinstance(expert_df, pd.DataFrame):
            expert_df = pd.DataFrame(expert_df)

        if len(expert_df) != len(df):
            raise ValueError(f"Expert feature rows mismatch: {len(expert_df)} vs {len(df)}")

        if not expert_df.index.equals(df.index):
            expert_df = expert_df.reset_index(drop=True)
            df = df.reset_index(drop=True)

        forbidden = {args.label_col, "timestamp", "building_id", "site_id"}
        cols_to_drop = [c for c in expert_df.columns if c in forbidden]
        if cols_to_drop:
            expert_df = expert_df.drop(columns=cols_to_drop)

        obj_cols = [c for c in expert_df.columns if expert_df[c].dtype == object]
        if obj_cols:
            expert_df[obj_cols] = expert_df[obj_cols].astype("category")
            categorical_cols = list(dict.fromkeys(categorical_cols + obj_cols))

        existing_cols = set(df.columns)
        rename_map = {}
        used_new = set()
        for c in expert_df.columns:
            new_c = c
            if new_c in existing_cols or new_c in used_new:
                base = f"expert__{c}"
                new_c = base
                k = 2
                while new_c in existing_cols or new_c in used_new:
                    new_c = f"{base}__{k}"
                    k += 1
                rename_map[c] = new_c
            used_new.add(new_c)

        if rename_map:
            expert_df = expert_df.rename(columns=rename_map)
            print(f"[Expert] renamed {len(rename_map)} colliding expert columns (prefixed with 'expert__').")

        expert_cols = list(expert_df.columns)
        df = pd.concat([df, expert_df], axis=1)
        print(
            f"[Expert] appended {len(expert_cols)} expert feature columns. total_cols={df.shape[1]}"
        )  # :contentReference[oaicite:1]{index=1}

    # Time split
    train_mask, val_mask, cutoff = utility.common_time_split(df, "timestamp", args.val_days)
    print(f"[Split] cutoff={cutoff}  train_rows={train_mask.sum()}  val_rows={val_mask.sum()}")

    # Target transform
    y = df[args.label_col].to_numpy()
    if args.use_log1p_target:
        y = np.log1p(y).astype(np.float32)

    # Prepare X/y (baseline + expert features)
    feature_cols = keep_cols + expert_cols
    df_model = df[feature_cols + [args.label_col] + drop_cols].copy()
    df_model[args.label_col] = y

    X_all, y_all, cat_present = utility.tree_prepare_tabular_matrices(
        df_model,
        label_col=args.label_col,
        drop_cols=drop_cols,
        categorical_cols=categorical_cols,
    )

    X_train = X_all.iloc[train_mask].copy()
    y_train = y_all[train_mask]
    X_val = X_all.iloc[val_mask].copy()
    y_val = y_all[val_mask]

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_present, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_present, reference=dtrain, free_raw_data=False)

    params = {
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
    if args.device_type == "gpu" and args.max_bin > 63:
        print("[Warn] GPU mode usually benefits from smaller --max_bin (e.g. 63).")

    evals_result = {}

    print("[Train] starting LightGBM training...")
    t0 = time.time()
    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=args.n_estimators,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.record_evaluation(evals_result),
            lgb.log_evaluation(period=args.log_period),
            lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=True),
        ],
    )
    dt = time.time() - t0
    print(f"[Train] done in {dt:.1f}s  best_iter={booster.best_iteration}")

    # Save model
    model_path = model_out_dir / "lgbm_model.txt"
    booster.save_model(str(model_path))
    print("[Save] model:", model_path)  # :contentReference[oaicite:2]{index=2}

    # NEW: save expert cache (pickle) next to model so inference can restore it as a dict
    if args.use_expert_features:
        expert_cache_path = model_out_dir / "expert_feature_cache.pkl"
        with open(expert_cache_path, "wb") as f:
            pickle.dump(expert_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("[Save] expert cache:", expert_cache_path)

    utility.common_save_json({"params": params, **vars(args)}, model_out_dir / "train_args_and_params.json")
    utility.common_save_json(evals_result, model_out_dir / "evals_result.json")

    utility.tree_write_tensorboard_evals(evals_result, tb_dir)
    print("[TB] wrote scalars to:", tb_dir)

    pred_val = booster.predict(X_val, num_iteration=booster.best_iteration)
    rmse_log = float(np.sqrt(np.mean((pred_val - y_val) ** 2)))
    print(f"[Val] RMSE(log1p)={rmse_log:.6f} (equals RMSLE on original scale if trained on log1p)")

    if args.use_log1p_target:
        y_val_orig = np.expm1(y_val)
        pred_val_orig = np.expm1(pred_val)
        rmsle = float(np.sqrt(np.mean((np.log1p(pred_val_orig) - np.log1p(y_val_orig)) ** 2)))
        print(f"[Val] RMSLE(original)={rmsle:.6f}")

    feature_list = list(X_train.columns)
    manifest = {"feature_columns": feature_list, "categorical_columns": cat_present}
    if expert_cache_path is not None:
        manifest["expert_cache_file"] = expert_cache_path.name
    utility.common_save_json(manifest, model_out_dir / "feature_manifest.json")
    print("[Save] feature manifest:", model_out_dir / "feature_manifest.json")


if __name__ == "__main__":
    main()
