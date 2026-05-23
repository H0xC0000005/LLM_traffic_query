from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import List
from datetime import datetime
import re
import numpy as np
import pandas as pd
import lightgbm as lgb

import utility
from supplementary_feature_registry import (
    get_supplementary_extractor,
    list_supplementary_extractors,
    normalize_supplementary_encoder_name,
)


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
    p.add_argument(
        "--keep_site_id", action="store_true", default=False, help="Include site_id as a categorical feature"
    )
    p.add_argument(
        "--keep_building_id", action="store_true", default=False, help="Include building_id as a categorical feature"
    )
    p.add_argument("--drop_site_id", action="store_true", default=False, help=argparse.SUPPRESS)
    p.add_argument("--drop_building_id", action="store_true", default=False, help=argparse.SUPPRESS)

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

    args = p.parse_args()
    if args.use_expert_features and args.supplementary_encoder == "none":
        args.supplementary_encoder = "expert"
    args.supplementary_encoder = normalize_supplementary_encoder_name(args.supplementary_encoder)
    return args


def _sanitize_run_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:180]


def build_default_run_name(args: argparse.Namespace) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [
        "lgbm",
        stamp,
        f"lr{args.learning_rate:g}",
        f"L{args.num_leaves}",
        f"minleaf{args.min_data_in_leaf}",
        args.supplementary_encoder,
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


def main() -> None:
    args = parse_args()
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

    train_mask, val_mask, cutoff = utility.common_time_split(df, "timestamp", args.val_days)
    print(f"[Split] cutoff={cutoff}  train_rows={train_mask.sum()}  val_rows={val_mask.sum()}")

    supplementary_cols: List[str] = []
    if args.supplementary_encoder != "none":
        print(f"[Supplementary] encoder={args.supplementary_encoder} (fit on train, transform on val)...")
        extractor = get_supplementary_extractor(args.supplementary_encoder)
        encoder_cache: dict = {
            "encoder_name": args.supplementary_encoder,
            "label_col": args.label_col,
            "exclude_building_id": bool(args.supplementary_exclude_building_id),
        }

        train_index = df.index[train_mask]
        val_index = df.index[val_mask]

        sup_train_input = df.loc[train_index].copy()
        if args.supplementary_exclude_building_id and "building_id" in sup_train_input.columns:
            sup_train_input = sup_train_input.drop(columns=["building_id"])

        sup_val_input = df.loc[val_index].copy()
        if args.label_col in sup_val_input.columns:
            sup_val_input = sup_val_input.drop(columns=[args.label_col])
        if args.supplementary_exclude_building_id and "building_id" in sup_val_input.columns:
            sup_val_input = sup_val_input.drop(columns=["building_id"])

        sup_train_df = _coerce_supplementary_df(
            extractor(
                sup_train_input,
                cache=encoder_cache,
                mode="fit_transform",
                label_col=args.label_col,
                exclude_building_id=args.supplementary_exclude_building_id,
            ),
            expected_rows=len(train_index),
            label_col=args.label_col,
        )
        sup_val_df = _coerce_supplementary_df(
            extractor(
                sup_val_input,
                cache=encoder_cache,
                mode="transform",
                label_col=args.label_col,
                exclude_building_id=args.supplementary_exclude_building_id,
            ),
            expected_rows=len(val_index),
            label_col=args.label_col,
        )

        if list(sup_train_df.columns) != list(sup_val_df.columns):
            missing_in_val = sorted(set(sup_train_df.columns) - set(sup_val_df.columns))
            missing_in_train = sorted(set(sup_val_df.columns) - set(sup_train_df.columns))
            raise ValueError(
                "Supplementary feature columns differ between train-fit and val-transform. "
                f"missing_in_val={missing_in_val[:20]} "
                f"missing_in_train={missing_in_train[:20]}"
            )

        existing_cols = set(df.columns)
        rename_map = {}
        used_new = set()
        for c in sup_train_df.columns:
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
            sup_train_df = sup_train_df.rename(columns=rename_map)
            sup_val_df = sup_val_df.rename(columns=rename_map)
            print(f"[Supplementary] renamed {len(rename_map)} colliding columns (prefixed with 'supp__').")

        sup_train_df.index = train_index
        sup_val_df.index = val_index
        supplementary_df = pd.concat([sup_train_df, sup_val_df], axis=0).reindex(df.index)

        obj_cols = [c for c in supplementary_df.columns if supplementary_df[c].dtype == object]
        if obj_cols:
            supplementary_df[obj_cols] = supplementary_df[obj_cols].astype("category")
            categorical_cols = list(dict.fromkeys(categorical_cols + obj_cols))

        supplementary_cols = list(supplementary_df.columns)
        df = pd.concat([df, supplementary_df], axis=1)
        print(f"[Supplementary] appended {len(supplementary_cols)} columns. total_cols={df.shape[1]}")

        cache_path = model_out_dir / f"{args.supplementary_encoder}_feature_cache.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(encoder_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Supplementary] saved cache: {cache_path}")
        pass

    y = df[args.label_col].to_numpy()
    if args.use_log1p_target:
        y = np.log1p(y).astype(np.float32)

    feature_cols = keep_cols + supplementary_cols
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

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=cat_present,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_val,
        label=y_val,
        categorical_feature=cat_present,
        reference=dtrain,
        free_raw_data=False,
    )

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

    model_path = model_out_dir / "lgbm_model.txt"
    booster.save_model(str(model_path))
    print("[Save] model:", model_path)

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
    manifest = {
        "feature_columns": feature_list,
        "categorical_columns": cat_present,
        "supplementary_encoder": args.supplementary_encoder,
        "supplementary_exclude_building_id": bool(args.supplementary_exclude_building_id),
    }
    if args.supplementary_encoder != "none":
        manifest["supplementary_cache_file"] = f"{args.supplementary_encoder}_feature_cache.pkl"
    utility.common_save_json(manifest, model_out_dir / "feature_manifest.json")
    print("[Save] feature manifest:", model_out_dir / "feature_manifest.json")


if __name__ == "__main__":
    main()
