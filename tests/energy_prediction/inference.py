from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

import utility
from expert_feature_extractor import extract_expert_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Inference for ASHRAE GEPIII LightGBM model (baseline + expert features)")

    # IO
    p.add_argument("--data_parquet", type=str, required=True, help="Path to engineered TEST parquet dataset dir/file")
    p.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing lgbm_model.txt and (optionally) feature_manifest.json + train_args_and_params.json",
    )
    p.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Output submission CSV path (must contain columns: row_id, meter_reading)",
    )

    # Columns
    p.add_argument("--row_id_col", type=str, default="row_id")
    p.add_argument("--label_col", type=str, default="meter_reading")

    # Expert features
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--use_expert_features",
        action="store_true",
        default=True,
        help="Compute expert features with extract_expert_features(df) and append before prediction (default: on)",
    )
    g.add_argument(
        "--no_expert_features",
        dest="use_expert_features",
        action="store_false",
        help="Disable expert feature extraction (only use features already in dataset)",
    )

    # Target inverse-transform
    t = p.add_mutually_exclusive_group()
    t.add_argument(
        "--use_log1p_target",
        action="store_true",
        default=None,
        help="Model predicts log1p(y); apply expm1 before saving submission (default: auto from train_args_and_params.json)",
    )
    t.add_argument(
        "--no_log1p_target",
        dest="use_log1p_target",
        action="store_false",
        default=None,
        help="Model predicts y directly; do not apply expm1 (default: auto from train_args_and_params.json)",
    )

    # Prediction knobs
    p.add_argument(
        "--num_iteration",
        type=int,
        default=-1,
        help="Number of boosting iterations to use. -1 means: use model.best_iteration if available else all.",
    )
    p.add_argument("--clip_negative", action="store_true", default=True, help="Clip predictions to >=0 (default: on)")

    # Misc
    p.add_argument("--engine", type=str, default="pyarrow", choices=["pyarrow", "fastparquet"])
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def _load_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_expected_features(model_dir: Path, booster: lgb.Booster) -> Tuple[List[str], List[str]]:
    """
    Returns (feature_columns, categorical_columns).

    Priority:
      1) model_dir/feature_manifest.json (written by your training pipeline)
      2) booster.feature_name() with empty categorical list
    """
    manifest_path = model_dir / "feature_manifest.json"
    manifest = _load_json_if_exists(manifest_path)
    if manifest is not None:
        feat_cols = manifest.get("feature_columns")
        cat_cols = manifest.get("categorical_columns", [])
        if not isinstance(feat_cols, list) or not all(isinstance(x, str) for x in feat_cols):
            raise ValueError(f"Invalid feature_manifest.json: feature_columns missing/invalid at {manifest_path}")
        if not isinstance(cat_cols, list) or not all(isinstance(x, str) for x in cat_cols):
            raise ValueError(f"Invalid feature_manifest.json: categorical_columns invalid at {manifest_path}")
        return feat_cols, cat_cols

    # Fallback
    return list(booster.feature_name()), []


def _resolve_use_log1p(model_dir: Path, cli_value: Optional[bool]) -> bool:
    """
    If user passed --use_log1p_target / --no_log1p_target, obey it.
    Otherwise try to read model_dir/train_args_and_params.json and use its 'use_log1p_target' value.
    Fallback: False.
    """
    if cli_value is not None:
        return bool(cli_value)

    train_args_path = model_dir / "train_args_and_params.json"
    train_args = _load_json_if_exists(train_args_path)
    if train_args is not None and "use_log1p_target" in train_args:
        return bool(train_args["use_log1p_target"])

    # Conservative default: no inverse transform if unknown.
    return False


def _attach_expert_features(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Mirrors the training-time expert feature appending logic from train_tree_lgbm.py:
      - extract_expert_features(df) -> expert_df (DataFrame-like)
      - drop forbidden cols
      - cast object -> category
      - rename colliding expert columns deterministically with 'expert__' prefix
      - concat to df (column-wise)
    """
    expert_df = extract_expert_features(df)
    if expert_df is None:
        raise ValueError("Expert feature extractor returned None; expected a DataFrame with expert features.")
    if not isinstance(expert_df, pd.DataFrame):
        expert_df = pd.DataFrame(expert_df)

    if len(expert_df) != len(df):
        raise ValueError(f"Expert feature rows mismatch: {len(expert_df)} vs {len(df)}")

    # Ensure positional alignment if indices differ.
    if not expert_df.index.equals(df.index):
        expert_df = expert_df.reset_index(drop=True)
        df = df.reset_index(drop=True)

    forbidden = {label_col, "timestamp", "building_id", "site_id"}
    cols_to_drop = [c for c in expert_df.columns if c in forbidden]
    if cols_to_drop:
        expert_df = expert_df.drop(columns=cols_to_drop)

    # Treat object cols as categoricals (LightGBM compatibility)
    obj_cols = [c for c in expert_df.columns if expert_df[c].dtype == object]
    if obj_cols:
        expert_df[obj_cols] = expert_df[obj_cols].astype("category")

    # Avoid name collisions with existing df columns (deterministic)
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

    return pd.concat([df, expert_df], axis=1)


def _build_feature_matrix(df: pd.DataFrame, feature_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    """
    Build X with columns exactly matching `feature_cols` in the same order.
    Missing columns are created as NaN.
    Ensures categorical cols are pandas 'category' dtype.
    """
    X = pd.DataFrame(index=df.index)

    # Add columns in the exact order
    for c in feature_cols:
        if c in df.columns:
            X[c] = df[c]
        else:
            X[c] = np.nan

    # Convert categoricals to category dtype
    cat_present = [c for c in categorical_cols if c in X.columns]
    for c in cat_present:
        if not pd.api.types.is_categorical_dtype(X[c]):
            X[c] = X[c].astype("category")

    # Safety: convert remaining object columns to category.
    obj_cols = [c for c in X.columns if X[c].dtype == "object"]
    if obj_cols:
        X[obj_cols] = X[obj_cols].astype("category")

    return X


def main() -> None:
    args = parse_args()
    utility.common_set_seed(args.seed)

    data_path = Path(args.data_parquet)
    model_dir = Path(args.model_dir)
    out_csv = Path(args.out_csv)
    utility.common_ensure_dir(out_csv.parent)

    model_path = model_dir / "lgbm_model.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    booster = lgb.Booster(model_file=str(model_path))

    # Expected feature schema from training
    feature_cols, categorical_cols = _resolve_expected_features(model_dir, booster)

    # Whether to apply expm1
    use_log1p = _resolve_use_log1p(model_dir, args.use_log1p_target)

    # Load test dataset.
    # If expert features are enabled, load full dataset so the extractor can access everything.
    if args.use_expert_features:
        df_test = utility.common_load_parquet_dataset(data_path, columns=None, engine=args.engine)
    else:
        needed = sorted(set(feature_cols + [args.row_id_col]))
        df_test = utility.common_load_parquet_dataset(data_path, columns=needed, engine=args.engine)

    if args.row_id_col not in df_test.columns:
        raise KeyError(
            f"Missing {args.row_id_col} in test dataset. "
            f"Ensure your test preprocessing keeps row_id for Kaggle submission alignment."
        )

    # Attach expert features
    if args.use_expert_features:
        df_test = _attach_expert_features(df_test, label_col=args.label_col)

    # Build model matrix matching training column order
    X_test = _build_feature_matrix(df_test, feature_cols=feature_cols, categorical_cols=categorical_cols)

    # Choose iteration
    num_iter: Optional[int]
    if args.num_iteration is not None and args.num_iteration > 0:
        num_iter = int(args.num_iteration)
    else:
        best_iter = getattr(booster, "best_iteration", 0) or 0
        num_iter = int(best_iter) if best_iter > 0 else None  # None => all trees

    # Predict
    pred = booster.predict(X_test, num_iteration=num_iter)

    if use_log1p:
        pred = np.expm1(pred)

    if args.clip_negative:
        pred = np.maximum(pred, 0.0)

    # Kaggle submission format
    sub = pd.DataFrame(
        {
            args.row_id_col: df_test[args.row_id_col].astype(np.int64),
            args.label_col: pred.astype(np.float64),
        }
    ).sort_values(args.row_id_col)

    sub.to_csv(out_csv, index=False)
    print(
        f"[OK] wrote submission: {out_csv}  rows={len(sub)}  "
        f"use_expert={args.use_expert_features}  log1p={use_log1p}"
    )


if __name__ == "__main__":
    main()
