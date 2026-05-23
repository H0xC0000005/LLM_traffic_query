from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# Common wheels (tree + NN)
# =========================


def common_set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def common_ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def common_save_json(obj: dict, path: Path) -> None:
    common_ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def common_load_parquet_dataset(
    parquet_path: Path,
    columns: Optional[List[str]] = None,
    engine: str = "pyarrow",
) -> pd.DataFrame:
    """
    Loads a (possibly partitioned) Parquet dataset directory or file.
    Use `columns=` aggressively to reduce memory.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet path not found: {parquet_path}")
    return pd.read_parquet(parquet_path, columns=columns, engine=engine)


def common_time_split(
    df: pd.DataFrame,
    timestamp_col: str,
    val_days: int,
) -> Tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    """
    Simple and effective: time-based split by cutoff = max_ts - val_days.
    Returns boolean masks (train_mask, val_mask) and cutoff timestamp.
    """
    if timestamp_col not in df.columns:
        raise KeyError(f"Missing timestamp_col: {timestamp_col}")

    ts = df[timestamp_col]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise TypeError(f"{timestamp_col} must be datetime64 dtype")

    max_ts = ts.max()
    cutoff = max_ts - pd.Timedelta(days=int(val_days))

    train_mask = (ts < cutoff).to_numpy()
    val_mask = (ts >= cutoff).to_numpy()

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise ValueError(
            f"Invalid split: train={train_mask.sum()} val={val_mask.sum()} "
            f"(cutoff={cutoff}, max_ts={max_ts}, val_days={val_days})"
        )
    return train_mask, val_mask, cutoff


def common_concat_expert_features(
    X: np.ndarray,
    expert_feat: Optional[np.ndarray],
) -> np.ndarray:
    """
    Reserved spot: concatenate expert features to the base feature matrix.
    expert_feat must be (N, K) numeric array aligned row-wise with X.
    """
    if expert_feat is None:
        return X
    if expert_feat.ndim != 2:
        raise ValueError(f"expert_feat must be 2D, got shape {expert_feat.shape}")
    if expert_feat.shape[0] != X.shape[0]:
        raise ValueError(f"Row mismatch: X={X.shape}, expert_feat={expert_feat.shape}")
    return np.concatenate([X, expert_feat], axis=1)


# =========================
# Tree-only wheels
# =========================


@dataclass
class TreeDataBundle:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    categorical_cols: List[str]


def tree_prepare_tabular_matrices(
    df: pd.DataFrame,
    label_col: str,
    drop_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Prepare X (as pandas DataFrame) and y (as numpy).
    - Drops drop_cols.
    - Ensures categorical cols are pandas 'category' dtype (LightGBM native categorical).
    Returns X, y, categorical_cols_present.
    """
    if label_col not in df.columns:
        raise KeyError(f"Missing label_col: {label_col}")

    y = df[label_col].to_numpy()

    X = df.drop(columns=[label_col] + [c for c in drop_cols if c in df.columns], errors="ignore").copy()

    cat_present = [c for c in categorical_cols if c in X.columns]
    for c in cat_present:
        # LightGBM expects integer codes for categorical. pandas 'category' works.
        if not pd.api.types.is_categorical_dtype(X[c]):
            X[c] = X[c].astype("category")

    # Safety: ensure no object dtype (LightGBM dislikes object)
    obj_cols = [c for c in X.columns if X[c].dtype == "object"]
    if obj_cols:
        raise TypeError(f"Object dtype columns found: {obj_cols}. Convert them to category or numeric.")

    return X, y, cat_present


def tree_write_tensorboard_evals(
    evals_result: Dict[str, Dict[str, List[float]]],
    log_dir: Path,
) -> None:
    """
    Writes LightGBM eval curves to TensorBoard scalars.
    evals_result format from lightgbm.record_evaluation:
      {'training': {'rmse': [..]}, 'valid_1': {'rmse': [..]}}
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        # If torch is not available, fall back to tensorboardX if installed
        try:
            from tensorboardX import SummaryWriter  # type: ignore
        except Exception as e:
            raise ImportError(
                "TensorBoard logging requires either torch (torch.utils.tensorboard) " "or tensorboardX installed."
            ) from e

    common_ensure_dir(log_dir)
    writer = SummaryWriter(log_dir=str(log_dir))

    try:
        for data_name, metrics in evals_result.items():
            for metric_name, values in metrics.items():
                tag = f"{data_name}/{metric_name}"
                for step, v in enumerate(values):
                    writer.add_scalar(tag, float(v), step)
    finally:
        writer.flush()
        writer.close()
