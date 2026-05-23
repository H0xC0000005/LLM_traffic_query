from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from expert_feature_extractor import extract_expert_features

_VALID_MODES = {"fit_transform", "transform"}


def _ensure_feature_df(feature_df: pd.DataFrame | np.ndarray, expected_rows: int) -> pd.DataFrame:
    if feature_df is None:
        raise ValueError("Supplementary extractor returned None.")
    if not isinstance(feature_df, pd.DataFrame):
        feature_df = pd.DataFrame(feature_df)
    if len(feature_df) != expected_rows:
        raise ValueError(f"Supplementary feature rows mismatch: {len(feature_df)} vs {expected_rows}")
    return feature_df.reset_index(drop=True)


def extract_expert_supplementary_features(
    df: pd.DataFrame,
    *,
    cache: dict | None = None,
    mode: str = "transform",
    label_col: str = "meter_reading",
    freeze_cache_during_transform: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Thin wrapper over the existing expert extractor with an explicit mode flag.

    Contract:
      - mode='fit_transform': df may include label_col and cache will be populated.
      - mode='transform': label_col is removed before feature extraction to prevent
        any target-derived block from fitting on validation/test rows.

    The returned DataFrame contains supplementary features only.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Valid modes: {sorted(_VALID_MODES)}")

    if cache is None:
        cache = {}

    meta = cache.setdefault("__supplementary_encoder_meta__", {})
    meta.setdefault("encoder_name", "expert")

    work_df = df.copy()
    if mode == "transform" and label_col in work_df.columns:
        work_df = work_df.drop(columns=[label_col])

    inner_cache = deepcopy(cache) if (mode == "transform" and freeze_cache_during_transform) else cache
    feature_df = extract_expert_features(work_df, cache=inner_cache, **kwargs)
    feature_df = _ensure_feature_df(feature_df, expected_rows=len(work_df))

    cols = list(feature_df.columns)
    if mode == "fit_transform":
        meta["feature_columns"] = cols
    else:
        expected_cols = meta.get("feature_columns")
        if expected_cols is not None and cols != expected_cols:
            raise ValueError(
                "Expert supplementary feature columns differ between fit and transform. "
                f"fit_cols={expected_cols[:20]} transform_cols={cols[:20]}"
            )
        meta.setdefault("feature_columns", cols)

    return feature_df
