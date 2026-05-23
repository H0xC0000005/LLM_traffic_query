from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_VALID_MODES = {"fit_transform", "transform"}
_DEFAULT_WEATHER_COLS = [
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
]


def _validate_mode(mode: str) -> None:
    if mode not in _VALID_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Valid modes: {sorted(_VALID_MODES)}")


def _as_frame(feature_df: pd.DataFrame | np.ndarray, expected_rows: int) -> pd.DataFrame:
    if feature_df is None:
        raise ValueError("Supplementary extractor returned None.")
    if not isinstance(feature_df, pd.DataFrame):
        feature_df = pd.DataFrame(feature_df)
    if len(feature_df) != expected_rows:
        raise ValueError(f"Supplementary feature rows mismatch: {len(feature_df)} vs {expected_rows}")
    return feature_df.reset_index(drop=True)


def _site_series(df: pd.DataFrame, site_col: str) -> pd.Series:
    if site_col in df.columns:
        return df[site_col].astype(str)
    return pd.Series(["__global__"] * len(df), index=df.index)


def _fit_missingness_state(
    df: pd.DataFrame,
    *,
    site_col: str,
    weather_cols: list[str],
) -> dict[str, Any]:
    site_key = _site_series(df, site_col)
    state: dict[str, Any] = {"weather_cols": list(weather_cols)}

    # any_missing = pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)
    # for col in weather_cols:
    #     miss = df[col].isna().astype(np.float32)
    #     any_missing = np.maximum(any_missing.to_numpy(), miss.to_numpy()).astype(np.float32)

    #     work = pd.DataFrame({"site_key": site_key, "miss": miss})
    #     grp = work.groupby("site_key", sort=False)["miss"].mean()
    #     state[col] = {
    #         "site_missing_rate": {str(k): float(v) for k, v in grp.items()},
    #         "global_missing_rate": float(miss.mean()),
    #     }
    any_missing = np.zeros(len(df), dtype=np.float32)
    for col in weather_cols:
        miss = df[col].isna().astype(np.float32)
        miss_arr = miss.to_numpy(dtype=np.float32, copy=False)
        any_missing = np.maximum(any_missing, miss_arr)

        work = pd.DataFrame({"site_key": site_key, "miss": miss})
        grp = work.groupby("site_key", sort=False)["miss"].mean()
        state[col] = {
            "site_missing_rate": {str(k): float(v) for k, v in grp.items()},
            "global_missing_rate": float(miss.mean()),
        }

    any_missing_series = pd.Series(any_missing, index=df.index)
    grp_any = (
        pd.DataFrame({"site_key": site_key, "miss_any": any_missing_series})
        .groupby("site_key", sort=False)["miss_any"]
        .mean()
    )
    state["site_any_missing_rate"] = {str(k): float(v) for k, v in grp_any.items()}
    state["global_any_missing_rate"] = float(any_missing_series.mean())
    return state


def extract_rank25_missingness_features(
    df: pd.DataFrame,
    *,
    cache: dict | None = None,
    mode: str = "transform",
    label_col: str = "meter_reading",
    site_col: str = "site_id",
    weather_cols: list[str] | None = None,
    **_: Any,
) -> pd.DataFrame:
    """Approximate rank-25 supplementary block.

    Implemented as a missingness-aware weather encoder with:
      - per-column is_missing indicators,
      - row-level missing-count and missing-fraction features,
      - train-fitted site-level missing-rate priors.

    Notes:
      - This block assumes weather NaNs are still present in the dataframe that
        reaches the encoder. If earlier preprocessing fully imputes weather,
        these features may become nearly constant.
    """
    _validate_mode(mode)
    if cache is None:
        cache = {}

    meta = cache.setdefault("rank25_missingness", {})
    weather_cols = list(_DEFAULT_WEATHER_COLS if weather_cols is None else weather_cols)
    weather_cols = [c for c in weather_cols if c in df.columns]
    if not weather_cols:
        raise ValueError("rank25_missingness requires at least one weather column in df.")

    if mode == "fit_transform":
        meta["state"] = _fit_missingness_state(df, site_col=site_col, weather_cols=weather_cols)
        feature_columns = []
        for col in weather_cols:
            feature_columns.extend([f"r25_is_missing_{col}", f"r25_site_missing_rate_{col}"])
        feature_columns.extend(["r25_missing_weather_count", "r25_missing_weather_frac", "r25_site_any_missing_rate"])
        meta["feature_columns"] = feature_columns
    else:
        if "state" not in meta:
            raise ValueError("rank25_missingness cache is empty. Run mode='fit_transform' first.")
        weather_cols = [c for c in meta["state"].get("weather_cols", weather_cols) if c in df.columns]
        if not weather_cols:
            raise ValueError("rank25_missingness transform received none of the fitted weather columns.")

    state = meta["state"]
    site_key = _site_series(df, site_col)
    out = pd.DataFrame(index=df.index)

    miss_cols = []
    for col in weather_cols:
        miss = df[col].isna().astype(np.int8)
        out[f"r25_is_missing_{col}"] = miss
        miss_cols.append(f"r25_is_missing_{col}")
        site_rate = site_key.map(state[col]["site_missing_rate"]).fillna(state[col]["global_missing_rate"])
        out[f"r25_site_missing_rate_{col}"] = site_rate.to_numpy(dtype=np.float32, copy=False)

    miss_matrix = out[miss_cols].to_numpy(dtype=np.float32, copy=False)
    out["r25_missing_weather_count"] = miss_matrix.sum(axis=1).astype(np.float32)
    out["r25_missing_weather_frac"] = (miss_matrix.mean(axis=1)).astype(np.float32)
    out["r25_site_any_missing_rate"] = (
        site_key.map(state["site_any_missing_rate"])
        .fillna(state["global_any_missing_rate"])
        .to_numpy(dtype=np.float32, copy=False)
    )

    out = _as_frame(out, expected_rows=len(df))
    expected_cols = meta.get("feature_columns")
    if expected_cols is not None and list(out.columns) != expected_cols:
        raise ValueError(
            "rank25_missingness feature columns differ from cached fit columns. "
            f"fit_cols={expected_cols[:20]} transform_cols={list(out.columns)[:20]}"
        )
    return out
