from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_VALID_MODES = {"fit_transform", "transform"}
_DEFAULT_WEATHER_COLS = [
    "air_temperature",
    "dew_temperature",
    "sea_level_pressure",
    "wind_speed",
    "cloud_coverage",
    "precip_depth_1_hr",
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


def _fit_site_weather_stats(
    df: pd.DataFrame,
    *,
    site_col: str,
    weather_cols: list[str],
) -> dict[str, Any]:
    site_key = _site_series(df, site_col)
    state: dict[str, Any] = {"weather_cols": list(weather_cols)}

    for col in weather_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        work = pd.DataFrame({"site_key": site_key, col: values}).dropna(subset=[col])
        if work.empty:
            state[col] = {
                "mean": {},
                "std": {},
                "min": {},
                "max": {},
                "global_mean": 0.0,
                "global_std": 1.0,
                "global_min": 0.0,
                "global_max": 0.0,
            }
            continue

        grp = work.groupby("site_key", sort=False)[col]
        means = grp.mean()
        stds = grp.std(ddof=0).replace(0, np.nan)
        mins = grp.min()
        maxs = grp.max()

        global_std = float(work[col].std(ddof=0)) if len(work) > 1 else 1.0
        if not np.isfinite(global_std) or global_std <= 0:
            global_std = 1.0

        state[col] = {
            "mean": {str(k): float(v) for k, v in means.items()},
            "std": {str(k): float(v) for k, v in stds.fillna(global_std).items()},
            "min": {str(k): float(v) for k, v in mins.items()},
            "max": {str(k): float(v) for k, v in maxs.items()},
            "global_mean": float(work[col].mean()),
            "global_std": global_std,
            "global_min": float(work[col].min()),
            "global_max": float(work[col].max()),
        }
    return state


def _map_site_stat(site_key: pd.Series, mapping: dict[str, float], default: float) -> np.ndarray:
    return site_key.map(mapping).fillna(default).to_numpy(dtype=np.float32, copy=False)


def extract_rank2_weather_stats_features(
    df: pd.DataFrame,
    *,
    cache: dict | None = None,
    mode: str = "transform",
    label_col: str = "meter_reading",
    site_col: str = "site_id",
    weather_cols: list[str] | None = None,
    **_: Any,
) -> pd.DataFrame:
    """Approximate rank-2 supplementary block.

    Implemented as a train-fitted weather-statistics encoder using simple per-site
    descriptive statistics of weather variables and anomaly features:
      - per-site mean and std,
      - row-wise z-score relative to site climate,
      - row-wise normalized position within site min/max range.

    This block intentionally avoids lags so it stays distinct from the proposed
    rank-4 lag/rolling block.
    """
    _validate_mode(mode)
    if cache is None:
        cache = {}

    meta = cache.setdefault("rank2_weather_stats", {})
    weather_cols = list(_DEFAULT_WEATHER_COLS if weather_cols is None else weather_cols)
    weather_cols = [c for c in weather_cols if c in df.columns]
    if not weather_cols:
        raise ValueError("rank2_weather_stats requires at least one weather column in df.")

    if mode == "fit_transform":
        meta["state"] = _fit_site_weather_stats(df, site_col=site_col, weather_cols=weather_cols)
        feature_columns = []
        for col in weather_cols:
            feature_columns.extend(
                [
                    f"r2_{col}_site_mean",
                    f"r2_{col}_site_std",
                    f"r2_{col}_site_z",
                    f"r2_{col}_site_pos",
                ]
            )
        meta["feature_columns"] = feature_columns
    else:
        if "state" not in meta:
            raise ValueError("rank2_weather_stats cache is empty. Run mode='fit_transform' first.")
        weather_cols = [c for c in meta["state"].get("weather_cols", weather_cols) if c in df.columns]
        if not weather_cols:
            raise ValueError("rank2_weather_stats transform received none of the fitted weather columns.")

    state = meta["state"]
    site_key = _site_series(df, site_col)
    out = pd.DataFrame(index=df.index)

    for col in weather_cols:
        stats = state[col]
        raw = pd.to_numeric(df[col], errors="coerce")
        site_mean = _map_site_stat(site_key, stats["mean"], stats["global_mean"])
        site_std = _map_site_stat(site_key, stats["std"], stats["global_std"])
        site_min = _map_site_stat(site_key, stats["min"], stats["global_min"])
        site_max = _map_site_stat(site_key, stats["max"], stats["global_max"])

        value = raw.fillna(stats["global_mean"]).to_numpy(dtype=np.float32, copy=False)
        denom_std = np.maximum(site_std.astype(np.float32), np.float32(1e-6))
        denom_rng = np.maximum((site_max - site_min).astype(np.float32), np.float32(1e-6))

        out[f"r2_{col}_site_mean"] = site_mean.astype(np.float32)
        out[f"r2_{col}_site_std"] = site_std.astype(np.float32)
        out[f"r2_{col}_site_z"] = ((value - site_mean) / denom_std).astype(np.float32)
        out[f"r2_{col}_site_pos"] = ((value - site_min) / denom_rng).astype(np.float32)

    out = _as_frame(out, expected_rows=len(df))
    expected_cols = meta.get("feature_columns")
    if expected_cols is not None and list(out.columns) != expected_cols:
        raise ValueError(
            "rank2_weather_stats feature columns differ from cached fit columns. "
            f"fit_cols={expected_cols[:20]} transform_cols={list(out.columns)[:20]}"
        )
    return out
