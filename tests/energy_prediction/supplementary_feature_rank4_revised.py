from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_VALID_MODES = {"fit_transform", "transform"}
_DEFAULT_LAG_WEATHER_COLS = [
    "air_temperature",
    "dew_temperature",
    "sea_level_pressure",
    "wind_speed",
]
_DEFAULT_AGG_WEATHER_COLS = [
    "air_temperature",
    "dew_temperature",
    "sea_level_pressure",
    "wind_speed",
    "cloud_coverage",
    "precip_depth_1_hr",
]
_DEFAULT_LAGS = [1, 3, 24]
_DEFAULT_WINDOWS = [3, 24]


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


def _key_join(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].astype(str).agg("||".join, axis=1)


def _prepare_weather_frame(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    site_col: str,
    weather_cols: list[str],
) -> pd.DataFrame:
    required = [c for c in [site_col, timestamp_col] if c in df.columns]
    if timestamp_col not in df.columns:
        raise KeyError(f"Missing required column '{timestamp_col}' for rank4 weather features.")
    cols = required + [c for c in weather_cols if c in df.columns]
    if site_col not in cols:
        temp = pd.DataFrame(index=df.index)
        temp[site_col] = "__global__"
        temp[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        for c in weather_cols:
            if c in df.columns:
                temp[c] = pd.to_numeric(df[c], errors="coerce")
        base = temp
    else:
        base = df[cols].copy()
        base[timestamp_col] = pd.to_datetime(base[timestamp_col], errors="coerce")
        for c in weather_cols:
            if c in base.columns:
                base[c] = pd.to_numeric(base[c], errors="coerce")

    weather = (
        base.drop_duplicates(subset=[site_col, timestamp_col], keep="first")
        .sort_values([site_col, timestamp_col])
        .reset_index(drop=True)
    )
    return weather


def _fit_aggregate_state(
    weather_df: pd.DataFrame,
    *,
    site_col: str,
    timestamp_col: str,
    weather_cols: list[str],
) -> dict[str, Any]:
    # OLD IMPLEMENTATION (kept for reference only)
    # work = weather_df[[site_col, timestamp_col] + weather_cols].copy()
    # ts = pd.to_datetime(work[timestamp_col], errors="coerce")
    # work["hour"] = ts.dt.hour.fillna(0).astype(np.int16)
    # work[site_col] = work[site_col].astype(str)
    # work["key"] = _key_join(work, [site_col, "hour"])
    #
    # state: dict[str, Any] = {"weather_cols": list(weather_cols)}
    # for col in weather_cols:
    #     sub = work[["key", col]].dropna(subset=[col])
    #     if sub.empty:
    #         state[col] = {
    #             "mean": {},
    #             "std": {},
    #             "global_mean": 0.0,
    #             "global_std": 1.0,
    #         }
    #         continue
    #     grp = sub.groupby("key", sort=False)[col]
    #     mean_map = grp.mean()
    #     std_map = grp.std(ddof=0).replace(0, np.nan)
    #     global_std = float(sub[col].std(ddof=0)) if len(sub) > 1 else 1.0
    #     if not np.isfinite(global_std) or global_std <= 0:
    #         global_std = 1.0
    #     state[col] = {
    #         "mean": {str(k): float(v) for k, v in mean_map.items()},
    #         "std": {str(k): float(v) for k, v in std_map.fillna(global_std).items()},
    #         "global_mean": float(sub[col].mean()),
    #         "global_std": global_std,
    #     }
    # return state

    # NEW IMPLEMENTATION:
    # Group directly on [site_col, hour] and keep a compact mapping DataFrame.
    work = weather_df[[site_col, timestamp_col] + weather_cols].copy()
    ts = pd.to_datetime(work[timestamp_col], errors="coerce")
    work["hour"] = ts.dt.hour.fillna(0).astype(np.int16)
    work[site_col] = work[site_col].astype(str)

    agg_df = work[[site_col, "hour"]].drop_duplicates(ignore_index=True)
    globals_state: dict[str, dict[str, float]] = {}

    for col in weather_cols:
        sub = work[[site_col, "hour", col]].dropna(subset=[col])
        if sub.empty:
            globals_state[col] = {"global_mean": 0.0, "global_std": 1.0}
            agg_df[f"r4_{col}_sitehour_mean"] = np.float32(0.0)
            agg_df[f"r4_{col}_sitehour_std"] = np.float32(1.0)
            continue

        grouped = sub.groupby([site_col, "hour"], sort=False)[col].agg(["mean", "std"]).reset_index()
        grouped["std"] = grouped["std"].fillna(0.0)
        global_std = float(sub[col].std(ddof=0)) if len(sub) > 1 else 1.0
        if not np.isfinite(global_std) or global_std <= 0:
            global_std = 1.0
        grouped["std"] = grouped["std"].replace(0, np.nan).fillna(global_std)
        grouped = grouped.rename(
            columns={
                "mean": f"r4_{col}_sitehour_mean",
                "std": f"r4_{col}_sitehour_std",
            }
        )
        agg_df = agg_df.merge(grouped, on=[site_col, "hour"], how="left", sort=False)
        globals_state[col] = {
            "global_mean": float(sub[col].mean()),
            "global_std": float(global_std),
        }

    return {
        "weather_cols": list(weather_cols),
        "agg_df": agg_df,
        "globals": globals_state,
    }


def _apply_aggregate_features(
    weather_df: pd.DataFrame,
    *,
    state: dict[str, Any],
    site_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    # OLD IMPLEMENTATION (kept for reference only)
    # out = weather_df[[site_col, timestamp_col]].copy()
    # ts = pd.to_datetime(out[timestamp_col], errors="coerce")
    # out["hour"] = ts.dt.hour.fillna(0).astype(np.int16)
    # out[site_col] = out[site_col].astype(str)
    # out["key"] = _key_join(out, [site_col, "hour"])
    # res = out[[site_col, timestamp_col]].copy()
    #
    # for col in state["weather_cols"]:
    #     raw = pd.to_numeric(weather_df[col], errors="coerce") if col in weather_df.columns else pd.Series(np.nan, index=weather_df.index)
    #     mean_map = state[col]["mean"]
    #     std_map = state[col]["std"]
    #     agg_mean = out["key"].map(mean_map).fillna(state[col]["global_mean"]).to_numpy(dtype=np.float32, copy=False)
    #     agg_std = out["key"].map(std_map).fillna(state[col]["global_std"]).to_numpy(dtype=np.float32, copy=False)
    #     value = raw.fillna(state[col]["global_mean"]).to_numpy(dtype=np.float32, copy=False)
    #     denom = np.maximum(agg_std, np.float32(1e-6))
    #     res[f"r4_{col}_sitehour_mean"] = agg_mean.astype(np.float32)
    #     res[f"r4_{col}_sitehour_std"] = agg_std.astype(np.float32)
    #     res[f"r4_{col}_sitehour_z"] = ((value - agg_mean) / denom).astype(np.float32)
    # return res

    # NEW IMPLEMENTATION:
    # Merge aggregate statistics back on [site_col, hour] instead of string-key mapping.
    keys = weather_df[[site_col, timestamp_col]].copy()
    ts = pd.to_datetime(keys[timestamp_col], errors="coerce")
    keys["hour"] = ts.dt.hour.fillna(0).astype(np.int16)
    keys[site_col] = keys[site_col].astype(str)

    agg_df = state["agg_df"]
    res = keys.merge(agg_df, on=[site_col, "hour"], how="left", sort=False)

    for col in state["weather_cols"]:
        g = state["globals"][col]
        mean_col = f"r4_{col}_sitehour_mean"
        std_col = f"r4_{col}_sitehour_std"
        raw = (
            pd.to_numeric(weather_df[col], errors="coerce")
            if col in weather_df.columns
            else pd.Series(np.nan, index=weather_df.index)
        )
        res[mean_col] = pd.to_numeric(res[mean_col], errors="coerce").fillna(g["global_mean"]).astype(np.float32)
        res[std_col] = pd.to_numeric(res[std_col], errors="coerce").fillna(g["global_std"]).astype(np.float32)
        value = raw.fillna(g["global_mean"]).to_numpy(dtype=np.float32, copy=False)
        denom = np.maximum(res[std_col].to_numpy(dtype=np.float32, copy=False), np.float32(1e-6))
        res[f"r4_{col}_sitehour_z"] = ((value - res[mean_col].to_numpy(dtype=np.float32, copy=False)) / denom).astype(
            np.float32
        )

    drop_cols = [c for c in [site_col, timestamp_col, "hour"] if c in res.columns]
    keep_cols = [c for c in res.columns if c not in drop_cols]
    # out = pd.concat([weather_df[[site_col, timestamp_col]].reset_index(drop=True), res[keep_cols].reset_index(drop=True)], axis=1)
    # Use `keys`, not `weather_df`, so site_col keeps the same dtype as the merged key path.
    out = pd.concat(
        [keys[[site_col, timestamp_col]].reset_index(drop=True), res[keep_cols].reset_index(drop=True)],
        axis=1,
    )
    return out


def _fit_history_state(
    weather_df: pd.DataFrame,
    *,
    site_col: str,
    lag_weather_cols: list[str],
    max_history_len: int,
) -> dict[str, Any]:
    history: dict[str, dict[str, list[float]]] = {}
    for site, grp in weather_df.groupby(site_col, sort=False):
        site_key = str(site)
        site_hist: dict[str, list[float]] = {}
        for col in lag_weather_cols:
            if col not in grp.columns:
                continue
            vals = pd.to_numeric(grp[col], errors="coerce")
            filled = vals.ffill().bfill()
            if filled.isna().all():
                tail = [0.0] * max_history_len
            else:
                arr = filled.to_numpy(dtype=float)
                tail = arr[-max_history_len:].tolist()
                if len(tail) < max_history_len:
                    tail = [arr[0]] * (max_history_len - len(tail)) + tail
            site_hist[col] = tail
        history[site_key] = site_hist
    return {"site_history": history, "max_history_len": int(max_history_len)}


def _apply_lag_features(
    weather_df: pd.DataFrame,
    *,
    site_col: str,
    timestamp_col: str,
    lag_weather_cols: list[str],
    lags: list[int],
    windows: list[int],
    history_state: dict[str, Any] | None,
) -> pd.DataFrame:
    # OLD IMPLEMENTATION (kept for reference only)
    # max_hist = max([0, *lags, *windows])
    # parts = []
    # weather_df = weather_df.sort_values([site_col, timestamp_col]).reset_index(drop=True)
    # weather_df[site_col] = weather_df[site_col].astype(str)
    #
    # for site, grp in weather_df.groupby(site_col, sort=False):
    #     site_key = str(site)
    #     g = grp[[site_col, timestamp_col] + [c for c in lag_weather_cols if c in grp.columns]].copy().reset_index(drop=True)
    #     site_out = g[[site_col, timestamp_col]].copy()
    #     hist_map = (history_state or {}).get("site_history", {}).get(site_key, {}) if history_state else {}
    #
    #     for col in lag_weather_cols:
    #         if col not in g.columns:
    #             continue
    #         series = pd.to_numeric(g[col], errors="coerce").ffill().bfill()
    #         if series.isna().all():
    #             arr = np.zeros(len(g), dtype=np.float32)
    #         else:
    #             arr = series.to_numpy(dtype=np.float32, copy=False)
    #
    #         prefix = hist_map.get(col)
    #         if prefix is None or len(prefix) == 0:
    #             pad_value = float(arr[0]) if len(arr) else 0.0
    #             prefix_arr = np.full(max_hist, pad_value, dtype=np.float32)
    #         else:
    #             prefix_arr = np.asarray(prefix, dtype=np.float32)
    #             if len(prefix_arr) < max_hist:
    #                 pad_value = prefix_arr[0] if len(prefix_arr) else (float(arr[0]) if len(arr) else 0.0)
    #                 prefix_arr = np.concatenate([np.full(max_hist - len(prefix_arr), pad_value, dtype=np.float32), prefix_arr])
    #             elif len(prefix_arr) > max_hist:
    #                 prefix_arr = prefix_arr[-max_hist:]
    #
    #         ext = np.concatenate([prefix_arr, arr])
    #         base = ext[max_hist: max_hist + len(arr)]
    #
    #         for lag in lags:
    #             lagged = ext[max_hist - lag: max_hist - lag + len(arr)]
    #             site_out[f"r4_{col}_lag{lag}"] = lagged.astype(np.float32)
    #             site_out[f"r4_{col}_diff_lag{lag}"] = (base - lagged).astype(np.float32)
    #
    #         for win in windows:
    #             roll_mean = np.empty(len(arr), dtype=np.float32)
    #             for i in range(len(arr)):
    #                 start = max_hist + i - win
    #                 end = max_hist + i
    #                 window = ext[start:end]
    #                 roll_mean[i] = float(window.mean()) if len(window) else float(base[i])
    #             site_out[f"r4_{col}_rollmean{win}"] = roll_mean
    #             site_out[f"r4_{col}_anom{win}"] = (base - roll_mean).astype(np.float32)
    #
    #     parts.append(site_out)
    #
    # out = pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame(columns=[site_col, timestamp_col])
    # return out

    # NEW IMPLEMENTATION:
    # Keep per-site history semantics, but replace the per-row rolling loop with vectorized shift/rolling.
    max_hist = max([0, *lags, *windows])
    parts = []
    weather_df = weather_df.sort_values([site_col, timestamp_col]).reset_index(drop=True)
    weather_df[site_col] = weather_df[site_col].astype(str)

    for site, grp in weather_df.groupby(site_col, sort=False):
        site_key = str(site)
        g = (
            grp[[site_col, timestamp_col] + [c for c in lag_weather_cols if c in grp.columns]]
            .copy()
            .reset_index(drop=True)
        )
        site_out = g[[site_col, timestamp_col]].copy()
        hist_map = (history_state or {}).get("site_history", {}).get(site_key, {}) if history_state else {}

        for col in lag_weather_cols:
            if col not in g.columns:
                continue
            series = pd.to_numeric(g[col], errors="coerce").ffill().bfill()
            if series.isna().all():
                arr = np.zeros(len(g), dtype=np.float32)
            else:
                arr = series.to_numpy(dtype=np.float32, copy=False)

            prefix = hist_map.get(col)
            if prefix is None or len(prefix) == 0:
                pad_value = float(arr[0]) if len(arr) else 0.0
                prefix_arr = np.full(max_hist, pad_value, dtype=np.float32)
            else:
                prefix_arr = np.asarray(prefix, dtype=np.float32)
                if len(prefix_arr) < max_hist:
                    pad_value = prefix_arr[0] if len(prefix_arr) else (float(arr[0]) if len(arr) else 0.0)
                    prefix_arr = np.concatenate(
                        [np.full(max_hist - len(prefix_arr), pad_value, dtype=np.float32), prefix_arr]
                    )
                elif len(prefix_arr) > max_hist:
                    prefix_arr = prefix_arr[-max_hist:]

            ext = np.concatenate([prefix_arr, arr])
            ext_s = pd.Series(ext)
            base = ext[max_hist : max_hist + len(arr)]

            for lag in lags:
                lagged = ext_s.shift(lag).iloc[max_hist : max_hist + len(arr)].to_numpy(dtype=np.float32, copy=False)
                site_out[f"r4_{col}_lag{lag}"] = lagged.astype(np.float32)
                site_out[f"r4_{col}_diff_lag{lag}"] = (base - lagged).astype(np.float32)

            for win in windows:
                roll_mean = ext_s.shift(1).rolling(win, min_periods=1).mean().iloc[max_hist : max_hist + len(arr)]
                roll_mean = roll_mean.to_numpy(dtype=np.float32, copy=False)
                site_out[f"r4_{col}_rollmean{win}"] = roll_mean
                site_out[f"r4_{col}_anom{win}"] = (base - roll_mean).astype(np.float32)

        parts.append(site_out)

    out = pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame(columns=[site_col, timestamp_col])
    return out


def _fit_te_state(
    df: pd.DataFrame,
    *,
    label_col: str,
    building_col: str,
    meter_col: str,
    primary_use_col: str,
    site_col: str,
    min_count_for_exact: int,
    exclude_building_id: bool = False,
) -> dict[str, Any]:
    if label_col not in df.columns:
        raise KeyError(f"rank4 target encoding requires '{label_col}' in fit_transform mode.")

    # OLD IMPLEMENTATION (kept for reference only)
    # work = df.copy()
    # work[label_col] = pd.to_numeric(work[label_col], errors="coerce")
    # work = work.dropna(subset=[label_col])
    # work["y_log1p"] = np.log1p(np.maximum(work[label_col].to_numpy(dtype=float), 0.0))
    # global_mean = float(work["y_log1p"].mean()) if len(work) else 0.0
    #
    # group_specs: dict[str, list[str]] = {}
    # if (not exclude_building_id) and all(c in work.columns for c in [building_col, meter_col]):
    #     group_specs["building_meter"] = [building_col, meter_col]
    # if all(c in work.columns for c in [primary_use_col, meter_col]):
    #     group_specs["primary_use_meter"] = [primary_use_col, meter_col]
    # if all(c in work.columns for c in [site_col, meter_col]):
    #     group_specs["site_meter"] = [site_col, meter_col]
    #
    # state: dict[str, Any] = {"global_mean": global_mean, "group_specs": {}, "exclude_building_id": bool(exclude_building_id)}
    # for name, cols in group_specs.items():
    #     sub = work[cols + ["y_log1p"]].dropna(subset=cols)
    #     if sub.empty:
    #         state["group_specs"][name] = {
    #             "cols": cols,
    #             "mean": {},
    #             "count": {},
    #             "global_mean": global_mean,
    #             "min_count_for_exact": int(min_count_for_exact),
    #         }
    #         continue
    #     key = sub[cols].astype(str).agg("||".join, axis=1)
    #     grp = sub.groupby(key, sort=False)["y_log1p"]
    #     mean_map = grp.mean()
    #     count_map = grp.size()
    #     state["group_specs"][name] = {
    #         "cols": cols,
    #         "mean": {str(k): float(v) for k, v in mean_map.items()},
    #         "count": {str(k): int(v) for k, v in count_map.items()},
    #         "global_mean": global_mean,
    #         "min_count_for_exact": int(min_count_for_exact),
    #     }
    # return state

    # NEW IMPLEMENTATION:
    # Group directly on native columns and keep a mapping DataFrame per TE family.
    work = df.copy()
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce")
    work = work.dropna(subset=[label_col])
    work["y_log1p"] = np.log1p(np.maximum(work[label_col].to_numpy(dtype=float), 0.0))
    global_mean = float(work["y_log1p"].mean()) if len(work) else 0.0

    group_specs: dict[str, list[str]] = {}
    if (not exclude_building_id) and all(c in work.columns for c in [building_col, meter_col]):
        group_specs["building_meter"] = [building_col, meter_col]
    if all(c in work.columns for c in [primary_use_col, meter_col]):
        group_specs["primary_use_meter"] = [primary_use_col, meter_col]
    if all(c in work.columns for c in [site_col, meter_col]):
        group_specs["site_meter"] = [site_col, meter_col]

    state: dict[str, Any] = {
        "global_mean": global_mean,
        "group_specs": {},
        "exclude_building_id": bool(exclude_building_id),
    }
    for name, cols in group_specs.items():
        sub = work[cols + ["y_log1p"]].dropna(subset=cols)
        if sub.empty:
            mapping_df = pd.DataFrame(columns=[*cols, "_te_mean", "_te_count"])
        else:
            mapping_df = sub.groupby(cols, sort=False)["y_log1p"].agg(["mean", "size"]).reset_index()
            mapping_df = mapping_df.rename(columns={"mean": "_te_mean", "size": "_te_count"})
        state["group_specs"][name] = {
            "cols": cols,
            "mapping_df": mapping_df,
            "global_mean": global_mean,
            "min_count_for_exact": int(min_count_for_exact),
        }
    return state


def _apply_te_features(df: pd.DataFrame, *, state: dict[str, Any]) -> pd.DataFrame:
    # OLD IMPLEMENTATION (kept for reference only)
    # out = pd.DataFrame(index=df.index)
    # global_mean = state["global_mean"]
    # for name, spec in state["group_specs"].items():
    #     cols = [c for c in spec["cols"] if c in df.columns]
    #     if len(cols) != len(spec["cols"]):
    #         raise KeyError(f"rank4 target encoding requires columns {spec['cols']} during transform.")
    #     key = df[cols].astype(str).agg("||".join, axis=1)
    #     mean = key.map(spec["mean"]).fillna(spec["global_mean"]).to_numpy(dtype=np.float32, copy=False)
    #     cnt = key.map(spec["count"]).fillna(0).to_numpy(dtype=np.float32, copy=False)
    #     shrink = np.minimum(cnt / max(float(spec["min_count_for_exact"]), 1.0), 1.0).astype(np.float32)
    #     smooth = (shrink * mean + (1.0 - shrink) * np.float32(global_mean)).astype(np.float32)
    #     out[f"r4_te_{name}_mean"] = smooth
    #     out[f"r4_te_{name}_count"] = cnt.astype(np.float32)
    # return out

    # NEW IMPLEMENTATION:
    # Vectorized merge on grouping columns instead of string-key mapping.
    out = pd.DataFrame(index=df.index)
    global_mean = state["global_mean"]
    for name, spec in state["group_specs"].items():
        cols = [c for c in spec["cols"] if c in df.columns]
        if len(cols) != len(spec["cols"]):
            raise KeyError(f"rank4 target encoding requires columns {spec['cols']} during transform.")

        mapping_df = spec["mapping_df"]
        if mapping_df.empty:
            mean = np.full(len(df), spec["global_mean"], dtype=np.float32)
            cnt = np.zeros(len(df), dtype=np.float32)
        else:
            merged = df[cols].merge(mapping_df, on=cols, how="left", sort=False)
            mean = merged["_te_mean"].fillna(spec["global_mean"]).to_numpy(dtype=np.float32, copy=False)
            cnt = merged["_te_count"].fillna(0).to_numpy(dtype=np.float32, copy=False)

        shrink = np.minimum(cnt / max(float(spec["min_count_for_exact"]), 1.0), 1.0).astype(np.float32)
        smooth = (shrink * mean + (1.0 - shrink) * np.float32(global_mean)).astype(np.float32)
        out[f"r4_te_{name}_mean"] = smooth
        out[f"r4_te_{name}_count"] = cnt.astype(np.float32)
    return out


def _merge_weather_back(
    df: pd.DataFrame,
    weather_features: pd.DataFrame,
    *,
    site_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    merged = df[[c for c in [site_col, timestamp_col] if c in df.columns]].copy()
    if site_col not in merged.columns:
        merged[site_col] = "__global__"
    else:
        merged[site_col] = merged[site_col].astype(str)
    merged[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    feat = weather_features.copy()
    feat[site_col] = feat[site_col].astype(str)
    feat[timestamp_col] = pd.to_datetime(feat[timestamp_col], errors="coerce")
    out = merged.merge(feat, on=[site_col, timestamp_col], how="left", sort=False)
    drop_cols = [c for c in [site_col, timestamp_col] if c in out.columns]
    return out.drop(columns=drop_cols)


def extract_rank4_features(
    df: pd.DataFrame,
    *,
    cache: dict | None = None,
    mode: str = "transform",
    label_col: str = "meter_reading",
    timestamp_col: str = "timestamp",
    site_col: str = "site_id",
    building_col: str = "building_id",
    primary_use_col: str = "primary_use",
    meter_col: str = "meter",
    lag_weather_cols: list[str] | None = None,
    agg_weather_cols: list[str] | None = None,
    lags: list[int] | None = None,
    windows: list[int] | None = None,
    min_count_for_exact: int = 5,
    exclude_building_id: bool = False,
    **_: Any,
) -> pd.DataFrame:
    """Approximate rank-4 supplementary block.

    Clean reimplementation of the public rank-4 description using three parts:
      - aggregate site-hour weather context,
      - past-only weather lag / rolling features,
      - train-fitted target encodings.

    When exclude_building_id=True, the building_id × meter target-encoding family is removed.
    """
    _validate_mode(mode)
    if cache is None:
        cache = {}
    meta = cache.setdefault("rank4", {})
    meta["exclude_building_id"] = bool(exclude_building_id)

    lag_weather_cols = list(_DEFAULT_LAG_WEATHER_COLS if lag_weather_cols is None else lag_weather_cols)
    agg_weather_cols = list(_DEFAULT_AGG_WEATHER_COLS if agg_weather_cols is None else agg_weather_cols)
    lag_weather_cols = [c for c in lag_weather_cols if c in df.columns]
    agg_weather_cols = [c for c in agg_weather_cols if c in df.columns]
    lags = sorted(set(_DEFAULT_LAGS if lags is None else lags))
    windows = sorted(set(_DEFAULT_WINDOWS if windows is None else windows))

    if not lag_weather_cols and not agg_weather_cols:
        raise ValueError("rank4 requires at least one weather column for aggregate or lag features.")

    weather_cols = sorted(set(lag_weather_cols + agg_weather_cols))
    weather_df = _prepare_weather_frame(df, timestamp_col=timestamp_col, site_col=site_col, weather_cols=weather_cols)
    max_history_len = max([0, *lags, *windows])

    if mode == "fit_transform":
        meta["agg_state"] = _fit_aggregate_state(
            weather_df,
            site_col=site_col,
            timestamp_col=timestamp_col,
            weather_cols=agg_weather_cols,
        )
        meta["history_state"] = _fit_history_state(
            weather_df,
            site_col=site_col,
            lag_weather_cols=lag_weather_cols,
            max_history_len=max_history_len,
        )
        meta["te_state"] = _fit_te_state(
            df,
            label_col=label_col,
            building_col=building_col,
            meter_col=meter_col,
            primary_use_col=primary_use_col,
            site_col=site_col,
            min_count_for_exact=min_count_for_exact,
            exclude_building_id=exclude_building_id,
        )
    else:
        for key in ["agg_state", "history_state", "te_state"]:
            if key not in meta:
                raise ValueError(f"rank4 cache is missing '{key}'. Run mode='fit_transform' first.")

    agg_features = _apply_aggregate_features(
        weather_df,
        state=meta["agg_state"],
        site_col=site_col,
        timestamp_col=timestamp_col,
    )
    lag_features = _apply_lag_features(
        weather_df,
        site_col=site_col,
        timestamp_col=timestamp_col,
        lag_weather_cols=lag_weather_cols,
        lags=lags,
        windows=windows,
        history_state=None if mode == "fit_transform" else meta["history_state"],
    )

    # defense against unexpected dtype changes in the key columns that would break the merge logic
    agg_features[site_col] = agg_features[site_col].astype(str)
    lag_features[site_col] = lag_features[site_col].astype(str)
    weather_feature_df = agg_features.merge(lag_features, on=[site_col, timestamp_col], how="outer", sort=False)
    weather_feature_df = _merge_weather_back(df, weather_feature_df, site_col=site_col, timestamp_col=timestamp_col)
    te_feature_df = _apply_te_features(df, state=meta["te_state"])

    out = pd.concat([weather_feature_df.reset_index(drop=True), te_feature_df.reset_index(drop=True)], axis=1)
    out = _as_frame(out, expected_rows=len(df))

    if mode == "fit_transform":
        meta["feature_columns"] = list(out.columns)
    else:
        expected_cols = meta.get("feature_columns")
        if expected_cols is not None and list(out.columns) != expected_cols:
            raise ValueError(
                "rank4 feature columns differ from cached fit columns. "
                f"fit_cols={expected_cols[:20]} transform_cols={list(out.columns)[:20]}"
            )
    return out
