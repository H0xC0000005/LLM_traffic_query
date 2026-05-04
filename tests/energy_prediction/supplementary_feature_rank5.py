from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_VALID_MODES = {"fit_transform", "transform"}

_DEFAULT_SITE_HOUR_OFFSETS = {
    0: -4,
    1: 0,
    2: -7,
    3: -4,
    4: -4,
    5: 0,
    6: -5,
    7: -5,
    8: -5,
    9: -6,
    10: -7,
    11: -5,
    12: 0,
    13: -6,
    14: -5,
    15: -5,
}


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


def _compute_local_time_frame(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    site_col: str,
    site_hour_offsets: dict[int, int],
) -> pd.DataFrame:
    if timestamp_col not in df.columns:
        raise KeyError(f"Missing required column '{timestamp_col}'")

    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    if site_col in df.columns:
        site_num = pd.to_numeric(df[site_col], errors="coerce")
        offset_hours = site_num.map(site_hour_offsets).fillna(0).astype(np.int16)
    else:
        offset_hours = pd.Series(np.zeros(len(df), dtype=np.int16), index=df.index)

    local_ts = ts + pd.to_timedelta(offset_hours.to_numpy(), unit="h")
    out = pd.DataFrame(index=df.index)
    out["r5_local_hour"] = local_ts.dt.hour.fillna(0).astype(np.int16)
    out["r5_local_weekday"] = local_ts.dt.weekday.fillna(0).astype(np.int16)
    out["r5_local_dayofyear"] = local_ts.dt.dayofyear.fillna(1).astype(np.int16)
    out["r5_is_weekend_local"] = (out["r5_local_weekday"] >= 5).astype(np.int8)

    if "is_holiday_any" in df.columns:
        hol = pd.to_numeric(df["is_holiday_any"], errors="coerce").fillna(0).astype(np.int8)
    elif "is_na_holiday" in df.columns or "is_eu_holiday" in df.columns:
        hol = (
            pd.to_numeric(df.get("is_na_holiday", 0), errors="coerce").fillna(0)
            + pd.to_numeric(df.get("is_eu_holiday", 0), errors="coerce").fillna(0)
        ).clip(0, 1).astype(np.int8)
    else:
        hol = pd.Series(np.zeros(len(df), dtype=np.int8), index=df.index)

    out["r5_is_day_off_or_holiday"] = np.maximum(out["r5_is_weekend_local"].to_numpy(), hol.to_numpy()).astype(np.int8)
    return out


def _fit_ratio_map(
    df: pd.DataFrame,
    *,
    entity_col: str,
    bucket_col: str,
    target_col: str,
) -> dict[str, Any]:
    work = df[[entity_col, bucket_col, target_col]].copy()
    work = work.dropna(subset=[entity_col, bucket_col, target_col])
    work[entity_col] = work[entity_col].astype(str)
    work[bucket_col] = work[bucket_col].astype(str)

    entity_med = work.groupby(entity_col, sort=False)[target_col].median()
    pair_med = work.groupby([entity_col, bucket_col], sort=False)[target_col].median()
    pair_cnt = work.groupby([entity_col, bucket_col], sort=False)[target_col].size()

    mapping = {}
    count_mapping = {}
    for (entity, bucket), val in pair_med.items():
        key = f"{entity}||{bucket}"
        mapping[key] = float(val)
        count_mapping[key] = int(pair_cnt.loc[(entity, bucket)])

    return {
        "entity_median": {str(k): float(v) for k, v in entity_med.items()},
        "pair_median": mapping,
        "pair_count": count_mapping,
        "global_default": float(work[target_col].median()) if len(work) else 0.0,
    }


def _apply_ratio_map(
    df: pd.DataFrame,
    *,
    entity_col: str,
    bucket_col: str,
    state: dict[str, Any],
    output_col: str,
    min_count_for_exact: int,
) -> pd.DataFrame:
    entity_key = df[entity_col].astype(str)
    bucket_key = df[bucket_col].astype(str)
    pair_key = entity_key + "||" + bucket_key

    pair_med = pair_key.map(state["pair_median"])
    pair_cnt = pair_key.map(state["pair_count"]).fillna(0)
    entity_med = entity_key.map(state["entity_median"]).fillna(state["global_default"])

    blended = np.where(
        pair_cnt.to_numpy(dtype=float) >= float(min_count_for_exact),
        pair_med.fillna(entity_med).to_numpy(dtype=float),
        entity_med.to_numpy(dtype=float),
    )
    denom = np.maximum(entity_med.to_numpy(dtype=float), 1e-6)
    ratio = (blended / denom).astype(np.float32)
    return pd.DataFrame({output_col: ratio}, index=df.index)


def _fit_quantile_map(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    target_col: str,
    q: float,
) -> dict[str, Any]:
    work = df[group_cols + [target_col]].copy().dropna(subset=group_cols + [target_col])
    group_key = work[group_cols].astype(str).agg("||".join, axis=1)
    quant = work.groupby(group_key, sort=False)[target_col].quantile(q)
    return {
        "mapping": {str(k): float(v) for k, v in quant.items()},
        "global_default": float(work[target_col].quantile(q)) if len(work) else 0.0,
    }


def _apply_quantile_map(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    state: dict[str, Any],
    output_col: str,
) -> pd.DataFrame:
    group_key = df[group_cols].astype(str).agg("||".join, axis=1)
    values = group_key.map(state["mapping"]).fillna(state["global_default"]).to_numpy(dtype=np.float32, copy=False)
    return pd.DataFrame({output_col: values}, index=df.index)


def extract_rank5_mma_features(
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
    min_count_for_exact: int = 5,
    site_hour_offsets: dict[int, int] | None = None,
    exclude_building_id: bool = False,
    **_: Any,
) -> pd.DataFrame:
    """Approximate rank-5 supplementary block.

    Implemented as one coherent feature block with:
      - site-local time features,
      - day-off / holiday flag,
      - primary-use schedule-shape ratio encodings,
      - optional building schedule-shape ratio encodings,
      - optional building-meter 5th / 95th percentile encodings.

    When exclude_building_id=True, all building-keyed derived features are removed.
    """
    _validate_mode(mode)
    if cache is None:
        cache = {}

    site_hour_offsets = dict(_DEFAULT_SITE_HOUR_OFFSETS if site_hour_offsets is None else site_hour_offsets)
    meta = cache.setdefault("rank5_mma", {})
    meta["exclude_building_id"] = bool(exclude_building_id)

    base = _compute_local_time_frame(
        df,
        timestamp_col=timestamp_col,
        site_col=site_col,
        site_hour_offsets=site_hour_offsets,
    )

    use_building = (not exclude_building_id) and (building_col in df.columns)
    work = df.copy()
    if mode == "fit_transform":
        if label_col not in work.columns:
            raise ValueError(f"'{label_col}' is required in mode='fit_transform'")
        target = np.log1p(pd.to_numeric(work[label_col], errors="coerce").fillna(0.0).clip(lower=0.0))
        fit_df = base.copy()
        if use_building:
            fit_df[building_col] = work[building_col].astype(str)
        fit_df[primary_use_col] = work[primary_use_col].astype(str) if primary_use_col in work.columns else "Unknown"
        fit_df[meter_col] = pd.to_numeric(work[meter_col], errors="coerce").fillna(0).astype(np.int16) if meter_col in work.columns else 0
        fit_df[label_col] = target.astype(np.float32)

        meta["site_hour_offsets"] = site_hour_offsets
        feature_columns = [
            "r5_local_hour",
            "r5_local_weekday",
            "r5_local_dayofyear",
            "r5_is_weekend_local",
            "r5_is_day_off_or_holiday",
            "r5_frac_primary_use_hour",
            "r5_frac_primary_use_weekday",
            "r5_frac_primary_use_dayofyear",
        ]
        meta["frac_primary_use_hour"] = _fit_ratio_map(fit_df, entity_col=primary_use_col, bucket_col="r5_local_hour", target_col=label_col)
        meta["frac_primary_use_weekday"] = _fit_ratio_map(fit_df, entity_col=primary_use_col, bucket_col="r5_local_weekday", target_col=label_col)
        meta["frac_primary_use_dayofyear"] = _fit_ratio_map(fit_df, entity_col=primary_use_col, bucket_col="r5_local_dayofyear", target_col=label_col)

        if use_building:
            feature_columns.extend(
                [
                    "r5_frac_building_hour",
                    "r5_frac_building_weekday",
                    "r5_frac_building_dayofyear",
                    "r5_building_meter_q95_log1p",
                    "r5_building_meter_q05_log1p",
                ]
            )
            meta["frac_building_hour"] = _fit_ratio_map(fit_df, entity_col=building_col, bucket_col="r5_local_hour", target_col=label_col)
            meta["frac_building_weekday"] = _fit_ratio_map(fit_df, entity_col=building_col, bucket_col="r5_local_weekday", target_col=label_col)
            meta["frac_building_dayofyear"] = _fit_ratio_map(fit_df, entity_col=building_col, bucket_col="r5_local_dayofyear", target_col=label_col)
            meta["building_meter_q95"] = _fit_quantile_map(fit_df, group_cols=[building_col, meter_col], target_col=label_col, q=0.95)
            meta["building_meter_q05"] = _fit_quantile_map(fit_df, group_cols=[building_col, meter_col], target_col=label_col, q=0.05)
        else:
            for key in [
                "frac_building_hour",
                "frac_building_weekday",
                "frac_building_dayofyear",
                "building_meter_q95",
                "building_meter_q05",
            ]:
                meta.pop(key, None)

        meta["feature_columns"] = feature_columns
    else:
        if not meta:
            raise ValueError("rank5_mma cache is empty. Run mode='fit_transform' first.")
        site_hour_offsets = dict(meta.get("site_hour_offsets", site_hour_offsets))
        base = _compute_local_time_frame(
            df,
            timestamp_col=timestamp_col,
            site_col=site_col,
            site_hour_offsets=site_hour_offsets,
        )
        use_building = (not meta.get("exclude_building_id", False)) and (building_col in df.columns) and ("frac_building_hour" in meta)

    feat = base.copy()
    if use_building:
        feat[building_col] = df[building_col].astype(str)
    feat[primary_use_col] = df[primary_use_col].astype(str) if primary_use_col in df.columns else "Unknown"
    feat[meter_col] = pd.to_numeric(df[meter_col], errors="coerce").fillna(0).astype(np.int16) if meter_col in df.columns else 0

    parts = []
    out = pd.DataFrame(index=df.index)
    out["r5_local_hour"] = feat["r5_local_hour"].astype(np.int16)
    out["r5_local_weekday"] = feat["r5_local_weekday"].astype(np.int16)
    out["r5_local_dayofyear"] = feat["r5_local_dayofyear"].astype(np.int16)
    out["r5_is_weekend_local"] = feat["r5_is_weekend_local"].astype(np.int8)
    out["r5_is_day_off_or_holiday"] = feat["r5_is_day_off_or_holiday"].astype(np.int8)
    parts.append(out)

    parts.extend(
        [
            _apply_ratio_map(feat, entity_col=primary_use_col, bucket_col="r5_local_hour", state=meta["frac_primary_use_hour"], output_col="r5_frac_primary_use_hour", min_count_for_exact=min_count_for_exact),
            _apply_ratio_map(feat, entity_col=primary_use_col, bucket_col="r5_local_weekday", state=meta["frac_primary_use_weekday"], output_col="r5_frac_primary_use_weekday", min_count_for_exact=min_count_for_exact),
            _apply_ratio_map(feat, entity_col=primary_use_col, bucket_col="r5_local_dayofyear", state=meta["frac_primary_use_dayofyear"], output_col="r5_frac_primary_use_dayofyear", min_count_for_exact=min_count_for_exact),
        ]
    )

    if use_building:
        parts.extend(
            [
                _apply_ratio_map(feat, entity_col=building_col, bucket_col="r5_local_hour", state=meta["frac_building_hour"], output_col="r5_frac_building_hour", min_count_for_exact=min_count_for_exact),
                _apply_ratio_map(feat, entity_col=building_col, bucket_col="r5_local_weekday", state=meta["frac_building_weekday"], output_col="r5_frac_building_weekday", min_count_for_exact=min_count_for_exact),
                _apply_ratio_map(feat, entity_col=building_col, bucket_col="r5_local_dayofyear", state=meta["frac_building_dayofyear"], output_col="r5_frac_building_dayofyear", min_count_for_exact=min_count_for_exact),
                _apply_quantile_map(feat, group_cols=[building_col, meter_col], state=meta["building_meter_q95"], output_col="r5_building_meter_q95_log1p"),
                _apply_quantile_map(feat, group_cols=[building_col, meter_col], state=meta["building_meter_q05"], output_col="r5_building_meter_q05_log1p"),
            ]
        )

    out = pd.concat(parts, axis=1)
    out = _as_frame(out, expected_rows=len(df))
    expected_cols = meta.get("feature_columns")
    if expected_cols is not None and list(out.columns) != expected_cols:
        raise ValueError(
            "rank5_mma feature columns differ from cached fit columns. "
            f"fit_cols={expected_cols[:20]} transform_cols={list(out.columns)[:20]}"
        )
    return out
