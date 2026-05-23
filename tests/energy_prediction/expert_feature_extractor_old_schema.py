import numpy as np
import pandas as pd
from typing import Any, Mapping, Sequence


def extract_expert_features(
    df: pd.DataFrame,
    *,
    cache: dict | None = None,
    meter_bases_c: Mapping[int, tuple[float, float]] = {
        0: (15.0, 22.0),
        1: (10.0, 15.5),
        2: (18.0, 28.0),
        3: (17.0, 27.0),
    },
    meter_response_weights: Mapping[int, tuple[float, float]] = {
        0: (0.80, 1.00),
        1: (0.15, 1.30),
        2: (1.35, 0.05),
        3: (1.10, 0.08),
    },
    dew_comfort_c: float = 12.0,
    humidity_weight: float = 0.35,
    wind_weight: float = 0.20,
    memory_halflife_hours: int = 6,
    psychrometric_config: Mapping[str, Any] | None = None,
    reliability_min_group_size: int = 64,
    reliability_robust_z_clip: float = 8.0,
    reliability_temp_gap_soft_cap: float = 8.0,
    reliability_pressure_valid_range: tuple[float, float] = (870.0, 1085.0),
    bp_heating_grid_c: Sequence[float] = (6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0),
    bp_cooling_grid_c: Sequence[float] = (14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0),
    bp_enthalpy_grid_kjkg: Sequence[float] = (25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0),
    bp_min_group_size: int = 600,
    bp_fit_sample_size: int = 20000,
    bp_shrinkage: float = 800.0,
    bp_softplus_scale: float = 2.75,
    bp_random_state: int = 42,
    **kwargs: Any,
) -> pd.DataFrame:
    """Combine four expert feature families into one leak-aware feature-only extractor.

    Parameters
    ----------
    df : pd.DataFrame
        Full input table. Row order and index are preserved. The function returns only engineered features.
        `meter_reading` is optional and is used only to fit cached balance-point thresholds.
    cache : dict | None
        Mutable cross-call state. Internal keys are namespaced under `cache["extract_expert_features"]`
        to avoid collisions with other pipelines.
    meter_bases_c, meter_response_weights, dew_comfort_c, humidity_weight, wind_weight, memory_halflife_hours
        Hyperparameters for the meter-aware thermal response block.
    psychrometric_config : Mapping[str, Any] | None
        Optional override dict for psychrometric weather-load settings.
    reliability_min_group_size, reliability_robust_z_clip, reliability_temp_gap_soft_cap, reliability_pressure_valid_range
        Hyperparameters for the contextual reliability/anomaly block.
    bp_heating_grid_c, bp_cooling_grid_c, bp_enthalpy_grid_kjkg, bp_min_group_size, bp_fit_sample_size,
    bp_shrinkage, bp_softplus_scale, bp_random_state
        Hyperparameters for the learned humidity-adjusted balance-point block.
    **kwargs : Any
        Reserved for forward compatibility.

    Returns
    -------
    pd.DataFrame
        Sixteen float32 feature columns aligned to `df.index`:
        - thermal_* : meter-aware thermal response features
        - wx_* : psychrometric weather-load features
        - *_score / *_anomaly : contextual reliability features
        - bp_* : learned balance-point features
    """
    if cache is None:
        cache = {}

    root_cache = cache.setdefault("extract_expert_features", {})
    eps = 1e-6

    # ------------------------------------------------------------------
    # Shared utilities used across the expert blocks
    # ------------------------------------------------------------------
    def _num(data: pd.Series | Any) -> pd.Series:
        return pd.to_numeric(data, errors="coerce")

    def _series(name: str, default: Any = np.nan, *, dtype: str | None = None) -> pd.Series:
        if name in df.columns:
            s = df[name]
        else:
            s = pd.Series(default, index=df.index)
        if dtype is not None:
            return s.astype(dtype)
        return s

    def _fill_with_median(s: pd.Series, default: float) -> pd.Series:
        s = _num(s)
        med = s.median(skipna=True)
        if pd.isna(med):
            med = default
        return s.fillna(float(med)).astype(float)

    def _softplus(x: np.ndarray, scale: float = 1.0, clip: float = 50.0) -> np.ndarray:
        z = np.clip(np.asarray(x, dtype=float) / max(scale, 1e-6), -clip, clip)
        return scale * np.log1p(np.exp(z))

    def _meter_key(s: pd.Series) -> pd.Series:
        return s.astype("string").fillna("__MISSING__")

    def _vapor_pressure_kpa_from_dew_c(dew_c: np.ndarray) -> np.ndarray:
        return 0.61094 * np.exp((17.625 * dew_c) / (dew_c + 243.04))

    def _humidity_ratio_from_kpa(dew_c: np.ndarray, pressure_hpa: np.ndarray) -> np.ndarray:
        p_kpa = np.maximum(np.asarray(pressure_hpa, dtype=float) * 0.1, 60.0)
        e_kpa = np.minimum(_vapor_pressure_kpa_from_dew_c(np.asarray(dew_c, dtype=float)), p_kpa - 1e-3)
        return 0.62198 * e_kpa / np.maximum(p_kpa - e_kpa, 1e-3)

    def _enthalpy_kjkg_from_kpa(air_c: np.ndarray, humidity_ratio: np.ndarray) -> np.ndarray:
        air_c = np.asarray(air_c, dtype=float)
        humidity_ratio = np.asarray(humidity_ratio, dtype=float)
        return 1.006 * air_c + humidity_ratio * (2501.0 + 1.86 * air_c)

    def _enthalpy_kjkg_from_hpa(t_c: pd.Series, td_c: pd.Series, p_hpa: pd.Series) -> pd.Series:
        # Tetens-style vapor pressure + humidity ratio + moist-air enthalpy
        t = _num(t_c).astype(float)
        td = _num(td_c).astype(float)
        p = _num(p_hpa).astype(float)

        e = 6.112 * np.exp((17.67 * td) / (td + 243.5))
        e = np.clip(e, 0.01, None)

        denom = np.maximum(p - e, 1.0)
        w = 0.621945 * e / denom
        w = np.clip(w, 0.0, 0.08)

        h = 1.006 * t + w * (2501.0 + 1.86 * t)
        return pd.Series(h, index=t.index)

    def _outside_interval(x: pd.Series, low: float, high: float) -> np.ndarray:
        x_num = _num(x).to_numpy(dtype=float)
        return np.where(x_num < low, low - x_num, np.where(x_num > high, x_num - high, 0.0))

    def _compress_score(x: pd.Series | np.ndarray, clip: float) -> np.ndarray:
        x = np.asarray(x, dtype="float64")
        x = np.clip(x, 0.0, clip)
        return np.tanh(x / 3.0)

    def _fit_group_stats(frame: pd.DataFrame, keys: list[str], cols: list[str], min_group_size: int) -> dict[str, Any]:
        grp = frame.groupby(keys, observed=True, dropna=False)
        med = grp[cols].median()

        tmp = frame[keys + cols].merge(
            med.reset_index(),
            on=keys,
            how="left",
            suffixes=("", "__med"),
        )

        abs_dev_cols: list[str] = []
        for c in cols:
            dev_col = f"{c}__absdev"
            tmp[dev_col] = (_num(tmp[c]) - _num(tmp[f"{c}__med"])).abs()
            abs_dev_cols.append(dev_col)

        mad = tmp.groupby(keys, observed=True, dropna=False)[abs_dev_cols].median()
        mad.columns = cols

        counts = grp.size().rename("__count")
        stats = med.copy()
        for c in cols:
            stats[f"{c}__med"] = med[c]
            stats[f"{c}__mad"] = mad[c]
        stats = stats[[col for c in cols for col in (f"{c}__med", f"{c}__mad")]]
        stats["__count"] = counts

        # Small groups fall back to global robust statistics.
        small_groups = stats["__count"] < int(min_group_size)
        stat_cols = [c for c in stats.columns if c != "__count"]
        stats.loc[small_groups, stat_cols] = np.nan

        global_med = frame[cols].median(numeric_only=True)
        global_mad = (frame[cols] - global_med).abs().median(numeric_only=True)
        global_mad = global_mad.replace(0.0, np.nan).fillna(1.0)

        return {
            "keys": keys,
            "cols": cols,
            "stats": stats.reset_index(),
            "global_med": global_med.to_dict(),
            "global_mad": global_mad.to_dict(),
        }

    def _score_from_group_stats(frame: pd.DataFrame, fit: dict[str, Any], clip: float) -> pd.Series:
        keys = fit["keys"]
        cols = fit["cols"]
        joined = frame[keys].merge(fit["stats"], on=keys, how="left")

        score_parts = []
        for c in cols:
            x = _num(frame[c])
            med = _num(joined[f"{c}__med"]).fillna(float(fit["global_med"][c]))
            mad = _num(joined[f"{c}__mad"]).fillna(float(fit["global_mad"][c]))
            mad = mad.replace(0.0, np.nan).fillna(1.0)

            x_filled = x.fillna(med)
            rz = (x_filled - med).abs() / (mad + eps)
            score_parts.append(_compress_score(rz, clip))

        return pd.Series(np.mean(np.vstack(score_parts), axis=0), index=frame.index)

    def _sample_frame(frame: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
        if len(frame) <= n:
            return frame
        return frame.sample(n=n, random_state=random_state)

    def _fit_thresholds(
        sample: pd.DataFrame,
        heat_grid: Sequence[float],
        cool_grid: Sequence[float],
        enth_grid: Sequence[float],
        fallback: dict[str, float],
    ) -> dict[str, float]:
        y = sample["_target"].to_numpy(dtype=float)
        t = sample["_temp"].to_numpy(dtype=float)
        h = sample["_enth"].to_numpy(dtype=float)

        if len(sample) < 50 or np.nanstd(y) < 1e-8:
            return dict(fallback)

        best_loss = np.inf
        best = dict(fallback)

        # Grid-search the threshold triplet exactly as the expert design intends.
        for hb in heat_grid:
            heat_load = np.clip(hb - t, 0.0, None)
            for cb in cool_grid:
                if cb <= hb + 1.0:
                    continue
                cool_load = np.clip(t - cb, 0.0, None)
                for eb in enth_grid:
                    latent_load = np.clip(h - eb, 0.0, None)
                    x = np.column_stack([np.ones_like(y), heat_load, cool_load, latent_load])
                    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
                    resid = y - x @ beta
                    loss = float(np.mean(resid * resid))
                    if loss < best_loss:
                        best_loss = loss
                        best = {"heat_bp": float(hb), "cool_bp": float(cb), "enth_bp": float(eb)}
        return best

    def _local_grid(center: float, full_grid: Sequence[float], radius: int = 1) -> tuple[float, ...]:
        arr = np.asarray(full_grid, dtype=float)
        idx = int(np.argmin(np.abs(arr - center)))
        lo = max(0, idx - radius)
        hi = min(len(arr), idx + radius + 1)
        return tuple(float(v) for v in arr[lo:hi])

    out_parts: list[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # Expert 1: meter-aware thermal response features
    # ------------------------------------------------------------------
    required_e1 = [
        "meter",
        "air_temperature",
        "dew_temperature",
        "wind_speed",
        "square_feet",
        "year_built",
        "floor_count",
        "is_day_off_or_holiday",
    ]
    missing_e1 = [c for c in required_e1 if c not in df.columns]
    if missing_e1:
        raise ValueError(f"Missing required columns for thermal response features: {missing_e1}")

    e1_cache = root_cache.setdefault("meter_aware_thermal_response", {})
    if "fit_state" not in e1_cache:
        square_feet_fit = _num(df["square_feet"])
        year_built_fit = _num(df["year_built"])
        floor_count_fit = _num(df["floor_count"])

        if "timestamp" in df.columns:
            ts_fit = pd.to_datetime(df["timestamp"], errors="coerce")
            year_ref = int(ts_fit.dt.year.dropna().max()) if ts_fit.notna().any() else 2020
        else:
            valid_years = year_built_fit.dropna()
            year_ref = int(valid_years.max()) if not valid_years.empty else 2020

        log_size_fit = np.log1p(square_feet_fit.clip(lower=0))
        e1_cache["fit_state"] = {
            "square_feet_median": float(square_feet_fit.median(skipna=True)) if square_feet_fit.notna().any() else 0.0,
            "year_built_median": (
                float(year_built_fit.median(skipna=True)) if year_built_fit.notna().any() else float(year_ref - 30)
            ),
            "floor_count_median": float(floor_count_fit.median(skipna=True)) if floor_count_fit.notna().any() else 3.0,
            "log_size_median": float(log_size_fit.median(skipna=True)) if np.isfinite(log_size_fit).any() else 1.0,
            "year_ref": year_ref,
        }

    e1_state = e1_cache["fit_state"]

    meter = _num(df["meter"]).fillna(-1).astype(int)
    air_t = _num(df["air_temperature"]).astype(float)
    dew_t = _num(df["dew_temperature"]).astype(float)
    wind = _num(df["wind_speed"]).astype(float)

    square_feet = _num(df["square_feet"]).fillna(e1_state["square_feet_median"]).clip(lower=0)
    year_built = _num(df["year_built"]).fillna(e1_state["year_built_median"])
    floor_count = _num(df["floor_count"]).fillna(e1_state["floor_count_median"]).clip(lower=1)
    is_day_off = pd.Series(df["is_day_off_or_holiday"], index=df.index).fillna(False).astype(bool)

    # Weather imputations remain current-call median based, matching the expert design.
    air_t = air_t.fillna(float(air_t.median(skipna=True)) if air_t.notna().any() else 20.0)
    dew_t = dew_t.fillna(float(dew_t.median(skipna=True)) if dew_t.notna().any() else 10.0)
    wind = wind.fillna(float(wind.median(skipna=True)) if wind.notna().any() else 0.0).clip(lower=0)

    heat_base_map = {k: v[0] for k, v in meter_bases_c.items()}
    cool_base_map = {k: v[1] for k, v in meter_bases_c.items()}
    heat_weight_map = {k: v[0] for k, v in meter_response_weights.items()}
    cool_weight_map = {k: v[1] for k, v in meter_response_weights.items()}

    heat_base = meter.map(heat_base_map).fillna(16.0).astype(float)
    cool_base = meter.map(cool_base_map).fillna(22.0).astype(float)
    heat_weight = meter.map(heat_weight_map).fillna(1.0).astype(float)
    cool_weight = meter.map(cool_weight_map).fillna(1.0).astype(float)

    # Metadata-derived archetype scaling.
    log_size = np.log1p(square_feet)
    size_scale = (log_size / max(e1_state["log_size_median"], 1e-6)).clip(lower=0.60, upper=1.80)
    building_age = (e1_state["year_ref"] - year_built).clip(lower=0, upper=120)
    age_scale = (1.0 + 0.0025 * building_age).clip(lower=0.90, upper=1.30)
    floor_scale = (1.0 + 0.03 * (floor_count - e1_state["floor_count_median"])).clip(lower=0.85, upper=1.25)
    archetype_scale = (size_scale * age_scale * floor_scale).clip(lower=0.60, upper=2.20)

    heat_raw = (heat_base - air_t).clip(lower=0.0)
    cool_raw = (air_t - cool_base).clip(lower=0.0)
    heat_schedule = np.where(is_day_off, 0.97, 1.03)
    cool_schedule = np.where(is_day_off, 0.92, 1.08)
    dew_excess = (dew_t - dew_comfort_c).clip(lower=0.0)
    wind_term = np.log1p(wind)

    thermal_heat_stress = np.log1p(heat_raw * heat_weight * archetype_scale * heat_schedule * (1.0 + 0.15 * wind_term))
    thermal_cool_stress = np.log1p(
        cool_raw * cool_weight * archetype_scale * cool_schedule * (1.0 + humidity_weight * dew_excess / 10.0)
    )
    moisture_infiltration_stress = np.log1p(dew_excess * archetype_scale * (1.0 + wind_weight * wind_term))

    # Past-only temperature memory feature.
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        temp_table = (
            pd.DataFrame({"timestamp": ts, "air_temperature": air_t})
            .dropna(subset=["timestamp"])
            .groupby("timestamp", sort=True)["air_temperature"]
            .mean()
        )
        past_baseline = (
            temp_table.ewm(
                halflife=max(int(memory_halflife_hours), 1),
                adjust=False,
                min_periods=1,
            )
            .mean()
            .shift(1)
        )

        fallback_temp = float(temp_table.median()) if not temp_table.empty else float(air_t.median())
        baseline_temp = pd.to_numeric(ts.map(past_baseline.to_dict()), errors="coerce").fillna(fallback_temp)
        temp_shock = (air_t - baseline_temp).abs()
    else:
        fallback_temp = float(air_t.median(skipna=True)) if air_t.notna().any() else 20.0
        temp_shock = (air_t - fallback_temp).abs()

    thermal_memory_shock = np.log1p(temp_shock * archetype_scale * (0.5 + 0.5 * (heat_weight + cool_weight)))

    out_parts.append(
        pd.DataFrame(
            {
                "thermal_heat_stress": thermal_heat_stress.astype("float32"),
                "thermal_cool_stress": thermal_cool_stress.astype("float32"),
                "moisture_infiltration_stress": moisture_infiltration_stress.astype("float32"),
                "thermal_memory_shock": thermal_memory_shock.astype("float32"),
            },
            index=df.index,
        )
    )

    # ------------------------------------------------------------------
    # Expert 2: psychrometric weather-load features
    # ------------------------------------------------------------------
    cfg: dict[str, Any] = {
        "cooling_base_c": 18.0,
        "heating_base_c": 15.0,
        "comfort_dew_c": 10.0,
        "neutral_temp_c": 18.0,
        "neutral_band_c": 4.0,
        "softness_c": 1.5,
        "pressure_ref_hpa": 1013.25,
        "wind_ref_mps": 5.0,
        "precip_ref_mm": 3.0,
        "meter_cooling_weight": {0: 0.80, 1: 1.20, 2: 0.05, 3: 0.15},
        "meter_heating_weight": {0: 0.25, 1: 0.05, 2: 1.20, 3: 1.00},
        "default_cooling_weight": 0.55,
        "default_heating_weight": 0.55,
        "primary_use_gain": {
            "education": 1.05,
            "office": 1.00,
            "lodging/residential": 1.10,
            "entertainment/public assembly": 1.12,
            "retail": 1.08,
            "parking": 0.82,
            "warehouse/storage": 0.86,
            "services": 0.96,
            "technology/science": 0.98,
            "health care": 1.06,
            "manufacturing/industrial": 0.94,
            "utility": 0.92,
            "public services": 1.00,
            "religious worship": 0.95,
            "food sales and service": 1.08,
            "other": 1.00,
            "unknown": 1.00,
        },
        "size_ref_sqft": None,
        "floor_ref": None,
        "vintage_ref_year": None,
    }
    if psychrometric_config is not None:
        cfg.update(dict(psychrometric_config))

    e2_cache = root_cache.setdefault("psychrometric_weather_load", {})

    air = _fill_with_median(_series("air_temperature"), 20.0)
    dew = _fill_with_median(_series("dew_temperature"), 10.0)
    press = _fill_with_median(_series("sea_level_pressure"), float(cfg["pressure_ref_hpa"]))
    wind2 = _fill_with_median(_series("wind_speed"), float(cfg["wind_ref_mps"]))
    precip = _fill_with_median(_series("precip_depth_1_hr"), 0.0).clip(lower=0.0)
    cloud = _fill_with_median(_series("cloud_coverage"), 4.0).clip(lower=0.0, upper=8.0)

    sqft = _fill_with_median(_series("square_feet"), 1.0).clip(lower=1.0)
    floors = _fill_with_median(_series("floor_count"), 1.0).clip(lower=1.0)
    year_built2 = _fill_with_median(_series("year_built"), 2000.0)
    primary_use = _series("primary_use", default="unknown").astype("string").fillna("unknown").str.strip().str.lower()
    meter_num = _num(_series("meter")).round().astype("Int64")

    if "size_ref_sqft" not in e2_cache:
        positive_sqft = sqft[sqft > 0]
        e2_cache["size_ref_sqft"] = float(
            cfg["size_ref_sqft"]
            if cfg["size_ref_sqft"] is not None
            else (positive_sqft.median() if len(positive_sqft) else 100000.0)
        )
    if "floor_ref" not in e2_cache:
        positive_floors = floors[floors > 0]
        e2_cache["floor_ref"] = float(
            cfg["floor_ref"]
            if cfg["floor_ref"] is not None
            else (positive_floors.median() if len(positive_floors) else 3.0)
        )
    if "vintage_ref_year" not in e2_cache:
        valid_year = year_built2[(year_built2 >= 1800) & (year_built2 <= 2100)]
        e2_cache["vintage_ref_year"] = float(
            cfg["vintage_ref_year"]
            if cfg["vintage_ref_year"] is not None
            else (valid_year.median() if len(valid_year) else 2000.0)
        )

    size_ref = max(float(e2_cache["size_ref_sqft"]), 1.0)
    floor_ref = max(float(e2_cache["floor_ref"]), 1.0)
    vintage_ref = float(e2_cache["vintage_ref_year"])

    size_factor = np.clip(np.log1p(sqft.to_numpy()) / np.log1p(size_ref), 0.70, 1.45)
    floor_factor = np.clip(np.sqrt(floors.to_numpy()) / np.sqrt(floor_ref), 0.85, 1.20)
    vintage_factor = np.clip(1.0 + ((vintage_ref - year_built2.to_numpy()) / 100.0), 0.85, 1.20)
    use_gain = primary_use.map(cfg["primary_use_gain"]).fillna(1.0).astype(float).to_numpy()
    structural_gain = np.clip(size_factor * floor_factor * vintage_factor * use_gain, 0.65, 1.85)

    cool_weight2 = (
        meter_num.map(cfg["meter_cooling_weight"]).fillna(float(cfg["default_cooling_weight"])).astype(float).to_numpy()
    )
    heat_weight2 = (
        meter_num.map(cfg["meter_heating_weight"]).fillna(float(cfg["default_heating_weight"])).astype(float).to_numpy()
    )
    latent_weight = np.clip(0.40 + 0.60 * cool_weight2, 0.30, 1.20)
    envelope_weight = np.clip(0.35 + 0.35 * cool_weight2 + 0.35 * heat_weight2, 0.35, 1.40)

    air_np = air.to_numpy()
    dew_np = dew.to_numpy()
    press_np = press.to_numpy()
    wind_np = wind2.to_numpy()
    precip_np = precip.to_numpy()
    cloud_np = cloud.to_numpy()

    humidity_ratio = _humidity_ratio_from_kpa(dew_np, press_np)
    enthalpy = _enthalpy_kjkg_from_kpa(air_np, humidity_ratio)

    comfort_w = _humidity_ratio_from_kpa(
        np.full(len(df), float(cfg["comfort_dew_c"]), dtype=float),
        np.full(len(df), float(cfg["pressure_ref_hpa"]), dtype=float),
    )
    comfort_h = _enthalpy_kjkg_from_kpa(
        np.full(len(df), float(cfg["cooling_base_c"]), dtype=float),
        comfort_w,
    )

    cooling_degree = _softplus(air_np - float(cfg["cooling_base_c"]), float(cfg["softness_c"]))
    heating_degree = _softplus(float(cfg["heating_base_c"]) - air_np, float(cfg["softness_c"]))
    latent_burden = _softplus(enthalpy - comfort_h, 3.0)

    envelope_core = _softplus(
        np.abs(air_np - float(cfg["neutral_temp_c"])) - float(cfg["neutral_band_c"]),
        float(cfg["softness_c"]),
    )
    envelope_multiplier = (
        1.0
        + 0.30 * np.clip(wind_np / float(cfg["wind_ref_mps"]), 0.0, 3.0)
        + 0.18 * np.clip(precip_np / float(cfg["precip_ref_mm"]), 0.0, 3.0)
        + 0.08 * (cloud_np / 8.0)
    )
    envelope_stress = envelope_core * envelope_multiplier

    cooling_support = 1.0 + 0.05 * np.clip(latent_burden, 0.0, 20.0)
    heating_support = 1.0 + 0.04 * np.clip(wind_np, 0.0, 15.0)

    out_parts.append(
        pd.DataFrame(
            {
                "wx_cooling_response": np.log1p(
                    cooling_degree * cooling_support * structural_gain * cool_weight2
                ).astype("float32"),
                "wx_heating_response": np.log1p(
                    heating_degree * heating_support * structural_gain * heat_weight2
                ).astype("float32"),
                "wx_latent_response": np.log1p(latent_burden * structural_gain * latent_weight).astype("float32"),
                "wx_envelope_response": np.log1p(envelope_stress * structural_gain * envelope_weight).astype("float32"),
            },
            index=df.index,
        )
    )

    # ------------------------------------------------------------------
    # Expert 3: contextual reliability / spike features
    # ------------------------------------------------------------------
    required_e3 = [
        "meter",
        "primary_use",
        "square_feet",
        "year_built",
        "floor_count",
        "air_temperature",
        "cloud_coverage",
        "dew_temperature",
        "precip_depth_1_hr",
        "sea_level_pressure",
        "wind_direction",
        "wind_speed",
        "hour",
        "is_day_off_or_holiday",
    ]
    missing_e3 = [c for c in required_e3 if c not in df.columns]
    if missing_e3:
        raise KeyError(f"Missing required columns for contextual reliability features: {missing_e3}")

    weather_context_keys = ["meter", "hour", "is_day_off_or_holiday"]
    weather_cols = [
        "air_temperature",
        "dew_temperature",
        "sea_level_pressure",
        "wind_speed",
        "cloud_coverage",
        "precip_depth_1_hr_log",
    ]
    structure_keys = ["primary_use"]
    structure_cols = ["log_square_feet", "building_age", "floor_count"]

    work = pd.DataFrame(index=df.index)
    work["square_feet"] = _num(df["square_feet"])
    work["year_built"] = _num(df["year_built"])
    work["floor_count"] = _num(df["floor_count"])
    work["air_temperature"] = _num(df["air_temperature"])
    work["cloud_coverage"] = _num(df["cloud_coverage"])
    work["dew_temperature"] = _num(df["dew_temperature"])
    work["precip_depth_1_hr"] = _num(df["precip_depth_1_hr"])
    work["sea_level_pressure"] = _num(df["sea_level_pressure"])
    work["wind_direction"] = _num(df["wind_direction"])
    work["wind_speed"] = _num(df["wind_speed"])
    work["meter"] = df["meter"]
    work["primary_use"] = df["primary_use"]
    work["hour"] = df["hour"]
    work["is_day_off_or_holiday"] = df["is_day_off_or_holiday"].fillna(False)

    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        row_year = df["timestamp"].dt.year.astype("float64")
        work["building_age"] = (row_year - work["year_built"]).clip(lower=0, upper=250)
    else:
        work["building_age"] = (2020.0 - work["year_built"]).clip(lower=0, upper=250)

    work["log_square_feet"] = np.log1p(work["square_feet"].clip(lower=0))
    work["precip_depth_1_hr_log"] = np.sign(work["precip_depth_1_hr"]) * np.log1p(work["precip_depth_1_hr"].abs())

    # Feature 1: metadata missingness
    meta_missing = pd.concat(
        [
            work["square_feet"].isna().astype("float64"),
            work["year_built"].isna().astype("float64"),
            work["floor_count"].isna().astype("float64"),
        ],
        axis=1,
    )
    meta_missing_score = meta_missing.mean(axis=1)

    # Feature 2: physical/range implausibility
    pressure_low, pressure_high = reliability_pressure_valid_range
    dew_excess2 = np.clip(
        (work["dew_temperature"] - work["air_temperature"]) / max(reliability_temp_gap_soft_cap, eps),
        a_min=0.0,
        a_max=None,
    )
    cloud_out = _outside_interval(work["cloud_coverage"], 0.0, 8.0) / 8.0
    wind_neg = np.clip(-work["wind_speed"].to_numpy(dtype=float) / 10.0, a_min=0.0, a_max=None)
    wind_dir_out = _outside_interval(work["wind_direction"], 0.0, 360.0) / 180.0
    pressure_out = _outside_interval(work["sea_level_pressure"], pressure_low, pressure_high) / 50.0
    precip_neg = np.clip(-work["precip_depth_1_hr"].to_numpy(dtype=float) / 10.0, a_min=0.0, a_max=None)

    implausibility_stack = np.vstack(
        [
            _compress_score(np.nan_to_num(dew_excess2, nan=0.0), reliability_robust_z_clip),
            _compress_score(np.nan_to_num(cloud_out, nan=0.0), reliability_robust_z_clip),
            _compress_score(np.nan_to_num(wind_neg, nan=0.0), reliability_robust_z_clip),
            _compress_score(np.nan_to_num(wind_dir_out, nan=0.0), reliability_robust_z_clip),
            _compress_score(np.nan_to_num(pressure_out, nan=0.0), reliability_robust_z_clip),
            _compress_score(np.nan_to_num(precip_neg, nan=0.0), reliability_robust_z_clip),
        ]
    )
    weather_implausibility_score = pd.Series(implausibility_stack.mean(axis=0), index=df.index)

    # Fit or reuse robust context stats.
    e3_cache = root_cache.setdefault("contextual_reliability_spike", {})
    if "weather_fit" not in e3_cache:
        e3_cache["weather_fit"] = _fit_group_stats(
            work[weather_context_keys + weather_cols].copy(),
            weather_context_keys,
            weather_cols,
            reliability_min_group_size,
        )
        e3_cache["structure_fit"] = _fit_group_stats(
            work[structure_keys + structure_cols].copy(),
            structure_keys,
            structure_cols,
            reliability_min_group_size,
        )

    contextual_weather_anomaly = _score_from_group_stats(
        work[weather_context_keys + weather_cols],
        e3_cache["weather_fit"],
        reliability_robust_z_clip,
    )
    contextual_structure_anomaly = _score_from_group_stats(
        work[structure_keys + structure_cols],
        e3_cache["structure_fit"],
        reliability_robust_z_clip,
    )

    out_parts.append(
        pd.DataFrame(
            {
                "meta_missing_score": meta_missing_score.astype("float32"),
                "weather_implausibility_score": weather_implausibility_score.astype("float32"),
                "contextual_weather_anomaly": contextual_weather_anomaly.astype("float32"),
                "contextual_structure_anomaly": contextual_structure_anomaly.astype("float32"),
            },
            index=df.index,
        )
    )

    # ------------------------------------------------------------------
    # Expert 4: meter-specific humidity-adjusted balance-point features
    # ------------------------------------------------------------------
    required_e4 = [
        "meter",
        "primary_use",
        "air_temperature",
        "dew_temperature",
        "sea_level_pressure",
        "square_feet",
        "year_built",
        "floor_count",
        "wind_speed",
        "cloud_coverage",
    ]
    missing_e4 = [c for c in required_e4 if c not in df.columns]
    if missing_e4:
        raise KeyError(f"Missing required columns for balance-point features: {missing_e4}")

    e4_cache = root_cache.setdefault("meter_specific_balance_point", {})
    meter_priors = {
        "0": {"heat_bp": 15.0, "cool_bp": 22.0, "enth_bp": 50.0},
        "1": {"heat_bp": 10.0, "cool_bp": 18.0, "enth_bp": 42.0},
        "2": {"heat_bp": 18.0, "cool_bp": 28.0, "enth_bp": 60.0},
        "3": {"heat_bp": 17.0, "cool_bp": 27.0, "enth_bp": 58.0},
        "__MISSING__": {"heat_bp": 16.0, "cool_bp": 22.0, "enth_bp": 50.0},
    }

    if "learned_state" not in e4_cache:
        learned: dict[str, Any] = {"meter_params": {}, "group_params": {}, "medians": {}}

        medians = {
            "air_temperature": float(_num(df["air_temperature"]).median()),
            "dew_temperature": float(_num(df["dew_temperature"]).median()),
            "sea_level_pressure": float(_num(df["sea_level_pressure"]).median()),
            "wind_speed": float(_num(df["wind_speed"]).median()),
            "cloud_coverage": float(_num(df["cloud_coverage"]).median()),
            "square_feet": float(_num(df["square_feet"]).median()),
            "floor_count": float(_num(df["floor_count"]).median()),
            "year_built": float(_num(df["year_built"]).median()),
        }
        learned["medians"] = medians

        if "meter_reading" in df.columns:
            target = _num(df["meter_reading"])
            fit_mask = target.notna()
            if fit_mask.any():
                fit_df = pd.DataFrame(
                    {
                        "_meter": _meter_key(df.loc[fit_mask, "meter"]),
                        "_primary_use": _meter_key(df.loc[fit_mask, "primary_use"]),
                        "_temp": _num(df.loc[fit_mask, "air_temperature"]).fillna(medians["air_temperature"]),
                        "_dew": _num(df.loc[fit_mask, "dew_temperature"]).fillna(medians["dew_temperature"]),
                        "_press": _num(df.loc[fit_mask, "sea_level_pressure"]).fillna(medians["sea_level_pressure"]),
                        "_target": target.loc[fit_mask].astype(float),
                    }
                )
                fit_df["_enth"] = _enthalpy_kjkg_from_hpa(fit_df["_temp"], fit_df["_dew"], fit_df["_press"])

                # Meter-level threshold fit.
                for meter_value, group in fit_df.groupby("_meter", observed=True, sort=False):
                    prior = dict(meter_priors.get(str(meter_value), meter_priors["__MISSING__"]))
                    sample = _sample_frame(group, bp_fit_sample_size, bp_random_state)
                    learned["meter_params"][str(meter_value)] = _fit_thresholds(
                        sample,
                        bp_heating_grid_c,
                        bp_cooling_grid_c,
                        bp_enthalpy_grid_kjkg,
                        prior,
                    )

                # Group-level threshold fit with shrinkage toward the meter-level prior.
                for (meter_value, primary_use_value), group in fit_df.groupby(
                    ["_meter", "_primary_use"], observed=True, sort=False
                ):
                    if len(group) < bp_min_group_size:
                        continue

                    meter_value = str(meter_value)
                    primary_use_value = str(primary_use_value)

                    meter_base = dict(
                        learned["meter_params"].get(
                            meter_value,
                            meter_priors.get(meter_value, meter_priors["__MISSING__"]),
                        )
                    )
                    local_heat = _local_grid(meter_base["heat_bp"], bp_heating_grid_c, radius=1)
                    local_cool = _local_grid(meter_base["cool_bp"], bp_cooling_grid_c, radius=1)
                    local_enth = _local_grid(meter_base["enth_bp"], bp_enthalpy_grid_kjkg, radius=1)

                    sample = _sample_frame(group, max(3000, bp_fit_sample_size // 2), bp_random_state)
                    local_est = _fit_thresholds(sample, local_heat, local_cool, local_enth, meter_base)

                    w = float(len(group)) / float(len(group) + bp_shrinkage)
                    learned["group_params"][(meter_value, primary_use_value)] = {
                        "heat_bp": w * local_est["heat_bp"] + (1.0 - w) * meter_base["heat_bp"],
                        "cool_bp": w * local_est["cool_bp"] + (1.0 - w) * meter_base["cool_bp"],
                        "enth_bp": w * local_est["enth_bp"] + (1.0 - w) * meter_base["enth_bp"],
                    }

        e4_cache["learned_state"] = learned

    learned = e4_cache["learned_state"]
    medians = learned.get("medians", {})
    medians = {
        "air_temperature": medians.get("air_temperature", float(_num(df["air_temperature"]).median())),
        "dew_temperature": medians.get("dew_temperature", float(_num(df["dew_temperature"]).median())),
        "sea_level_pressure": medians.get("sea_level_pressure", float(_num(df["sea_level_pressure"]).median())),
        "wind_speed": medians.get("wind_speed", float(_num(df["wind_speed"]).median())),
        "cloud_coverage": medians.get("cloud_coverage", float(_num(df["cloud_coverage"]).median())),
        "square_feet": medians.get("square_feet", float(_num(df["square_feet"]).median())),
        "floor_count": medians.get("floor_count", float(_num(df["floor_count"]).median())),
        "year_built": medians.get("year_built", float(_num(df["year_built"]).median())),
    }

    meter_params = learned.get("meter_params", {})
    group_params = learned.get("group_params", {})

    # Build vectorized lookup tables for fitted or prior thresholds.
    meter_rows = []
    all_meter_keys = set(meter_priors.keys()) | set(meter_params.keys())
    for mk in all_meter_keys:
        base = dict(meter_priors.get(mk, meter_priors["__MISSING__"]))
        base.update(meter_params.get(mk, {}))
        meter_rows.append(
            {
                "_meter": mk,
                "heat_bp_meter": float(base["heat_bp"]),
                "cool_bp_meter": float(base["cool_bp"]),
                "enth_bp_meter": float(base["enth_bp"]),
            }
        )
    meter_param_df = pd.DataFrame(meter_rows)

    group_rows = []
    for (mk, pk), vals in group_params.items():
        group_rows.append(
            {
                "_meter": mk,
                "_primary_use": pk,
                "heat_bp_group": float(vals["heat_bp"]),
                "cool_bp_group": float(vals["cool_bp"]),
                "enth_bp_group": float(vals["enth_bp"]),
            }
        )
    group_param_df = pd.DataFrame(
        group_rows,
        columns=["_meter", "_primary_use", "heat_bp_group", "cool_bp_group", "enth_bp_group"],
    )

    keys = pd.DataFrame(
        {
            "_meter": _meter_key(df["meter"]),
            "_primary_use": _meter_key(df["primary_use"]),
        },
        index=df.index,
    )

    params = keys.merge(group_param_df, on=["_meter", "_primary_use"], how="left")
    params = params.merge(meter_param_df, on="_meter", how="left")
    params.index = df.index

    heat_bp = (
        _num(params["heat_bp_group"])
        .fillna(_num(params["heat_bp_meter"]))
        .fillna(meter_priors["__MISSING__"]["heat_bp"])
    )
    cool_bp = (
        _num(params["cool_bp_group"])
        .fillna(_num(params["cool_bp_meter"]))
        .fillna(meter_priors["__MISSING__"]["cool_bp"])
    )
    enth_bp = (
        _num(params["enth_bp_group"])
        .fillna(_num(params["enth_bp_meter"]))
        .fillna(meter_priors["__MISSING__"]["enth_bp"])
    )

    temp = _num(df["air_temperature"]).fillna(medians["air_temperature"]).astype(float)
    dew4 = _num(df["dew_temperature"]).fillna(medians["dew_temperature"]).astype(float)
    press4 = _num(df["sea_level_pressure"]).fillna(medians["sea_level_pressure"]).astype(float)
    wind4 = _num(df["wind_speed"]).fillna(medians["wind_speed"]).astype(float)
    cloud4 = _num(df["cloud_coverage"]).fillna(medians["cloud_coverage"]).astype(float)
    sqft4 = _num(df["square_feet"]).fillna(medians["square_feet"]).clip(lower=1.0).astype(float)
    floors4 = _num(df["floor_count"]).fillna(medians["floor_count"]).clip(lower=1.0).astype(float)
    year_built4 = _num(df["year_built"]).fillna(medians["year_built"]).astype(float)

    enth = _enthalpy_kjkg_from_hpa(temp, dew4, press4).astype(float)

    # Metadata modulation remains dense and low-cardinality.
    size_norm = (np.log1p(sqft4) - np.log1p(max(medians["square_feet"], 1.0))) / 1.25
    size_norm = np.clip(size_norm, -2.5, 2.5)

    floor_norm = (floors4 - max(medians["floor_count"], 1.0)) / max(medians["floor_count"], 1.0)
    floor_norm = np.clip(floor_norm, -2.5, 2.5)

    age_norm = (medians["year_built"] - year_built4) / 30.0
    age_norm = np.clip(age_norm, -2.5, 2.5)

    heat_factor = 1.0 + 0.06 * size_norm + 0.04 * floor_norm + 0.03 * np.maximum(age_norm, 0.0)
    cool_factor = 1.0 + 0.05 * size_norm + 0.05 * floor_norm
    latent_factor = 1.0 + 0.04 * size_norm

    heat_input = ((heat_bp - temp) + 0.20 * wind4 - 0.05 * cloud4) / bp_softplus_scale
    cool_input = (temp - cool_bp) / bp_softplus_scale
    latent_input = (enth - enth_bp) / (bp_softplus_scale * 3.5)

    bp_heat_drive = _softplus(heat_input.to_numpy(dtype=float), scale=1.0, clip=30.0) * np.asarray(
        heat_factor, dtype=float
    )
    bp_cool_drive = _softplus(cool_input.to_numpy(dtype=float), scale=1.0, clip=30.0) * np.asarray(
        cool_factor, dtype=float
    )
    bp_latent_drive = _softplus(latent_input.to_numpy(dtype=float), scale=1.0, clip=30.0) * np.asarray(
        latent_factor, dtype=float
    )

    band_width = np.maximum((cool_bp - heat_bp).to_numpy(dtype=float), 1.5)
    band_center = 0.5 * (heat_bp.to_numpy(dtype=float) + cool_bp.to_numpy(dtype=float))
    bp_thermal_position = np.tanh((temp.to_numpy(dtype=float) - band_center) / band_width)

    out_parts.append(
        pd.DataFrame(
            {
                "bp_heat_drive": bp_heat_drive.astype(np.float32),
                "bp_cool_drive": bp_cool_drive.astype(np.float32),
                "bp_latent_drive": bp_latent_drive.astype(np.float32),
                "bp_thermal_position": bp_thermal_position.astype(np.float32),
            },
            index=df.index,
        )
    )

    out = pd.concat(out_parts, axis=1)
    if len(out) != len(df):
        raise RuntimeError("Feature extractor changed the row count, which is not allowed.")
    return out
