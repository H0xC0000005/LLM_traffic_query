import numpy as np
import pandas as pd
from typing import Any, MutableMapping, Dict, Tuple


def extract_expert_features(
    df: pd.DataFrame,
    *,
    cache: dict | None = None,
    # --- Expert 1: segmented change-point signature ---
    sig_n_size_bins: int = 10,
    sig_min_samples_segment: int = 200,
    sig_min_samples_meter_primary: int = 500,
    sig_min_samples_meter: int = 1000,
    sig_max_points_per_group: int = 5000,
    sig_tb_heat_grid: "np.ndarray | None" = None,
    sig_tb_cool_grid: "np.ndarray | None" = None,
    sig_random_seed: int = 0,
    # --- Expert 2: Time-of-week × temperature schedule signature ---
    towt_sqft_quantiles: int = 4,
    towt_smoothing_k: float = 20.0,
    towt_ridge_alpha: float = 1.0,
    towt_min_segment_count: int = 200,
    towt_clip_quantile: float = 0.99,
    # --- Expert 3: archetype empirical bayes encoding ---
    eb_n_splits: int = 5,
    eb_n_bins_sqft: int = 8,
    eb_alpha_meter: float = 50.0,
    eb_alpha_cohort: float = 100.0,
    eb_alpha_fine: float = 150.0,
    eb_random_state: int = 42,
    # --- Expert 4: data quality ---
    dq_robust_z_thresh: float = 6.0,
) -> pd.DataFrame:
    """Combine expert features for building energy regression: (1) cached segmented 2-change-point signature vs
    temperature, (2) cached time-of-week schedule+weather signature, (3) leakage-safe Empirical-Bayes archetype
    encodings (OOF on train, cached apply on test), and (4) unsupervised data-quality metrics.

    Returns only engineered features (same row count/index as df) and never mutates df in-place. For inference without
    'meter_reading', pass the same cache previously fitted on training data.
    """

    if cache is None:
        cache = {}

    # Namespace the cache to avoid collisions with other pipelines.
    ns: MutableMapping[str, Any] = cache.setdefault("extract_expert_features_v2", {})
    c_sig: dict = ns.setdefault("segmented_change_point_signature_features", {})
    c_towt: dict = ns.setdefault("towt_schedule_signature_features", {})
    c_eb: dict = ns.setdefault("archetype_empirical_bayes_encoding", {})
    c_dq: dict = ns.setdefault("data_quality_missingness_validity", {})

    # Some experts require log_sqft explicitly; compute a local copy if absent (do not mutate df).
    if "log_sqft" not in df.columns and "square_feet" in df.columns:
        dfx = df.copy()
        dfx["log_sqft"] = np.log1p(pd.to_numeric(dfx["square_feet"], errors="coerce"))
    else:
        dfx = df

    # ---------------------------------------------------------------------
    # Expert 1: segmented_change_point_signature_features (verbatim algorithm)
    # ---------------------------------------------------------------------
    def segmented_change_point_signature_features(
        df: pd.DataFrame,
        *,
        cache: dict,
        n_size_bins: int = 10,
        min_samples_segment: int = 200,
        min_samples_meter_primary: int = 500,
        min_samples_meter: int = 1000,
        max_points_per_group: int = 5000,
        tb_heat_grid: np.ndarray | None = None,
        tb_cool_grid: np.ndarray | None = None,
        random_seed: int = 0,
    ) -> pd.DataFrame:
        cache_key = "segmented_signature_v1"
        req_cols = {"meter", "primary_use", "air_temperature", "is_business_hours"}
        missing = req_cols - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

        def _make_grids():
            h = tb_heat_grid
            c = tb_cool_grid
            if h is None:
                h = np.arange(6.0, 19.0, 2.0)  # 6,8,...,18
            if c is None:
                c = np.arange(18.0, 31.0, 2.0)  # 18,20,...,30
            return np.asarray(h, dtype=float), np.asarray(c, dtype=float)

        def _fit_params(T: np.ndarray, y: np.ndarray, h_grid: np.ndarray, c_grid: np.ndarray):
            n = T.size
            if n < 10:
                return None

            y = y.astype(float, copy=False)
            T = T.astype(float, copy=False)
            yTy = float(np.dot(y, y))
            sy = float(y.sum())

            best = None
            best_mse = np.inf

            for tbH in h_grid:
                xH = np.maximum(0.0, tbH - T)
                sH = float(xH.sum())
                sHH = float(np.dot(xH, xH))
                sHy = float(np.dot(xH, y))

                for tbC in c_grid:
                    if tbH > tbC:
                        continue

                    xC = np.maximum(0.0, T - tbC)
                    sC = float(xC.sum())
                    sCC = float(np.dot(xC, xC))
                    sHC = float(np.dot(xH, xC))
                    sCy = float(np.dot(xC, y))

                    XTX = np.array([[n, sH, sC], [sH, sHH, sHC], [sC, sHC, sCC]], dtype=float)
                    XTy = np.array([sy, sHy, sCy], dtype=float)

                    try:
                        beta = np.linalg.solve(XTX, XTy)
                    except np.linalg.LinAlgError:
                        continue

                    rss = yTy - float(beta @ XTy)
                    mse = rss / max(n, 1)
                    if mse < best_mse:
                        best_mse = mse
                        best = (float(tbH), float(tbC), float(beta[0]), float(beta[1]), float(beta[2]))

            if best is None:
                return None

            tbH, tbC, b0, bH, bC = best
            return (tbH, tbC, b0, max(bH, 0.0), max(bC, 0.0))

        def _iter_groups(code: np.ndarray):
            order = np.argsort(code, kind="mergesort")
            sorted_code = code[order]
            if sorted_code.size == 0:
                return
            boundaries = np.flatnonzero(sorted_code[1:] != sorted_code[:-1]) + 1
            starts = np.r_[0, boundaries]
            ends = np.r_[boundaries, sorted_code.size]
            for s, e in zip(starts, ends):
                yield int(sorted_code[s]), order[s:e]

        pu_raw = df["primary_use"].fillna("Unknown").astype(str)

        if cache_key in cache:
            state = cache[cache_key]
            pu_categories = state["primary_use_categories"]
            sqft_edges = np.asarray(state["sqft_edges"], dtype=float)
            B = int(state["n_size_bins_effective"])
            P = int(state["n_primary_categories"])
            params = state["params"]
        else:
            pu_unique = pd.Index(pu_raw.unique())
            if "Unknown" not in pu_unique:
                pu_unique = pu_unique.insert(len(pu_unique), "Unknown")
            pu_categories = list(pu_unique)

            if "log_sqft" in df.columns:
                size_x = pd.to_numeric(df["log_sqft"], errors="coerce")
            elif "square_feet" in df.columns:
                size_x = np.log1p(pd.to_numeric(df["square_feet"], errors="coerce"))
            else:
                raise KeyError("Need either 'log_sqft' or 'square_feet' to create size bins.")

            x = size_x.to_numpy(dtype=float, copy=False)
            x = x[np.isfinite(x)]
            if x.size == 0:
                sqft_edges = np.array([-np.inf, np.inf], dtype=float)
            else:
                qs = np.linspace(0.0, 1.0, n_size_bins + 1)
                edges = np.unique(np.quantile(x, qs))
                if edges.size < 2:
                    sqft_edges = np.array([-np.inf, np.inf], dtype=float)
                else:
                    edges[0] = -np.inf
                    edges[-1] = np.inf
                    sqft_edges = edges.astype(float)

            B = max(int(sqft_edges.size - 1), 1)
            P = len(pu_categories)
            params = None

        pu_aligned = pu_raw.where(pu_raw.isin(pu_categories), "Unknown")
        pu_code = pd.Categorical(pu_aligned, categories=pu_categories).codes.astype(np.int32)

        if "log_sqft" in df.columns:
            size_series = pd.to_numeric(df["log_sqft"], errors="coerce")
        else:
            size_series = np.log1p(pd.to_numeric(df["square_feet"], errors="coerce"))

        bin_code = pd.cut(size_series, bins=sqft_edges, labels=False, include_lowest=True)
        bin_code = pd.to_numeric(bin_code, errors="coerce").fillna(0).astype(np.int32).to_numpy()

        meter = pd.to_numeric(df["meter"], errors="coerce").fillna(0).astype(np.int32).clip(0, 3).to_numpy()
        bh = pd.to_numeric(df["is_business_hours"], errors="coerce").fillna(0).astype(np.int32).clip(0, 1).to_numpy()

        seg_code = (((meter * P + pu_code) * B + bin_code) * 2 + bh).astype(np.int64)
        n_segments_total = int(4 * P * B * 2)

        if params is None:
            if "meter_reading" not in df.columns:
                raise ValueError(
                    "No cached signature parameters found and 'meter_reading' is absent. "
                    "Run once on training data containing 'meter_reading' to fit and cache parameters."
                )

            rng = np.random.default_rng(random_seed)
            h_grid, c_grid = _make_grids()

            y_raw = pd.to_numeric(df["meter_reading"], errors="coerce").fillna(0.0).clip(lower=0.0)
            y = np.log1p(y_raw).to_numpy(dtype=float, copy=False)
            T = pd.to_numeric(df["air_temperature"], errors="coerce").to_numpy(dtype=float, copy=False)

            valid = np.isfinite(T) & np.isfinite(y)
            if int(valid.sum()) < 50:
                raise ValueError("Not enough valid rows to fit signature parameters.")

            seg_valid = seg_code[valid]
            meter_valid = meter[valid]
            pu_valid = pu_code[valid]
            mp_code_valid = (meter_valid * P + pu_valid).astype(np.int64)

            yv = y[valid]
            Tv = T[valid]

            global_params = _fit_params(Tv, yv, h_grid, c_grid)
            if global_params is None:
                global_params = (18.0, 18.0, float(np.nanmean(yv)), 0.0, 0.0)

            meter_params = np.full((4, 5), np.nan, dtype=float)
            for m in range(4):
                idx = np.flatnonzero(meter_valid == m)
                if idx.size >= min_samples_meter:
                    take = (
                        idx
                        if idx.size <= max_points_per_group
                        else rng.choice(idx, size=max_points_per_group, replace=False)
                    )
                    p = _fit_params(Tv[take], yv[take], h_grid, c_grid)
                    if p is not None:
                        meter_params[m, :] = p

            mp_params = np.full((4 * P, 5), np.nan, dtype=float)
            for mp, idx in _iter_groups(mp_code_valid):
                if idx.size >= min_samples_meter_primary:
                    take = (
                        idx
                        if idx.size <= max_points_per_group
                        else rng.choice(idx, size=max_points_per_group, replace=False)
                    )
                    p = _fit_params(Tv[take], yv[take], h_grid, c_grid)
                    if p is not None and 0 <= mp < mp_params.shape[0]:
                        mp_params[mp, :] = p

            seg_params = np.full((n_segments_total, 5), np.nan, dtype=float)
            for s, idx in _iter_groups(seg_valid):
                if idx.size >= min_samples_segment:
                    take = (
                        idx
                        if idx.size <= max_points_per_group
                        else rng.choice(idx, size=max_points_per_group, replace=False)
                    )
                    p = _fit_params(Tv[take], yv[take], h_grid, c_grid)
                    if p is not None and 0 <= s < n_segments_total:
                        seg_params[s, :] = p

            for m in range(4):
                m_p = meter_params[m, :]
                use_m = np.isfinite(m_p).all()
                for pcode in range(P):
                    mp = m * P + pcode
                    mp_p = mp_params[mp, :]
                    use_mp = np.isfinite(mp_p).all()
                    for b in range(B):
                        for bhv in (0, 1):
                            s = ((m * P + pcode) * B + b) * 2 + bhv
                            if np.isfinite(seg_params[s, :]).all():
                                continue
                            if use_mp:
                                seg_params[s, :] = mp_p
                            elif use_m:
                                seg_params[s, :] = m_p
                            else:
                                seg_params[s, :] = global_params

            params = {
                "tb_heat": seg_params[:, 0].astype(float),
                "tb_cool": seg_params[:, 1].astype(float),
                "b0": seg_params[:, 2].astype(float),
                "bH": seg_params[:, 3].astype(float),
                "bC": seg_params[:, 4].astype(float),
            }

            cache[cache_key] = {
                "primary_use_categories": pu_categories,
                "sqft_edges": sqft_edges.tolist(),
                "n_size_bins_effective": B,
                "n_primary_categories": P,
                "params": params,
                "meta": {
                    "fit_on_log1p": True,
                    "min_samples_segment": int(min_samples_segment),
                    "min_samples_meter_primary": int(min_samples_meter_primary),
                    "min_samples_meter": int(min_samples_meter),
                    "max_points_per_group": int(max_points_per_group),
                    "tb_heat_grid": h_grid.tolist(),
                    "tb_cool_grid": c_grid.tolist(),
                    "random_seed": int(random_seed),
                },
            }

        seg_code_safe = np.clip(seg_code, 0, len(params["tb_heat"]) - 1)
        tbH = params["tb_heat"][seg_code_safe]
        tbC = params["tb_cool"][seg_code_safe]
        b0 = params["b0"][seg_code_safe]
        bH = params["bH"][seg_code_safe]
        bC = params["bC"][seg_code_safe]

        T_all = pd.to_numeric(df["air_temperature"], errors="coerce").to_numpy(dtype=float, copy=False)
        T_filled = np.where(np.isfinite(T_all), T_all, (tbH + tbC) / 2.0)

        hdh_bal = np.maximum(0.0, tbH - T_filled)
        cdh_bal = np.maximum(0.0, T_filled - tbC)

        sig_pred_log1p = b0 + bH * hdh_bal + bC * cdh_bal
        lin_weather = bH * hdh_bal + bC * cdh_bal
        sig_weather_intensity = np.abs(lin_weather) / (np.abs(b0) + np.abs(lin_weather) + 1e-6)
        sig_deadband = tbC - tbH

        return pd.DataFrame(
            {
                "hdh_bal": hdh_bal.astype(np.float32),
                "cdh_bal": cdh_bal.astype(np.float32),
                "sig_pred_log1p": sig_pred_log1p.astype(np.float32),
                "sig_weather_intensity": sig_weather_intensity.astype(np.float32),
                "sig_deadband": sig_deadband.astype(np.float32),
            },
            index=df.index,
        )

    # ---------------------------------------------------------------------
    # Expert 2: towt_schedule_signature_features (verbatim algorithm)
    # ---------------------------------------------------------------------
    def towt_schedule_signature_features(
        df: pd.DataFrame,
        *,
        cache: dict,
        sqft_quantiles: int = 4,
        smoothing_k: float = 20.0,
        ridge_alpha: float = 1.0,
        min_segment_count: int = 200,
        clip_quantile: float = 0.99,
    ) -> pd.DataFrame:
        required = [
            "meter",
            "primary_use",
            "log_sqft",
            "year_built_clipped",
            "dayofweek",
            "hour_sin",
            "hour_cos",
            "is_business_hours",
            "CDH_18C",
            "HDH_18C",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        def _hour_from_sincos(sin_v: np.ndarray, cos_v: np.ndarray) -> np.ndarray:
            angle = (np.arctan2(sin_v, cos_v) + 2.0 * np.pi) % (2.0 * np.pi)
            hour = angle * 24.0 / (2.0 * np.pi)
            return np.floor(hour + 1e-6).astype(np.int16) % 24

        def _year_bin(series: pd.Series) -> pd.Series:
            b = pd.cut(series, bins=[1919, 1945, 1979, 2016], labels=["y1", "y2", "y3"], include_lowest=True)
            return b.astype("object").where(~b.isna(), "yna")

        def _make_sqft_bins(log_sqft: pd.Series, edges: np.ndarray | None) -> tuple[pd.Series, np.ndarray]:
            x = log_sqft.astype(float)
            if edges is None:
                x_nonan = x.dropna()
                if x_nonan.empty:
                    edges = np.array([0.0, 1.0], dtype=float)
                else:
                    qs = np.linspace(0.0, 1.0, max(2, int(sqft_quantiles)) + 1)
                    edges = np.unique(np.quantile(x_nonan.to_numpy(), qs))
                    if edges.size < 2:
                        m = float(np.nanmedian(x_nonan.to_numpy()))
                        edges = np.array([m - 1.0, m + 1.0], dtype=float)
            edges = np.unique(edges)
            if edges.size < 2:
                edges = np.array([0.0, 1.0], dtype=float)

            bins = pd.cut(x, bins=edges, include_lowest=True, labels=False)
            bins = bins.astype("float").where(~bins.isna(), -1.0).astype(np.int16)
            return bins, edges

        def _segment_key(meter: pd.Series, use: pd.Series, sqft_bin: pd.Series, ybin: pd.Series) -> pd.Series:
            return (
                "m"
                + meter.astype(int).astype(str)
                + "|u"
                + use.astype(str)
                + "|s"
                + sqft_bin.astype(int).astype(str)
                + "|y"
                + ybin.astype(str)
            )

        def _meter_key(meter: pd.Series) -> pd.Series:
            return "m" + meter.astype(int).astype(str)

        def _fit_ridge_no_intercept(x1, x2, y, alpha):
            s11 = float(np.dot(x1, x1))
            s22 = float(np.dot(x2, x2))
            s12 = float(np.dot(x1, x2))
            t1 = float(np.dot(x1, y))
            t2 = float(np.dot(x2, y))
            a11 = s11 + alpha
            a22 = s22 + alpha
            det = a11 * a22 - s12 * s12
            if det <= 1e-12:
                return 0.0, 0.0
            b1 = (t1 * a22 - t2 * s12) / det
            b2 = (t2 * a11 - t1 * s12) / det
            return float(b1), float(b2)

        hour = _hour_from_sincos(df["hour_sin"].to_numpy(dtype=float), df["hour_cos"].to_numpy(dtype=float))
        day = df["dayofweek"].astype(int).to_numpy()
        tow = (day * 24 + hour).astype(np.int16)

        ybin = _year_bin(df["year_built_clipped"])
        model_key = "towt_schedule_signature_v1"
        state = cache.setdefault(model_key, {})

        edges = state.get("sqft_edges", None)
        sqft_bin, edges = _make_sqft_bins(df["log_sqft"], edges)
        state["sqft_edges"] = edges

        seg = _segment_key(df["meter"], df["primary_use"], sqft_bin, ybin)
        mkey = _meter_key(df["meter"])

        is_bh = df["is_business_hours"].astype(int).to_numpy() == 1

        cdh = df["CDH_18C"].astype(float).to_numpy()
        hdh = df["HDH_18C"].astype(float).to_numpy()

        has_label = "meter_reading" in df.columns

        if has_label:
            y_raw = df["meter_reading"].astype(float).to_numpy()
            y = np.log1p(np.maximum(y_raw, 0.0))

            if "cdh_clip" not in state or "hdh_clip" not in state:
                cdh_clip = float(np.nanquantile(cdh, clip_quantile)) if np.isfinite(cdh).any() else 0.0
                hdh_clip = float(np.nanquantile(hdh, clip_quantile)) if np.isfinite(hdh).any() else 0.0
                state["cdh_clip"] = max(0.0, cdh_clip)
                state["hdh_clip"] = max(0.0, hdh_clip)

            cdh = np.clip(cdh, 0.0, state["cdh_clip"])
            hdh = np.clip(hdh, 0.0, state["hdh_clip"])

            tmp = pd.DataFrame({"seg": seg, "mkey": mkey, "tow": tow, "y": y})

            seg_stats = tmp.groupby("seg")["y"].agg(seg_mean="mean", seg_count="size")
            m_stats = tmp.groupby("mkey")["y"].agg(m_mean="mean", m_count="size")
            global_mean = float(tmp["y"].mean()) if len(tmp) else 0.0

            g = tmp.groupby(["seg", "tow"])["y"].agg(["mean", "count"])
            seg_mean_map = seg_stats["seg_mean"]
            base_seg = (g["count"] * g["mean"] + smoothing_k * g.index.get_level_values(0).map(seg_mean_map)) / (
                g["count"] + smoothing_k
            )
            base_seg.name = "base"

            gm = tmp.groupby(["mkey", "tow"])["y"].agg(["mean", "count"])
            m_mean_map = m_stats["m_mean"]
            base_m = (gm["count"] * gm["mean"] + smoothing_k * gm.index.get_level_values(0).map(m_mean_map)) / (
                gm["count"] + smoothing_k
            )
            base_m.name = "base"

            def _schedule_amp(base_series: pd.Series, overall_mean: pd.Series) -> pd.Series:
                n_obs = base_series.groupby(level=0).size()
                sum_v = base_series.groupby(level=0).sum()
                sum_v2 = (base_series**2).groupby(level=0).sum()
                all_bins = 168.0
                om = overall_mean.reindex(sum_v.index).astype(float)
                miss = (all_bins - n_obs.reindex(sum_v.index).astype(float)).clip(lower=0.0)
                mean_all = (sum_v + miss * (om)) / all_bins
                mean2_all = (sum_v2 + miss * (om**2)) / all_bins
                var = (mean2_all - mean_all**2).clip(lower=0.0)
                return np.sqrt(var)

            seg_amp = _schedule_amp(base_seg, seg_stats["seg_mean"])
            m_amp = _schedule_amp(base_m, m_stats["m_mean"])

            idx_seg = pd.MultiIndex.from_arrays([seg.to_numpy(), tow])
            base_seg_lookup = base_seg.reindex(idx_seg).to_numpy()
            base_seg_lookup = np.where(
                np.isfinite(base_seg_lookup), base_seg_lookup, seg.map(seg_stats["seg_mean"]).to_numpy(dtype=float)
            )
            base_seg_lookup = np.where(
                np.isfinite(base_seg_lookup), base_seg_lookup, mkey.map(m_stats["m_mean"]).to_numpy(dtype=float)
            )
            base_seg_lookup = np.where(np.isfinite(base_seg_lookup), base_seg_lookup, global_mean)

            resid = y - base_seg_lookup

            resid_std = pd.Series(resid).groupby(seg).std().fillna(0.0)
            m_resid_std = pd.Series(resid).groupby(mkey).std().fillna(0.0)
            global_resid_std = float(np.nanstd(resid)) if resid.size else 0.0

            X = pd.DataFrame(
                {"seg": seg, "mkey": mkey, "bh": is_bh.astype(np.int8), "cdh": cdh, "hdh": hdh, "r": resid}
            )

            def _fit_group_slopes(group: pd.DataFrame) -> tuple[float, float]:
                if len(group) < 30:
                    return 0.0, 0.0
                b1, b2 = _fit_ridge_no_intercept(
                    group["cdh"].to_numpy(dtype=float),
                    group["hdh"].to_numpy(dtype=float),
                    group["r"].to_numpy(dtype=float),
                    ridge_alpha,
                )
                return float(np.clip(b1, -2.0, 2.0)), float(np.clip(b2, -2.0, 2.0))

            seg_bh = X[X["bh"] == 1].groupby("seg", sort=False).apply(_fit_group_slopes)
            seg_off = X[X["bh"] == 0].groupby("seg", sort=False).apply(_fit_group_slopes)
            seg_bh = seg_bh.apply(lambda t: pd.Series({"b_cdh_bh": t[0], "b_hdh_bh": t[1]}))
            seg_off = seg_off.apply(lambda t: pd.Series({"b_cdh_off": t[0], "b_hdh_off": t[1]}))
            seg_slopes = seg_bh.join(seg_off, how="outer").fillna(0.0)

            m_bh = X[X["bh"] == 1].groupby("mkey", sort=False).apply(_fit_group_slopes)
            m_off = X[X["bh"] == 0].groupby("mkey", sort=False).apply(_fit_group_slopes)
            m_bh = m_bh.apply(lambda t: pd.Series({"b_cdh_bh": t[0], "b_hdh_bh": t[1]}))
            m_off = m_off.apply(lambda t: pd.Series({"b_cdh_off": t[0], "b_hdh_off": t[1]}))
            m_slopes = m_bh.join(m_off, how="outer").fillna(0.0)

            state["fitted"] = True
            state["min_segment_count"] = int(min_segment_count)
            state["global_mean"] = float(global_mean)
            state["global_resid_std"] = float(global_resid_std)
            state["seg_stats"] = seg_stats
            state["m_stats"] = m_stats
            state["base_seg"] = base_seg
            state["base_m"] = base_m
            state["seg_amp"] = seg_amp
            state["m_amp"] = m_amp
            state["resid_std"] = resid_std
            state["m_resid_std"] = m_resid_std
            state["seg_slopes"] = seg_slopes
            state["m_slopes"] = m_slopes

        if not state.get("fitted", False):
            raise ValueError("Feature cache is not fitted yet. Call on training data (with meter_reading) first.")

        cdh = np.clip(cdh, 0.0, float(state.get("cdh_clip", 0.0)))
        hdh = np.clip(hdh, 0.0, float(state.get("hdh_clip", 0.0)))

        seg_stats = state["seg_stats"]
        m_stats = state["m_stats"]
        base_seg = state["base_seg"]
        base_m = state["base_m"]
        seg_amp = state["seg_amp"]
        m_amp = state["m_amp"]
        resid_std = state["resid_std"]
        m_resid_std = state["m_resid_std"]
        seg_slopes = state["seg_slopes"]
        m_slopes = state["m_slopes"]

        min_seg = int(state.get("min_segment_count", min_segment_count))
        global_mean = float(state.get("global_mean", 0.0))
        global_resid_std = float(state.get("global_resid_std", 0.0))

        seg_count_map = seg.map(seg_stats["seg_count"]).to_numpy(dtype=float)
        use_seg = np.isfinite(seg_count_map) & (seg_count_map >= min_seg)

        idx_seg = pd.MultiIndex.from_arrays([seg.to_numpy(), tow])
        idx_m = pd.MultiIndex.from_arrays([mkey.to_numpy(), tow])

        base_seg_lookup = base_seg.reindex(idx_seg).to_numpy()
        base_m_lookup = base_m.reindex(idx_m).to_numpy()

        seg_mean_lookup = seg.map(seg_stats["seg_mean"]).to_numpy(dtype=float)
        m_mean_lookup = mkey.map(m_stats["m_mean"]).to_numpy(dtype=float)

        base_seg_lookup = np.where(np.isfinite(base_seg_lookup), base_seg_lookup, seg_mean_lookup)
        base_m_lookup = np.where(np.isfinite(base_m_lookup), base_m_lookup, m_mean_lookup)
        base_seg_lookup = np.where(np.isfinite(base_seg_lookup), base_seg_lookup, global_mean)
        base_m_lookup = np.where(np.isfinite(base_m_lookup), base_m_lookup, global_mean)

        baseline = np.where(use_seg, base_seg_lookup, base_m_lookup)

        amp_seg = seg.map(seg_amp).to_numpy(dtype=float)
        amp_m = mkey.map(m_amp).to_numpy(dtype=float)
        amp = np.where(np.isfinite(amp_seg) & use_seg, amp_seg, amp_m)
        amp = np.where(np.isfinite(amp), amp, 0.0)

        rstd_seg = seg.map(resid_std).to_numpy(dtype=float)
        rstd_m = mkey.map(m_resid_std).to_numpy(dtype=float)
        rstd = np.where(np.isfinite(rstd_seg) & use_seg, rstd_seg, rstd_m)
        rstd = np.where(np.isfinite(rstd), rstd, global_resid_std)

        def _map_slopes(key_series: pd.Series, table: pd.DataFrame, col: str) -> np.ndarray:
            out = key_series.map(table[col]).to_numpy(dtype=float)
            return np.where(np.isfinite(out), out, 0.0)

        b_cdh_bh_seg = _map_slopes(seg, seg_slopes, "b_cdh_bh")
        b_hdh_bh_seg = _map_slopes(seg, seg_slopes, "b_hdh_bh")
        b_cdh_off_seg = _map_slopes(seg, seg_slopes, "b_cdh_off")
        b_hdh_off_seg = _map_slopes(seg, seg_slopes, "b_hdh_off")

        b_cdh_bh_m = _map_slopes(mkey, m_slopes, "b_cdh_bh")
        b_hdh_bh_m = _map_slopes(mkey, m_slopes, "b_hdh_bh")
        b_cdh_off_m = _map_slopes(mkey, m_slopes, "b_cdh_off")
        b_hdh_off_m = _map_slopes(mkey, m_slopes, "b_hdh_off")

        b_cdh_bh = np.where(use_seg, b_cdh_bh_seg, b_cdh_bh_m)
        b_hdh_bh = np.where(use_seg, b_hdh_bh_seg, b_hdh_bh_m)
        b_cdh_off = np.where(use_seg, b_cdh_off_seg, b_cdh_off_m)
        b_hdh_off = np.where(use_seg, b_hdh_off_seg, b_hdh_off_m)

        b_cdh = np.where(is_bh, b_cdh_bh, b_cdh_off)
        b_hdh = np.where(is_bh, b_hdh_bh, b_hdh_off)
        pred_log1p = baseline + b_cdh * cdh + b_hdh * hdh

        hot_delta = b_cdh_bh - b_cdh_off
        cold_delta = b_hdh_bh - b_hdh_off

        return pd.DataFrame(
            {
                "towt_pred_log1p": pred_log1p.astype(np.float32),
                "towt_seg_schedule_amp": amp.astype(np.float32),
                "towt_seg_resid_scale": rstd.astype(np.float32),
                "towt_seg_hot_slope_delta": hot_delta.astype(np.float32),
                "towt_seg_cold_slope_delta": cold_delta.astype(np.float32),
            },
            index=df.index,
        )

    # ---------------------------------------------------------------------
    # Expert 3: archetype_empirical_bayes_encoding (verbatim algorithm)
    # ---------------------------------------------------------------------
    def archetype_empirical_bayes_encoding(
        df: pd.DataFrame,
        *,
        cache: dict,
        n_splits: int = 5,
        n_bins_sqft: int = 8,
        alpha_meter: float = 50.0,
        alpha_cohort: float = 100.0,
        alpha_fine: float = 150.0,
        random_state: int = 42,
    ) -> pd.DataFrame:
        def _safe_log_sqft(dfx: pd.DataFrame) -> pd.Series:
            if "log_sqft" in dfx.columns:
                return pd.to_numeric(dfx["log_sqft"], errors="coerce")
            return np.log1p(pd.to_numeric(dfx["square_feet"], errors="coerce"))

        def _compute_bin_edges(log_sqft: pd.Series, n_bins: int) -> np.ndarray:
            x = log_sqft.to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if x.size < 10:
                lo, hi = (0.0, 20.0) if x.size == 0 else (float(np.nanmin(x)), float(np.nanmax(x)))
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    lo, hi = 0.0, 20.0
                edges = np.linspace(lo, hi, max(2, n_bins + 1))
            else:
                qs = np.linspace(0.0, 1.0, n_bins + 1)
                edges = np.nanquantile(x, qs)
                edges = np.unique(edges)
                if edges.size < 3:
                    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
                    if lo == hi:
                        lo, hi = lo - 1.0, hi + 1.0
                    edges = np.linspace(lo, hi, max(2, n_bins + 1))
            edges = edges.astype(float)
            edges[0] -= 1e-6
            edges[-1] += 1e-6
            return edges

        def _assign_size_bin(log_sqft: pd.Series, edges: np.ndarray) -> pd.Series:
            b = pd.cut(log_sqft, bins=edges, labels=False, include_lowest=True)
            return b.astype("float").fillna(-1).astype(int)

        def _fit_stats(dfx: pd.DataFrame, y_log: pd.Series, edges: np.ndarray) -> dict:
            d = pd.DataFrame(
                {
                    "meter": pd.to_numeric(dfx["meter"], errors="coerce").astype("Int64"),
                    "primary_use": dfx["primary_use"].astype("string").fillna("__NA__"),
                    "size_bin": _assign_size_bin(_safe_log_sqft(dfx), edges),
                    "y": pd.to_numeric(y_log, errors="coerce"),
                }
            ).dropna(subset=["meter", "y"])
            d["meter"] = d["meter"].astype(int)

            mu0 = float(d["y"].mean())
            var0 = float(d["y"].var(ddof=0))
            if not np.isfinite(var0) or var0 <= 1e-12:
                var0 = 1.0

            m = d.groupby("meter")["y"].agg(
                n="count", mean="mean", var=lambda s: float(s.var(ddof=0) if s.size > 1 else 0.0)
            )
            m["var"] = m["var"].clip(lower=0.0)
            m["post_mean"] = (m["n"] * m["mean"] + alpha_meter * mu0) / (m["n"] + alpha_meter)
            m["post_var"] = (m["n"] * m["var"] + alpha_meter * var0) / (m["n"] + alpha_meter)

            c = d.groupby(["meter", "primary_use"])["y"].agg(
                n="count", mean="mean", var=lambda s: float(s.var(ddof=0) if s.size > 1 else 0.0)
            )
            c["var"] = c["var"].clip(lower=0.0)
            c = c.join(m[["post_mean", "post_var"]], on="meter", rsuffix="_meter")
            c["prior_mean"] = c["post_mean"]
            c["prior_var"] = c["post_var"]
            c["post_mean"] = (c["n"] * c["mean"] + alpha_cohort * c["prior_mean"]) / (c["n"] + alpha_cohort)
            c["post_var"] = (c["n"] * c["var"] + alpha_cohort * c["prior_var"]) / (c["n"] + alpha_cohort)
            c = c.drop(columns=["prior_mean", "prior_var"])

            f = d.groupby(["meter", "primary_use", "size_bin"])["y"].agg(
                n="count", mean="mean", var=lambda s: float(s.var(ddof=0) if s.size > 1 else 0.0)
            )
            f["var"] = f["var"].clip(lower=0.0)
            f = f.join(c[["post_mean", "post_var"]], on=["meter", "primary_use"], rsuffix="_cohort")
            f["prior_mean"] = f["post_mean"]
            f["prior_var"] = f["post_var"]
            f["post_mean"] = (f["n"] * f["mean"] + alpha_fine * f["prior_mean"]) / (f["n"] + alpha_fine)
            f["post_var"] = (f["n"] * f["var"] + alpha_fine * f["prior_var"]) / (f["n"] + alpha_fine)
            f = f.drop(columns=["prior_mean", "prior_var"])

            max_n_fine = int(f["n"].max()) if len(f) else 0
            return {
                "edges": edges,
                "mu0": mu0,
                "var0": var0,
                "meter_tbl": m[["n", "post_mean", "post_var"]],
                "cohort_tbl": c[["n", "post_mean", "post_var"]],
                "fine_tbl": f[["n", "post_mean", "post_var"]],
                "max_n_fine": max_n_fine,
            }

        def _apply_stats(dfx: pd.DataFrame, stats: dict) -> pd.DataFrame:
            edges = stats["edges"]
            mu0 = stats["mu0"]
            var0 = stats["var0"]

            base = pd.DataFrame(
                {
                    "meter": pd.to_numeric(dfx["meter"], errors="coerce").astype("Int64"),
                    "primary_use": dfx["primary_use"].astype("string").fillna("__NA__"),
                    "size_bin": _assign_size_bin(_safe_log_sqft(dfx), edges),
                },
                index=dfx.index,
            )
            base["meter_int"] = base["meter"].fillna(-9999).astype(int)

            m = stats["meter_tbl"]
            meter_join = (
                base[["meter_int"]]
                .rename(columns={"meter_int": "meter"})
                .merge(m.reset_index(), on="meter", how="left")
                .set_index(base.index)
            )
            meter_mean = meter_join["post_mean"].fillna(mu0)
            meter_var = meter_join["post_var"].fillna(var0)

            c = stats["cohort_tbl"]
            cohort_join = (
                base[["meter_int", "primary_use"]]
                .rename(columns={"meter_int": "meter"})
                .merge(c.reset_index(), on=["meter", "primary_use"], how="left")
                .set_index(base.index)
            )
            cohort_mean = cohort_join["post_mean"].fillna(meter_mean)
            cohort_var = cohort_join["post_var"].fillna(meter_var)

            f = stats["fine_tbl"]
            fine_join = (
                base[["meter_int", "primary_use", "size_bin"]]
                .rename(columns={"meter_int": "meter"})
                .merge(f.reset_index(), on=["meter", "primary_use", "size_bin"], how="left")
                .set_index(base.index)
            )
            fine_mean = fine_join["post_mean"].fillna(cohort_mean)
            fine_var = fine_join["post_var"].fillna(cohort_var)

            n_fine = fine_join["n"].fillna(0.0)
            denom = np.log1p(max(stats.get("max_n_fine", 0), 1))
            neff = (np.log1p(n_fine) / denom).clip(lower=0.0, upper=1.0)

            return pd.DataFrame(
                {
                    "archetype_log_mean": fine_mean.astype("float32"),
                    "archetype_log_std": np.sqrt(np.maximum(fine_var.to_numpy(dtype=float), 1e-12)).astype("float32"),
                    "archetype_neff": neff.astype("float32"),
                    "archetype_size_adj": (fine_mean - cohort_mean).fillna(0.0).astype("float32"),
                },
                index=dfx.index,
            )

        def _make_stratified_folds_by_meter(dfx: pd.DataFrame, k: int, seed: int) -> list[np.ndarray]:
            rng = np.random.default_rng(seed)
            folds = [[] for _ in range(k)]
            for _, idx in dfx.groupby("meter").indices.items():
                idx = np.asarray(list(idx), dtype=int)
                if idx.size == 0:
                    continue
                rng.shuffle(idx)
                parts = np.array_split(idx, k)
                for i, part in enumerate(parts):
                    if part.size:
                        folds[i].append(part)
            return [np.concatenate(p) if len(p) else np.array([], dtype=int) for p in folds]

        has_y = "meter_reading" in df.columns and pd.to_numeric(df["meter_reading"], errors="coerce").notna().any()

        enc_key = "archetype_empirical_bayes_encoding"
        if enc_key not in cache:
            edges = _compute_bin_edges(_safe_log_sqft(df), n_bins_sqft)
            cache[enc_key] = {"edges": edges}
        else:
            edges = cache[enc_key].get("edges")
            if edges is None:
                edges = _compute_bin_edges(_safe_log_sqft(df), n_bins_sqft)
                cache[enc_key]["edges"] = edges

        if has_y:
            y_log = np.log1p(pd.to_numeric(df["meter_reading"], errors="coerce"))
            folds = _make_stratified_folds_by_meter(df, max(2, int(n_splits)), random_state)

            feat_oof = pd.DataFrame(
                index=df.index,
                columns=["archetype_log_mean", "archetype_log_std", "archetype_neff", "archetype_size_adj"],
                dtype="float32",
            )

            all_idx = np.arange(len(df), dtype=int)
            for val_idx in folds:
                if val_idx.size == 0:
                    continue
                val_mask = np.zeros(len(df), dtype=bool)
                val_mask[val_idx] = True
                trn_idx = all_idx[~val_mask]
                stats = _fit_stats(df.iloc[trn_idx], y_log.iloc[trn_idx], edges)
                feat_oof.iloc[val_idx] = _apply_stats(df.iloc[val_idx], stats).to_numpy()

            cache[enc_key].update({"stats": _fit_stats(df, y_log, edges)})
            return feat_oof

        if "stats" not in cache.get(enc_key, {}):
            raise ValueError(
                "No cached training stats found for archetype encoding. "
                "Call this function once on training data that includes 'meter_reading' first."
            )
        return _apply_stats(df, cache[enc_key]["stats"])

    # ---------------------------------------------------------------------
    # Expert 4: data_quality_missingness_validity (verbatim algorithm)
    # ---------------------------------------------------------------------
    def data_quality_missingness_validity(
        df: pd.DataFrame,
        *,
        cache: MutableMapping[str, Any],
        robust_z_thresh: float = 6.0,
    ) -> pd.DataFrame:
        weather_cols = [
            c
            for c in [
                "air_temperature",
                "dew_temperature",
                "wind_speed",
                "cloud_coverage",
                "precip_depth_1_hr",
                "sea_level_pressure",
                "wind_direction",
            ]
            if c in df.columns
        ]
        building_nullable_cols = [c for c in ["floor_count", "year_built_clipped"] if c in df.columns]

        dq_missing_weather_rate = (
            df[weather_cols].isna().mean(axis=1) if weather_cols else pd.Series(0.0, index=df.index)
        )
        dq_missing_building_rate = (
            df[building_nullable_cols].isna().mean(axis=1) if building_nullable_cols else pd.Series(0.0, index=df.index)
        )

        checks = []

        def s(col: str) -> pd.Series:
            return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)

        air, dew = s("air_temperature"), s("dew_temperature")
        wspd, pres = s("wind_speed"), s("sea_level_pressure")
        cloud, precip = s("cloud_coverage"), s("precip_depth_1_hr")
        wdir, floors = s("wind_direction"), s("floor_count")
        yb = s("year_built_clipped")

        checks.append((air.notna()) & ((air < -80.0) | (air > 60.0)))
        checks.append((dew.notna()) & ((dew < -100.0) | (dew > 50.0)))
        checks.append((air.notna() & dew.notna()) & (dew > air + 0.5))
        checks.append((wspd.notna()) & ((wspd < 0.0) | (wspd > 80.0)))
        checks.append((pres.notna()) & ((pres < 800.0) | (pres > 1100.0)))
        checks.append((cloud.notna()) & ((cloud < 0.0) | (cloud > 9.0)))
        checks.append((precip.notna()) & ((precip < 0.0) | (precip > 500.0)))
        checks.append((wdir.notna()) & ((wdir < 0.0) | (wdir > 360.0)))
        checks.append((floors.notna()) & ((floors < 1.0) | (floors > 200.0)))
        checks.append((yb.notna()) & ((yb < 1920.0) | (yb > 2016.0)))

        invalid_count = np.zeros(len(df), dtype=np.float32)
        for cond in checks:
            invalid_count += cond.fillna(False).to_numpy(dtype=np.float32)
        dq_invalid_rate = invalid_count / max(len(checks), 1)

        outlier_cols = [
            c
            for c in ["air_temperature", "dew_temperature", "wind_speed", "sea_level_pressure", "precip_depth_1_hr"]
            if c in df.columns
        ]

        def _median_mad(x: pd.Series) -> Tuple[float, float]:
            x = pd.to_numeric(x, errors="coerce")
            x = x[np.isfinite(x)]
            if x.size < 50:
                return (np.nan, np.nan)
            med = float(np.nanmedian(x))
            mad = float(np.nanmedian(np.abs(x - med)))
            return (med, mad)

        stats: Dict[str, Tuple[float, float]] = cache.get("dq_robust_stats", {})
        if not isinstance(stats, dict):
            stats = {}
        for c in outlier_cols:
            if c not in stats or stats[c] is None or not np.isfinite(stats[c][0]) or not np.isfinite(stats[c][1]):
                stats[c] = _median_mad(df[c])
        cache["dq_robust_stats"] = stats

        outlier_count = np.zeros(len(df), dtype=np.float32)
        used_outlier_cols = 0
        for c in outlier_cols:
            med, mad = stats.get(c, (np.nan, np.nan))
            if not (np.isfinite(med) and np.isfinite(mad)) or mad <= 0.0:
                continue
            x = pd.to_numeric(df[c], errors="coerce")
            rz = 0.6745 * (x - med) / (mad + 1e-9)
            flag = (x.notna()) & (np.abs(rz) > robust_z_thresh)
            outlier_count += flag.to_numpy(dtype=np.float32)
            used_outlier_cols += 1
        dq_outlier_rate = outlier_count / max(used_outlier_cols, 1)

        badness = (
            0.70 * dq_missing_weather_rate.to_numpy(dtype=np.float32)
            + 0.30 * dq_missing_building_rate.to_numpy(dtype=np.float32)
            + 0.80 * dq_invalid_rate.astype(np.float32)
            + 0.50 * dq_outlier_rate.astype(np.float32)
        )
        dq_quality_score = np.exp(-badness).astype(np.float32)

        return pd.DataFrame(
            {
                "dq_missing_weather_rate": dq_missing_weather_rate.astype(np.float32),
                "dq_missing_building_rate": dq_missing_building_rate.astype(np.float32),
                "dq_invalid_rate": dq_invalid_rate.astype(np.float32),
                "dq_outlier_rate": dq_outlier_rate.astype(np.float32),
                "dq_quality_score": dq_quality_score,
            },
            index=df.index,
        )

    # ----------------------------
    # Run each expert block (no in-place mutation of df)
    # ----------------------------
    f_sig = segmented_change_point_signature_features(
        dfx,
        cache=c_sig,
        n_size_bins=int(sig_n_size_bins),
        min_samples_segment=int(sig_min_samples_segment),
        min_samples_meter_primary=int(sig_min_samples_meter_primary),
        min_samples_meter=int(sig_min_samples_meter),
        max_points_per_group=int(sig_max_points_per_group),
        tb_heat_grid=sig_tb_heat_grid,
        tb_cool_grid=sig_tb_cool_grid,
        random_seed=int(sig_random_seed),
    )

    f_towt = towt_schedule_signature_features(
        dfx,
        cache=c_towt,
        sqft_quantiles=int(towt_sqft_quantiles),
        smoothing_k=float(towt_smoothing_k),
        ridge_alpha=float(towt_ridge_alpha),
        min_segment_count=int(towt_min_segment_count),
        clip_quantile=float(towt_clip_quantile),
    )

    f_eb = archetype_empirical_bayes_encoding(
        dfx,
        cache=c_eb,
        n_splits=int(eb_n_splits),
        n_bins_sqft=int(eb_n_bins_sqft),
        alpha_meter=float(eb_alpha_meter),
        alpha_cohort=float(eb_alpha_cohort),
        alpha_fine=float(eb_alpha_fine),
        random_state=int(eb_random_state),
    )

    f_dq = data_quality_missingness_validity(dfx, cache=c_dq, robust_z_thresh=float(dq_robust_z_thresh))

    return pd.concat([f_sig, f_towt, f_eb, f_dq], axis=1)
