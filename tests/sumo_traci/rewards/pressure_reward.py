from __future__ import annotations

from typing import Any

import numpy as np

from .base import downstream_count_ratio_norm, get_unique_tls_movements, lane_value_map


def pressure_reward(
    *,
    tls_id: str,
    sim_time: float,
    scene_stats: Any,
    cache: dict | None = None,
    gamma_dt: float | None = None,
    init_only: bool = False,
    upstream_key: str = "count_ratio_norm",
    veh_equiv_len_m: float = 7.5,
    clip_occ: float = 1.0,
    aggregate: str = "presslight",
    positive_only: bool = False,
    normalize_by_movement_count: bool = True,
    **_: Any,
) -> float:
    """PressLight-style pressure reward over scene statistics."""
    _ = sim_time, gamma_dt
    if init_only:
        return 0.0
    if cache is None:
        cache = {}

    agg_key = str(aggregate).strip().lower()
    upstream = lane_value_map(scene_stats, upstream_key)
    movements = get_unique_tls_movements(str(tls_id), cache)
    if not movements:
        return 0.0

    downstream_cache: dict[str, float] = {}
    pressures: list[float] = []
    for in_lane, out_lane in movements:
        if out_lane not in downstream_cache:
            downstream_cache[out_lane] = downstream_count_ratio_norm(
                out_lane,
                veh_equiv_len_m=float(veh_equiv_len_m),
                clip_occ=float(clip_occ),
            )
        p = float(upstream.get(in_lane, 0.0)) - float(downstream_cache[out_lane])
        if positive_only and p < 0.0:
            p = 0.0
        pressures.append(float(p))

    pv = np.asarray(pressures, dtype=np.float64)
    n = max(1, pv.size)
    if agg_key == "mean_abs":
        penalty = float(np.mean(np.abs(pv)))
    elif agg_key == "sum_abs":
        penalty = float(np.sum(np.abs(pv)))
        if normalize_by_movement_count:
            penalty /= float(n)
    elif agg_key == "mean":
        penalty = float(np.mean(pv))
    elif agg_key == "sum":
        penalty = float(np.sum(pv))
        if normalize_by_movement_count:
            penalty /= float(n)
    elif agg_key == "presslight":
        penalty = float(abs(np.sum(pv)))
        if normalize_by_movement_count:
            penalty /= float(n)
    else:
        raise ValueError(
            f"aggregate must be one of 'mean_abs', 'sum_abs', 'mean', 'sum', 'presslight', got {aggregate}"
        )
    return -float(penalty)
