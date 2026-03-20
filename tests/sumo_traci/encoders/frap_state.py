from __future__ import annotations

"""FRAP-inspired fixed-vector encoder for a fixed PPO backbone.

This is *not* a faithful reproduction of FRAP's learned phase-competition model.
Instead, it turns the FRAP intuition into a deterministic feature block:

    [ per-phase demand | pairwise phase-demand differences | optional signal context ]

The goal is to compare phase-competition-aware representation quality under the same
strong PPO host, without introducing FRAP's custom learned competition module.
"""

from itertools import combinations
from typing import Any

import numpy as np

from controllers.base import get_controller_structure
from .base import append_signal_context, lane_value_map


def _phase_demands(
    tls_id: str,
    *,
    scene_stats: Any,
    cache: dict,
    demand_key: str = "count_ratio_norm",
    min_major_green_s: float = 5.0,
    normalize_by_served_lane_count: bool = True,
) -> np.ndarray:
    """Compute one deterministic demand scalar for each major action.

    The default demand is the sum of the chosen upstream lane feature over incoming lanes
    served by each major action. Optional normalization keeps phase scales comparable when
    different major actions serve different numbers of lanes.
    """
    struct = get_controller_structure(tls_id, cache, min_major_green_s=float(min_major_green_s))
    lane_demands = lane_value_map(scene_stats, demand_key)

    out: list[float] = []
    for action_idx in range(len(struct.action_to_phase)):
        lanes = tuple(struct.action_to_in_lanes.get(int(action_idx), ()))
        total = float(sum(float(lane_demands.get(str(lane), 0.0)) for lane in lanes))
        if normalize_by_served_lane_count:
            total /= float(max(1, len(lanes)))
        out.append(total)
    return np.asarray(out, dtype=np.float32)


def _pairwise_phase_differences(phase_demands: np.ndarray) -> np.ndarray:
    """Unordered pairwise signed demand differences in a deterministic action-index order."""
    vals = np.asarray(phase_demands, dtype=np.float32).reshape(-1)
    feats: list[float] = []
    for i, j in combinations(range(vals.shape[0]), 2):
        feats.append(float(vals[i] - vals[j]))
    return np.asarray(feats, dtype=np.float32)


def frap_state_encoder(
    tls_id: str,
    *,
    scene_stats: Any,
    cache: dict | None = None,
    demand_key: str = "count_ratio_norm",
    min_major_green_s: float = 5.0,
    normalize_by_served_lane_count: bool = True,
    include_pairwise_differences: bool = True,
    include_signal_context: bool = True,
    include_current_phase_onehot: bool = True,
    include_major_since_served: bool = True,
    demand_scale: float = 1.0,
    soft_sat_demand: bool = False,
    **kwargs,
) -> np.ndarray:
    if cache is None:
        cache = {}
    if scene_stats is None:
        raise AssertionError("scene_stats must not be None")

    phase_demands = _phase_demands(
        tls_id,
        scene_stats=scene_stats,
        cache=cache,
        demand_key=demand_key,
        min_major_green_s=float(min_major_green_s),
        normalize_by_served_lane_count=bool(normalize_by_served_lane_count),
    )

    if soft_sat_demand:
        scale = max(1e-6, float(demand_scale))
        phase_demands = np.tanh((phase_demands / scale).astype(np.float32)).astype(np.float32)

    features: list[float] = phase_demands.reshape(-1).tolist()

    if include_pairwise_differences:
        pairwise = _pairwise_phase_differences(phase_demands)
        features.extend(pairwise.reshape(-1).tolist())

    if include_signal_context:
        features.extend(
            append_signal_context(
                scene_stats,
                include_current_phase_onehot=bool(include_current_phase_onehot),
                include_major_since_served=bool(include_major_since_served),
            )
        )

    return np.asarray(features, dtype=np.float32)
