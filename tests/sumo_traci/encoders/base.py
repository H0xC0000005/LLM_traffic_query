from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
try:
    import libsumo as traci
except ModuleNotFoundError:  # pragma: no cover
    import traci

from utility import _soft_sat, get_tls_phase_plan
from controllers.base import get_controller_structure


def scene_lane_ids(scene_stats: Any) -> tuple[str, ...]:
    if hasattr(scene_stats, "lane_ids"):
        lane_ids = getattr(scene_stats, "lane_ids")
    elif isinstance(scene_stats, Mapping):
        if "lane_ids" in scene_stats:
            lane_ids = scene_stats["lane_ids"]
        elif "meta" in scene_stats and "lane_ids" in scene_stats["meta"]:
            lane_ids = scene_stats["meta"]["lane_ids"]
        else:
            raise KeyError("scene_stats does not expose lane_ids")
    else:
        raise TypeError("scene_stats must expose lane_ids")
    return tuple(str(x) for x in lane_ids)


def scene_per_lane(scene_stats: Any, key: str) -> np.ndarray:
    if hasattr(scene_stats, "per_lane"):
        per_lane = getattr(scene_stats, "per_lane")
    elif isinstance(scene_stats, Mapping):
        per_lane = scene_stats["per_lane"]
    else:
        raise TypeError("scene_stats must expose per_lane")
    return np.asarray(per_lane[key], dtype=np.float32).reshape(-1)


def scene_global(scene_stats: Any, key: str, default: Any = None) -> Any:
    if hasattr(scene_stats, "global_stats"):
        gs = getattr(scene_stats, "global_stats")
    elif isinstance(scene_stats, Mapping):
        gs = scene_stats.get("global_stats", {})
    else:
        return default
    return gs.get(key, default)


def scene_signal_context(scene_stats: Any) -> dict[str, Any]:
    if hasattr(scene_stats, "extras"):
        ex = getattr(scene_stats, "extras")
    elif isinstance(scene_stats, Mapping):
        ex = scene_stats.get("extras", {})
    else:
        return {}
    ctx = ex.get("signal_context", {})
    if not isinstance(ctx, Mapping):
        return {}
    return dict(ctx)


def lane_value_map(scene_stats: Any, key: str) -> dict[str, float]:
    lane_ids = scene_lane_ids(scene_stats)
    values = scene_per_lane(scene_stats, key)
    if values.shape[0] != len(lane_ids):
        raise ValueError(
            f"scene_stats.per_lane[{key!r}] length {values.shape[0]} != len(lane_ids) {len(lane_ids)}"
        )
    return {lane_ids[i]: float(values[i]) for i in range(len(lane_ids))}


def downstream_norm_for_lane(
    lane_id: str,
    *,
    moving_speed_threshold: float = 0.1,
    veh_equiv_len_m: float = 7.5,
    clip_occ: float = 1.0,
) -> float:
    """Proxy normalized downstream demand for one outgoing lane.

    This matches the same count/capacity + soft-saturation normalization family used by the
    portable scene snapshot, so state substitutions remain numerically consistent with the rest
    of the pipeline.
    """
    veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
    veh_count = len(veh_ids)
    lane_len = float(traci.lane.getLength(lane_id))
    lane_cap = max(1.0, lane_len / max(1e-6, float(veh_equiv_len_m)))
    return float(_soft_sat(float(veh_count) / lane_cap, sat=float(clip_occ)))


def phase_signed_pressures(
    tls_id: str,
    *,
    scene_stats: Any,
    cache: MutableMapping[str, Any],
    upstream_key: str = "count_ratio_norm",
    veh_equiv_len_m: float = 7.5,
    clip_occ: float = 1.0,
    min_major_green_s: float = 5.0,
) -> tuple[np.ndarray, dict[int, tuple[tuple[str, str], ...]]]:
    """Compute signed pressure per major action.

    For each major action, sum signed movement pressures over its green movements:

        p(action) = sum_{(in,out) served by action} [u(in) - d(out)]

    where ``u`` comes from the chosen upstream key in scene_stats and ``d`` is queried on the
    downstream lane with the same normalization family.
    """
    struct = get_controller_structure(tls_id, cache, min_major_green_s=float(min_major_green_s))
    upstream = lane_value_map(scene_stats, upstream_key)
    out: list[float] = []
    for action_idx in range(len(struct.action_to_phase)):
        total = 0.0
        for in_lane, out_lane in struct.action_to_movements.get(int(action_idx), ()):  # type: ignore[arg-type]
            u = float(upstream.get(str(in_lane), 0.0))
            d = downstream_norm_for_lane(
                str(out_lane),
                veh_equiv_len_m=float(veh_equiv_len_m),
                clip_occ=float(clip_occ),
            )
            total += (u - d)
        out.append(float(total))
    return np.asarray(out, dtype=np.float32), struct.action_to_movements


def phase_effective_running_vehicles(
    tls_id: str,
    *,
    scene_stats: Any,
    cache: MutableMapping[str, Any],
    speed_gate_ratio: float = 0.15,
    min_major_green_s: float = 5.0,
) -> np.ndarray:
    """Low-cost ATS-style proxy for effective running vehicles per major action.

    We approximate "running" demand on a served incoming lane by combining normalized count and
    normalized speed; a lane contributes more when it both contains vehicles and those vehicles are
    not fully stopped. The phase feature is the sum over incoming lanes served by that major action.
    """
    struct = get_controller_structure(tls_id, cache, min_major_green_s=float(min_major_green_s))
    n_occ = lane_value_map(scene_stats, "count_ratio_norm")
    v_norm = lane_value_map(scene_stats, "speed_norm")
    out: list[float] = []
    gate = float(speed_gate_ratio)
    for action_idx in range(len(struct.action_to_phase)):
        total = 0.0
        for in_lane in struct.action_to_in_lanes.get(int(action_idx), ()):  # type: ignore[arg-type]
            cnt = float(n_occ.get(str(in_lane), 0.0))
            vel = float(v_norm.get(str(in_lane), 0.0))
            # Keep the feature low when traffic is fully stopped, larger when vehicles are present
            # and moving. This is intentionally simple and scene-stats-compatible.
            eff = cnt * max(0.0, vel - gate) / max(1e-6, 1.0 - gate)
            total += eff
        out.append(float(total))
    return np.asarray(out, dtype=np.float32)


def append_signal_context(
    scene_stats: Any,
    *,
    include_current_phase_onehot: bool = True,
    include_major_since_served: bool = True,
) -> list[float]:
    """Append the same signal-context terms used by the current baseline when available."""
    ctx = scene_signal_context(scene_stats)
    out: list[float] = []
    if include_current_phase_onehot and "current_phase_onehot" in ctx:
        out.extend(np.asarray(ctx["current_phase_onehot"], dtype=np.float32).reshape(-1).tolist())
    if include_major_since_served and "time_since_major_green_norm" in ctx:
        out.extend(np.asarray(ctx["time_since_major_green_norm"], dtype=np.float32).reshape(-1).tolist())
    return out
