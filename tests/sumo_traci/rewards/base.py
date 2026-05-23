from __future__ import annotations

from typing import Any, Mapping, MutableMapping

import numpy as np

try:
    import libsumo as traci
except ModuleNotFoundError:  # pragma: no cover
    import traci

from utility import _soft_sat


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


def lane_value_map(scene_stats: Any, key: str) -> dict[str, float]:
    lane_ids = scene_lane_ids(scene_stats)
    values = scene_per_lane(scene_stats, key)
    if values.shape[0] != len(lane_ids):
        raise ValueError(
            f"scene_stats.per_lane[{key!r}] length {values.shape[0]} != len(lane_ids) {len(lane_ids)}"
        )
    return {lane_ids[i]: float(values[i]) for i in range(len(lane_ids))}


def scene_global(scene_stats: Any, key: str, default: Any = None) -> Any:
    if hasattr(scene_stats, "global_stats"):
        gs = getattr(scene_stats, "global_stats")
    elif isinstance(scene_stats, Mapping):
        gs = scene_stats.get("global_stats", {})
    else:
        return default
    return gs.get(key, default)


def downstream_count_ratio_norm(
    lane_id: str,
    *,
    veh_equiv_len_m: float = 7.5,
    clip_occ: float = 1.0,
) -> float:
    veh_count = float(traci.lane.getLastStepVehicleNumber(str(lane_id)))
    lane_len = float(traci.lane.getLength(str(lane_id)))
    lane_cap = max(1.0, lane_len / max(1e-6, float(veh_equiv_len_m)))
    return float(_soft_sat(veh_count / lane_cap, sat=float(clip_occ)))


def get_unique_tls_movements(
    tls_id: str,
    cache: MutableMapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    """Return unique (incoming, outgoing) movements controlled by one TLS program."""
    tls_id = str(tls_id)
    program_id = str(traci.trafficlight.getProgram(tls_id))
    reward_cache = cache.setdefault("_reward_structure", {})
    cached = reward_cache.get(tls_id)
    if cached is not None and cached.get("program_id") == program_id:
        return tuple(cached["movements"])

    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    seen: set[tuple[str, str]] = set()
    movements: list[tuple[str, str]] = []
    for group in controlled_links:
        for in_lane, out_lane, _via_lane in group:
            mv = (str(in_lane), str(out_lane))
            if mv not in seen:
                seen.add(mv)
                movements.append(mv)

    reward_cache[tls_id] = {
        "program_id": program_id,
        "movements": tuple(movements),
    }
    return tuple(movements)
