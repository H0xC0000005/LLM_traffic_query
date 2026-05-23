from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Protocol, Sequence

import numpy as np

try:
    import libsumo as traci
except ModuleNotFoundError as e:  # pragma: no cover
    print(f"Error importing libsumo: {e}. libsumo is prefered over traci and is a must in this version")
    raise e

from utility import get_tls_phase_plan, tls_current_major_phase, _soft_sat


@dataclass(frozen=True)
class ControlDecision:
    """
    One decision emitted by a controller backend.

    Parameters
    ----------
    action:
        Major-phase action index, matching the action semantics already used by the
        PPO policy and ``tls_action_to_major_phase(...)``.
    hold_s:
        Optional macro-action hold time for this decision. If ``None``, the caller
        should keep using its existing default hold time.
    info:
        Optional debug metadata for logging.
    """

    action: int
    hold_s: float | None = None
    info: dict[str, Any] = field(default_factory=dict)


class BaseSignalController(ABC):
    """Minimal controller interface for rule-based external baselines."""

    @abstractmethod
    def reset(self, *, tls_id: str, cache: dict | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def choose_action(
        self,
        tls_id: str,
        *,
        scene_stats: Any,
        sim_time: float,
        cache: dict,
    ) -> ControlDecision:
        raise NotImplementedError


class SignalControllerProtocol(Protocol):
    def reset(self, *, tls_id: str, cache: dict | None = None) -> None: ...

    def choose_action(
        self,
        tls_id: str,
        *,
        scene_stats: Any,
        sim_time: float,
        cache: dict,
    ) -> ControlDecision: ...


@dataclass(frozen=True)
class ControllerStructure:
    """
    Cached structural information for one TLS program under the macro-action scheme.

    Notes
    -----
    - ``action_to_phase`` follows the same order as ``plan.major_greens``.
    - ``action_to_in_lanes`` lists incoming lanes served by each major action.
    - ``action_to_movements`` lists (incoming, outgoing) lane movements that have
      a green signal under each major action.
    """

    program_id: str
    action_to_phase: tuple[int, ...]
    phase_to_action: dict[int, int]
    action_to_in_lanes: dict[int, tuple[str, ...]]
    action_to_movements: dict[int, tuple[tuple[str, str], ...]]


def _scene_lane_ids(scene_stats: Any) -> tuple[str, ...]:
    if hasattr(scene_stats, "lane_ids"):
        lane_ids = getattr(scene_stats, "lane_ids")
    elif isinstance(scene_stats, Mapping):
        lane_ids = scene_stats["lane_ids"]
    else:
        raise TypeError("scene_stats must expose lane_ids")
    return tuple(str(x) for x in lane_ids)


def _scene_per_lane(scene_stats: Any, key: str) -> np.ndarray:
    if hasattr(scene_stats, "per_lane"):
        per_lane = getattr(scene_stats, "per_lane")
    elif isinstance(scene_stats, Mapping):
        per_lane = scene_stats["per_lane"]
    else:
        raise TypeError("scene_stats must expose per_lane")
    return np.asarray(per_lane[key], dtype=np.float32).reshape(-1)


def lane_value_map(scene_stats: Any, key: str) -> dict[str, float]:
    lane_ids = _scene_lane_ids(scene_stats)
    values = _scene_per_lane(scene_stats, key)
    if values.shape[0] != len(lane_ids):
        raise ValueError(f"scene_stats.per_lane[{key!r}] length {values.shape[0]} != len(lane_ids) {len(lane_ids)}")
    return {lane_ids[i]: float(values[i]) for i in range(len(lane_ids))}


def scene_global(scene_stats: Any, key: str, default: Any = None) -> Any:
    if hasattr(scene_stats, "global_stats"):
        gs = getattr(scene_stats, "global_stats")
    elif isinstance(scene_stats, Mapping):
        gs = scene_stats.get("global_stats", {})
    else:
        return default
    return gs.get(key, default)


def get_controller_structure(
    tls_id: str,
    cache: MutableMapping[str, Any],
    *,
    min_major_green_s: float = 5.0,
) -> ControllerStructure:
    """
    Build and cache a movement/served-lane view for rule-based controllers.

    The mapping follows the existing macro-action abstraction:
    action index -> major green phase -> served incoming lanes / movements.
    """
    program_id = str(traci.trafficlight.getProgram(tls_id))
    ctl_cache = cache.setdefault("_controller_structure", {})
    cached: Optional[ControllerStructure] = ctl_cache.get(tls_id)
    if cached is not None and cached.program_id == program_id:
        return cached

    plan = get_tls_phase_plan(tls_id, cache, min_major_green_s=float(min_major_green_s))
    action_to_phase = tuple(int(x) for x in plan.major_greens)
    phase_to_action = {int(ph): int(i) for i, ph in enumerate(action_to_phase)}

    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    signal_links: list[tuple[int, str, str]] = []
    sigpos = 0
    for link_group in controlled_links:
        for in_lane, out_lane, _via_lane in link_group:
            signal_links.append((sigpos, str(in_lane), str(out_lane)))
            sigpos += 1

    phase_state = {int(idx): str(state) for idx, _dur, state in plan.phases}
    action_to_in_lanes: dict[int, tuple[str, ...]] = {}
    action_to_movements: dict[int, tuple[tuple[str, str], ...]] = {}

    for action_idx, phase_idx in enumerate(action_to_phase):
        st = phase_state[int(phase_idx)]
        seen_lanes: set[str] = set()
        seen_movements: set[tuple[str, str]] = set()
        in_lanes: list[str] = []
        movements: list[tuple[str, str]] = []
        for sig_idx, in_lane, out_lane in signal_links:
            if 0 <= sig_idx < len(st) and st[sig_idx] in ("G", "g"):
                if in_lane not in seen_lanes:
                    seen_lanes.add(in_lane)
                    in_lanes.append(in_lane)
                mv = (in_lane, out_lane)
                if mv not in seen_movements:
                    seen_movements.add(mv)
                    movements.append(mv)
        action_to_in_lanes[int(action_idx)] = tuple(in_lanes)
        action_to_movements[int(action_idx)] = tuple(movements)

    out = ControllerStructure(
        program_id=program_id,
        action_to_phase=action_to_phase,
        phase_to_action=phase_to_action,
        action_to_in_lanes=action_to_in_lanes,
        action_to_movements=action_to_movements,
    )
    ctl_cache[tls_id] = out
    return out


def current_major_action_index(
    tls_id: str,
    cache: MutableMapping[str, Any],
    *,
    min_major_green_s: float = 5.0,
) -> int:
    struct = get_controller_structure(tls_id, cache, min_major_green_s=float(min_major_green_s))
    current_major = int(tls_current_major_phase(tls_id, cache, min_major_green_s=float(min_major_green_s)))
    return int(struct.phase_to_action.get(current_major, 0))


def downstream_count_ratio_norm(
    lane_id: str,
    *,
    veh_equiv_len_m: float = 7.5,
    clip_occ: float = 1.0,
) -> float:
    """Mirror the encoder's softly saturated normalized lane count formula."""
    veh_count = float(traci.lane.getLastStepVehicleNumber(str(lane_id)))
    lane_len = float(traci.lane.getLength(str(lane_id)))
    lane_cap = max(1.0, lane_len / max(1e-6, float(veh_equiv_len_m)))
    return float(_soft_sat(veh_count / lane_cap, sat=float(clip_occ)))
