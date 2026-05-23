from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import libsumo as traci
from utility import _soft_sat, TSCSceneSnapshot


def _clip01_unit(x: float) -> float:
    x = float(x)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def collect_tsc_scene_snapshot(
    tls_id: str,
    *,
    moving_speed_threshold: float = 0.1,
    stopped_speed_threshold: float = 0.1,
    cache: dict | None = None,
    veh_equiv_len_m: float = 7.5,
    wait_ref_s: float = 60.0,
    since_served_ref_s: float = 60.0,
    clip_occ: float = 1.0,
    min_major_green_s: float = 5.0,
    include_signal_context: bool = True,
) -> TSCSceneSnapshot:
    """
    Query SUMO/TraCI once and build a reusable portable scene snapshot.

    The snapshot preserves the canonical lane order previously used by
    ``encode_tsc_state_vector_bounded_v2`` and stores both raw and normalized
    per-lane quantities. The normalization formulas are intentionally identical to the
    existing encoder so that downstream state vectors and rewards remain numerically
    unchanged when they consume this precomputed snapshot.

    Parameters
    ----------
    tls_id : str
        Traffic light system ID in SUMO/TraCI.
    moving_speed_threshold : float, default=0.1
        Vehicles with speed strictly above this threshold are treated as moving when
        computing mean moving speed.
    stopped_speed_threshold : float, default=0.1
        Vehicles with speed at or below this threshold are treated as stopped/queued
        when computing queue length and mean stopped waiting time.
    cache : dict | None, default=None
        Mutable cache reused across calls to preserve lane ordering, lane->signal-index
        mapping, and per-major-phase last-served timestamps.
    veh_equiv_len_m : float, default=7.5
        Effective vehicle length used to estimate lane capacity.
    wait_ref_s : float, default=60.0
        Reference time used for the softly saturated normalized wait feature.
    since_served_ref_s : float, default=60.0
        Reference time used for the softly saturated major-phase starvation feature.
    clip_occ : float, default=1.0
        Soft-saturation target for occupancy-like ratios.
    min_major_green_s : float, default=5.0
        Forwarded to ``utility.get_tls_phase_plan(...)``.
    include_signal_context : bool, default=True
        If True, populate ``extras["signal_context"]`` with the current phase one-hot,
        major-phase IDs, and time-since-major-green features needed by the current
        encoder and starvation reward.

    Returns
    -------
    TSCSceneSnapshot
        Per-frame portable scene snapshot.
    """
    if cache is None:
        cache = {}

    # local import to avoid accidental circular import at module import time
    from utility import get_tls_phase_plan

    sim_t = float(traci.simulation.getTime())

    if ("lane_ids" not in cache) or ("lane_to_sigidx" not in cache):
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)

        lane_to_sigidx: dict[str, list[int]] = {}
        incoming_lanes: set[str] = set()
        sigpos = 0

        for link_group in controlled_links:
            for in_lane, _out_lane, _via_lane in link_group:
                incoming_lanes.add(in_lane)
                lane_to_sigidx.setdefault(in_lane, []).append(sigpos)
                sigpos += 1

        lane_ids = sorted(list(incoming_lanes), key=lambda x: (traci.lane.getEdgeID(x), x))
        cache["lane_ids"] = lane_ids
        cache["lane_to_sigidx"] = lane_to_sigidx

    lane_ids: list[str] = cache["lane_ids"]
    lane_to_sigidx: dict[str, list[int]] = cache["lane_to_sigidx"]

    ryg = traci.trafficlight.getRedYellowGreenState(tls_id)
    ryg_len = len(ryg)

    queue_count: list[float] = []
    vehicle_count: list[float] = []
    mean_moving_speed_mps: list[float] = []
    mean_wait_stopped_s: list[float] = []
    lane_length_m: list[float] = []
    lane_capacity_veh: list[float] = []
    speed_limit_mps: list[float] = []
    queue_ratio_norm: list[float] = []
    count_ratio_norm: list[float] = []
    speed_norm: list[float] = []
    wait_norm: list[float] = []
    is_green: list[float] = []

    for ln in lane_ids:
        veh_ids = traci.lane.getLastStepVehicleIDs(ln)
        veh_count = len(veh_ids)

        queue = 0
        moving_speeds_sum = 0.0
        moving_count = 0
        waiting_time_static_sum = 0.0

        for vid in veh_ids:
            spd = float(traci.vehicle.getSpeed(vid))
            if spd <= stopped_speed_threshold:
                queue += 1
                waiting_time_static_sum += float(traci.vehicle.getWaitingTime(vid))
            if spd > moving_speed_threshold:
                moving_speeds_sum += spd
                moving_count += 1

        mean_speed_moving = (moving_speeds_sum / moving_count) if moving_count > 0 else 0.0
        mean_wait_stopped = (waiting_time_static_sum / float(queue)) if queue > 0 else 0.0

        lane_len = float(traci.lane.getLength(ln))
        lane_cap = max(1.0, lane_len / max(1e-6, float(veh_equiv_len_m)))
        v_limit = float(traci.lane.getMaxSpeed(ln))

        q_occ = _soft_sat(float(queue) / lane_cap, sat=float(clip_occ))
        n_occ = _soft_sat(float(veh_count) / lane_cap, sat=float(clip_occ))
        v_norm = 0.0 if v_limit <= 1e-6 else float(mean_speed_moving) / v_limit
        v_norm = _clip01_unit(v_norm)
        w_norm = _soft_sat(mean_wait_stopped / max(1e-6, float(wait_ref_s)), sat=1.0)

        green_now = 0.0
        for idx in lane_to_sigidx.get(ln, []):
            if 0 <= idx < ryg_len and ryg[idx] in ("G", "g"):
                green_now = 1.0
                break

        queue_count.append(float(queue))
        vehicle_count.append(float(veh_count))
        mean_moving_speed_mps.append(float(mean_speed_moving))
        mean_wait_stopped_s.append(float(mean_wait_stopped))
        lane_length_m.append(lane_len)
        lane_capacity_veh.append(lane_cap)
        speed_limit_mps.append(v_limit)
        queue_ratio_norm.append(q_occ)
        count_ratio_norm.append(n_occ)
        speed_norm.append(v_norm)
        wait_norm.append(w_norm)
        is_green.append(green_now)

    per_lane = {
        "queue_count": np.asarray(queue_count, dtype=np.float32),
        "vehicle_count": np.asarray(vehicle_count, dtype=np.float32),
        "mean_moving_speed_mps": np.asarray(mean_moving_speed_mps, dtype=np.float32),
        "mean_wait_stopped_s": np.asarray(mean_wait_stopped_s, dtype=np.float32),
        "lane_length_m": np.asarray(lane_length_m, dtype=np.float32),
        "lane_capacity_veh": np.asarray(lane_capacity_veh, dtype=np.float32),
        "speed_limit_mps": np.asarray(speed_limit_mps, dtype=np.float32),
        "queue_ratio_norm": np.asarray(queue_ratio_norm, dtype=np.float32),
        "count_ratio_norm": np.asarray(count_ratio_norm, dtype=np.float32),
        "speed_norm": np.asarray(speed_norm, dtype=np.float32),
        "wait_norm": np.asarray(wait_norm, dtype=np.float32),
        "is_green": np.asarray(is_green, dtype=np.float32),
    }

    global_stats: dict[str, Any] = {
        "current_program_id": str(traci.trafficlight.getProgram(tls_id)),
        "num_lanes": int(len(lane_ids)),
    }

    extras: dict[str, Any] = {
        "normalization": {
            "moving_speed_threshold": float(moving_speed_threshold),
            "stopped_speed_threshold": float(stopped_speed_threshold),
            "veh_equiv_len_m": float(veh_equiv_len_m),
            "wait_ref_s": float(wait_ref_s),
            "since_served_ref_s": float(since_served_ref_s),
            "clip_occ": float(clip_occ),
        }
    }

    if include_signal_context:
        plan = get_tls_phase_plan(tls_id, cache, min_major_green_s=float(min_major_green_s))
        num_phases = int(len(plan.phases))
        major_phases = tuple(int(p) for p in plan.major_greens)
        major_set = set(major_phases)

        prog_key = "_v2_major_last_prog_id"
        last_key = "_v2_major_last_t"
        if (cache.get(prog_key) != plan.program_id) or (last_key not in cache):
            cache[prog_key] = plan.program_id
            cache[last_key] = {ph: sim_t for ph in major_phases}

        major_last_t: dict[int, float] = cache[last_key]
        current_phase = int(traci.trafficlight.getPhase(tls_id))
        one_hot = np.zeros(num_phases, dtype=np.float32)
        if 0 <= current_phase < num_phases:
            one_hot[current_phase] = 1.0

        if current_phase in major_set:
            major_last_t[current_phase] = sim_t

        denom = max(1e-6, float(since_served_ref_s))
        time_since_major_green_s = []
        time_since_major_green_norm = []
        for ph in major_phases:
            dt = sim_t - float(major_last_t.get(ph, sim_t))
            time_since_major_green_s.append(float(dt))
            time_since_major_green_norm.append(_soft_sat(dt / denom, sat=1.0))

        extras["signal_context"] = {
            "current_phase_index": int(current_phase),
            "num_phases": int(num_phases),
            "phase_one_hot": one_hot,
            "major_phases": major_phases,
            "time_since_major_green_s": np.asarray(time_since_major_green_s, dtype=np.float32),
            "time_since_major_green_norm": np.asarray(time_since_major_green_norm, dtype=np.float32),
        }

    return TSCSceneSnapshot(
        tls_id=str(tls_id),
        sim_time=float(sim_t),
        lane_ids=tuple(lane_ids),
        per_lane=per_lane,
        global_stats=global_stats,
        extras=extras,
    )


def encode_tsc_state_vector_bounded_v2(
    tls_id: str,
    *,
    moving_speed_threshold: float = 0.1,
    stopped_speed_threshold: float = 0.1,
    cache: dict | None = None,
    scene_stats: TSCSceneSnapshot | None = None,
    # -------------------- normalization / bounding knobs --------------------
    veh_equiv_len_m: float = 7.5,
    wait_ref_s: float = 60.0,
    since_served_ref_s: float = 60.0,
    clip_occ: float = 1.0,
    # -----------------------------------------------------------------------
    min_major_green_s: float = 5.0,
    **kwargs: Any,
) -> np.ndarray:
    """
    Encode a bounded (normalized + softly saturated) state vector for RL traffic signal control.

    The vector layout is intentionally unchanged from the previous implementation. When
    ``scene_stats`` is supplied, the encoder consumes the precomputed portable scene
    snapshot instead of issuing live TraCI lane queries. If ``scene_stats`` is omitted,
    a fresh snapshot is collected with ``collect_tsc_scene_snapshot(...)`` using the same
    normalization knobs as before.

    Parameters
    ----------
    tls_id : str
        Traffic light system ID in SUMO/TraCI.
    moving_speed_threshold : float, default=0.1
        Forwarded to ``collect_tsc_scene_snapshot(...)`` when ``scene_stats`` is not
        provided.
    stopped_speed_threshold : float, default=0.1
        Forwarded to ``collect_tsc_scene_snapshot(...)`` when ``scene_stats`` is not
        provided.
    cache : dict | None, default=None
        Mutable cache reused across calls. If ``scene_stats`` is omitted, the cache keeps
        lane ordering, signal-index mapping, and per-major-phase last-served timestamps.
    scene_stats : TSCSceneSnapshot | None, default=None
        Optional precomputed scene snapshot. If provided, it must be compatible with the
        same normalization choices used here.
    veh_equiv_len_m, wait_ref_s, since_served_ref_s, clip_occ, min_major_green_s
        Same semantics as before; used only when ``scene_stats`` must be collected inside
        this function.

    Returns
    -------
    np.ndarray
        1-D ``float32`` feature vector with the exact same layout as before:
          1) per-lane [queue_ratio_norm, count_ratio_norm, speed_norm, wait_norm]
          2) per-lane is_green flags
          3) current phase one-hot
          4) time-since-major-green normalized features
    """
    if cache is None:
        cache = {}

    if scene_stats is None:
        scene_stats = collect_tsc_scene_snapshot(
            tls_id,
            moving_speed_threshold=moving_speed_threshold,
            stopped_speed_threshold=stopped_speed_threshold,
            cache=cache,
            veh_equiv_len_m=veh_equiv_len_m,
            wait_ref_s=wait_ref_s,
            since_served_ref_s=since_served_ref_s,
            clip_occ=clip_occ,
            min_major_green_s=min_major_green_s,
            include_signal_context=True,
        )

    sig_ctx = dict(scene_stats.extras.get("signal_context", {}))
    num_phases = int(sig_ctx.get("num_phases", 0))
    phase_one_hot = np.asarray(sig_ctx.get("phase_one_hot", []), dtype=np.float32).reshape(-1)
    if phase_one_hot.shape[0] != num_phases:
        phase_one_hot = np.zeros(num_phases, dtype=np.float32)

    since_major = np.asarray(sig_ctx.get("time_since_major_green_norm", []), dtype=np.float32).reshape(-1)

    features: list[float] = []
    q_occ = scene_stats.per_lane["queue_ratio_norm"]
    n_occ = scene_stats.per_lane["count_ratio_norm"]
    v_norm = scene_stats.per_lane["speed_norm"]
    w_norm = scene_stats.per_lane["wait_norm"]
    for i in range(scene_stats.num_lanes):
        features.extend(
            [
                float(q_occ[i]),
                float(n_occ[i]),
                float(v_norm[i]),
                float(w_norm[i]),
            ]
        )

    features.extend(float(x) for x in scene_stats.per_lane["is_green"])
    features.extend(float(x) for x in phase_one_hot)
    features.extend(float(x) for x in since_major)

    return np.asarray(features, dtype=np.float32)
