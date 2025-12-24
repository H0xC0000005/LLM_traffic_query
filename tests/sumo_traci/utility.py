# utility.py

from __future__ import annotations

from typing import Dict, Optional, Sequence, Set
import traci

def reward_avg_queue_from_encoded_state(
    state_vec: Sequence[float],
    *,
    num_lanes: int,
    lane_block_size: int = 4,
    queue_offset_in_block: int = 0,
    reduce: str = "mean",
) -> float:
    """
    Compute a reward from an encoded state vector based on (average) queue length.

    Assumptions about your encoding (matching your current encoder, excluding time_in_phase):
      - Per incoming lane block: [queue, veh_count, mean_speed, waiting_time]  (lane_block_size=4)
      - Then: is_green_now per lane  (num_lanes values)
      - Then: phase one-hot          (remaining values)
      - time_in_phase has been dropped before calling this function.

    Args:
      state_vec: encoded feature vector (time_in_phase already removed if you dropped it)
      num_lanes: number of incoming lanes encoded (must match your encoder's lane order)
      lane_block_size: number of features per lane in the lane block (default 4)
      queue_offset_in_block: index of queue feature within each lane block (default 0)
      reduce: "mean" or "sum" over lanes

    Returns:
      Reward as float. Typical choice is negative mean queue:  reward = -avg_queue
    """
    if num_lanes <= 0:
        raise ValueError("num_lanes must be > 0")

    expected_min_len = num_lanes * lane_block_size
    if len(state_vec) < expected_min_len:
        raise ValueError(
            f"state_vec too short: len={len(state_vec)} < {expected_min_len} "
            f"(num_lanes={num_lanes}, lane_block_size={lane_block_size})"
        )

    queues = []
    base = 0
    for i in range(num_lanes):
        idx = base + i * lane_block_size + queue_offset_in_block
        q = float(state_vec[idx])
        # queues should never be negative; clamp defensively
        if q < 0.0:
            q = 0.0
        queues.append(q)

    if reduce == "mean":
        val = sum(queues) / float(num_lanes)
    elif reduce == "sum":
        val = sum(queues)
    else:
        raise ValueError("reduce must be 'mean' or 'sum'")
    r = - (0.5 * val + 0.5 * max(queues))
    return r


def _get_tls_out_lanes(tls_id: str) -> list[str]:
    """
    Outgoing lanes (downstream of the intersection) derived from controlled links.
    Each link tuple is (inLane, outLane, viaLane).
    """
    out_lanes: Set[str] = set()
    for conn_group in traci.trafficlight.getControlledLinks(tls_id):
        for link in conn_group:
            if not link:
                continue
            out_lane = link[1]
            if out_lane:
                out_lanes.add(out_lane)
    return sorted(out_lanes)


def throughput_tracker_step(
    tls_id: str,
    cache: Dict,
    *,
    out_lanes: Optional[Sequence[str]] = None,
) -> None:
    """
    Call this EVERY simulationStep() to accumulate which vehicles newly ENTERED
    the downstream lanes since the last decision reward was computed.

    It counts a vehicle exactly once at the moment it first appears on ANY downstream lane.

    cache keys used:
      - "_tp_out_lanes": list[str]
      - "_tp_seen_total": set[str]
      - "_tp_new_since_last": set[str]
    """
    if out_lanes is None:
        out_lanes = cache.get("_tp_out_lanes")
        if out_lanes is None:
            out_lanes = _get_tls_out_lanes(tls_id)
            cache["_tp_out_lanes"] = out_lanes

    seen_total: Set[str] = cache.setdefault("_tp_seen_total", set())
    new_since_last: Set[str] = cache.setdefault("_tp_new_since_last", set())

    for ln in out_lanes:
        for vid in traci.lane.getLastStepVehicleIDs(ln):
            if vid not in seen_total:
                seen_total.add(vid)
                new_since_last.add(vid)


def reward_throughput_per_second_on_decision(
    sim_time: float,
    cache: Dict,
) -> float:
    """
    Call this at EACH DECISION POINT to get:
        reward = (# vehicles that newly entered downstream lanes since last decision) / dt

    cache keys used:
      - "_tp_last_decision_t": float
      - "_tp_new_since_last": set[str]  (cleared after reward is computed)

    Returns:
      throughput_veh_per_sec (float)

    Notes:
      - First call returns 0.0 (no previous decision interval).
      - This relies on throughput_tracker_step(...) being called every simulation step
        during the interval; otherwise fast vehicles may be missed.
    """
    last_t = cache.get("_tp_last_decision_t")
    cache["_tp_last_decision_t"] = float(sim_time)

    if last_t is None:
        # initialize, no interval yet
        cache["_tp_new_since_last"] = set()
        return 0.0

    dt = float(sim_time) - float(last_t)
    if dt <= 0.0:
        cache["_tp_new_since_last"] = set()
        return 0.0

    new_since_last: Set[str] = cache.get("_tp_new_since_last", set())
    count = float(len(new_since_last))

    # reset interval counter (keep seen_total so vehicles are not double-counted)
    cache["_tp_new_since_last"] = set()

    return count / dt