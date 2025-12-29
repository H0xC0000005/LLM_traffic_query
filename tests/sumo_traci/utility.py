# utility.py

from __future__ import annotations

from typing import Dict, Optional, Sequence, Set, Tuple
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


def reward_top2_queue_from_encoded_state(
    state_vec: Sequence[float],
    *,
    num_lanes: int,
    lane_block_size: int = 4,
    queue_offset_in_block: int = 0,
    weights: Tuple[float, float] = (0.7, 0.3),
    power: float = 1.0,
    scale: float = 1.0,
    clip_nonnegative: bool = True,
) -> float:
    """
    Reward based on the TOP-2 longest queues across incoming lanes.

    Assumptions about encoding (matching your current encoder, time_in_phase dropped):
      - Per incoming lane block: [queue, veh_count, mean_speed, waiting_time] (lane_block_size=4)
      - Then: is_green_now per lane (num_lanes values)
      - Then: phase one-hot (remaining values)

    Args:
      state_vec: encoded feature vector.
      num_lanes: number of incoming lanes encoded.
      lane_block_size: number of features per lane in the lane block.
      queue_offset_in_block: index of queue feature within each lane block.
      weights: (w1, w2) weights for largest and 2nd-largest queues. Should sum to 1.0 (recommended).
      power: >=1.0. If >1, penalizes long queues more heavily (starvation/spillback).
      scale: optional scaling divisor applied to queues before computing penalty (useful when combining with throughput).
      clip_nonnegative: clamp negative queue values to 0 defensively.

    Returns:
      Reward (float): higher is better. Default is negative penalty:
        reward = - (w1*q1^p + w2*q2^p), with optional scaling.

    Notes:
      - If num_lanes == 1, q2 is taken equal to q1.
      - If scale <= 0, scale is treated as 1.0.
    """
    if num_lanes <= 0:
        raise ValueError("num_lanes must be > 0")

    expected_min_len = num_lanes * lane_block_size
    if len(state_vec) < expected_min_len:
        raise ValueError(
            f"state_vec too short: len={len(state_vec)} < {expected_min_len} "
            f"(num_lanes={num_lanes}, lane_block_size={lane_block_size})"
        )

    w1, w2 = float(weights[0]), float(weights[1])
    if w1 < 0.0 or w2 < 0.0:
        raise ValueError("weights must be nonnegative")
    if power < 1.0:
        raise ValueError("power must be >= 1.0")

    s = float(scale) if float(scale) > 0.0 else 1.0

    # Extract queue per lane
    queues = []
    base = 0
    for i in range(num_lanes):
        idx = base + i * lane_block_size + queue_offset_in_block
        q = float(state_vec[idx])
        if clip_nonnegative and q < 0.0:
            q = 0.0
        queues.append(q / s)

    # Top-2 largest
    q_sorted = sorted(queues, reverse=True)
    q1 = q_sorted[0]
    q2 = q_sorted[1] if len(q_sorted) >= 2 else q_sorted[0]

    penalty = w1 * (q1 ** power) + w2 * (q2 ** power)
    return -penalty


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

    # NEW: integer counter since last reward call
    count_since_last: int = cache.setdefault("_tp_count_since_last", 0)

    for ln in out_lanes:
        for vid in traci.lane.getLastStepVehicleIDs(ln):
            if vid not in seen_total:
                seen_total.add(vid)
                count_since_last += 1

    # write back updated counter
    cache["_tp_count_since_last"] = count_since_last

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
        cache["_tp_count_since_last"] = 0
        return 0.0

    dt = float(sim_time) - float(last_t)
    if dt <= 0.0:
        cache["_tp_count_since_last"] = 0
        return 0.0

    count = float(cache.get("_tp_count_since_last", 0))

    # reset interval counter (keep seen_total so vehicles are not double-counted)
    cache["_tp_count_since_last"] = 0

    return count / dt



def reward_throughput_plus_top2_queue(
    *,
    tls_id: str,
    sim_time: float,
    state_vec: Sequence[float],
    cache: Dict,
    num_lanes: int,
    # --- normalization references (set these so scales match) ---
    throughput_ref_veh_per_s: float,
    queue_ref_veh: float,
    # --- weighting ---
    w_throughput: float = 1.0,
    w_queue: float = 1.0,
    # --- top-2 queue shaping ---
    top2_weights: Tuple[float, float] = (0.7, 0.3),
    queue_power: float = 1.0,
    # --- encoding layout ---
    lane_block_size: int = 4,
    queue_offset_in_block: int = 0,
    # --- optional clipping of final reward ---
    reward_clip: Optional[Tuple[float, float]] = (-1.0, 1.0),
) -> float:
    """
    Composite reward:
      + normalized average throughput over the whole decision interval
      - normalized top-2 longest queues (anti-starvation)

    Requires:
      - throughput_tracker_step(tls_id, cache) is called EVERY simulationStep()
      - this function is called at each decision point (to close the interval)

    Normalization:
      throughput_norm = clip(throughput / throughput_ref_veh_per_s, 0, 1)
      queue_norm is handled by passing scale=queue_ref_veh into the top-2 queue reward
        (so queues are effectively measured as a fraction of queue_ref_veh)

    Final:
      r = w_throughput * throughput_norm + w_queue * queue_reward
      where queue_reward <= 0

    Choose refs:
      - throughput_ref_veh_per_s: a reasonable "good" throughput (empirical 90-95% percentile under fixed-time works well)
      - queue_ref_veh: lane storage cap in vehicles (or a conservative threshold where you consider it 'too long')
    """
    def _clip(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x

    if throughput_ref_veh_per_s <= 0.0:
        raise ValueError("throughput_ref_veh_per_s must be > 0")
    if queue_ref_veh <= 0.0:
        raise ValueError("queue_ref_veh must be > 0")

    # 1) Throughput averaged over the whole span since last decision (veh/s)
    thr = reward_throughput_per_second_on_decision(sim_time=sim_time, cache=cache)
    thr_norm = _clip(float(thr) / float(throughput_ref_veh_per_s), 0.0, 1.0)

    # 2) Top-2 queue penalty (negative), normalized by queue_ref_veh via scale
    q_reward = reward_top2_queue_from_encoded_state(
        state_vec,
        num_lanes=num_lanes,
        lane_block_size=lane_block_size,
        queue_offset_in_block=queue_offset_in_block,
        weights=top2_weights,
        power=queue_power,
        scale=queue_ref_veh,          # <-- key: normalize queue magnitudes
        clip_nonnegative=True,
    )
    # q_reward is <= 0

    r = float(w_throughput) * thr_norm + float(w_queue) * float(q_reward)

    if reward_clip is not None:
        lo, hi = float(reward_clip[0]), float(reward_clip[1])
        if lo > hi:
            lo, hi = hi, lo
        r = _clip(r, lo, hi)

    return float(r)