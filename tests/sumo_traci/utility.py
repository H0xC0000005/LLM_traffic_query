# utility.py

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple
from dataclasses import dataclass
from sumolib import checkBinary
import libsumo as traci


"""
essential data structures 
"""

# -------------------- Fixed-time baseline schedule --------------------
# Proof-of-concept schedule: list of (phase_index, duration_seconds).
#
# IMPORTANT:
#   - Phase indices must exist in your TLS program.
#   - If your TLS has different phase ordering, update this list.
#   - If a phase index is out of range, it will be wrapped modulo action_dim.
FIXED_SCHEDULE: List[Tuple[int, float]] = [
    (0, 40.0),
    (1, 20.0),
    (2, 40.0),
    (3, 20.0),
]


@dataclass
class TLSControllerState:
    next_decision_time: float = 0.0
    pending_state: Optional[np.ndarray] = None
    pending_action: Optional[int] = None
    pending_epsilon: float = 0.0
    in_control_when_pending: bool = False
    next_target_update_time: float = 0.0
    pending_segments: Deque[Tuple[int, float]] = field(
        default_factory=deque
    )  # (phase_idx, duration_s)
    segment_end_time: float = 0.0  # when current segment ends
    # log_step: int = 0


"""
phase management (finding out major green phases and register auxilliary phases)
"""

# [NEW BLOCK] TLS phase-plan helpers (major greens + auxiliary phases)


@dataclass(frozen=True)
class TLSPhasePlan:
    program_id: str
    phases: List[Tuple[int, float, str]]  # (idx, duration_s, state_str)
    major_greens: List[
        int
    ]  # indices of "major green" phases (agent actions map to these)
    owner_major: List[
        int
    ]  # owner_major[phase_idx] -> major green phase idx that owns it
    aux_after_major: Dict[
        int, List[int]
    ]  # major green idx -> aux phase indices after it until next major
    phase_duration: Dict[int, float]  # phase_idx -> configured duration (seconds)


def _get_active_program_logic(tls_id: str):
    """Return the active ProgramLogic object for tls_id."""
    program_id = traci.trafficlight.getProgram(tls_id)
    logics = traci.trafficlight.getAllProgramLogics(tls_id)

    logic = None
    for lg in logics:
        try:
            if lg.getSubID() == program_id:
                logic = lg
                break
        except Exception:
            if getattr(lg, "programID", None) == program_id:
                logic = lg
                break
    if logic is None:
        logic = logics[0]
    return program_id, logic


def _default_is_major_green(
    state: str, duration_s: float, *, min_major_green_s: float
) -> bool:
    """
    Heuristic "major green":
      - contains any green signal (G/g)
      - contains NO yellow (y/Y)
      - duration >= min_major_green_s  (filters out short clearance phases)
    """
    has_green = ("G" in state) or ("g" in state)
    has_yellow = ("y" in state) or ("Y" in state)
    return (
        has_green
        and (not has_yellow)
        and (float(duration_s) >= float(min_major_green_s))
    )


def get_tls_phase_plan(
    tls_id: str,
    cache: Dict[str, Any],
    *,
    min_major_green_s: float = 5.0,
    is_major_green: Optional[Callable[[str, float], bool]] = None,
) -> TLSPhasePlan:
    """
    Build (and cache) a relaxed phase plan:
      - agent chooses only major green phases
      - any phases between two major greens are treated as auxiliary phases owned by the earlier major green
    """
    program_id = traci.trafficlight.getProgram(tls_id)
    key = "_tls_phase_plan"

    # reuse cached plan if program unchanged
    if key in cache:
        plan: TLSPhasePlan = cache[key]
        if plan.program_id == program_id:
            return plan

    program_id, logic = _get_active_program_logic(tls_id)
    try:
        phases_obj = logic.getPhases()
    except Exception:
        phases_obj = getattr(logic, "phases")

    phases: List[Tuple[int, float, str]] = []
    phase_duration: Dict[int, float] = {}
    for i, ph in enumerate(phases_obj):
        dur = float(getattr(ph, "duration"))
        st = str(getattr(ph, "state"))
        phases.append((int(i), dur, st))
        phase_duration[int(i)] = dur

    if is_major_green is None:

        def is_major_green_local(s: str, d: float) -> bool:
            return _default_is_major_green(
                s, d, min_major_green_s=float(min_major_green_s)
            )

        is_major_green = is_major_green_local

    major_greens = [idx for (idx, dur, st) in phases if is_major_green(st, dur)]
    if not major_greens:
        raise RuntimeError(
            f"[{tls_id}] No major green phases found. "
            f"Adjust min_major_green_s or provide is_major_green()."
        )

    n = len(phases)
    major_set = set(major_greens)

    # owner_major: each phase belongs to the most recent major green in cyclic order
    owner_major: List[int] = [-1] * n
    last_major = major_greens[-1]  # for phases before first major in list
    for i in range(n):
        if i in major_set:
            last_major = i
        owner_major[i] = int(last_major)

    # aux_after_major: phases strictly between major k and next major (cyclic)
    aux_after_major: Dict[int, List[int]] = {mg: [] for mg in major_greens}
    for j, mg in enumerate(major_greens):
        nxt = major_greens[(j + 1) % len(major_greens)]
        aux: List[int] = []
        k = (mg + 1) % n
        while k != nxt:
            aux.append(int(k))
            k = (k + 1) % n
        aux_after_major[int(mg)] = aux

    plan = TLSPhasePlan(
        program_id=str(program_id),
        phases=phases,
        major_greens=[int(x) for x in major_greens],
        owner_major=owner_major,
        aux_after_major=aux_after_major,
        phase_duration=phase_duration,
    )
    cache[key] = plan
    return plan


def tls_major_action_dim(
    tls_id: str, cache: Dict[str, Any], *, min_major_green_s: float = 5.0
) -> int:
    plan = get_tls_phase_plan(tls_id, cache, min_major_green_s=min_major_green_s)
    return int(len(plan.major_greens))


def tls_action_to_major_phase(
    tls_id: str, cache: Dict[str, Any], action: int, *, min_major_green_s: float = 5.0
) -> int:
    plan = get_tls_phase_plan(tls_id, cache, min_major_green_s=min_major_green_s)
    a = int(action)
    if a < 0 or a >= len(plan.major_greens):
        raise ValueError(
            f"action out of range: {a} (num_major={len(plan.major_greens)})"
        )
    return int(plan.major_greens[a])


def tls_current_major_phase(
    tls_id: str,
    cache: Dict[str, Any],
    current_phase: Optional[int] = None,
    *,
    min_major_green_s: float = 5.0,
) -> int:
    plan = get_tls_phase_plan(tls_id, cache, min_major_green_s=min_major_green_s)
    cur = int(
        traci.trafficlight.getPhase(tls_id) if current_phase is None else current_phase
    )
    if cur < 0 or cur >= len(plan.owner_major):
        # fall back to the first major green
        return int(plan.major_greens[0])
    return int(plan.owner_major[cur])


def tls_build_switch_segments(
    tls_id: str,
    cache: Dict[str, Any],
    *,
    target_major_phase: int,
    hold_s: float,
    current_phase: Optional[int] = None,
    min_major_green_s: float = 5.0,
    min_aux_dur_s: float = 0.1,
) -> List[Tuple[int, float]]:
    """
    Build segments for one macro-action:
      - if switching away from current major: play all auxiliary phases owned by current major (configured durations)
      - then play target major green for hold_s (hold_s excludes aux time)
    Returns list[(phase_idx, duration_s)].
    """
    plan = get_tls_phase_plan(tls_id, cache, min_major_green_s=min_major_green_s)
    cur_major = tls_current_major_phase(
        tls_id, cache, current_phase=current_phase, min_major_green_s=min_major_green_s
    )
    tgt = int(target_major_phase)

    segs: List[Tuple[int, float]] = []
    if tgt != int(cur_major):
        for aux_idx in plan.aux_after_major.get(int(cur_major), []):
            dur = float(plan.phase_duration.get(int(aux_idx), 0.0))
            if dur < float(min_aux_dur_s):
                dur = float(min_aux_dur_s)
            segs.append((int(aux_idx), dur))

    segs.append((tgt, float(hold_s)))
    return segs


def tls_set_phase_frozen(tls_id: str, phase_idx: int) -> None:
    """Set a TLS phase and prevent SUMO from auto-advancing."""
    traci.trafficlight.setPhase(tls_id, int(phase_idx))
    traci.trafficlight.setPhaseDuration(tls_id, 1e6)


def tls_advance_pending_segments(
    *,
    tls_id: str,
    pending_segments: Deque[Tuple[int, float]],
    segment_end_time: float,
    sim_t: float,
) -> float:
    """
    If the currently-playing segment has ended (sim_t >= segment_end_time),
    pop the next (phase, dur) from pending_segments, set it frozen, and return
    the new segment_end_time. If nothing remains, return 0.0.
    """
    if segment_end_time <= 0.0:
        return 0.0
    if sim_t < float(segment_end_time):
        return float(segment_end_time)

    if not pending_segments:
        return 0.0

    next_phase, next_dur = pending_segments.popleft()
    tls_set_phase_frozen(tls_id, int(next_phase))
    return float(sim_t) + float(next_dur)


"""
helper functions
"""


# =======================
# [NEW] shared queue extractor
# =======================
def _extract_queues_from_encoded_state(
    state_vec: Sequence[float],
    *,
    num_lanes: int,
    lane_block_size: int,
    queue_offset_in_block: int,
    scale: float = 1.0,
    clip_nonnegative: bool = True,
) -> List[float]:
    """
    Extract per-lane queue lengths from the encoded state vector.

    Returns a list of length num_lanes.
    If scale > 0, each queue is divided by scale (useful for normalization).
    """
    if num_lanes <= 0:
        raise ValueError("num_lanes must be > 0")

    expected_min_len = num_lanes * lane_block_size
    if len(state_vec) < expected_min_len:
        raise ValueError(
            f"state_vec too short: len={len(state_vec)} < {expected_min_len} "
            f"(num_lanes={num_lanes}, lane_block_size={lane_block_size})"
        )

    s = float(scale) if float(scale) > 0.0 else 1.0

    queues: List[float] = []
    for i in range(num_lanes):
        idx = i * lane_block_size + queue_offset_in_block
        q = float(state_vec[idx])
        if clip_nonnegative and q < 0.0:
            q = 0.0
        queues.append(q / s)
    return queues


def start_sumo(
    sumocfg: str, *, gui: bool, delay_ms: int, sumo_seed: int, traffic_scale: float
) -> None:
    binary = checkBinary("sumo-gui" if gui else "sumo")
    cmd: List[str] = [
        binary,
        "-c",
        sumocfg,
        "--start",
        "--no-step-log",
        "true",
        "--delay",
        str(delay_ms),
        "--seed",
        str(int(sumo_seed)),
        "--scale",
        str(float(traffic_scale)),
    ]
    traci.start(cmd)


def get_phase_count(tls_id: str) -> int:
    current_program = traci.trafficlight.getProgram(tls_id)
    logics = traci.trafficlight.getAllProgramLogics(tls_id)

    logic = None
    for lg in logics:
        try:
            if lg.getSubID() == current_program:
                logic = lg
                break
        except Exception:
            if getattr(lg, "programID", None) == current_program:
                logic = lg
                break
    if logic is None:
        logic = logics[0]

    try:
        phases = logic.getPhases()
    except Exception:
        phases = getattr(logic, "phases")
    return int(len(phases))


"""
various reward functions
"""


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
    r = -(0.5 * val + 0.5 * max(queues))
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
    w1, w2 = float(weights[0]), float(weights[1])
    if w1 < 0.0 or w2 < 0.0:
        raise ValueError("weights must be nonnegative")
    if power < 1.0:
        raise ValueError("power must be >= 1.0")

    queues = _extract_queues_from_encoded_state(
        state_vec,
        num_lanes=num_lanes,
        lane_block_size=lane_block_size,
        queue_offset_in_block=queue_offset_in_block,
        scale=scale,
        clip_nonnegative=clip_nonnegative,
    )
    # Top-2 largest
    q_sorted = sorted(queues, reverse=True)
    q1 = q_sorted[0]
    q2 = q_sorted[1] if len(q_sorted) >= 2 else q_sorted[0]

    penalty = w1 * (q1**power) + w2 * (q2**power)
    return -penalty


# =======================
# [NEW] softmax-weighted queue penalty (smooth "top" approximation)
# =======================
def reward_softmax_queue_from_encoded_state(
    state_vec: Sequence[float],
    *,
    num_lanes: int,
    lane_block_size: int = 4,
    queue_offset_in_block: int = 0,
    power: float = 1.0,
    scale: float = 1.0,
    softmax_beta: float = 5.0,
    clip_nonnegative: bool = True,
) -> float:
    """
    Smooth alternative to "top-k" queue penalty: softmax-weighted penalty over ALL lanes.

      weights_i = softmax(beta * q_i)
      penalty   = sum_i weights_i * (q_i ** power)
      reward    = -penalty

    - weights sum to 1 (normalization)
    - as beta -> +inf, weights concentrate on the maximum queue (approaches max-like behavior)
    - power > 1 penalizes long queues super-linearly

    All queues are first normalized by `scale` (same as in reward_top2_queue_from_encoded_state).
    """
    if power < 1.0:
        raise ValueError("power must be >= 1.0")

    qs = _extract_queues_from_encoded_state(
        state_vec,
        num_lanes=num_lanes,
        lane_block_size=lane_block_size,
        queue_offset_in_block=queue_offset_in_block,
        scale=scale,
        clip_nonnegative=clip_nonnegative,
    )

    beta = float(softmax_beta)
    if beta <= 0.0:
        # beta<=0: raise error instead of uniform weights
        raise ValueError("softmax_beta must be > 0.0")
    else:
        logits = beta * np.asarray(qs, dtype=np.float64)
        logits = logits - float(np.max(logits))  # stabilize
        exps = np.exp(logits)
        weights = exps / float(np.sum(exps) + 1e-12)

    q_pow = np.asarray(qs, dtype=np.float64) ** float(power)
    penalty = float(np.sum(weights * q_pow))
    return -penalty


# =======================
# [NEW] softmax-weighted waiting-time barrier (smooth "max-wait" approximation)
# =======================
def reward_softmax_wait_barrier_from_encoded_state(
    state_vec: Sequence[float],
    *,
    num_lanes: int,
    lane_block_size: int = 4,
    wait_offset_in_block: int = 3,  # [queue, veh_count, speed, waiting_time]
    wait_ref_s: float = 60.0,  # normalize waits by this (seconds)
    softmax_beta: float = 10.0,
    barrier_start_s: float = 30.0,  # no penalty until soft-wait exceeds this
    barrier_power: float = 1.0,
    clip_nonnegative: bool = True,
) -> float:
    """
    Smooth waiting-time barrier reward (negative).

    Steps:
      1) Extract per-lane waiting times from encoded state
      2) Normalize by wait_ref_s
      3) Compute softmax-weighted mean wait (smooth max-like)
      4) Apply a threshold barrier: penalty = max(0, soft_wait - start)^power
      5) reward = -penalty

    Returns:
      wait_reward <= 0
    """
    if num_lanes <= 0:
        raise ValueError("num_lanes must be > 0")
    if wait_ref_s <= 0.0:
        raise ValueError("wait_ref_s must be > 0")
    if softmax_beta <= 0.0:
        raise ValueError("softmax_beta must be > 0")
    if barrier_power < 1.0:
        raise ValueError("barrier_power must be >= 1.0")

    expected_min_len = num_lanes * lane_block_size
    if len(state_vec) < expected_min_len:
        raise ValueError(
            f"state_vec too short: len={len(state_vec)} < {expected_min_len} "
            f"(num_lanes={num_lanes}, lane_block_size={lane_block_size})"
        )

    # ---- extract + normalize waiting times ----
    waits = []
    inv_ref = 1.0 / float(wait_ref_s)
    for i in range(num_lanes):
        idx = i * lane_block_size + wait_offset_in_block
        w = float(state_vec[idx])
        if clip_nonnegative and w < 0.0:
            w = 0.0
        waits.append(w * inv_ref)  # normalized wait

    waits = np.asarray(waits, dtype=np.float64)

    # ---- softmax weights over waits (smooth max) ----
    beta = float(softmax_beta)
    logits = beta * waits
    logits = logits - float(np.max(logits))  # stabilize
    exps = np.exp(logits)
    weights = exps / float(np.sum(exps) + 1e-12)

    soft_wait = float(np.sum(weights * waits))

    # ---- barrier threshold in normalized units ----
    start = float(barrier_start_s) * inv_ref
    overflow = soft_wait - start
    if overflow <= 0.0:
        return 0.0

    penalty = overflow ** float(barrier_power)
    return -float(penalty)


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


# =======================
# [NEW] composite reward variant using softmax queue term
# =======================
def reward_throughput_plus_softmax_queue(
    *,
    tls_id: str,
    sim_time: float,
    state_vec: Sequence[float],
    cache: Dict,
    num_lanes: int,
    throughput_ref_veh_per_s: float,
    queue_ref_veh: float,
    w_throughput: float = 1.0,
    w_queue: float = 1.0,
    queue_power: float = 1.0,
    softmax_beta: float = 5.0,
    lane_block_size: int = 4,
    queue_offset_in_block: int = 0,
    reward_clip: Optional[Tuple[float, float]] = (-1.0, 1.0),
) -> float:
    def _clip(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x

    if throughput_ref_veh_per_s <= 0.0:
        raise ValueError("throughput_ref_veh_per_s must be > 0")
    if queue_ref_veh <= 0.0:
        raise ValueError("queue_ref_veh must be > 0")

    thr = reward_throughput_per_second_on_decision(sim_time=sim_time, cache=cache)
    thr_norm = _clip(float(thr) / float(throughput_ref_veh_per_s), 0.0, 1.0)

    q_reward = reward_softmax_queue_from_encoded_state(
        state_vec,
        num_lanes=num_lanes,
        lane_block_size=lane_block_size,
        queue_offset_in_block=queue_offset_in_block,
        power=queue_power,
        scale=queue_ref_veh,
        softmax_beta=softmax_beta,
        clip_nonnegative=True,
    )

    print(
        f">> reward: thr={thr:.3f} (norm {thr_norm:.3f}), softmax_queue_reward={q_reward:.5f}"
    )

    r = float(w_throughput) * thr_norm + float(w_queue) * float(q_reward)

    if reward_clip is not None:
        lo, hi = float(reward_clip[0]), float(reward_clip[1])
        if lo > hi:
            lo, hi = hi, lo
        r = _clip(r, lo, hi)

    return float(r)


# =======================
# [NEW] composite reward: throughput + softmax queue + delta softmax queue + softmax wait barrier
# =======================
def reward_throughput_plus_softmax_queue_deltaq_plus_softmax_wait_barrier_v2(
    *,
    tls_id: str,
    sim_time: float,
    state_vec: Sequence[float],
    cache: Dict,
    num_lanes: int,
    throughput_ref_veh_per_s: float,
    queue_ref_veh: float,
    wait_ref_s: float = 60.0,
    wait_barrier_start_s: float = 30.0,
    # ---- weights (recommended defaults below) ----
    w_throughput: float = 1.0,
    w_queue: float = 1.0,
    w_delta_queue: float = 0.5,
    w_wait_barrier: float = 0.5,
    # ---- queue term params ----
    queue_power: float = 1.0,
    softmax_queue_beta: float = 5.0,
    # ---- wait term params ----
    softmax_wait_beta: float = 10.0,
    wait_barrier_power: float = 1.0,
    # ---- encoding layout ----
    lane_block_size: int = 4,
    queue_offset_in_block: int = 0,
    wait_offset_in_block: int = 3,
) -> float:
    """
    Reward terms:
      1) Throughput term (normalized by throughput_ref_veh_per_s, NO CLIP)
      2) Softmax queue penalty (negative)
      3) Delta softmax queue improvement: (prev_penalty - cur_penalty) (positive if improved)
      4) Softmax wait barrier penalty (negative): smooth "max-wait" above a threshold

    Notes:
      - No final reward clipping: uses full reward range.
      - Delta term uses cached previous softmax-queue penalty per TLS.
    """
    if throughput_ref_veh_per_s <= 0.0:
        raise ValueError("throughput_ref_veh_per_s must be > 0")
    if queue_ref_veh <= 0.0:
        raise ValueError("queue_ref_veh must be > 0")
    if wait_ref_s <= 0.0:
        raise ValueError("wait_ref_s must be > 0")

    # 1) Throughput (veh/s) over last decision interval, normalized (no clamp)
    thr = reward_throughput_per_second_on_decision(sim_time=sim_time, cache=cache)
    thr_norm = float(thr) / float(throughput_ref_veh_per_s)

    # 2) Absolute softmax queue reward (negative)
    q_reward = reward_softmax_queue_from_encoded_state(
        state_vec,
        num_lanes=num_lanes,
        lane_block_size=lane_block_size,
        queue_offset_in_block=queue_offset_in_block,
        power=queue_power,
        scale=queue_ref_veh,
        softmax_beta=softmax_queue_beta,
        clip_nonnegative=True,
    )
    # convert reward -> penalty scalar for delta
    q_penalty = -float(q_reward)  # >= 0

    # 3) Delta softmax queue improvement (positive if queues improved)
    prev_key = f"_rw_prev_softmax_q_penalty::{tls_id}"
    prev_penalty = cache.get(prev_key, None)
    cache[prev_key] = q_penalty

    if prev_penalty is None:
        delta_q = 0.0
    else:
        delta_q = float(prev_penalty) - float(q_penalty)  # >0 if improved

    # 4) Softmax wait barrier (negative)
    wait_reward = reward_softmax_wait_barrier_from_encoded_state(
        state_vec,
        num_lanes=num_lanes,
        lane_block_size=lane_block_size,
        wait_offset_in_block=wait_offset_in_block,
        wait_ref_s=wait_ref_s,
        softmax_beta=softmax_wait_beta,
        barrier_start_s=wait_barrier_start_s,
        barrier_power=wait_barrier_power,
        clip_nonnegative=True,
    )

    # Final combined reward (no clipping)
    r = (
        float(w_throughput) * float(thr_norm)
        + float(w_queue) * float(q_reward)
        + float(w_delta_queue) * float(delta_q)
        + float(w_wait_barrier) * float(wait_reward)
    )
    print(
        f">> reward: thr={thr:.3f} (norm {thr_norm:.3f}), q={q_reward:.3f}, delta_q={delta_q:.3f}, wait_barrier={wait_reward:.3f}"
    )
    return float(r)
