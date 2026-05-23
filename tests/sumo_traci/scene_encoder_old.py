import numpy as np
import libsumo as traci
from utility import _soft_sat


def encode_tsc_state_vector_bounded_v2(
    tls_id: str,
    *,
    moving_speed_threshold: float = 0.1,
    stopped_speed_threshold: float = 0.1,
    cache: dict | None = None,
    # -------------------- normalization / bounding knobs --------------------
    veh_equiv_len_m: float = 7.5,  # ~ vehicle length + min gap (used to estimate lane capacity)
    wait_ref_s: float = 60.0,  # scale for waiting-time squashing (soft saturation)
    since_served_ref_s: float = 60.0,  # scale for time-since-served squashing (per major green)
    clip_occ: float = 1.0,  # soft-saturate occupancy-like ratios to [0, clip_occ]
    # -----------------------------------------------------------------------
    min_major_green_s: float = 5.0,  # forwarded to utility.get_tls_phase_plan()
) -> np.ndarray:
    """
    Encode a bounded (normalized + softly saturated) state vector for RL traffic signal control.

    The state is built from SUMO/TraCI observations at a given traffic light system (TLS) and is
    intended to be numerically stable for learning: occupancy-like ratios and time-based features
    are “soft-saturated” via `_soft_sat(...)`, and speed is clipped to [0, 1].

    Parameters
    ----------
    tls_id : str
        Traffic light system ID in SUMO/TraCI.
    moving_speed_threshold : float, default=0.1
        A vehicle is considered *moving* if `speed > moving_speed_threshold` (m/s). Used for the
        mean-moving-speed feature.
    stopped_speed_threshold : float, default=0.1
        A vehicle is considered *stopped/queued* if `speed <= stopped_speed_threshold` (m/s). Used
        for queue length and stopped-vehicle waiting-time aggregation.
    cache : dict | None, default=None
        Mutable cache reused across calls to avoid recomputing lane ordering/link mappings and to
        track per-major-phase “last served” timestamps. If None, an empty dict is created.

    veh_equiv_len_m : float, default=7.5
        Effective vehicle length (vehicle length + minimum gap) used to estimate lane capacity as:
        `lane_cap_veh = lane_length_m / veh_equiv_len_m` (min 1 vehicle).
    wait_ref_s : float, default=60.0
        Reference time (seconds) for squashing stopped-vehicle mean waiting time:
        `w_norm = _soft_sat(mean_wait_stopped / wait_ref_s, sat=1.0)`.
    since_served_ref_s : float, default=60.0
        Reference time (seconds) for squashing “time since last served” for each major green phase:
        `_soft_sat(dt / since_served_ref_s, sat=1.0)`.
    clip_occ : float, default=1.0
        Soft saturation target for occupancy-like ratios (queue/capacity and count/capacity), i.e.
        `_soft_sat(ratio, sat=clip_occ)`.
    min_major_green_s : float, default=5.0
        Forwarded to `utility.get_tls_phase_plan(...)` when determining major green phases.

    Returns
    -------
    np.ndarray
        1-D `float32` feature vector. Define:
          - L = number of incoming lanes controlled by `tls_id` (sorted by `(edge_id, lane_id)`),
          - P = number of phases in the active TLS program (`len(plan.phases)`),
          - M = number of major green phases (`len(plan.major_greens)`).

        Layout (concatenated in this order):

        1) Per-lane features (for each lane i in 0..L-1), 4 values:
           - q_occ : soft-saturated queue ratio = queue_veh / lane_cap_veh, sat=`clip_occ`
           - n_occ : soft-saturated count ratio = veh_count / lane_cap_veh, sat=`clip_occ`
           - v_norm: mean speed of *moving* vehicles divided by lane speed limit, clipped to [0, 1]
           - w_norm: soft-saturated mean waiting time of *stopped* vehicles / `wait_ref_s`, sat=1

        2) Per-lane green flags, L values:
           - is_green[i] = 1 if any controlled signal index for the lane is 'G' or 'g' in the current
             RYG string, else 0.

        3) Current phase one-hot, P values:
           - One-hot encoding of `traci.trafficlight.getPhase(tls_id)` over all phases.

        4) Time-since-served for major greens, M values:
           - For each major green phase `ph`, append `_soft_sat((sim_t - last_served[ph]) /
             since_served_ref_s, sat=1.0)`.
           - `last_served[ph]` is updated only when the *current* phase equals `ph`.

        Total length: `4*L + L + P + M = 5*L + P + M`.

    Notes
    -----
    - Requires an active TraCI connection and a `_soft_sat(x, sat)` function available in the
      module scope. `_soft_sat` is expected to monotonically squash (typically nonnegative) inputs
      to a bounded range approximately within [0, sat].
    - Phase plan information is obtained via a local import of `utility.get_tls_phase_plan(...)`.
    - Internal cache keys used: `"lane_ids"`, `"lane_to_sigidx"`, `"_v2_major_last_prog_id"`,
      `"_v2_major_last_t"`. If the TLS program ID changes, per-major-phase last-served times are
      reinitialized to the current simulation time.

    Raises
    ------
    traci.TraCIException
        If `tls_id` is invalid or if TraCI queries fail.
    """
    if cache is None:
        cache = {}

    # local import to avoid any accidental circular dependency at module import time
    from utility import get_tls_phase_plan

    def _clip01_unit(x: float) -> float:
        x = float(x)
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    # ----------------------------
    # episode reset detection
    # DEPRECATED: consider moving reset authority to run_ppo_tsc.py
    # ----------------------------
    sim_t = float(traci.simulation.getTime())
    # if ("_last_sim_t" in cache) and (sim_t < float(cache["_last_sim_t"])):
    #     cache.clear()
    # cache["_last_sim_t"] = sim_t

    # ----------------------------
    # lane ordering + link-index map (keep local caching as before)
    # ----------------------------
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

        lane_ids = sorted(
            list(incoming_lanes), key=lambda x: (traci.lane.getEdgeID(x), x)
        )
        cache["lane_ids"] = lane_ids
        cache["lane_to_sigidx"] = lane_to_sigidx

    lane_ids: list[str] = cache["lane_ids"]
    lane_to_sigidx: dict[str, list[int]] = cache["lane_to_sigidx"]

    # ----------------------------
    # phase-plan (REUSE utility.py)
    # ----------------------------
    plan = get_tls_phase_plan(tls_id, cache, min_major_green_s=float(min_major_green_s))
    num_phases = int(len(plan.phases))
    major_phases = [int(p) for p in plan.major_greens]
    major_set = set(major_phases)

    # init / reset per-major last-served times if program changed
    prog_key = "_v2_major_last_prog_id"
    last_key = "_v2_major_last_t"
    if (cache.get(prog_key) != plan.program_id) or (last_key not in cache):
        cache[prog_key] = plan.program_id
        cache[last_key] = {ph: sim_t for ph in major_phases}

    major_last_t: dict[int, float] = cache[last_key]

    # ----------------------------
    # per-lane features
    # ----------------------------
    ryg = traci.trafficlight.getRedYellowGreenState(tls_id)
    ryg_len = len(ryg)

    features: list[float] = []
    is_green_flags: list[float] = []

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

        mean_speed_moving = (
            (moving_speeds_sum / moving_count) if moving_count > 0 else 0.0
        )

        # lane capacity estimate
        lane_len_m = float(traci.lane.getLength(ln))
        lane_cap_veh = max(1.0, lane_len_m / max(1e-6, float(veh_equiv_len_m)))

        # Candidate A: soft saturation instead of hard clip / tanh
        q_occ = _soft_sat(float(queue) / lane_cap_veh, sat=float(clip_occ))
        n_occ = _soft_sat(float(veh_count) / lane_cap_veh, sat=float(clip_occ))

        v_limit = float(traci.lane.getMaxSpeed(ln))
        v_norm = 0.0 if v_limit <= 1e-6 else float(mean_speed_moving) / v_limit
        v_norm = _clip01_unit(v_norm)

        mean_wait_stopped = (
            (waiting_time_static_sum / float(queue)) if queue > 0 else 0.0
        )
        w_norm = _soft_sat(mean_wait_stopped / max(1e-6, float(wait_ref_s)), sat=1.0)

        features.extend([q_occ, n_occ, v_norm, w_norm])

        green_now = 0.0
        for idx in lane_to_sigidx.get(ln, []):
            if 0 <= idx < ryg_len and ryg[idx] in ("G", "g"):
                green_now = 1.0
                break
        is_green_flags.append(green_now)

    features.extend(is_green_flags)

    # ----------------------------
    # phase one-hot over ALL phases
    # ----------------------------
    current_phase = int(traci.trafficlight.getPhase(tls_id))
    one_hot = [0.0] * num_phases
    if 0 <= current_phase < num_phases:
        one_hot[current_phase] = 1.0
    features.extend(one_hot)

    # ----------------------------
    # Candidate E: time-since-served per MAJOR green phase
    # update last-served only when the current phase IS a major green
    # ----------------------------
    if current_phase in major_set:
        major_last_t[current_phase] = sim_t

    denom = max(1e-6, float(since_served_ref_s))
    for ph in major_phases:
        dt = sim_t - float(major_last_t.get(ph, sim_t))
        features.append(_soft_sat(dt / denom, sat=1.0))

    return np.asarray(features, dtype=np.float32)
