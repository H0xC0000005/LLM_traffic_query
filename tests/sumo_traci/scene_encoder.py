import numpy as np
import libsumo as traci


def encode_tsc_state_vector_bounded(
    tls_id: str,
    *,
    moving_speed_threshold: float = 0.1,
    stopped_speed_threshold: float = 0.1,
    cache: dict | None = None,
    # -------------------- NEW: normalization / bounding knobs --------------------
    veh_equiv_len_m: float = 7.5,  # ~ vehicle length + min gap (used to estimate lane capacity)
    wait_ref_s: float = 60.0,  # scale for waiting-time squashing (tanh)
    clip_occ: float = 1.0,  # cap occupancy-like ratios to [0, clip_occ]
    # ----------------------------------------------------------------------------
) -> np.ndarray:
    """
    BOUNDED + NORMALIZED encoding (per incoming lane, stable order):
      Per lane block:
        1) q_occ     = clip(queue / lane_cap_veh, 0..1)
        2) n_occ     = clip(veh_count / lane_cap_veh, 0..1)
        3) v_norm    = clip(mean_speed_moving / lane_speed_limit, 0..1)
        4) w_norm    = tanh(mean_wait_stopped / wait_ref_s)          in [0,1)
           where mean_wait_stopped is averaged over stopped vehicles only
      Then:
        is_green_now per lane (0/1)  [unchanged]
        phase one-hot               [unchanged]
      (time_in_phase remains excluded, as you decided to drop it)

    Key changes vs your previous encoder:
      - queue and veh_count are normalized by an estimated lane capacity (bounded)
      - mean_speed is normalized by lane speed limit (bounded)
      - waiting_time is changed from SUM(wait) to MEAN(wait of stopped vehicles),
        then squashed with tanh to keep bounded
    """
    if cache is None:
        cache = {}

    def _clip01(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x > clip_occ:
            return float(clip_occ)
        return float(x)

    # Detect episode reset by time going backwards (e.g., after restart)
    # DEPRECATED: consider moving reset authority to run_ppo_tsc.py
    sim_t = float(traci.simulation.getTime())
    # if ("_last_sim_t" in cache) and (sim_t < float(cache["_last_sim_t"])):
    #     cache.clear()
    # cache["_last_sim_t"] = sim_t

    # ----- Initialize / cache lane ordering and link-index mapping -----
    if ("lane_ids" not in cache) or ("lane_to_sigidx" not in cache):
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)

        lane_to_sigidx: dict[str, list[int]] = {}
        incoming_lanes: set[str] = set()
        sigpos = 0

        for link_group in controlled_links:
            for in_lane, _out_lane, _via_lane in link_group:
                if in_lane:
                    edge_id = traci.lane.getEdgeID(in_lane)
                    if not edge_id.startswith(":") and not in_lane.startswith(":"):
                        incoming_lanes.add(in_lane)
                        lane_to_sigidx.setdefault(in_lane, []).append(sigpos)
                sigpos += 1

        lane_ids = sorted(incoming_lanes, key=lambda ln: (traci.lane.getEdgeID(ln), ln))

        cache["lane_ids"] = lane_ids
        cache["lane_to_sigidx"] = lane_to_sigidx

    lane_ids: list[str] = cache["lane_ids"]
    lane_to_sigidx: dict[str, list[int]] = cache["lane_to_sigidx"]

    # ----- Cache phase count (one-hot length) for the current program -----
    current_program = traci.trafficlight.getProgram(tls_id)
    if (cache.get("program_id") != current_program) or ("num_phases" not in cache):
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
        cache["program_id"] = current_program
        cache["num_phases"] = int(len(phases))

    num_phases: int = int(cache["num_phases"])

    # ----- Per-lane features -----
    features: list[float] = []

    ryg = traci.trafficlight.getRedYellowGreenState(tls_id)
    ryg_len = len(ryg)

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

        # -------------------- NEW: capacity-normalized, bounded queue/count --------------------
        # Estimate lane capacity in vehicles using lane length / vehicle-equivalent length.
        # This makes queue/count comparable across different lane lengths and keeps bounded.
        lane_len_m = float(traci.lane.getLength(ln))
        lane_cap_veh = max(1.0, lane_len_m / max(1e-6, float(veh_equiv_len_m)))

        q_occ = _clip01(float(queue) / lane_cap_veh)  # in [0,1]
        n_occ = _clip01(float(veh_count) / lane_cap_veh)  # in [0,1]

        # -------------------- NEW: speed normalized by speed limit (bounded) -------------------
        v_limit = float(traci.lane.getMaxSpeed(ln))
        v_norm = 0.0 if v_limit <= 1e-6 else float(mean_speed_moving) / v_limit
        if v_norm < 0.0:
            v_norm = 0.0
        if v_norm > 1.0:
            v_norm = 1.0

        # -------------------- NEW: waiting_time SUM -> MEAN(stopped) then tanh squash ---------
        # Previous: waiting_time_static_sum could grow without bound with time and queue size.
        # Now: mean over stopped vehicles, then squash to [0,1) using tanh.
        mean_wait_stopped = (
            (waiting_time_static_sum / float(queue)) if queue > 0 else 0.0
        )
        w_norm = float(
            np.tanh(mean_wait_stopped / max(1e-6, float(wait_ref_s)))
        )  # in [0,1)

        # Final bounded lane block
        features.extend([q_occ, n_occ, v_norm, w_norm])
        # -------------------------------------------------------------------------------------

        # is_green_now per lane: unchanged
        green_now = 0.0
        for idx in lane_to_sigidx.get(ln, []):
            if 0 <= idx < ryg_len and ryg[idx] in ("G", "g"):
                green_now = 1.0
                break
        is_green_flags.append(green_now)

    # Append [is_green_now] per lane (unchanged)
    features.extend(is_green_flags)

    # Phase one-hot (unchanged)
    current_phase = int(traci.trafficlight.getPhase(tls_id))
    one_hot = [0.0] * num_phases
    if 0 <= current_phase < num_phases:
        one_hot[current_phase] = 1.0
    features.extend(one_hot)

    # time_in_phase intentionally omitted (as per your decision)

    return np.asarray(features, dtype=np.float32)


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
    BOUNDED + NORMALIZED encoding, v2 (refactored to reuse utility.py):
      - Candidate A: soft saturation for occ & wait
      - Candidate E: append bounded time-since-served per major green phase

    Uses:
      - utility.get_tls_phase_plan() to get phases + major greens + ownership
    """
    if cache is None:
        cache = {}

    # local import to avoid any accidental circular dependency at module import time
    from utility import get_tls_phase_plan

    def _soft_sat(x: float, *, sat: float = 1.0) -> float:
        """Soft saturating map ~identity near 0, asymptote to `sat` for large x."""
        x = float(x)
        if x <= 0.0:
            return 0.0
        sat = max(1e-6, float(sat))
        y = sat * (1.0 - float(np.exp(-x / sat)))
        if y < 0.0:
            y = 0.0
        if y > sat:
            y = sat
        return float(y)

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
