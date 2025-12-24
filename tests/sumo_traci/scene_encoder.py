import numpy as np
import traci

def encode_tsc_state_vector(
    tls_id: str,
    *,
    moving_speed_threshold: float = 0.1,
    stopped_speed_threshold: float = 0.1,
    cache: dict | None = None,
):
    """
    Compute an RL state encoding vector for a single TLS at the current SUMO frame.

    Encoding (fixed order):
      1) Per incoming lane block, lanes sorted by (edge_id, lane_id):
         [queue, veh_count, mean_speed_moving_only, waiting_time_static_only]
         - queue: number of vehicles on lane with speed <= stopped_speed_threshold
         - veh_count: number of vehicles currently on lane
         - mean_speed_moving_only: mean speed over vehicles with speed > moving_speed_threshold
         - waiting_time_static_only: sum of vehicle.getWaitingTime(veh) over vehicles with speed <= stopped_speed_threshold
      2) is_green_now per lane (same lane order): 1.0 if any controlled connection from that lane is green (G/g), else 0.0
      3) phase one-hot (length = number of phases in the currently active TLS program)
      4) time-in-phase: elapsed seconds since the current phase became active

    Args:
      tls_id: traffic light ID
      moving_speed_threshold: vehicles with speed > this are considered "moving" for mean_speed
      stopped_speed_threshold: vehicles with speed <= this are considered "static" for queue and waiting_time
      cache: mutable dict to keep stable lane ordering, TLS link mapping, and time-in-phase tracking across calls

    Returns:
      1D numpy float32 array.
    """

    if cache is None:
        cache = {}

    # Detect episode reset by time going backwards (e.g., after traci.load / restart)
    sim_t = float(traci.simulation.getTime())
    if ("_last_sim_t" in cache) and (sim_t < float(cache["_last_sim_t"])):
        cache.clear()
    cache["_last_sim_t"] = sim_t

    # ----- Initialize / cache lane ordering and link-index mapping -----
    if ("lane_ids" not in cache) or ("lane_to_sigidx" not in cache):
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)

        lane_to_sigidx: dict[str, list[int]] = {}
        incoming_lanes: set[str] = set()
        sigpos = 0  # position in rYG state string

        for link_group in controlled_links:
            for (in_lane, _out_lane, _via_lane) in link_group:
                # Each connection consumes one character in the state string.
                if in_lane:
                    edge_id = traci.lane.getEdgeID(in_lane)
                    # Skip internal edges/lanes
                    if not edge_id.startswith(":") and not in_lane.startswith(":"):
                        incoming_lanes.add(in_lane)
                        lane_to_sigidx.setdefault(in_lane, []).append(sigpos)
                sigpos += 1

        # Stable order: sort by (edge_id, lane_id)
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

    # ----- Time-in-phase tracking -----
    current_phase = int(traci.trafficlight.getPhase(tls_id))
    if ("_last_phase" not in cache) or ("_phase_start_t" not in cache):
        cache["_last_phase"] = current_phase
        cache["_phase_start_t"] = sim_t
    elif current_phase != int(cache["_last_phase"]):
        cache["_last_phase"] = current_phase
        cache["_phase_start_t"] = sim_t

    time_in_phase = sim_t - float(cache["_phase_start_t"])

    # ----- Per-lane features -----
    features: list[float] = []

    # Pre-fetch current signal state string once
    ryg = traci.trafficlight.getRedYellowGreenState(tls_id)
    ryg_len = len(ryg)

    # Lane blocks: [queue, veh_count, mean_speed(moving only), waiting_time(static only)]
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

        mean_speed_moving = (moving_speeds_sum / moving_count) if moving_count > 0 else 0.0

        features.extend([float(queue), float(veh_count), float(mean_speed_moving), float(waiting_time_static_sum)])

        # is_green_now per lane: green if ANY controlled connection from this lane is green (G/g)
        green_now = 0.0
        for idx in lane_to_sigidx.get(ln, []):
            if 0 <= idx < ryg_len and ryg[idx] in ("G", "g"):
                green_now = 1.0
                break
        is_green_flags.append(green_now)

    # Append [is_green_now] per lane
    features.extend(is_green_flags)

    # ----- Phase one-hot + time-in-phase -----
    one_hot = [0.0] * num_phases
    if 0 <= current_phase < num_phases:
        one_hot[current_phase] = 1.0
    features.extend(one_hot)
    # features.append(float(time_in_phase))

    print(f">>>>> time_in_phase: {time_in_phase}, current_phase: {current_phase}, last_phase: {cache['_last_phase']}, phase_start_t: {cache['_phase_start_t']}")

    return np.asarray(features, dtype=np.float32)
