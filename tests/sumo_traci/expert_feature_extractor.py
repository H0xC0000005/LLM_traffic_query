from typing import Dict, Any, List, Tuple, Optional
import math
import libsumo as traci


def _parse_intersection_encoding_to_bearings(
    intersection_encoding: str,
) -> Dict[str, float]:
    """
    Parse the compact intersection encoding into a mapping from leg name
    (e.g. 'N','E','S','W') to nominal bearing in degrees.
    """
    s = intersection_encoding.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(";") if p.strip()]
    leg_bearings: Dict[str, float] = {}
    for part in parts:
        if ":" in part:
            left, _right = part.split(":", 1)
        else:
            left = part
        left = left.strip()
        if not left:
            continue

        # Leg name: up to first '@' or '^' or ':' (defensive)
        end_idx = len(left)
        for ch in ("@", "^", ":"):
            idx = left.find(ch)
            if idx != -1 and idx < end_idx:
                end_idx = idx
        leg_name = left[:end_idx].strip()

        # Bearing, if present after '@'
        bearing: Optional[float] = None
        at_idx = left.find("@")
        if at_idx != -1:
            end_b_idx = len(left)
            for ch in ("^", ":"):
                idx2 = left.find(ch, at_idx + 1)
                if idx2 != -1 and idx2 < end_b_idx:
                    end_b_idx = idx2
            bearing_str = left[at_idx + 1 : end_b_idx].strip()
            try:
                bearing = float(bearing_str)
            except Exception:
                bearing = None

        if leg_name and bearing is not None:
            leg_bearings[leg_name] = bearing
    return leg_bearings


def _angular_diff(a: float, b: float) -> float:
    """Minimal absolute difference between two angles in degrees."""
    d = abs(a - b) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return d


def _get_or_build_topology(
    tls_id: str,
    intersection_encoding: str,
    cache: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build or retrieve from cache the intersection topology for this tls_id:
    - lane_ids_per_approach: {N,E,S,W -> [inbound lane ids]}
    - lane_group_ids: list of approach-level lane group ids [N,E,S,W]
    - lane_group_definitions: mapping group id -> inbound lane ids
    - movement_definitions: {approach_id -> {lanes, signal_indices}}
    - lane_id_to_approach: {lane_id -> 'N'|'E'|'S'|'W'|'U'}
    - controlled_lanes_ordered: list of lanes in state-string order
    """
    topo_all = cache.setdefault("tsc_topology_by_tls", {})
    if tls_id in topo_all:
        return topo_all[tls_id]

    leg_bearings = _parse_intersection_encoding_to_bearings(intersection_encoding)

    # Controlled lanes for this tls, ordered consistently with the RYG state string.
    controlled_lanes_ordered: List[str] = list(
        traci.trafficlight.getControlledLanes(tls_id)
    )
    unique_lanes = sorted(set(controlled_lanes_ordered))

    # Map each inbound lane to an approach based on its edge heading.
    lane_id_to_approach: Dict[str, str] = {}
    inbound_lanes_per_approach: Dict[str, List[str]] = {
        "N": [],
        "E": [],
        "S": [],
        "W": [],
    }

    for lane_id in unique_lanes:
        approach = "U"
        try:
            edge_id = traci.lane.getEdgeID(lane_id)
            # Use edge heading in degrees; 0° = east, 90° = north (per SUMO docs)
            edge_angle = float(traci.edge.getAngle(edge_id))
            if math.isfinite(edge_angle) and leg_bearings:
                # Convert SUMO heading to compass with 0°=north, clockwise positive.
                compass_bearing = (90.0 - edge_angle) % 360.0
                # Assign to closest leg bearing
                best_leg = None
                best_diff = 1e9
                for leg, leg_bearing in leg_bearings.items():
                    d = _angular_diff(compass_bearing, leg_bearing)
                    if d < best_diff:
                        best_diff = d
                        best_leg = leg
                # Only accept if we found a named leg
                if best_leg in ("N", "E", "S", "W"):
                    approach = best_leg
        except Exception:
            # Fallback: try first character of lane id
            if lane_id:
                first_char = lane_id[0].upper()
                if first_char in ("N", "E", "S", "W"):
                    approach = first_char
                else:
                    approach = "U"

        lane_id_to_approach[lane_id] = approach
        if approach in inbound_lanes_per_approach:
            inbound_lanes_per_approach[approach].append(lane_id)

    # Define lane groups as approach-level groups (aggregating L/T/TR for this assembly).
    lane_group_ids: List[str] = ["N", "E", "S", "W"]
    lane_group_definitions: Dict[str, List[str]] = {
        lg: inbound_lanes_per_approach.get(lg, []) for lg in lane_group_ids
    }

    # Build movement_definitions collapsed to approach-level movements.
    links = traci.trafficlight.getControlledLinks(tls_id)
    movement_definitions: Dict[str, Dict[str, Any]] = {}
    for approach in ("N", "E", "S", "W"):
        lanes = inbound_lanes_per_approach.get(approach, [])
        signal_indices: List[int] = []
        if lanes:
            lane_set = set(lanes)
            for idx, link_tuples in enumerate(links):
                if not link_tuples:
                    continue
                for in_lane, _out_lane, _via in link_tuples:
                    if in_lane in lane_set:
                        signal_indices.append(idx)
                        break
        movement_definitions[approach] = {
            "lanes": lanes,
            "signal_indices": signal_indices,
            "stopline_positions": {},  # not used explicitly in this assembly
        }

    topo: Dict[str, Any] = {
        "lane_ids_per_approach": inbound_lanes_per_approach,
        "lane_group_ids": lane_group_ids,
        "lane_group_definitions": lane_group_definitions,
        "movement_definitions": movement_definitions,
        "lane_id_to_approach": lane_id_to_approach,
        "controlled_lanes_ordered": controlled_lanes_ordered,
    }
    topo_all[tls_id] = topo
    return topo


def _initialize_green_utilization_state(
    tls_id: str,
    cache: Dict[str, Any],
    saturation_flow_per_lane: float,
    topology: Dict[str, Any],
) -> None:
    """
    Initialize per-tls green utilization state: service phases, approach mapping,
    and accumulators. Derived from expert_01, with topology-based lane->approach
    mapping and active-program resolution.
    """
    gu_state_all = cache.setdefault("green_utilization_state", {})
    if tls_id in gu_state_all:
        return

    controlled_lanes: List[str] = list(
        topology.get("controlled_lanes_ordered")
        or traci.trafficlight.getControlledLanes(tls_id)
    )

    lane_index_to_id: Dict[int, str] = {
        idx: lid for idx, lid in enumerate(controlled_lanes)
    }
    lane_id_to_approach: Dict[str, str] = topology.get("lane_id_to_approach", {})
    lane_index_to_approach: Dict[int, str] = {
        idx: lane_id_to_approach.get(lid, "U") for idx, lid in lane_index_to_id.items()
    }

    # Determine active signal program definition.
    program_defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
    active_program_id = traci.trafficlight.getProgram(tls_id)
    program_def = None
    for logic in program_defs:
        if getattr(logic, "programID", None) == active_program_id:
            program_def = logic
            break
    if program_def is None and program_defs:
        program_def = program_defs[0]
    raw_phases = getattr(program_def, "phases", []) if program_def is not None else []

    service_phases: Dict[int, Dict[str, Any]] = {}
    for raw_idx, phase in enumerate(raw_phases):
        state = phase.state
        has_green = any(c in ("g", "G") for c in state)
        if not has_green:
            continue

        service_phase_id = raw_idx
        approach_lanes: Dict[str, List[str]] = {"N": [], "E": [], "S": [], "W": []}
        for sig_idx, sig_char in enumerate(state):
            if sig_char not in ("g", "G"):
                continue
            lane_id = lane_index_to_id.get(sig_idx)
            if lane_id is None:
                continue
            appr = lane_index_to_approach.get(sig_idx, "U")
            if appr in approach_lanes:
                approach_lanes[appr].append(lane_id)

        served_approaches = {
            appr: lanes for appr, lanes in approach_lanes.items() if lanes
        }
        if not served_approaches:
            continue

        service_phases[service_phase_id] = {
            "raw_phase_index": raw_idx,
            "served_approaches": served_approaches,
        }

    # Initialize accumulators for each service_phase_id and approach.
    accumulators: Dict[Tuple[int, str], Dict[str, float]] = {}
    for sp_id, sp_info in service_phases.items():
        for appr, lanes in sp_info["served_approaches"].items():
            key = (sp_id, appr)
            n_lanes = float(len(lanes))
            accumulators[key] = {
                "total_green_time": 0.0,
                "effective_service_time": 0.0,
                "wasted_green_time": 0.0,
                "total_discharged_vehicles": 0.0,
                "num_lanes": n_lanes,
            }

    # Previous lane vehicle counts for approximate departure estimation.
    prev_lane_counts: Dict[str, float] = {}
    for lane_id in controlled_lanes:
        try:
            prev_lane_counts[lane_id] = float(
                traci.lane.getLastStepVehicleNumber(lane_id)
            )
        except Exception:
            prev_lane_counts[lane_id] = 0.0

    gu_state_all[tls_id] = {
        "lane_index_to_id": lane_index_to_id,
        "lane_index_to_approach": lane_index_to_approach,
        "service_phases": service_phases,
        "accumulators": accumulators,
        "prev_lane_counts": prev_lane_counts,
        "saturation_flow_per_lane": float(saturation_flow_per_lane),
    }


def _update_green_utilization_and_get_matrices(
    tls_id: str,
    cache: Dict[str, Any],
    sim_step_duration: float,
    saturation_flow_per_lane: float,
    topology: Dict[str, Any],
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """
    Update green utilization accumulators for current step and emit matrices:
    (green_utilization_ratio, wasted_green_seconds, discharge_flow_ratio).
    Derived from expert_01's per_phase_green_utilization_features.
    """
    _initialize_green_utilization_state(
        tls_id, cache, saturation_flow_per_lane, topology
    )
    gu_state = cache["green_utilization_state"][tls_id]

    lane_index_to_id: Dict[int, str] = gu_state["lane_index_to_id"]
    service_phases: Dict[int, Dict[str, Any]] = gu_state["service_phases"]
    accumulators: Dict[Tuple[int, str], Dict[str, float]] = gu_state["accumulators"]
    prev_lane_counts: Dict[str, float] = gu_state["prev_lane_counts"]
    sat_flow_lane: float = float(gu_state["saturation_flow_per_lane"])

    current_raw_phase: int = int(traci.trafficlight.getPhase(tls_id))
    current_state: str = traci.trafficlight.getRedYellowGreenState(tls_id)

    if current_raw_phase in service_phases:
        sp_info = service_phases[current_raw_phase]
        served_approaches: Dict[str, List[str]] = sp_info["served_approaches"]

        lane_departures: Dict[str, float] = {}
        lane_vehicle_numbers: Dict[str, float] = {}
        for _idx, lane_id in lane_index_to_id.items():
            try:
                curr_num = float(traci.lane.getLastStepVehicleNumber(lane_id))
            except Exception:
                curr_num = 0.0
            prev_num = float(prev_lane_counts.get(lane_id, curr_num))
            departed = max(prev_num - curr_num, 0.0)
            lane_departures[lane_id] = departed
            lane_vehicle_numbers[lane_id] = curr_num
            prev_lane_counts[lane_id] = curr_num

        dt = float(sim_step_duration)
        for appr, appr_lanes in served_approaches.items():
            key = (current_raw_phase, appr)
            if key not in accumulators:
                continue

            green_lanes_for_approach: List[str] = []
            for sig_idx, sig_char in enumerate(current_state):
                if sig_char not in ("g", "G"):
                    continue
                lane_id = lane_index_to_id.get(sig_idx)
                if lane_id is None:
                    continue
                if lane_id in appr_lanes:
                    green_lanes_for_approach.append(lane_id)

            if not green_lanes_for_approach:
                continue

            total_departed = 0.0
            total_lane_vehicles = 0.0
            for lane_id in green_lanes_for_approach:
                total_departed += lane_departures.get(lane_id, 0.0)
                total_lane_vehicles += lane_vehicle_numbers.get(lane_id, 0.0)

            acc = accumulators[key]
            acc["total_green_time"] += dt
            if (total_lane_vehicles > 0.0) or (total_departed > 0.0):
                acc["effective_service_time"] += dt
            else:
                acc["wasted_green_time"] += dt
            acc["total_discharged_vehicles"] += total_departed

    approaches_order = ["N", "E", "S", "W"]
    phase_ids_sorted = sorted(service_phases.keys())

    green_utilization_matrix: List[List[float]] = []
    wasted_green_matrix: List[List[float]] = []
    discharge_flow_ratio_matrix: List[List[float]] = []

    for sp_id in phase_ids_sorted:
        row_util: List[float] = []
        row_waste: List[float] = []
        row_flow_ratio: List[float] = []
        for appr in approaches_order:
            key = (sp_id, appr)
            if key not in accumulators:
                row_util.append(0.0)
                row_waste.append(0.0)
                row_flow_ratio.append(0.0)
                continue

            acc = accumulators[key]
            total_green = acc["total_green_time"]
            eff_green = acc["effective_service_time"]
            wasted_green = acc["wasted_green_time"]
            total_discharged = acc["total_discharged_vehicles"]
            n_lanes = max(acc.get("num_lanes", 0.0), 0.0)

            if total_green > 0.0:
                util_ratio = eff_green / total_green
            else:
                util_ratio = 0.0

            if (total_green > 0.0) and (sat_flow_lane > 0.0) and (n_lanes > 0.0):
                observed_flow = total_discharged / total_green
                capacity = sat_flow_lane * n_lanes
                flow_ratio = observed_flow / capacity
            else:
                flow_ratio = 0.0

            row_util.append(float(util_ratio))
            row_waste.append(float(wasted_green))
            row_flow_ratio.append(float(flow_ratio))

        green_utilization_matrix.append(row_util)
        wasted_green_matrix.append(row_waste)
        discharge_flow_ratio_matrix.append(row_flow_ratio)

    return green_utilization_matrix, wasted_green_matrix, discharge_flow_ratio_matrix


# === Expert 02: approach headway profile (unchanged except for being reused) ===

def compute_approach_headway_profile(
    traci_module,
    lane_ids_per_approach: Dict[str, List[str]],
    prediction_horizon_s: float = 10.0,
    queue_speed_threshold_mps: float = 0.5,
    platoon_headway_threshold_s: float = 2.0,
    queue_region_m: float = 30.0,
) -> Dict[str, Any]:
    """
    Compute short-horizon arrival headway features per approach from SUMO state.
    (Expert_02 implementation, slightly wrapped to accept global traci as argument.)
    """
    approach_order = ["N", "E", "S", "W"]

    next_arrival_tta_s: List[float] = []
    mean_imminent_headway_s: List[float] = []
    std_imminent_headway_s: List[float] = []
    platoon_short_headway_count: List[int] = []
    predicted_arrival_ttas_s: List[List[float]] = []

    max_headways_for_stats = 3

    for approach_id in approach_order:
        lane_ids = lane_ids_per_approach.get(approach_id, [])
        arrival_ttas: List[float] = []

        for lane_id in lane_ids:
            try:
                veh_ids = traci_module.lane.getLastStepVehicleIDs(lane_id)
            except Exception:
                continue

            try:
                lane_length = float(traci_module.lane.getLength(lane_id))
            except Exception:
                continue

            for veh_id in veh_ids:
                try:
                    pos = float(traci_module.vehicle.getLanePosition(veh_id))
                    speed = float(traci_module.vehicle.getSpeed(veh_id))
                except Exception:
                    continue

                distance_to_stopline = lane_length - pos
                if distance_to_stopline <= 0.0:
                    continue

                if (
                    speed <= queue_speed_threshold_mps
                    and distance_to_stopline <= queue_region_m
                ):
                    continue

                if speed <= 1e-3:
                    continue

                tta = distance_to_stopline / speed
                if 0.0 < tta <= prediction_horizon_s:
                    arrival_ttas.append(tta)

        arrival_ttas.sort()
        predicted_arrival_ttas_s.append(arrival_ttas)

        if arrival_ttas:
            next_tta = arrival_ttas[0]
        else:
            next_tta = prediction_horizon_s
        next_arrival_tta_s.append(next_tta)

        if len(arrival_ttas) >= 2:
            headways: List[float] = []
            for i in range(len(arrival_ttas) - 1):
                hw = arrival_ttas[i + 1] - arrival_ttas[i]
                if hw > 0.0:
                    headways.append(hw)

            short_hw_count = sum(
                1 for hw in headways if hw <= platoon_headway_threshold_s
            )
            platoon_short_headway_count.append(short_hw_count)

            if headways:
                selected_headways = headways[:max_headways_for_stats]
                n = len(selected_headways)
                if n >= 1:
                    mean_hw = sum(selected_headways) / float(n)
                else:
                    mean_hw = prediction_horizon_s

                if n >= 2:
                    var_hw = (
                        sum((hw - mean_hw) ** 2 for hw in selected_headways)
                        / float(n)
                    )
                    std_hw = var_hw ** 0.5
                else:
                    std_hw = 0.0
            else:
                mean_hw = prediction_horizon_s
                std_hw = 0.0
        else:
            platoon_short_headway_count.append(0)
            mean_hw = prediction_horizon_s
            std_hw = 0.0

        mean_imminent_headway_s.append(mean_hw)
        std_imminent_headway_s.append(std_hw)

    return {
        "approach_order": approach_order,
        "approach_next_arrival_tta_s": next_arrival_tta_s,
        "approach_mean_imminent_headway_s": mean_imminent_headway_s,
        "approach_std_imminent_headway_s": std_imminent_headway_s,
        "approach_platoon_short_headway_count": platoon_short_headway_count,
        "approach_predicted_arrival_ttas_s": predicted_arrival_ttas_s,
    }


# === Expert 03: green-onset discharge dynamics (verbatim, with traci module) ===

def compute_green_onset_discharge_features(
    traci_module,
    tls_id: str,
    current_time: float,
    movement_definitions: Dict[str, Dict[str, Any]],
    cache: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Expert_03 implementation, slightly adapted to receive the traci module as an argument.
    """
    if config is None:
        config = {}
    queue_detection_distance = float(config.get("queue_detection_distance", 50.0))
    queue_speed_threshold = float(config.get("queue_speed_threshold", 0.1))
    headway_K = int(config.get("headway_K", 5))
    early_window = float(config.get("early_window", 10.0))

    if "prev_signal_state" not in cache:
        cache["prev_signal_state"] = traci_module.trafficlight.getRedYellowGreenState(
            tls_id
        )

    if "green_episodes" not in cache:
        cache["green_episodes"] = {}

    prev_state = cache["prev_signal_state"]
    curr_state = traci_module.trafficlight.getRedYellowGreenState(tls_id)
    cache["prev_signal_state"] = curr_state

    green_episodes = cache["green_episodes"]

    def movement_has_green(state_str: str, signal_indices: List[int]) -> bool:
        for idx in signal_indices:
            if 0 <= idx < len(state_str):
                c = state_str[idx]
                if c == "g" or c == "G":
                    return True
        return False

    # Detect green onsets and update episode structures.
    for movement_id, mdef in movement_definitions.items():
        signal_indices = mdef.get("signal_indices", [])
        if movement_id not in green_episodes:
            green_episodes[movement_id] = {
                "active": False,
                "onset_time": 0.0,
                "initial_queue_members": set(),
                "initial_queue_length": 0,
                "crossing_times": [],
            }

        episode = green_episodes[movement_id]
        was_green = movement_has_green(prev_state, signal_indices)
        is_green = movement_has_green(curr_state, signal_indices)

        if (not was_green) and is_green:
            episode["active"] = True
            episode["onset_time"] = current_time
            episode["initial_queue_members"] = set()
            episode["initial_queue_length"] = 0
            episode["crossing_times"] = []

            lanes = mdef.get("lanes", [])
            stopline_positions = mdef.get("stopline_positions", {})

            for lane_id in lanes:
                lane_len = traci_module.lane.getLength(lane_id)
                stopline_pos = stopline_positions.get(lane_id, lane_len)

                veh_ids = traci_module.lane.getLastStepVehicleIDs(lane_id)
                for veh_id in veh_ids:
                    lane_pos = traci_module.vehicle.getLanePosition(veh_id)
                    speed = traci_module.vehicle.getSpeed(veh_id)
                    dist_to_stop = max(0.0, stopline_pos - lane_pos)
                    if (
                        dist_to_stop <= queue_detection_distance
                        and speed <= queue_speed_threshold
                    ):
                            episode["initial_queue_members"].add(veh_id)

            episode["initial_queue_length"] = len(episode["initial_queue_members"])

        if was_green and (not is_green):
            episode["active"] = False

    # Track departure of initial-queue vehicles as they leave inbound lanes.
    inbound_presence_by_movement: Dict[str, set] = {}
    for movement_id, mdef in movement_definitions.items():
        lanes = mdef.get("lanes", [])
        present = set()
        for lane_id in lanes:
            for veh_id in traci_module.lane.getLastStepVehicleIDs(lane_id):
                present.add(veh_id)
        inbound_presence_by_movement[movement_id] = present

    for movement_id, episode in green_episodes.items():
        if not episode["active"]:
            continue

        current_inbound_veh_ids = inbound_presence_by_movement.get(movement_id, set())
        initial_members = episode.get("initial_queue_members", set())
        recorded_veh_ids = {vid for (vid, _) in episode.get("crossing_times", [])}

        for veh_id in initial_members:
            if veh_id not in current_inbound_veh_ids and veh_id not in recorded_veh_ids:
                episode["crossing_times"].append((veh_id, current_time))

    # Compute features.
    features_by_movement: Dict[str, Dict[str, Any]] = {}
    for movement_id, episode in green_episodes.items():
        onset_time = float(episode.get("onset_time", 0.0))
        initial_queue_length = int(episode.get("initial_queue_length", 0))
        crossing_times_raw = episode.get("crossing_times", [])

        sorted_crossings = sorted(crossing_times_raw, key=lambda x: x[1])
        times_only = [t for (_, t) in sorted_crossings]

        headways: List[float] = [0.0 for _ in range(headway_K)]
        start_delay = 0.0

        if times_only and onset_time is not None:
            start_delay = max(0.0, times_only[0] - onset_time)
            headways[0] = start_delay

            for i in range(1, min(headway_K, len(times_only))):
                h = max(0.0, times_only[i] - times_only[i - 1])
                headways[i] = h
        else:
            start_delay = 0.0

        discharged_in_window = 0
        effective_window = 0.0
        if onset_time is not None and current_time > onset_time:
            effective_window = min(early_window, current_time - onset_time)
            window_end_time = onset_time + early_window
            for t in times_only:
                if t <= window_end_time:
                    discharged_in_window += 1

        if effective_window > 0.0:
            discharge_rate_first_10s = discharged_in_window / effective_window
        else:
            discharge_rate_first_10s = 0.0

        if initial_queue_length > 0 and len(times_only) >= initial_queue_length:
            last_crossing_time = times_only[initial_queue_length - 1]
            time_to_clear_initial_queue = max(0.0, last_crossing_time - onset_time)
            queue_cleared_flag = 1
        else:
            if onset_time is not None:
                time_to_clear_initial_queue = max(0.0, current_time - onset_time)
            else:
                time_to_clear_initial_queue = 0.0
            queue_cleared_flag = 0

        if initial_queue_length > 0 and onset_time is not None:
            window_end_time_10 = onset_time + 10.0
            discharged_in_10s = 0
            for t in times_only:
                if t <= window_end_time_10:
                    discharged_in_10s += 1
            fraction_discharged_10s = min(
                1.0, discharged_in_10s / float(initial_queue_length)
            )
        else:
            fraction_discharged_10s = 0.0

        features_by_movement[movement_id] = {
            "initial_queue_length_at_green": initial_queue_length,
            "headways_first_5_vehicles_from_queue": headways,
            "start_vehicle_delay_first_from_green": start_delay,
            "discharge_rate_first_10s": discharge_rate_first_10s,
            "time_to_clear_initial_queue": time_to_clear_initial_queue,
            "fraction_of_initial_queue_discharged_in_10s": fraction_discharged_10s,
            "queue_cleared_flag": int(queue_cleared_flag),
        }

    return features_by_movement


# === Expert 04: lane-group saturation ratio ===

def compute_lane_group_saturation_short_horizon(
    lane_group_ids: List[str],
    lane_group_definitions: Dict[str, List[str]],
    detection_zone_length_m: float = 150.0,
    saturation_flow_rate_per_lane_veh_per_s: float = 0.5,
    decision_horizon_seconds: float = 20.0,
) -> Tuple[List[float], List[int], List[float]]:
    """
    Expert_04 implementation, slightly simplified to assume global libsumo as traci.
    """
    sat_ratios: List[float] = []
    demands_veh: List[int] = []
    capacities_veh: List[float] = []

    capacity_per_lane_veh = (
        saturation_flow_rate_per_lane_veh_per_s * decision_horizon_seconds
    )

    for lg_id in lane_group_ids:
        lane_ids = lane_group_definitions.get(lg_id, [])
        demand_count = 0
        for lane_id in lane_ids:
            try:
                veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                lane_length = traci.lane.getLength(lane_id)
            except Exception:
                continue

            for veh_id in veh_ids:
                try:
                    pos_on_lane = traci.vehicle.getLanePosition(veh_id)
                except Exception:
                    continue

                distance_to_stop = lane_length - pos_on_lane
                if 0.0 <= distance_to_stop <= detection_zone_length_m:
                    demand_count += 1

        num_lanes_in_group = len(lane_ids)
        capacity_veh = capacity_per_lane_veh * float(num_lanes_in_group)

        if capacity_veh > 0.0:
            sat_ratio = float(demand_count) / capacity_veh
        else:
            sat_ratio = 0.0

        sat_ratios.append(sat_ratio)
        demands_veh.append(demand_count)
        capacities_veh.append(capacity_veh)

    return sat_ratios, demands_veh, capacities_veh


def tsc_isolated_intersection_feature_vector(
    tls_id: str,
    *,
    cache: Optional[Dict[str, Any]] = None,
    intersection_encoding: str = "{N@0^(-5,+5):L|T|TR; E@90^(-1,+1):L|T|TR; S@190^(-5,+5):L|T|TR;W@270^(-1,+1):L|T|TR}",
    sim_step_duration: float = 1.0,
    saturation_flow_per_lane: float = 0.5,
    arrival_prediction_horizon_s: float = 10.0,
    arrival_queue_speed_threshold_mps: float = 0.5,
    arrival_platoon_headway_threshold_s: float = 2.0,
    arrival_queue_region_m: float = 30.0,
    discharge_config: Optional[Dict[str, Any]] = None,
    saturation_detection_zone_length_m: float = 150.0,
    saturation_flow_rate_per_lane_veh_per_s: float = 0.5,
    saturation_decision_horizon_seconds: float = 20.0,
    max_tracked_service_phases: int = 8,
    headway_K: int = 5,
    **kwargs: Any,
) -> List[float]:
    """
    Compute a combined feature vector for a single isolated SUMO intersection,
    aggregating expert-designed green utilization, arrival headway, discharge
    dynamics, and lane-group saturation features using only current simulation
    state and a small external cache.

    Parameters
    ----------
    tls_id : str
        Traffic light system identifier for the isolated intersection.
    cache : dict, optional
        Mutable dictionary preserved across calls. Used to store:
        - 'tsc_topology_by_tls' (lane and approach topology),
        - 'green_utilization_state' (expert_01 accumulators),
        - 'prev_signal_state' and 'green_episodes' (expert_03).
        If None, a new local cache is created for this call only.
    intersection_encoding : str
        Compact encoding of the intersection geometry and lane permissions.
        Defaults to the scenario instance provided in the problem statement.
    sim_step_duration : float
        SUMO step length in seconds, used for time accumulation in green
        utilization metrics.
    saturation_flow_per_lane : float
        Nominal saturation discharge flow per inbound lane (veh/s) for
        expert_01's discharge flow ratio.
    arrival_prediction_horizon_s : float
        Short-term horizon for predicting arrivals in expert_02 (seconds).
    arrival_queue_speed_threshold_mps : float
        Speed threshold (m/s) below which vehicles near the stop line are
        treated as queued and excluded from arrival headway calculations.
    arrival_platoon_headway_threshold_s : float
        Headway threshold (seconds) defining "short" headways for platoon
        intensity in expert_02.
    arrival_queue_region_m : float
        Spatial region upstream of the stop line (meters) where slow vehicles
        are treated as standing queue in expert_02.
    discharge_config : dict, optional
        Optional configuration for expert_03. If provided, overrides queue
        detection distance, queue speed threshold, headway_K, and
        early_window. The headway_K argument to this function is always
        enforced into this config to keep the feature dimension fixed.
    saturation_detection_zone_length_m : float
        Upstream distance (meters) used to count demand vehicles for
        expert_04.
    saturation_flow_rate_per_lane_veh_per_s : float
        Assumed saturation flow per lane (veh/s) for expert_04.
    saturation_decision_horizon_seconds : float
        Time horizon (seconds) over which hypothetical lane-group capacity
        is evaluated for expert_04.
    max_tracked_service_phases : int
        Maximum number of service phases to encode per expert_01 matrices.
        Excess phases are discarded; missing phases are padded with zeros.
    headway_K : int
        Number of initial discharge headways to record per approach for
        expert_03. Feature layout assumes this value (default 5).
    **kwargs : Any
        Ignored compatibility arguments; reserved for future use.

    Returns
    -------
    List[float]
        Flat feature vector of length 168 when using default
        max_tracked_service_phases=8 and headway_K=5, with the following
        layout:
        - indices [0,96): expert_01 phase-approach utilization/waste/flow
        - indices [96,112): expert_02 approach headway profile summaries
        - indices [112,156): expert_03 approach-level discharge blocks
        - indices [156,168): expert_04 approach-level saturation ratios,
          demands, and capacities.

    Cache usage
    -----------
    The function never resets the caller-provided cache. It lazily
    initializes and updates topology, green utilization accumulators, and
    green-onset episodes inside the cache under names that are specific to
    this feature family and tls_id.
    """
    if cache is None:
        cache = {}

    # Build or retrieve intersection topology (lane->approach, groups, movements).
    topology = _get_or_build_topology(tls_id, intersection_encoding, cache)

    feature_vector: List[float] = []

    # === Expert 01: green utilization per phase and approach ===
    gu_green, gu_waste, gu_flow = _update_green_utilization_and_get_matrices(
        tls_id=tls_id,
        cache=cache,
        sim_step_duration=sim_step_duration,
        saturation_flow_per_lane=saturation_flow_per_lane,
        topology=topology,
    )

    approaches_order = ["N", "E", "S", "W"]
    actual_phases = len(gu_green)
    num_tracked = max_tracked_service_phases

    # Helper to read from possibly shorter matrices with zero padding.
    def _get_mtx(mtx: List[List[float]], p_idx: int, a_idx: int) -> float:
        if 0 <= p_idx < len(mtx):
            row = mtx[p_idx]
            if 0 <= a_idx < len(row):
                return float(row[a_idx])
        return 0.0

    # 1. green_utilization_ratio
    for p in range(num_tracked):
        for a_idx, _appr in enumerate(approaches_order):
            feature_vector.append(_get_mtx(gu_green, p, a_idx))

    # 2. wasted_green_seconds
    for p in range(num_tracked):
        for a_idx, _appr in enumerate(approaches_order):
            feature_vector.append(_get_mtx(gu_waste, p, a_idx))

    # 3. discharge_flow_ratio
    for p in range(num_tracked):
        for a_idx, _appr in enumerate(approaches_order):
            feature_vector.append(_get_mtx(gu_flow, p, a_idx))

    # At this point, the expert_01 block has length 3 * num_tracked * 4.
    # With default num_tracked=8, this is 96 dimensions.

    # === Expert 02: approach arrival headway profile ===
    lane_ids_per_approach: Dict[str, List[str]] = topology["lane_ids_per_approach"]
    headway_profile = compute_approach_headway_profile(
        traci_module=traci,
        lane_ids_per_approach=lane_ids_per_approach,
        prediction_horizon_s=arrival_prediction_horizon_s,
        queue_speed_threshold_mps=arrival_queue_speed_threshold_mps,
        platoon_headway_threshold_s=arrival_platoon_headway_threshold_s,
        queue_region_m=arrival_queue_region_m,
    )

    next_tta = headway_profile["approach_next_arrival_tta_s"]
    mean_hw = headway_profile["approach_mean_imminent_headway_s"]
    std_hw = headway_profile["approach_std_imminent_headway_s"]
    platoon_cnt = headway_profile["approach_platoon_short_headway_count"]

    for v in next_tta:
        feature_vector.append(float(v))
    for v in mean_hw:
        feature_vector.append(float(v))
    for v in std_hw:
        feature_vector.append(float(v))
    for v in platoon_cnt:
        feature_vector.append(float(v))

    # Expert_02 contributes 4 * 4 = 16 dimensions.

    # === Expert 03: queue discharge dynamics at green onset ===
    current_time = float(traci.simulation.getTime())
    movement_definitions = topology["movement_definitions"]

    # Ensure discharge_config honors headway_K used for feature layout.
    if discharge_config is None:
        discharge_config = {}
    else:
        discharge_config = dict(discharge_config)  # shallow copy to avoid side effects
    discharge_config.setdefault("headway_K", headway_K)

    discharge_features = compute_green_onset_discharge_features(
        traci_module=traci,
        tls_id=tls_id,
        current_time=current_time,
        movement_definitions=movement_definitions,
        cache=cache,
        config=discharge_config,
    )

    approaches_for_discharge = ["N", "E", "S", "W"]
    # Per approach, we flatten:
    # [initial_queue_length, headways[0..K-1],
    #  start_delay, discharge_rate, time_to_clear, fraction_discharged_10s, queue_cleared_flag]
    for appr in approaches_for_discharge:
        feats = discharge_features.get(
            appr,
            {
                "initial_queue_length_at_green": 0,
                "headways_first_5_vehicles_from_queue": [0.0] * headway_K,
                "start_vehicle_delay_first_from_green": 0.0,
                "discharge_rate_first_10s": 0.0,
                "time_to_clear_initial_queue": 0.0,
                "fraction_of_initial_queue_discharged_in_10s": 0.0,
                "queue_cleared_flag": 0,
            },
        )
        feature_vector.append(float(feats["initial_queue_length_at_green"]))
        headways_list = list(feats["headways_first_5_vehicles_from_queue"])
        if len(headways_list) < headway_K:
            headways_list = headways_list + [0.0] * (headway_K - len(headways_list))
        else:
            headways_list = headways_list[:headway_K]
        for hw in headways_list:
            feature_vector.append(float(hw))
        feature_vector.append(float(feats["start_vehicle_delay_first_from_green"]))
        feature_vector.append(float(feats["discharge_rate_first_10s"]))
        feature_vector.append(float(feats["time_to_clear_initial_queue"]))
        feature_vector.append(
            float(feats["fraction_of_initial_queue_discharged_in_10s"])
        )
        feature_vector.append(float(feats["queue_cleared_flag"]))

    # With default headway_K=5, expert_03 contributes 4 * (1 + 5 + 5) = 44 dimensions.

    # === Expert 04: lane-group saturation ratio per approach ===
    lane_group_ids: List[str] = topology["lane_group_ids"]
    lane_group_definitions: Dict[str, List[str]] = topology["lane_group_definitions"]

    sat_ratios, demands_veh, capacities_veh = (
        compute_lane_group_saturation_short_horizon(
            lane_group_ids=lane_group_ids,
            lane_group_definitions=lane_group_definitions,
            detection_zone_length_m=saturation_detection_zone_length_m,
            saturation_flow_rate_per_lane_veh_per_s=saturation_flow_rate_per_lane_veh_per_s,
            decision_horizon_seconds=saturation_decision_horizon_seconds,
        )
    )

    for v in sat_ratios:
        feature_vector.append(float(v))
    for v in demands_veh:
        feature_vector.append(float(v))
    for v in capacities_veh:
        feature_vector.append(float(v))

    # With 4 lane groups [N,E,S,W], expert_04 contributes 12 dimensions.

    return feature_vector
