from __future__ import annotations

from typing import Dict, List, Any, Tuple, Optional
import math

import numpy as np

try:
    import libsumo as traci  # preferred
except ImportError:  # pragma: no cover - fallback for non-libsumo setups
    import traci  # type: ignore


def tsc_isolated_intersection_feature_vector(
    tls_id: str,
    *,
    cache: dict | None = None,
    intersection_encoding: str = "{N@0^(-5,+5):L|T|TR; E@90^(-1,+1):L|T|TR; S@190^(-5,+5):L|T|TR;W@270^(-1,+1):L|T|TR}",
    num_distance_bins: int = 3,
    max_detection_distance: float = 120.0,
    reference_green_duration: float = 10.0,
    reference_transition_time: float = 10.0,
    min_service_green_default: float = 10.0,
    max_service_phases: int = 8,
    fairness_window_length: float = 300.0,
    max_fairness_ratio: float = 5.0,
    platoon_horizons: Optional[List[Tuple[float, float]]] = None,
    **kwargs: Any,
) -> List[float]:
    """
    Compute a fixed-length feature vector for a single SUMO/libsumo-controlled
    intersection, combining lane-level spatial distributions, short-term arrival
    forecasts, phase-transition costs, and per-movement service history.

    Parameters
    ----------
    tls_id : str
        Traffic light system identifier in the running SUMO/libsumo simulation.
    cache : dict | None, optional
        External mutable cache for preserving phase and service-history state
        across timesteps and episodes. If None, a new dict is used.
    intersection_encoding : str, optional
        Compact textual encoding of the intersection geometry and leg bearings.
        Defaults to the scenario encoding specified in the task.
    num_distance_bins : int, optional
        Number of longitudinal bins per inbound lane for spatial density/speed
        features (expert_01). Default is 3.
    max_detection_distance : float, optional
        Upstream distance (m) from the stop line considered for spatial features
        (expert_01). Default is 120 m.
    reference_green_duration : float, optional
        Reference green duration (s) for normalizing phase elapsed time
        (expert_02). Default is 10 s.
    reference_transition_time : float, optional
        Reference transition time (s) for normalizing transition costs
        (expert_02). Default is 10 s.
    min_service_green_default : float, optional
        Fallback minimum green (s) per logical service phase when not specified
        in the TLS program (expert_02). Default is 10 s.
    max_service_phases : int, optional
        Maximum number of logical service phases encoded in transition-cost
        features; extra slots are zero-padded (expert_02). Default is 8.
    fairness_window_length : float, optional
        Sliding-window length (s) for service-history/fairness features
        (expert_04). Default is 300 s.
    max_fairness_ratio : float, optional
        Cap on red/green fairness ratio (expert_04). Default is 5.0.
    platoon_horizons : list[(float,float)] | None, optional
        Optional custom horizons for arrival/platoon features (expert_03). If
        None, defaults to [(0,5),(5,10),(10,20)] seconds.
    **kwargs : Any
        Ignored; present only for compatibility with generic callers.

    Returns
    -------
    list[float]
        Concatenated feature vector of length 271 for the given intersection,
        ordered by expert groups:
        - expert_01 (0..143): density, mean_speed, movement_weights
        - expert_03 (144..227): predicted_arrivals, platoon_presence, max_platoon_size
        - expert_02 (228..238): transition context and padded transition_costs
        - expert_04 (239..270): per-movement service history and fairness.

    Feature layout and expert mapping
    ---------------------------------
    See `expert_feature_mapping` in the YAML output for precise index ranges
    and semantics per expert group.

    Cache usage
    -----------
    The external `cache` dict is used under the following keys (per tls_id):
    - "phase_program_cache": static TLS program structure and raw-to-service
      phase mapping (expert_02).
    - "phase_transition_history": elapsed time in current service phase and
      last update time (expert_02).
    - "movement_service_history_cache": nested dict holding per-(leg,movement)
      sliding-window green history (expert_04).
    The cache is never cleared inside this function; callers are responsible
    for resetting it between episodes or scenario changes.
    """
    # ------------------------------------------------------------------
    # Local cache handling
    # ------------------------------------------------------------------
    if cache is None:
        cache = {}
    _ = kwargs  # explicitly ignore compatibility kwargs

    # ------------------------------------------------------------------
    # Helper functions: geometry and topology
    # ------------------------------------------------------------------
    def parse_intersection_bearings(encoding: str) -> Dict[str, float]:
        """Parse leg bearings (degrees) from the compact intersection encoding."""
        inner = encoding.strip().strip("{}").strip()
        bearings: Dict[str, float] = {}
        if not inner:
            return bearings
        for part in inner.split(";"):
            part = part.strip()
            if not part:
                continue
            header = part.split(":", 1)[0].strip()
            if not header:
                continue
            leg_char = header[0].upper()
            bearing: Optional[float] = None
            if "@" in header:
                after_at = header.split("@", 1)[1]
                end_idx = len(after_at)
                for sep in ("^", ":"):
                    idx = after_at.find(sep)
                    if idx != -1:
                        end_idx = min(end_idx, idx)
                bearing_str = after_at[:end_idx].strip()
                if bearing_str:
                    try:
                        bearing = float(bearing_str)
                    except ValueError:
                        bearing = None
            if bearing is not None:
                bearings[leg_char] = bearing
        # Ensure all canonical legs exist with some reasonable defaults
        defaults = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}
        for leg, b in defaults.items():
            bearings.setdefault(leg, b)
        return bearings

    def infer_leg_from_angle(angle: float, leg_bearings: Dict[str, float]) -> str:
        """Map a lane heading angle to the nearest leg label using circular distance."""
        a = angle % 360.0
        best_leg = None
        best_diff = None
        for leg, bearing in leg_bearings.items():
            diff = abs(a - bearing)
            diff = min(diff, 360.0 - diff)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_leg = leg
        return best_leg if best_leg is not None else "N"

    def movement_from_legs(in_leg: str, out_leg: str) -> Optional[str]:
        """Infer L/T/R movement label from inbound/outbound leg pair for right-hand traffic."""
        mapping = {
            ("N", "E"): "L",
            ("N", "S"): "T",
            ("N", "W"): "R",
            ("E", "S"): "L",
            ("E", "W"): "T",
            ("E", "N"): "R",
            ("S", "W"): "L",
            ("S", "N"): "T",
            ("S", "E"): "R",
            ("W", "N"): "L",
            ("W", "E"): "T",
            ("W", "S"): "R",
        }
        return mapping.get((in_leg, out_leg))

    def analyze_intersection_topology(
        tls_id_local: str,
        leg_bearings: Dict[str, float],
    ) -> Tuple[
        Dict[str, Dict[str, Any]],
        Dict[str, Tuple[str, str]],
        List[str],
        Dict[str, List[Tuple[str, str]]],
    ]:
        """
        Derive lane_metadata, lane_to_movement, ordered inbound lanes, and
        inbound->(out_lane, movement) pairs from SUMO/libsumo.
        """
        lane_metadata: Dict[str, Dict[str, Any]] = {}
        lane_to_movement: Dict[str, Tuple[str, str]] = {}
        inbound_lane_order: List[str] = []
        inbound_to_out_pairs: Dict[str, List[Tuple[str, str]]] = {}

        try:
            controlled_links = traci.trafficlight.getControlledLinks(tls_id_local)
        except Exception:
            # No TLS info; return empty structures
            return lane_metadata, lane_to_movement, inbound_lane_order, inbound_to_out_pairs

        inbound_lanes: set[str] = set()
        outbound_lanes: set[str] = set()
        raw_inbound_to_outlanes: Dict[str, set[str]] = {}

        for link_list in controlled_links:
            for link in link_list:
                if not link or len(link) < 2:
                    continue
                in_lane = link[0]
                out_lane = link[1]
                if not in_lane or ":" in in_lane:
                    continue  # skip internal/virtual lanes as inbound
                inbound_lanes.add(in_lane)
                if out_lane and ":" not in out_lane:
                    outbound_lanes.add(out_lane)
                    raw_inbound_to_outlanes.setdefault(in_lane, set()).add(out_lane)

        # Angle-based leg assignment
        lane_leg: Dict[str, str] = {}
        for lane_id in inbound_lanes.union(outbound_lanes):
            try:
                angle = float(traci.lane.getAngle(lane_id))
            except Exception:
                angle = 0.0
            lane_leg[lane_id] = infer_leg_from_angle(angle, leg_bearings)

        # Movement sets and inbound->(out_lane, movement) mapping
        inbound_lane_to_movs: Dict[str, set[str]] = {lid: set() for lid in inbound_lanes}
        inbound_to_out_pairs = {lid: [] for lid in inbound_lanes}

        for in_lane, out_lanes in raw_inbound_to_outlanes.items():
            in_leg = lane_leg.get(in_lane)
            if in_leg is None:
                continue
            for out_lane in out_lanes:
                out_leg = lane_leg.get(out_lane)
                if out_leg is None or out_leg == in_leg:
                    continue
                mov = movement_from_legs(in_leg, out_leg)
                if mov is None:
                    continue
                inbound_lane_to_movs[in_lane].add(mov)
                inbound_to_out_pairs[in_lane].append((out_lane, mov))

        # Primary movement per lane (used where a single movement label is needed)
        lane_primary_movement: Dict[str, str] = {}
        for lane_id in inbound_lanes:
            movs = inbound_lane_to_movs.get(lane_id, set())
            if "T" in movs:
                primary = "T"
            elif "L" in movs:
                primary = "L"
            elif "R" in movs:
                primary = "R"
            else:
                primary = "T"
            lane_primary_movement[lane_id] = primary
            if not movs:
                inbound_lane_to_movs[lane_id].add(primary)

        # Deterministic ordering of inbound lanes by (leg, lane_id)
        legs_order = ["N", "E", "S", "W"]
        def leg_rank(leg: str) -> int:
            return legs_order.index(leg) if leg in legs_order else len(legs_order)

        inbound_lane_order = sorted(
            inbound_lanes,
            key=lambda lid: (leg_rank(lane_leg.get(lid, "N")), lid),
        )

        # Build lane_metadata and lane_to_movement
        for idx, lane_id in enumerate(inbound_lane_order):
            try:
                length = float(traci.lane.getLength(lane_id))
            except Exception:
                length = 0.0
            try:
                speed_limit = float(traci.lane.getMaxSpeed(lane_id))
            except Exception:
                speed_limit = 13.9  # ~50 km/h default
            leg = lane_leg.get(lane_id, "N")
            movs_sorted = sorted(list(inbound_lane_to_movs.get(lane_id, set())))
            lane_metadata[lane_id] = {
                "length": length,
                "speed_limit": speed_limit,
                "index": idx,
                "leg": leg,
                "movements": movs_sorted,
                "is_inbound": True,
            }
            lane_to_movement[lane_id] = (leg, lane_primary_movement.get(lane_id, "T"))

        return lane_metadata, lane_to_movement, inbound_lane_order, inbound_to_out_pairs

    def build_vehicle_descriptions(
        lane_metadata_local: Dict[str, Dict[str, Any]],
        inbound_lane_order_local: List[str],
        inbound_to_out_pairs_local: Dict[str, List[Tuple[str, str]]],
    ) -> List[Dict[str, Any]]:
        """
        Build a list of vehicle dicts with lane_id, lane_pos, speed, and
        intended L/T/R movement from the SUMO/libsumo vehicle state.
        """
        vehicles: List[Dict[str, Any]] = []

        # Precompute edge IDs for outgoing lanes
        out_lanes: set[str] = set()
        for pairs in inbound_to_out_pairs_local.values():
            for out_lane, _ in pairs:
                out_lanes.add(out_lane)
        lane_to_edge: Dict[str, Optional[str]] = {}
        for out_lane in out_lanes:
            try:
                lane_to_edge[out_lane] = traci.lane.getEdgeID(out_lane)
            except Exception:
                lane_to_edge[out_lane] = None

        for lane_id in inbound_lane_order_local:
            try:
                veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            except Exception:
                continue

            pairs = inbound_to_out_pairs_local.get(lane_id, [])
            # Map next-edge -> preferred movement
            edge_to_movement: Dict[str, str] = {}
            for out_lane, mov in pairs:
                edge_id = lane_to_edge.get(out_lane)
                if edge_id is None:
                    continue
                if edge_id not in edge_to_movement:
                    edge_to_movement[edge_id] = mov
                else:
                    # Prefer straight over turns if conflicting
                    existing = edge_to_movement[edge_id]
                    if existing != "T" and mov == "T":
                        edge_to_movement[edge_id] = mov

            lane_movements = lane_metadata_local.get(lane_id, {}).get("movements", [])

            for veh_id in veh_ids:
                try:
                    lane_pos = float(traci.vehicle.getLanePosition(veh_id))
                    speed = float(traci.vehicle.getSpeed(veh_id))
                    route = list(traci.vehicle.getRoute(veh_id))
                    route_index = int(traci.vehicle.getRouteIndex(veh_id))
                except Exception:
                    continue

                movement: Optional[str] = None
                if route_index >= 0 and route_index + 1 < len(route):
                    next_edge = route[route_index + 1]
                    movement = edge_to_movement.get(next_edge)

                if movement is None:
                    # Fallback: primary movement from lane metadata
                    if "T" in lane_movements:
                        movement = "T"
                    elif "L" in lane_movements:
                        movement = "L"
                    elif "R" in lane_movements:
                        movement = "R"
                    else:
                        movement = "T"

                vehicles.append(
                    {
                        "lane_id": lane_id,
                        "lane_pos": lane_pos,
                        "speed": speed,
                        "movement": movement,
                    }
                )

        return vehicles

    # ------------------------------------------------------------------
    # Expert 01: lane-level spatial distribution & movement demand
    # ------------------------------------------------------------------
    def compute_lane_spatial_movement_profile(
        lane_metadata_local: Dict[str, Dict[str, Any]],
        vehicles_local: List[Dict[str, Any]],
        num_distance_bins_local: int = 3,
        max_detection_distance_local: float = 120.0,
        ref_max_veh_per_bin_local: float = 10.0,
        ref_max_veh_per_lane_local: float = 20.0,
    ) -> Dict[str, List[float]]:
        """
        Expert_01 implementation: compute lane-level spatial density, speed,
        and movement-demand features (slightly adapted to use local args).
        """
        # Identify inbound lanes and build index mapping
        inbound_lane_ids = [
            lid
            for lid, meta in lane_metadata_local.items()
            if meta.get("is_inbound", False)
        ]
        if not inbound_lane_ids:
            return {"density": [], "mean_speed": [], "movement_weights": []}

        inbound_lane_ids.sort(key=lambda lid: lane_metadata_local[lid]["index"])
        num_lanes = len(inbound_lane_ids)

        density_counts = np.zeros((num_lanes, num_distance_bins_local), dtype=float)
        speed_sums = np.zeros((num_lanes, num_distance_bins_local), dtype=float)
        speed_counts = np.zeros((num_lanes, num_distance_bins_local), dtype=float)

        movements_order = ["L", "T", "R"]
        movement_counts = np.zeros((num_lanes, len(movements_order)), dtype=float)

        bin_width = max_detection_distance_local / float(max(num_distance_bins_local, 1))
        lane_index_map: Dict[str, int] = {
            lid: idx for idx, lid in enumerate(inbound_lane_ids)
        }

        # Aggregate vehicles
        for v in vehicles_local:
            lane_id = v.get("lane_id")
            if lane_id not in lane_index_map:
                continue

            meta = lane_metadata_local[lane_id]
            lane_len = float(meta["length"])
            lane_pos = float(v.get("lane_pos", 0.0))
            speed = float(v.get("speed", 0.0))
            movement = v.get("movement")

            dist_to_stop = max(lane_len - lane_pos, 0.0)
            if dist_to_stop > max_detection_distance_local:
                continue

            lane_idx = lane_index_map[lane_id]
            bin_idx = int(dist_to_stop / bin_width)
            if bin_idx >= num_distance_bins_local:
                bin_idx = num_distance_bins_local - 1

            density_counts[lane_idx, bin_idx] += 1.0

            speed_limit = float(meta.get("speed_limit", 13.9))
            norm_speed = (speed / speed_limit) if speed_limit > 0.0 else 0.0

            speed_sums[lane_idx, bin_idx] += norm_speed
            speed_counts[lane_idx, bin_idx] += 1.0

            if movement in movements_order:
                m_idx = movements_order.index(movement)
                movement_counts[lane_idx, m_idx] += 1.0

        if ref_max_veh_per_bin_local <= 0.0:
            ref_max_veh_per_bin_local = 10.0
        density = density_counts / ref_max_veh_per_bin_local
        density = np.clip(density, 0.0, 1.5)

        mean_speed = np.zeros_like(speed_sums)
        nonzero_mask = speed_counts > 0.0
        mean_speed[nonzero_mask] = (
            speed_sums[nonzero_mask] / speed_counts[nonzero_mask]
        )
        mean_speed = np.clip(mean_speed, 0.0, 1.5)

        if ref_max_veh_per_lane_local <= 0.0:
            ref_max_veh_per_lane_local = 20.0
        movement_norm_counts = movement_counts / ref_max_veh_per_lane_local
        movement_norm_counts = np.clip(movement_norm_counts, 0.0, 1.0)

        row_sums = movement_counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            movement_props = np.where(
                row_sums > 0.0,
                movement_counts / (row_sums + 1e-6),
                0.0,
            )

        movement_features = np.concatenate(
            [movement_norm_counts, movement_props], axis=1
        )

        return {
            "density": density.flatten(order="C").tolist(),
            "mean_speed": mean_speed.flatten(order="C").tolist(),
            "movement_weights": movement_features.flatten(order="C").tolist(),
        }

    # ------------------------------------------------------------------
    # Expert 03: short-horizon arrivals & platoons
    # ------------------------------------------------------------------
    def compute_short_horizon_arrival_platoon_features(
        lane_metadata_local: Dict[str, Dict[str, Any]],
        vehicles_local: List[Dict[str, Any]],
        horizons_local: Optional[List[Tuple[float, float]]] = None,
        headway_threshold: float = 2.0,
        speed_threshold: float = 0.5,
        max_prediction_time: float = 20.0,
        ref_count_per_horizon: float = 5.0,
        ref_max_platoon_size: float = 10.0,
    ) -> Dict[str, List[float]]:
        """
        Expert_03 implementation: predict short-horizon arrivals and platoons
        per leg/movement.
        """
        if horizons_local is None:
            horizons_local = [(0.0, 5.0), (5.0, 10.0), (10.0, 20.0)]

        legs_order = ["N", "E", "S", "W"]
        movements_order = ["L", "T", "R"]
        num_legs = len(legs_order)
        num_movements = len(movements_order)
        num_horizons = len(horizons_local)

        arrivals_by_group: Dict[Tuple[str, str], List[float]] = {}

        for v in vehicles_local:
            lane_id = v.get("lane_id")
            meta = lane_metadata_local.get(lane_id)
            if not meta or not meta.get("is_inbound", False):
                continue

            leg = meta["leg"]
            if leg not in legs_order:
                continue

            movement = v.get("movement")
            if movement not in movements_order:
                continue

            lane_len = float(meta["length"])
            lane_pos = float(v.get("lane_pos", 0.0))
            speed = float(v.get("speed", 0.0))

            remaining_distance = max(lane_len - lane_pos, 0.0)
            if speed <= speed_threshold:
                continue

            arrival_time = (
                remaining_distance / speed if speed > 0.0 else max_prediction_time + 1.0
            )
            if arrival_time < 0.0 or arrival_time > max_prediction_time:
                continue

            key = (leg, movement)
            arrivals_by_group.setdefault(key, []).append(arrival_time)

        total_groups = num_legs * num_movements
        predicted_arrivals = [0.0] * (total_groups * num_horizons)
        platoon_presence = [0.0] * (total_groups * num_horizons)
        max_platoon_size = [0.0] * total_groups

        if ref_count_per_horizon <= 0.0:
            ref_count_per_horizon = 5.0
        if ref_max_platoon_size <= 0.0:
            ref_max_platoon_size = 10.0

        def group_index(leg: str, movement: str) -> int:
            leg_idx = legs_order.index(leg)
            mov_idx = movements_order.index(movement)
            return leg_idx * num_movements + mov_idx

        for leg in legs_order:
            for movement in movements_order:
                key = (leg, movement)
                group_idx = group_index(leg, movement)
                times = sorted(arrivals_by_group.get(key, []))

                if not times:
                    continue

                headways = [
                    times[i] - times[i - 1] for i in range(1, len(times))
                ]

                platoon_sizes: List[int] = []
                current_platoon_size = 1
                for hw in headways:
                    if hw <= headway_threshold:
                        current_platoon_size += 1
                    else:
                        platoon_sizes.append(current_platoon_size)
                        current_platoon_size = 1
                platoon_sizes.append(current_platoon_size)

                max_size = max(platoon_sizes) if platoon_sizes else 0
                norm_max_size = max_size / ref_max_platoon_size
                if norm_max_size > 1.0:
                    norm_max_size = 1.0
                max_platoon_size[group_idx] = norm_max_size

                for h_idx, (t_start, t_end) in enumerate(horizons_local):
                    count = sum(1 for t in times if t_start <= t < t_end)
                    norm_count = count / ref_count_per_horizon
                    if norm_count > 2.0:
                        norm_count = 2.0
                    arr_offset = group_idx * num_horizons + h_idx
                    predicted_arrivals[arr_offset] = norm_count

                    has_platoon = 1.0 if any(s >= 2 for s in platoon_sizes) else 0.0
                    platoon_presence[arr_offset] = has_platoon

        return {
            "predicted_arrivals": predicted_arrivals,
            "platoon_presence": platoon_presence,
            "max_platoon_size": max_platoon_size,
        }

    # ------------------------------------------------------------------
    # Expert 02: phase transition costs & timing
    # ------------------------------------------------------------------
    def compute_phase_transition_cost_features(
        mapping_raw_to_service_phase: List[Optional[int]],
        raw_phase_durations: List[float],
        current_raw_phase_index: int,
        elapsed_time_in_current_raw_phase: float,
        elapsed_time_in_current_service_phase: float,
        minimum_service_green_per_phase: List[float],
        reference_green_duration_local: float = 10.0,
        reference_transition_time_local: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Expert_02 implementation: encode phase transition context and costs.
        """
        num_raw = len(mapping_raw_to_service_phase)
        num_service = len(minimum_service_green_per_phase)

        if 0 <= current_raw_phase_index < num_raw:
            current_service_phase = mapping_raw_to_service_phase[current_raw_phase_index]
        else:
            current_service_phase = None

        is_in_transition = 1 if current_service_phase is None else 0

        if reference_green_duration_local <= 0.0:
            reference_green_duration_local = max(
                10.0, max(raw_phase_durations) if raw_phase_durations else 10.0
            )
        norm_elapsed_raw = max(elapsed_time_in_current_raw_phase, 0.0) / reference_green_duration_local
        if norm_elapsed_raw > 2.0:
            norm_elapsed_raw = 2.0

        if (
            current_service_phase is not None
            and 0 <= current_service_phase < num_service
        ):
            min_green = minimum_service_green_per_phase[current_service_phase]
            if min_green < 0.0:
                min_green = 0.0
            remaining_min_green = max(
                min_green - elapsed_time_in_current_service_phase, 0.0
            )
            if min_green > 0.0:
                norm_remaining_min_green = remaining_min_green / min_green
            else:
                norm_remaining_min_green = 0.0
        else:
            remaining_min_green = 0.0
            norm_remaining_min_green = 0.0

        def estimate_transition_time_to_service(target_service: int) -> float:
            if current_service_phase == target_service and remaining_min_green > 0.0:
                return reference_transition_time_local * 2.0

            time_sum = 0.0
            idx = current_raw_phase_index
            max_steps = num_raw * 2 if num_raw > 0 else 0

            for _ in range(max_steps):
                idx = (idx + 1) % num_raw
                mapped = mapping_raw_to_service_phase[idx]
                dur = (
                    raw_phase_durations[idx]
                    if 0 <= idx < len(raw_phase_durations)
                    else 0.0
                )

                if mapped is None:
                    time_sum += dur
                elif mapped == target_service:
                    break
                else:
                    # Other service phase: skip its green duration
                    pass
            else:
                time_sum = reference_transition_time_local

            return max(time_sum, 0.0)

        transition_costs: List[float] = []
        if reference_transition_time_local <= 0.0:
            reference_transition_time_local = 10.0

        if num_raw == 0 or num_service == 0:
            transition_costs = [0.0 for _ in range(num_service)]
        else:
            for sp in range(num_service):
                cost = estimate_transition_time_to_service(sp)
                norm_cost = cost / reference_transition_time_local
                if norm_cost > 2.0:
                    norm_cost = 2.0
                transition_costs.append(norm_cost)

        return {
            "is_in_transition_phase": is_in_transition,
            "elapsed_time_in_current_phase": norm_elapsed_raw,
            "remaining_minimum_green": norm_remaining_min_green,
            "transition_costs": transition_costs,
        }

    def build_phase_program_structures(
        tls_id_local: str,
        min_service_green_default_local: float,
    ) -> Tuple[List[Optional[int]], List[float], List[float]]:
        """
        Build mapping_raw_to_service_phase, raw_phase_durations, and
        minimum_service_green_per_phase from the current TLS program.
        """
        try:
            program_id = traci.trafficlight.getProgram(tls_id_local)
            logics = traci.trafficlight.getCompleteRedYellowGreenDefinition(
                tls_id_local
            )
        except Exception:
            return [], [], []

        chosen_logic = None
        for logic in logics:
            if getattr(logic, "programID", None) == program_id:
                chosen_logic = logic
                break
        if chosen_logic is None and logics:
            chosen_logic = logics[0]
        if chosen_logic is None:
            return [], [], []

        phases = list(getattr(chosen_logic, "phases", []))
        num_raw = len(phases)
        raw_phase_durations = [
            float(getattr(ph, "duration", 0.0)) for ph in phases
        ]
        raw_phase_min_durs = [
            float(getattr(ph, "minDur", 0.0)) for ph in phases
        ]
        states = [str(getattr(ph, "state", "")) for ph in phases]

        mapping_raw_to_service_phase: List[Optional[int]] = [None] * num_raw
        canonical_state_to_service_index: Dict[str, int] = {}
        service_phase_to_raw_indices: Dict[int, List[int]] = {}
        num_service = 0

        for i, state in enumerate(states):
            is_service = any(ch in ("g", "G", "s") for ch in state)
            if not is_service:
                continue
            canonical_state = "".join("G" if ch == "g" else ch for ch in state)
            srv_idx = canonical_state_to_service_index.get(canonical_state)
            if srv_idx is None:
                srv_idx = num_service
                canonical_state_to_service_index[canonical_state] = srv_idx
                service_phase_to_raw_indices[srv_idx] = []
                num_service += 1
            mapping_raw_to_service_phase[i] = srv_idx
            service_phase_to_raw_indices[srv_idx].append(i)

        minimum_service_green_per_phase: List[float] = []
        for srv_idx in range(num_service):
            raw_idxs = service_phase_to_raw_indices.get(srv_idx, [])
            min_green = 0.0
            if raw_idxs:
                greens = [
                    raw_phase_min_durs[j]
                    for j in raw_idxs
                    if raw_phase_min_durs[j] > 0.0
                ]
                if greens:
                    min_green = max(greens)
                else:
                    dur_candidates = [
                        raw_phase_durations[j]
                        for j in raw_idxs
                        if raw_phase_durations[j] > 0.0
                    ]
                    if dur_candidates:
                        min_green = min(dur_candidates)
            if min_green <= 0.0:
                min_green = min_service_green_default_local
            minimum_service_green_per_phase.append(min_green)

        return mapping_raw_to_service_phase, raw_phase_durations, minimum_service_green_per_phase

    # ------------------------------------------------------------------
    # Expert 04: per-movement service history & fairness
    # ------------------------------------------------------------------
    def update_service_history_and_compute_features(
        current_phase_state: Dict[str, str],
        lane_to_movement: Dict[str, Tuple[str, str]],
        history_cache: Dict[str, Any],
        time_step_length: float = 1.0,
        window_length: float = 300.0,
        max_fairness_ratio_local: float = 5.0,
    ) -> Dict[str, List[float]]:
        """
        Expert_04 implementation: update per-movement service history and
        compute temporal fairness features (slightly adapted).
        """
        if not history_cache.get("initialized", False):
            movements_set = set(lane_to_movement.values())
            legs_order = ["N", "E", "S", "W"]
            movements_order_local = ["L", "T", "R"]
            movements_order: List[Tuple[str, str]] = []
            for leg in legs_order:
                for mov in movements_order_local:
                    if (leg, mov) in movements_set:
                        movements_order.append((leg, mov))

            if not movements_order:
                history_cache["initialized"] = True
                history_cache["window_length"] = window_length
                history_cache["time_step_length"] = time_step_length
                history_cache["window_steps"] = 0
                history_cache["step_index"] = 0
                history_cache["movements_order"] = []
                history_cache["movement_to_lanes"] = {}
                history_cache["per_movement"] = {}
                return {
                    "time_since_last_green": [],
                    "cumulative_green_recent": [],
                    "cumulative_red_recent": [],
                    "fairness_ratio": [],
                }

            window_steps = int(
                round(window_length / max(time_step_length, 1e-6))
            )
            if window_steps <= 0:
                window_steps = 1

            movement_to_lanes: Dict[Tuple[str, str], List[str]] = {}
            for lane_id, mv in lane_to_movement.items():
                if mv not in movements_order:
                    continue
                movement_to_lanes.setdefault(mv, []).append(lane_id)

            per_movement: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for mv in movements_order:
                per_movement[mv] = {
                    "time_since_last_green": 0.0,
                    "green_steps": [0] * window_steps,
                }

            history_cache["initialized"] = True
            history_cache["window_length"] = window_length
            history_cache["time_step_length"] = time_step_length
            history_cache["window_steps"] = window_steps
            history_cache["step_index"] = 0
            history_cache["movements_order"] = movements_order
            history_cache["movement_to_lanes"] = movement_to_lanes
            history_cache["per_movement"] = per_movement

        window_steps = history_cache["window_steps"]
        if window_steps <= 0:
            return {
                "time_since_last_green": [],
                "cumulative_green_recent": [],
                "cumulative_red_recent": [],
                "fairness_ratio": [],
            }

        movements_order = history_cache["movements_order"]
        movement_to_lanes = history_cache["movement_to_lanes"]
        per_movement = history_cache["per_movement"]
        step_index = history_cache.get("step_index", 0)

        pointer = step_index % window_steps

        for mv in movements_order:
            lanes = movement_to_lanes.get(mv, [])
            is_green = any(
                current_phase_state.get(lane_id, "r") in ("G", "g")
                for lane_id in lanes
            )

            if is_green:
                per_movement[mv]["time_since_last_green"] = 0.0
            else:
                per_movement[mv]["time_since_last_green"] += time_step_length

            green_steps = per_movement[mv]["green_steps"]
            if len(green_steps) != window_steps:
                per_movement[mv]["green_steps"] = [0] * window_steps
                green_steps = per_movement[mv]["green_steps"]
            green_steps[pointer] = 1 if is_green else 0

        step_index += 1
        history_cache["step_index"] = step_index

        norm_time_since_last_green: List[float] = []
        norm_cum_green: List[float] = []
        norm_cum_red: List[float] = []
        fairness_ratios: List[float] = []

        effective_window = history_cache["window_length"]
        valid_steps = min(step_index, window_steps)
        total_time_in_window = valid_steps * time_step_length

        for mv in movements_order:
            record = per_movement[mv]
            time_since_last = record["time_since_last_green"]

            if effective_window > 0.0:
                norm_tslg = min(time_since_last / effective_window, 1.0)
            else:
                norm_tslg = 0.0
            norm_time_since_last_green.append(norm_tslg)

            green_steps = record["green_steps"]
            sum_green_steps = sum(green_steps[:valid_steps])
            green_time = sum_green_steps * time_step_length
            red_time = max(total_time_in_window - green_time, 0.0)

            if effective_window > 0.0:
                norm_green = min(green_time / effective_window, 1.0)
                norm_red = min(red_time / effective_window, 1.0)
            else:
                norm_green = 0.0
                norm_red = 0.0
            norm_cum_green.append(norm_green)
            norm_cum_red.append(norm_red)

            eps = 1e-3
            ratio = (red_time + eps) / (green_time + eps)
            if ratio > max_fairness_ratio_local:
                ratio = max_fairness_ratio_local
            fairness_ratios.append(ratio)

        return {
            "time_since_last_green": norm_time_since_last_green,
            "cumulative_green_recent": norm_cum_green,
            "cumulative_red_recent": norm_cum_red,
            "fairness_ratio": fairness_ratios,
        }

    # ------------------------------------------------------------------
    # Main computation
    # ------------------------------------------------------------------
    # 1) Geometry and vehicles
    leg_bearings = parse_intersection_bearings(intersection_encoding)
    lane_metadata, lane_to_movement, inbound_lane_order, inbound_to_out_pairs = (
        analyze_intersection_topology(tls_id, leg_bearings)
    )
    vehicles = build_vehicle_descriptions(
        lane_metadata, inbound_lane_order, inbound_to_out_pairs
    )

    # Expected inbound lanes from scenario encoding: 4 legs × 3 lanes per leg
    expected_inbound_lanes = 12
    expected_density_len = expected_inbound_lanes * num_distance_bins
    expected_speed_len = expected_inbound_lanes * num_distance_bins
    expected_movement_len = expected_inbound_lanes * 6  # 3 movements × 2 stats

    # 2) Expert 01 features
    lane_spatial_feats = compute_lane_spatial_movement_profile(
        lane_metadata,
        vehicles,
        num_distance_bins_local=num_distance_bins,
        max_detection_distance_local=max_detection_distance,
    )

    density_vec = np.array(
        lane_spatial_feats.get("density", []), dtype=float
    ).flatten()
    mean_speed_vec = np.array(
        lane_spatial_feats.get("mean_speed", []), dtype=float
    ).flatten()
    movement_vec = np.array(
        lane_spatial_feats.get("movement_weights", []), dtype=float
    ).flatten()

    if density_vec.size < expected_density_len:
        density_vec = np.pad(
            density_vec, (0, expected_density_len - density_vec.size)
        )
    elif density_vec.size > expected_density_len:
        density_vec = density_vec[:expected_density_len]

    if mean_speed_vec.size < expected_speed_len:
        mean_speed_vec = np.pad(
            mean_speed_vec, (0, expected_speed_len - mean_speed_vec.size)
        )
    elif mean_speed_vec.size > expected_speed_len:
        mean_speed_vec = mean_speed_vec[:expected_speed_len]

    if movement_vec.size < expected_movement_len:
        movement_vec = np.pad(
            movement_vec, (0, expected_movement_len - movement_vec.size)
        )
    elif movement_vec.size > expected_movement_len:
        movement_vec = movement_vec[:expected_movement_len]

    expert01_features = (
        density_vec.tolist() + mean_speed_vec.tolist() + movement_vec.tolist()
    )

    # 3) Expert 03 features
    if platoon_horizons is None:
        horizons_local = None
        max_pred_time = 20.0
    else:
        horizons_local = platoon_horizons
        max_pred_time = max((h[1] for h in platoon_horizons), default=20.0)

    arrival_feats = compute_short_horizon_arrival_platoon_features(
        lane_metadata,
        vehicles,
        horizons_local=horizons_local,
        max_prediction_time=max_pred_time,
    )
    expert03_features = (
        [float(x) for x in arrival_feats["predicted_arrivals"]]
        + [float(x) for x in arrival_feats["platoon_presence"]]
        + [float(x) for x in arrival_feats["max_platoon_size"]]
    )

    # 4) Expert 02 features (phase program & transition costs)
    phase_program_cache_root = cache.setdefault("phase_program_cache", {})
    phase_program_cache = phase_program_cache_root.get(tls_id)
    try:
        current_program_id = traci.trafficlight.getProgram(tls_id)
    except Exception:
        current_program_id = None

    if (
        phase_program_cache is None
        or phase_program_cache.get("program_id") != current_program_id
    ):
        (
            mapping_raw_to_service_phase,
            raw_phase_durations,
            minimum_service_green_per_phase,
        ) = build_phase_program_structures(
            tls_id, min_service_green_default
        )
        phase_program_cache = {
            "program_id": current_program_id,
            "mapping_raw_to_service_phase": mapping_raw_to_service_phase,
            "raw_phase_durations": raw_phase_durations,
            "minimum_service_green_per_phase": minimum_service_green_per_phase,
        }
        phase_program_cache_root[tls_id] = phase_program_cache
    else:
        mapping_raw_to_service_phase = phase_program_cache[
            "mapping_raw_to_service_phase"
        ]
        raw_phase_durations = phase_program_cache["raw_phase_durations"]
        minimum_service_green_per_phase = phase_program_cache[
            "minimum_service_green_per_phase"
        ]

    try:
        current_raw_phase_index = int(traci.trafficlight.getPhase(tls_id))
    except Exception:
        current_raw_phase_index = 0
    try:
        elapsed_raw = float(traci.trafficlight.getSpentDuration(tls_id))
    except Exception:
        elapsed_raw = 0.0

    # Maintain elapsed time in current service phase via cache and simulation time
    phase_hist_root = cache.setdefault("phase_transition_history", {})
    phase_hist = phase_hist_root.setdefault(tls_id, {})
    sim_time = float(traci.simulation.getTime())

    last_update_time = phase_hist.get("last_update_time", sim_time)
    dt = max(sim_time - float(last_update_time), 0.0)
    current_service_phase = (
        mapping_raw_to_service_phase[current_raw_phase_index]
        if 0 <= current_raw_phase_index < len(mapping_raw_to_service_phase)
        else None
    )

    if not phase_hist.get("initialized", False):
        elapsed_service = elapsed_raw if current_service_phase is not None else 0.0
        phase_hist["initialized"] = True
    else:
        elapsed_service = float(phase_hist.get("elapsed_in_service_phase", 0.0))
        prev_service = phase_hist.get("last_service_phase_index", None)
        if current_service_phase is not None:
            if prev_service == current_service_phase:
                elapsed_service = max(elapsed_service + dt, 0.0)
            else:
                elapsed_service = max(elapsed_raw, 0.0)
        else:
            # In transition; keep elapsed_service as last value
            elapsed_service = elapsed_service

    phase_hist["last_update_time"] = sim_time
    phase_hist["last_raw_phase_index"] = current_raw_phase_index
    phase_hist["last_service_phase_index"] = current_service_phase
    phase_hist["elapsed_in_service_phase"] = elapsed_service

    elapsed_service_for_feature = (
        elapsed_service if current_service_phase is not None else 0.0
    )

    phase_feats = compute_phase_transition_cost_features(
        mapping_raw_to_service_phase,
        raw_phase_durations,
        current_raw_phase_index,
        elapsed_raw,
        elapsed_service_for_feature,
        minimum_service_green_per_phase,
        reference_green_duration_local=reference_green_duration,
        reference_transition_time_local=reference_transition_time,
    )

    # Pad transition_costs to fixed max_service_phases length
    raw_tc = phase_feats.get("transition_costs", [])
    padded_tc: List[float] = [0.0] * max_service_phases
    for i in range(min(len(raw_tc), max_service_phases)):
        padded_tc[i] = float(raw_tc[i])

    expert02_features = [
        float(phase_feats.get("is_in_transition_phase", 0)),
        float(phase_feats.get("elapsed_time_in_current_phase", 0.0)),
        float(phase_feats.get("remaining_minimum_green", 0.0)),
    ] + padded_tc

    # 5) Expert 04 features (service history & fairness)
    # Build per-lane current phase state from TLS string and controlled lanes
    current_phase_state: Dict[str, str] = {}
    try:
        tls_state = traci.trafficlight.getRedYellowGreenState(tls_id)
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
    except Exception:
        tls_state = ""
        controlled_lanes = ()

    def state_rank(ch: str) -> int:
        if ch in ("G", "g"):
            return 3
        if ch in ("y", "Y"):
            return 2
        return 1

    for idx, lane_id in enumerate(controlled_lanes):
        if idx >= len(tls_state):
            break
        color = tls_state[idx]
        prev = current_phase_state.get(lane_id)
        if prev is None or state_rank(color) > state_rank(prev):
            current_phase_state[lane_id] = color

    # Filter to inbound lanes only
    inbound_set = set(inbound_lane_order)
    current_phase_state = {
        lid: col for lid, col in current_phase_state.items() if lid in inbound_set
    }

    # History cache per tls_id
    svc_hist_root = cache.setdefault("movement_service_history_cache", {})
    svc_history_cache = svc_hist_root.setdefault(tls_id, {})

    try:
        dt_sim = float(traci.simulation.getDeltaT())
    except Exception:
        dt_sim = 1.0

    svc_feats = update_service_history_and_compute_features(
        current_phase_state=current_phase_state,
        lane_to_movement=lane_to_movement,
        history_cache=svc_history_cache,
        time_step_length=dt_sim,
        window_length=fairness_window_length,
        max_fairness_ratio_local=max_fairness_ratio,
    )

    # Reorder expert_04 features into canonical (N,E,S,W × L,T,R) order and pad
    legs_order = ["N", "E", "S", "W"]
    movs_order = ["L", "T", "R"]
    canonical_movements: List[Tuple[str, str]] = [
        (leg, mov) for leg in legs_order for mov in movs_order
    ]

    movements_order_cached: List[Tuple[str, str]] = svc_history_cache.get(
        "movements_order", []
    )
    movement_index_map: Dict[Tuple[str, str], int] = {
        mv: idx for idx, mv in enumerate(movements_order_cached)
    }

    def reorder_and_pad(vec: List[float]) -> List[float]:
        full = [0.0] * len(canonical_movements)
        for full_idx, mv in enumerate(canonical_movements):
            src_idx = movement_index_map.get(mv)
            if src_idx is not None and src_idx < len(vec):
                full[full_idx] = float(vec[src_idx])
            else:
                full[full_idx] = 0.0
        return full

    tslg_full = reorder_and_pad(svc_feats.get("time_since_last_green", []))
    cum_green_full = reorder_and_pad(svc_feats.get("cumulative_green_recent", []))
    cum_red_full = reorder_and_pad(svc_feats.get("cumulative_red_recent", []))
    fairness_full = reorder_and_pad(svc_feats.get("fairness_ratio", []))

    expert04_features = tslg_full + cum_green_full + cum_red_full + fairness_full

    # ------------------------------------------------------------------
    # Concatenate all expert features into final vector
    # ------------------------------------------------------------------
    feature_vector = (
        expert01_features
        + expert03_features
        + expert02_features
        + expert04_features
    )

    # Ensure pure float output
    return [float(x) for x in feature_vector]
