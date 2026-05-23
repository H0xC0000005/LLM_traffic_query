def tsc_isolated_intersection_feature_vector(
    tls_id: str,
    *,
    cache: dict | None = None,
    intersection_encoding: str = "{N@335:->3; E@90^(+1,+1):T|TR; S@155^:LT|T|TRx; W@270^(+1,+1):L60|T|T}",
    jam_vehicle_spacing_m: float = 7.5,
    wait_cap_s: float = 90.0,
    ema_alpha: float = 0.65,
    detection_distance_m: float = 120.0,
    prediction_horizon_s: float = 12.0,
    base_critical_gap_s: float = 5.1,
    follow_up_gap_s: float = 2.4,
    min_running_speed_mps: float = 4.0,
    startup_queue_distance_m: float = 25.0,
    commit_distance_m: float = 35.0,
    nominal_transition_s: float = 7.0,
    nominal_startup_loss_s: float = 2.5,
    sat_headway_s: float = 2.0,
    depart_zone_m: float = 7.0,
    startup_equiv_veh_per_lane: int = 4,
    startup_cap_s: float = 4.0,
    clearance_cap_s: float = 4.0,
    **kwargs,
) -> list[float]:
    """Combine the four expert TSC feature extractors into one ML-ready numeric vector.

    Args:
        tls_id: SUMO/libsumo traffic-light id for the isolated junction.
        cache: Caller-owned mutable cache that persists across simulation steps.
        intersection_encoding: Compact geometry/lane encoding for the analyzed junction.
        Remaining parameters are expert defaults kept configurable.
    Returns:
        A 23-D list[float]:
        9 critical-lane stress + 3 permissive-left gap + 3 transition loss + 8 cycle green-loss features.
    """
    import math
    import re

    try:
        import libsumo as traci
    except ImportError:
        import traci  # type: ignore

    if cache is None:
        cache = {}

    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    def _norm_occ(x: float) -> float:
        return _clip01(x / 100.0 if x > 1.0 else x)

    def _bearing_deg(dx: float, dy: float) -> float:
        # Convert Cartesian heading to the encoding convention: North=0, clockwise positive.
        return (90.0 - math.degrees(math.atan2(dy, dx))) % 360.0

    def _polyline_heading(shape: list[tuple[float, float]], toward_end: bool) -> float:
        # Use the first non-degenerate segment near the lane start/end to infer travel heading.
        if len(shape) < 2:
            return 0.0
        pairs = zip(shape[:-1], shape[1:]) if not toward_end else zip(shape[-2::-1], shape[:0:-1])
        for a, b in pairs:
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                return _bearing_deg(dx, dy)
        return 0.0

    def _circular_diff_deg(a: float, b: float) -> float:
        return abs((a - b + 180.0) % 360.0 - 180.0)

    def _parse_intersection(enc: str) -> list[dict[str, object]]:
        # Shared parser for all expert components.
        text = enc.strip()
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1]

        legs: list[dict[str, object]] = []
        head_re = re.compile(
            r"^\s*(?P<name>[A-Za-z0-9_]+)\s*@\s*(?P<bearing>\d+)" r"(?:\s*\^\s*(?P<offset>\([^)]*\))?)?\s*$"
        )

        for raw_leg in text.split(";"):
            raw_leg = raw_leg.strip()
            if not raw_leg or ":" not in raw_leg:
                continue

            head, body = raw_leg.split(":", 1)
            m = head_re.match(head.strip())
            if not m:
                raise ValueError(f"Cannot parse leg header: {head!r}")

            offset_text = m.group("offset") or ""
            din = 0.0
            dout = 0.0
            if offset_text.startswith("(") and offset_text.endswith(")"):
                vals = [v.strip() for v in offset_text[1:-1].split(",") if v.strip()]
                if len(vals) == 1:
                    din = dout = float(vals[0])
                elif len(vals) >= 2:
                    din = float(vals[0])
                    dout = float(vals[1])

            spec = body.strip()
            outbound_only = spec.startswith("->")
            no_outbound = spec.endswith("x") and not outbound_only

            inbound_tokens: list[str] = []
            lane_defs: list[dict[str, object]] = []
            outbound_lane_count = 0

            if outbound_only:
                out_m = re.match(r"->\s*(\d+)", spec)
                outbound_lane_count = int(out_m.group(1)) if out_m else 0
            else:
                lane_text = spec[:-1].strip() if no_outbound else spec
                inbound_tokens = [tok.strip() for tok in lane_text.split("|") if tok.strip()]
                outbound_lane_count = 0 if no_outbound else len(inbound_tokens)
                for tok in inbound_tokens:
                    lane_m = re.fullmatch(r"([LTR]+)(\d+(?:\.\d+)?)?", tok)
                    if not lane_m:
                        raise ValueError(f"Unsupported lane token {tok!r} on leg {m.group('name')!r}")
                    lane_defs.append(
                        {
                            "token": tok,
                            "dirs": lane_m.group(1),
                            "dedicated_bay_m": float(lane_m.group(2)) if lane_m.group(2) else None,
                        }
                    )

            legs.append(
                {
                    "name": m.group("name"),
                    "bearing": float(m.group("bearing")),
                    "din": din,
                    "dout": dout,
                    "outbound_only": outbound_only,
                    "no_outbound": no_outbound,
                    "outbound_exists": outbound_lane_count > 0,
                    "inbound_tokens": inbound_tokens,
                    "lane_defs": lane_defs,
                    "storage_lengths": [
                        float(x) for tok in inbound_tokens for x in re.findall(r"(\d+(?:\.\d+)?)", tok)
                    ],
                }
            )
        return legs

    def _lane_moves(token: str) -> set[str]:
        return {ch for ch in re.sub(r"\d+(?:\.\d+)?", "", token) if ch in {"L", "T", "R"}}

    def _lane_number(token: str) -> float | None:
        m = re.search(r"(\d+(?:\.\d+)?)", token)
        return float(m.group(1)) if m else None

    def _match_leg(candidate_bearing: float, eligible_legs: list[dict[str, object]]) -> str:
        if not eligible_legs:
            raise ValueError("No eligible legs available for bearing matching.")
        return min(
            eligible_legs,
            key=lambda leg: _circular_diff_deg(candidate_bearing, float(leg["bearing"])),
        )[
            "name"
        ]  # type: ignore[index]

    legs = _parse_intersection(intersection_encoding)
    legs_by_name = {str(leg["name"]): leg for leg in legs}
    inbound_legs = [leg for leg in legs if leg["inbound_tokens"]]
    outbound_legs = [leg for leg in legs if leg["outbound_exists"]]

    tls_state = traci.trafficlight.getRedYellowGreenState(tls_id)
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)

    # Resolve all controlled inbound/outbound lanes directly from the live junction.
    unique_incoming_lanes: set[str] = set()
    unique_outgoing_lanes: set[str] = set()
    for link_group in controlled_links:
        for link in link_group:
            if not link:
                continue
            if link[0]:
                unique_incoming_lanes.add(link[0])
            if len(link) > 1 and link[1]:
                unique_outgoing_lanes.add(link[1])

    inbound_lane_to_leg: dict[str, str] = {}
    for lane_id in unique_incoming_lanes:
        shape = list(traci.lane.getShape(lane_id))
        inbound_heading = _polyline_heading(shape, toward_end=True)
        candidate_leg_bearing = (inbound_heading + 180.0) % 360.0
        inbound_lane_to_leg[lane_id] = _match_leg(candidate_leg_bearing, inbound_legs)

    outbound_lane_to_leg: dict[str, str] = {}
    edge_to_outbound_leg: dict[str, str] = {}
    for lane_id in unique_outgoing_lanes:
        shape = list(traci.lane.getShape(lane_id))
        outbound_bearing = _polyline_heading(shape, toward_end=False)
        leg_name = _match_leg(outbound_bearing, outbound_legs)
        outbound_lane_to_leg[lane_id] = leg_name
        try:
            edge_to_outbound_leg[traci.lane.getEdgeID(lane_id)] = leg_name
        except Exception:
            pass

    # Order inbound lanes left-to-right to match the encoding token order.
    lane_ids_by_leg: dict[str, list[str]] = {}
    for leg in inbound_legs:
        leg_name = str(leg["name"])
        leg_lanes = [ln for ln, mapped_leg in inbound_lane_to_leg.items() if mapped_leg == leg_name]
        approach_heading = (float(leg["bearing"]) + 180.0) % 360.0
        heading_rad = math.radians(90.0 - approach_heading)
        left_unit = (-math.sin(heading_rad), math.cos(heading_rad))

        stop_points = []
        for lane_id in leg_lanes:
            shape = list(traci.lane.getShape(lane_id))
            pt = shape[-1] if shape else (0.0, 0.0)
            stop_points.append((lane_id, pt))
        cx = sum(pt[0] for _, pt in stop_points) / max(1, len(stop_points))
        cy = sum(pt[1] for _, pt in stop_points) / max(1, len(stop_points))

        def _lane_left_to_right_key(lane_id: str) -> tuple[float, float]:
            shape = list(traci.lane.getShape(lane_id))
            pt = shape[-1] if shape else (cx, cy)
            lateral = (pt[0] - cx) * left_unit[0] + (pt[1] - cy) * left_unit[1]
            try:
                lane_index = float(traci.lane.getIndex(lane_id))
            except Exception:
                lane_index = 0.0
            return (lateral, lane_index)

        leg_lanes.sort(key=_lane_left_to_right_key, reverse=True)
        if len(leg_lanes) != len(leg["inbound_tokens"]):
            raise ValueError(
                f"Resolved {len(leg_lanes)} controlled inbound lanes for leg {leg_name}, "
                f"but the encoding specifies {len(leg['inbound_tokens'])}."
            )
        lane_ids_by_leg[leg_name] = leg_lanes

    # Resolve each visible approaching vehicle's next outbound leg from its live route.
    vehicle_next_leg: dict[str, str] = {}
    for leg_name, lane_ids in lane_ids_by_leg.items():
        for lane_id in lane_ids:
            for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                try:
                    route = list(traci.vehicle.getRoute(veh_id))
                    route_index = int(traci.vehicle.getRouteIndex(veh_id))
                except Exception:
                    route = []
                    route_index = -1
                if 0 <= route_index < len(route) - 1:
                    next_edge = route[route_index + 1]
                    if next_edge in edge_to_outbound_leg:
                        vehicle_next_leg[veh_id] = edge_to_outbound_leg[next_edge]

    def _inbound_heading(leg_name: str) -> float:
        return (float(legs_by_name[leg_name]["bearing"]) + 180.0) % 360.0

    def _abs_heading_diff(a: float, b: float) -> float:
        return abs((a - b + 180.0) % 360.0 - 180.0)

    def _infer_straight_dest(approach_leg: str) -> str | None:
        h = _inbound_heading(approach_leg)
        candidates = [leg for leg in outbound_legs if leg["name"] != approach_leg]
        if not candidates:
            return None
        return min(candidates, key=lambda leg: _abs_heading_diff(h, float(leg["bearing"])))["name"]  # type: ignore[index]

    def _infer_left_dest(approach_leg: str) -> str | None:
        h = _inbound_heading(approach_leg)
        candidates: list[tuple[float, str]] = []
        for leg in outbound_legs:
            if leg["name"] == approach_leg:
                continue
            left_angle = (h - float(leg["bearing"])) % 360.0
            if 15.0 <= left_angle <= 165.0:
                candidates.append((left_angle, str(leg["name"])))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    feature_vector: list[float] = []

    # ---------------- Expert 1: critical lane-group persistent storage stress ----------------
    expert1_cache = cache.setdefault(
        "tsc_isolated_intersection_feature_vector__critical_lanegroup_storage_stress",
        {},
    )
    lane_green_ratios: dict[str, list[float]] = {}
    for sig_idx, link_group in enumerate(controlled_links):
        signal_char = tls_state[sig_idx] if sig_idx < len(tls_state) else "r"
        is_green = 1.0 if signal_char in ("g", "G") else 0.0
        for link in link_group:
            if link and link[0]:
                lane_green_ratios.setdefault(link[0], []).append(is_green)

    for leg in inbound_legs:
        leg_name = str(leg["name"])
        best_lane_features = {"persist": 0.0, "storage": 0.0, "residual": 0.0, "score": -1.0}

        for lane_id, lane_def in zip(lane_ids_by_leg[leg_name], leg["lane_defs"]):
            lane_len_m = float(traci.lane.getLength(lane_id))
            vmax = max(0.1, float(traci.lane.getMaxSpeed(lane_id)))
            mean_speed = float(traci.lane.getLastStepMeanSpeed(lane_id))
            veh_n = float(traci.lane.getLastStepVehicleNumber(lane_id))
            halt_n = float(traci.lane.getLastStepHaltingNumber(lane_id))
            occ = _norm_occ(float(traci.lane.getLastStepOccupancy(lane_id)))
            waiting_time_total = float(traci.lane.getWaitingTime(lane_id))
            mean_vehicle_len = float(traci.lane.getLastStepLength(lane_id)) if veh_n > 0 else 5.5

            effective_storage_m = lane_len_m
            if lane_def["dedicated_bay_m"] is not None:
                effective_storage_m = min(effective_storage_m, float(lane_def["dedicated_bay_m"]))

            effective_vehicle_spacing = max(jam_vehicle_spacing_m, mean_vehicle_len + 2.0)
            storage_slots = max(1.0, effective_storage_m / effective_vehicle_spacing)

            queue_frac = _clip01(halt_n / storage_slots)
            wait_per_vehicle = waiting_time_total / max(veh_n, 1.0)
            wait_norm = _clip01(wait_per_vehicle / wait_cap_s)
            speed_norm = _clip01(mean_speed / vmax)

            lane_states = lane_green_ratios.get(lane_id, [])
            green_ratio = sum(lane_states) / len(lane_states) if lane_states else 0.0

            # Preserve the expert overload logic: blocked queue under red + residual queue under service.
            red_deficit = queue_frac * (1.0 - green_ratio)
            green_residual = queue_frac * green_ratio * (1.0 - speed_norm)
            instant_overload = _clip01(0.50 * red_deficit + 0.30 * green_residual + 0.20 * wait_norm)

            prev = expert1_cache.get(lane_id, {})
            prev_persist = float(prev.get("persist", instant_overload))
            prev_green = float(prev.get("green_ratio", 0.0))
            prev_queue = float(prev.get("queue_frac", 0.0))

            persist = _clip01(ema_alpha * prev_persist + (1.0 - ema_alpha) * instant_overload)
            storage = _clip01(0.70 * queue_frac + 0.30 * occ)
            carryover_hint = queue_frac if (prev_green > 0.0 and prev_queue > 0.20 and queue_frac > 0.10) else 0.0
            residual = _clip01(0.50 * green_residual + 0.35 * persist + 0.15 * carryover_hint)
            critical_score = 0.45 * persist + 0.35 * storage + 0.20 * residual

            expert1_cache[lane_id] = {"persist": persist, "queue_frac": queue_frac, "green_ratio": green_ratio}

            if critical_score > best_lane_features["score"]:
                best_lane_features = {
                    "persist": persist,
                    "storage": storage,
                    "residual": residual,
                    "score": critical_score,
                }

        feature_vector.extend(
            [
                round(best_lane_features["persist"], 6),
                round(best_lane_features["storage"], 6),
                round(best_lane_features["residual"], 6),
            ]
        )

    # ---------------- Expert 2: opposing-gap sufficiency for the two relevant left turns ----------------
    def _vehicle_eta_to_stopline(lane_id: str, veh_id: str) -> tuple[float, float]:
        lane_len = float(traci.lane.getLength(lane_id))
        lane_pos = float(traci.vehicle.getLanePosition(veh_id))
        dist_to_stop = max(0.0, lane_len - lane_pos)
        speed = float(traci.vehicle.getSpeed(veh_id))
        lane_vmax = float(traci.lane.getMaxSpeed(lane_id))
        if dist_to_stop <= 8.0:
            eta = 0.0
        else:
            eff_speed = speed if speed > 1.0 else max(min_running_speed_mps, 0.5 * lane_vmax)
            eta = dist_to_stop / max(0.1, eff_speed)
        return dist_to_stop, eta

    def _lane_arrival_times(
        lane_id: str,
        target_leg: str,
        max_dist_m: float,
        discharge_headway_s: float = 1.2,
    ) -> list[float]:
        vehs = list(traci.lane.getLastStepVehicleIDs(lane_id))
        vehs.sort(key=lambda vid: float(traci.vehicle.getLanePosition(vid)), reverse=True)
        arrivals: list[float] = []
        last_t = -1e9
        for vid in vehs:
            if vehicle_next_leg.get(vid) != target_leg:
                continue
            dist, eta = _vehicle_eta_to_stopline(lane_id, vid)
            if dist > max_dist_m:
                continue
            eta = max(eta, last_t + discharge_headway_s) if arrivals else eta
            if eta <= prediction_horizon_s:
                arrivals.append(eta)
                last_t = eta
        return arrivals

    def _left_demand_equiv(approach_leg: str, dest_leg: str, lane_indices: list[int], max_dist_m: float) -> float:
        demand = 0.0
        for idx in lane_indices:
            lane_id = lane_ids_by_leg[approach_leg][idx]
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                if vehicle_next_leg.get(vid) != dest_leg:
                    continue
                dist, eta = _vehicle_eta_to_stopline(lane_id, vid)
                if dist > max_dist_m:
                    continue
                if eta <= 2.0:
                    demand += 1.00
                elif eta <= 6.0:
                    demand += 0.80
                elif eta <= prediction_horizon_s:
                    demand += 0.55
                else:
                    demand += 0.25
        return demand

    def _left_turn_angle_deg(approach_leg: str, dest_leg: str) -> float:
        return (_inbound_heading(approach_leg) - float(legs_by_name[dest_leg]["bearing"])) % 360.0

    def _adjusted_critical_gap(approach_leg: str, dest_leg: str) -> float:
        left_angle = _left_turn_angle_deg(approach_leg, dest_leg)
        skew_factor = 1.0 + 0.15 * abs(left_angle - 90.0) / 90.0
        offset_factor = 1.0 + 0.02 * (
            abs(float(legs_by_name[approach_leg]["din"])) + abs(float(legs_by_name[dest_leg]["dout"]))
        )
        return base_critical_gap_s * skew_factor * offset_factor

    def _service_slots(arrivals: list[float], critical_gap_s: float) -> float:
        arrivals = sorted(t for t in arrivals if 0.0 <= t <= prediction_horizon_s)
        if not arrivals:
            gaps = [prediction_horizon_s]
        else:
            gaps = [arrivals[0]]
            for i in range(1, len(arrivals)):
                gaps.append(arrivals[i] - arrivals[i - 1])
            gaps.append(prediction_horizon_s - arrivals[-1])

        slots = 0.0
        for gap in gaps:
            if gap >= critical_gap_s:
                slots += 1.0 + max(0.0, (gap - critical_gap_s) / max(0.1, follow_up_gap_s))
        return slots

    gap_scores: list[float] = []
    movement_specs = [("S", _infer_left_dest("S"), "W"), ("W", _infer_left_dest("W"), "E")]
    for approach_leg, dest_leg, conflict_leg in movement_specs:
        if approach_leg not in legs_by_name or dest_leg is None or conflict_leg not in legs_by_name:
            gap_scores.append(0.5)
            continue

        approach_lane_tokens = list(legs_by_name[approach_leg]["inbound_tokens"])
        conflict_lane_tokens = list(legs_by_name[conflict_leg]["inbound_tokens"])

        left_lane_indices = [i for i, tok in enumerate(approach_lane_tokens) if "L" in _lane_moves(tok)]
        if not left_lane_indices:
            gap_scores.append(0.5)
            continue

        pocket_caps = [
            _lane_number(approach_lane_tokens[i])
            for i in left_lane_indices
            if _lane_number(approach_lane_tokens[i]) is not None
        ]
        observed_left_zone_m = min(detection_distance_m, min(pocket_caps)) if pocket_caps else detection_distance_m

        conflict_straight_dest = _infer_straight_dest(conflict_leg)
        if conflict_straight_dest is None:
            gap_scores.append(0.5)
            continue

        conflict_lane_indices = [i for i, tok in enumerate(conflict_lane_tokens) if "T" in _lane_moves(tok)]
        demand_equiv = _left_demand_equiv(approach_leg, dest_leg, left_lane_indices, observed_left_zone_m)

        conflicting_arrivals: list[float] = []
        for idx in conflict_lane_indices:
            conflicting_arrivals.extend(
                _lane_arrival_times(
                    lane_ids_by_leg[conflict_leg][idx],
                    conflict_straight_dest,
                    detection_distance_m,
                )
            )

        critical_gap_s = _adjusted_critical_gap(approach_leg, dest_leg)
        service_slots = _service_slots(conflicting_arrivals, critical_gap_s)
        balance = (service_slots - demand_equiv) / 1.25
        gap_scores.append(_clip01(1.0 / (1.0 + math.exp(-balance))))

    mean_gap_score = sum(gap_scores) / len(gap_scores) if gap_scores else 0.5
    feature_vector.extend([round(gap_scores[0], 6), round(gap_scores[1], 6), round(mean_gap_score, 6)])

    # ---------------- Expert 3: transition clearance risk + transition/startup loss ----------------
    expert3_cache = cache.setdefault(
        "tsc_isolated_intersection_feature_vector__transition_clearance_loss",
        {},
    )

    def _is_green(ch: str) -> bool:
        return ch in ("g", "G")

    def _is_yellow(ch: str) -> bool:
        return ch in ("y", "Y")

    def _compute_geometry_severity() -> float:
        if not legs:
            return 0.0

        offset_score = sum(min(1.0, (abs(float(leg["din"])) + abs(float(leg["dout"]))) / 4.0) for leg in legs) / len(
            legs
        )

        opposite_pair_skews = []
        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                diff = _circular_diff_deg(float(legs[i]["bearing"]), float(legs[j]["bearing"]))
                if diff > 135.0:
                    opposite_pair_skews.append(min(1.0, abs(180.0 - diff) / 25.0))
        skew_score = sum(opposite_pair_skews) / len(opposite_pair_skews) if opposite_pair_skews else 0.0

        one_way_score = sum(1.0 for leg in legs if bool(leg["outbound_only"]) or bool(leg["no_outbound"])) / len(legs)

        storage_terms = []
        for leg in legs:
            for storage_len in leg["storage_lengths"]:
                storage_terms.append(max(0.0, 1.0 - min(float(storage_len), 120.0) / 120.0))
        storage_score = sum(storage_terms) / len(storage_terms) if storage_terms else 0.0

        return _clip01(0.35 * skew_score + 0.25 * offset_score + 0.25 * one_way_score + 0.15 * storage_score)

    def _iter_unique_start_lanes(signal_indices: list[int]) -> list[str]:
        seen = set()
        lanes: list[str] = []
        for idx in signal_indices:
            if idx >= len(controlled_links):
                continue
            for link in controlled_links[idx]:
                if link and link[0] and link[0] not in seen:
                    seen.add(link[0])
                    lanes.append(link[0])
        return lanes

    def _estimate_max_clearance_time(signal_indices: list[int], geometry_severity: float) -> float:
        max_time = 0.0
        fallback_crossing_m = 12.0 + 8.0 * geometry_severity

        for idx in signal_indices:
            if idx >= len(controlled_links):
                continue
            for link in controlled_links[idx]:
                if not link:
                    continue
                in_lane = link[0]
                out_lane = link[1] if len(link) > 1 else ""
                via_lane = link[2] if len(link) > 2 else ""

                if via_lane:
                    try:
                        via_len = float(traci.lane.getLength(via_lane))
                    except Exception:
                        via_len = fallback_crossing_m
                    for veh_id in traci.lane.getLastStepVehicleIDs(via_lane):
                        try:
                            remaining = max(0.0, via_len - float(traci.vehicle.getLanePosition(veh_id)))
                            speed = max(1.5, float(traci.vehicle.getSpeed(veh_id)))
                            max_time = max(max_time, remaining / speed)
                        except Exception:
                            continue

                if not in_lane:
                    continue

                try:
                    lane_len = float(traci.lane.getLength(in_lane))
                except Exception:
                    continue

                for veh_id in traci.lane.getLastStepVehicleIDs(in_lane):
                    try:
                        lane_pos = float(traci.vehicle.getLanePosition(veh_id))
                        dist_to_stop = max(0.0, lane_len - lane_pos)
                        speed = max(0.0, float(traci.vehicle.getSpeed(veh_id)))
                        veh_len = max(4.5, float(traci.vehicle.getLength(veh_id)))
                    except Exception:
                        continue

                    commit_zone_m = max(8.0, 2.0 * speed + 0.5 * veh_len)
                    if dist_to_stop > min(commit_distance_m, commit_zone_m + 10.0):
                        continue

                    crossing_m = fallback_crossing_m
                    if via_lane:
                        try:
                            crossing_m = max(crossing_m, float(traci.lane.getLength(via_lane)))
                        except Exception:
                            pass
                    elif out_lane:
                        try:
                            crossing_m = max(crossing_m, 0.35 * float(traci.lane.getLength(out_lane)))
                        except Exception:
                            pass

                    traversal_speed = max(4.0, speed)
                    max_time = max(max_time, (dist_to_stop + crossing_m) / traversal_speed)

        return max_time

    geometry_severity = _compute_geometry_severity()
    now = float(traci.simulation.getTime())
    next_switch = float(traci.trafficlight.getNextSwitch(tls_id))
    remaining_in_phase_s = max(0.0, next_switch - now)

    if "prev_state" not in expert3_cache:
        expert3_cache.update(
            {
                "prev_state": tls_state,
                "yellow_start_time": None,
                "green_start_time": None,
                "last_transition_duration_s": 0.0,
                "startup_queue_present": False,
                "startup_watch": {},
                "startup_loss_s": 0.0,
                "ending_signal_indices": [],
                "clearance_risk": 0.35 + 0.30 * geometry_severity,
            }
        )

    prev_state = str(expert3_cache["prev_state"])
    if tls_state != prev_state:
        ending_signal_indices = [
            i for i, (old, new) in enumerate(zip(prev_state, tls_state)) if _is_green(old) and not _is_green(new)
        ]
        starting_signal_indices = [
            i for i, (old, new) in enumerate(zip(prev_state, tls_state)) if not _is_green(old) and _is_green(new)
        ]

        if ending_signal_indices:
            expert3_cache["yellow_start_time"] = now
            expert3_cache["ending_signal_indices"] = ending_signal_indices

        if starting_signal_indices:
            yellow_start_time = expert3_cache.get("yellow_start_time")
            if isinstance(yellow_start_time, (int, float)):
                expert3_cache["last_transition_duration_s"] = max(0.0, now - float(yellow_start_time))
            expert3_cache["green_start_time"] = now

            startup_watch: dict[str, set[str]] = {}
            startup_queue_present = False
            for lane_id in _iter_unique_start_lanes(starting_signal_indices):
                watched: set[str] = set()
                try:
                    lane_len = float(traci.lane.getLength(lane_id))
                    veh_ids = list(traci.lane.getLastStepVehicleIDs(lane_id))
                    halting = int(traci.lane.getLastStepHaltingNumber(lane_id))
                except Exception:
                    lane_len = 0.0
                    veh_ids = []
                    halting = 0

                for veh_id in veh_ids:
                    try:
                        dist_to_stop = max(0.0, lane_len - float(traci.vehicle.getLanePosition(veh_id)))
                        if dist_to_stop <= startup_queue_distance_m:
                            watched.add(veh_id)
                    except Exception:
                        continue

                startup_watch[lane_id] = watched
                if halting > 0 or watched:
                    startup_queue_present = True

            expert3_cache["startup_watch"] = startup_watch
            expert3_cache["startup_queue_present"] = startup_queue_present
            expert3_cache["startup_loss_s"] = None if startup_queue_present else 0.0

    transition_active = any(_is_yellow(ch) for ch in tls_state) or (
        expert3_cache.get("yellow_start_time") is not None
        and (
            expert3_cache.get("green_start_time") is None
            or float(expert3_cache["green_start_time"]) < float(expert3_cache["yellow_start_time"])
        )
    )

    ending_signal_indices = list(expert3_cache.get("ending_signal_indices", []))
    if transition_active and ending_signal_indices:
        max_clearance_time = _estimate_max_clearance_time(ending_signal_indices, geometry_severity)
        time_margin_s = remaining_in_phase_s - max_clearance_time
        expert3_cache["clearance_risk"] = _clip01(0.5 - time_margin_s / 6.0 + 0.15 * geometry_severity)
    else:
        baseline_risk = 0.25 + 0.30 * geometry_severity
        expert3_cache["clearance_risk"] = _clip01(0.85 * float(expert3_cache["clearance_risk"]) + 0.15 * baseline_risk)

    startup_loss_s = expert3_cache.get("startup_loss_s")
    if expert3_cache.get("startup_queue_present") and expert3_cache.get("green_start_time") is not None:
        green_elapsed_s = max(0.0, now - float(expert3_cache["green_start_time"]))
        if startup_loss_s is None:
            discharged = False
            startup_watch = expert3_cache.get("startup_watch", {})
            if isinstance(startup_watch, dict):
                for lane_id, watched_ids in startup_watch.items():
                    for veh_id in list(watched_ids):
                        try:
                            current_lane = traci.vehicle.getLaneID(veh_id)
                            speed = float(traci.vehicle.getSpeed(veh_id))
                            if current_lane != lane_id or speed > 2.0:
                                discharged = True
                                break
                        except Exception:
                            discharged = True
                            break
                    if discharged:
                        break

            if discharged:
                expert3_cache["startup_loss_s"] = max(0.0, green_elapsed_s - 0.7)
            elif green_elapsed_s >= 6.0:
                expert3_cache["startup_loss_s"] = nominal_startup_loss_s

    if transition_active and expert3_cache.get("yellow_start_time") is not None:
        transition_duration_s = max(0.0, now - float(expert3_cache["yellow_start_time"]))
    else:
        transition_duration_s = float(expert3_cache.get("last_transition_duration_s", 0.0))

    transition_loss_norm = _clip01(transition_duration_s / nominal_transition_s)
    startup_loss_s = expert3_cache.get("startup_loss_s")
    if expert3_cache.get("startup_queue_present"):
        if startup_loss_s is None and expert3_cache.get("green_start_time") is not None:
            ongoing_loss_s = max(0.0, now - float(expert3_cache["green_start_time"]) - 0.7)
            startup_loss_norm = _clip01(ongoing_loss_s / nominal_startup_loss_s)
        else:
            startup_loss_norm = _clip01(float(startup_loss_s) / nominal_startup_loss_s)
    else:
        startup_loss_norm = 0.0

    effective_green_loss = _clip01(0.55 * transition_loss_norm + 0.45 * startup_loss_norm)
    expert3_cache["prev_state"] = tls_state

    feature_vector.extend(
        [
            round(float(expert3_cache["clearance_risk"]), 6),
            round(float(effective_green_loss), 6),
            round(float(geometry_severity), 6),
        ]
    )

    # ---------------- Expert 4: latest cycle startup/clearance loss profile ----------------
    expert4_cache = cache.setdefault(
        "tsc_isolated_intersection_feature_vector__cycle_green_loss_profile",
        {},
    )
    output_legs = [str(leg["name"]) for leg in inbound_legs]

    def _ang_diff(a: float, b: float) -> float:
        d = abs((a - b) % 360.0)
        return min(d, 360.0 - d)

    def _clearance_cap_for_leg(leg_name: str) -> float:
        leg = legs_by_name.get(leg_name, {})
        bearing = float(leg.get("bearing", 0.0))
        skew = min(_ang_diff(bearing, c) for c in (0.0, 90.0, 180.0, 270.0))
        offset_mag = abs(float(leg.get("din", 0.0))) + abs(float(leg.get("dout", 0.0)))
        factor = 1.0 + 0.35 * min(1.0, skew / 30.0) + 0.20 * min(1.0, offset_mag / 2.0)
        return clearance_cap_s * factor

    def _green_inbound_lanes() -> set[str]:
        green_lanes: set[str] = set()
        for idx, link_group in enumerate(controlled_links):
            if idx >= len(tls_state) or tls_state[idx] not in ("g", "G", "s"):
                continue
            for conn in link_group:
                if conn and conn[0]:
                    green_lanes.add(conn[0])
        return green_lanes

    if tls_id not in expert4_cache:
        expert4_cache[tls_id] = {
            "prev_time": now,
            "prev_lane_ids": {},
            "prev_lane_pos": {},
            "episodes": {leg: None for leg in output_legs},
            "profile": {leg: {"startup": 0.25, "clearance": 0.25} for leg in output_legs},
        }

    state4 = expert4_cache[tls_id]
    prev_time = float(state4.get("prev_time", now))
    step_dt = max(0.1, now - prev_time) if now > prev_time else 0.1

    current_lane_ids: dict[str, set[str]] = {}
    current_lane_pos: dict[str, dict[str, float]] = {}
    leg_departures = {leg: 0 for leg in output_legs}
    leg_halts = {leg: 0 for leg in output_legs}

    for leg_name in output_legs:
        for lane_id in lane_ids_by_leg.get(leg_name, []):
            veh_ids = list(traci.lane.getLastStepVehicleIDs(lane_id))
            veh_set = set(veh_ids)
            pos_map: dict[str, float] = {}
            for vid in veh_ids:
                try:
                    pos_map[vid] = float(traci.vehicle.getLanePosition(vid))
                except Exception:
                    continue

            current_lane_ids[lane_id] = veh_set
            current_lane_pos[lane_id] = pos_map
            leg_halts[leg_name] += int(traci.lane.getLastStepHaltingNumber(lane_id))

            prev_ids = state4["prev_lane_ids"].get(lane_id, set())
            prev_pos = state4["prev_lane_pos"].get(lane_id, {})
            lane_len = float(traci.lane.getLength(lane_id))
            for vid in prev_ids - veh_set:
                if prev_pos.get(vid, -1e9) >= lane_len - depart_zone_m:
                    leg_departures[leg_name] += 1

    green_lanes = _green_inbound_lanes()
    for leg_name in output_legs:
        active_lanes = [ln for ln in lane_ids_by_leg.get(leg_name, []) if ln in green_lanes]
        served_now = len(active_lanes) > 0
        episode = state4["episodes"].get(leg_name)

        if served_now and episode is None:
            onset_queue = sum(int(traci.lane.getLastStepHaltingNumber(ln)) for ln in active_lanes)
            episode = {
                "start_time": now,
                "active_lanes": tuple(active_lanes),
                "n_lanes": max(1, len(active_lanes)),
                "onset_queue": max(0, onset_queue),
                "departures": 0,
                "startup_norm": 0.0 if onset_queue <= 0 else state4["profile"][leg_name]["startup"],
                "startup_frozen": onset_queue <= 0,
                "last_depart_time": None,
                "clearance_cap_s": _clearance_cap_for_leg(leg_name),
            }
            state4["episodes"][leg_name] = episode

        if served_now and episode is not None:
            episode["active_lanes"] = tuple(active_lanes)
            episode["n_lanes"] = max(1, len(active_lanes))
            episode["departures"] += int(leg_departures[leg_name])
            if leg_departures[leg_name] > 0:
                episode["last_depart_time"] = now

            if not episode["startup_frozen"]:
                target_departures = min(
                    int(episode["onset_queue"]),
                    int(startup_equiv_veh_per_lane * episode["n_lanes"]),
                )
                if target_departures <= 0:
                    episode["startup_norm"] = 0.0
                    episode["startup_frozen"] = True
                else:
                    startup_window_s = target_departures * sat_headway_s / max(episode["n_lanes"], 1)
                    elapsed_s = max(0.0, now - episode["start_time"])
                    effective_elapsed_s = min(elapsed_s, startup_window_s)
                    observed = min(int(episode["departures"]), target_departures)
                    ideal_equiv_s = observed * sat_headway_s / max(episode["n_lanes"], 1)
                    startup_loss_s = max(0.0, effective_elapsed_s - ideal_equiv_s)
                    episode["startup_norm"] = _clip01(startup_loss_s / max(startup_cap_s, 1e-6))
                    if observed >= target_departures or effective_elapsed_s >= startup_window_s:
                        episode["startup_frozen"] = True

            pending_clearance_norm = 0.0
            if leg_halts[leg_name] <= 0 and episode["last_depart_time"] is not None:
                pending_clearance_norm = _clip01(
                    (now - episode["last_depart_time"]) / max(episode["clearance_cap_s"], 1e-6)
                )

            state4["profile"][leg_name] = {
                "startup": _clip01(episode["startup_norm"]),
                "clearance": pending_clearance_norm,
            }

        elif (not served_now) and episode is not None:
            last_depart_time = (
                episode["last_depart_time"] if episode["last_depart_time"] is not None else episode["start_time"]
            )
            tail_gap_s = max(0.0, now - last_depart_time)
            clearance_loss_s = min(episode["clearance_cap_s"], step_dt + tail_gap_s)
            state4["profile"][leg_name] = {
                "startup": _clip01(episode["startup_norm"]),
                "clearance": _clip01(clearance_loss_s / max(episode["clearance_cap_s"], 1e-6)),
            }
            state4["episodes"][leg_name] = None

    state4["prev_time"] = now
    state4["prev_lane_ids"] = current_lane_ids
    state4["prev_lane_pos"] = current_lane_pos

    global_startup = _clip01(
        sum(float(state4["profile"][leg]["startup"]) for leg in output_legs) / max(1, len(output_legs))
    )
    global_clearance = _clip01(
        sum(float(state4["profile"][leg]["clearance"]) for leg in output_legs) / max(1, len(output_legs))
    )

    feature_vector.extend([round(global_startup, 6), round(global_clearance, 6)])
    for leg_name in output_legs:
        feature_vector.extend(
            [
                round(_clip01(float(state4["profile"][leg_name]["startup"])), 6),
                round(_clip01(float(state4["profile"][leg_name]["clearance"])), 6),
            ]
        )

    return feature_vector
