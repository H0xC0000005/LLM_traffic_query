from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import math


import libsumo as traci


def tsc_isolated_intersection_feature_vector(
    tls_id: str,
    *,
    # --- Spillback-adjusted capacity factor (expert1) ---
    spillback_jam_frac: float = 0.80,
    recv_occ_thresh: float = 0.85,
    recv_speed_thresh_mps: float = 1.0,
    blocked_penalty: float = 0.70,
    ema_alpha: float = 0.35,
    cache: Optional[Dict[str, Any]] = None,
    # --- Permissive left-turn gap risk (expert2) ---
    approach_dist_m: float = 80.0,
    min_speed_mps_gap: float = 0.1,
    delta_max_s: float = 10.0,
    tau_s: float = 2.0,
    # --- TSP priority budget pressure (expert3) ---
    lookahead_s: float = 60.0,
    max_priority_budget_s: float = 8.0,
    per_vehicle_max_ext_s: float = 6.0,
    per_vehicle_max_eg_s: float = 6.0,
    min_speed_mps_tsp: float = 2.0,
    bus_type_prefixes: Sequence[str] = ("bus", "pt_bus"),
    emergency_type_prefixes: Sequence[str] = (
        "emergency",
        "ambulance",
        "fire",
        "police",
    ),
    emergency_budget_weight: float = 2.0,
    cap_pressure: float = 5.0,
    # --- Green-window overlap score (expert4) ---
    horizon_s: int = 90,
    dt_s: int = 1,
    approach_speed_mps: float = 13.9,
    min_moving_speed_mps: float = 0.5,
    startup_lost_time_s: float = 2.0,
) -> List[float]:
    """
    Build lane/edge-to-leg mappings from the TLS controlled links + lane bearings, then compute four
    expert features at the current timestep: (1) spillback-adjusted capacity factors, (2) permissive
    left-turn gap risk vs opposing through, (3) TSP priority budget pressure, and (4) green-window
    overlap score. Returns a concatenated numeric vector in a fixed order.

    Output order (length 27):
      [capFac_N,capFac_NE,capFac_E,capFac_S,capFac_W] +
      [risk_N,minGap_N, ..., risk_W,minGap_W] +
      [p_N,p_NE,p_E,p_S,p_W,p_ALL] +
      [GWOS_N,GWOS_NE,GWOS_E,GWOS_S,GWOS_W,GWOS_global_mean]
    """
    # --- Scene-specific leg bearings (from encoding used by experts) ---
    leg_bearing_deg: Dict[str, float] = {
        "N": 30.0,
        "NE": 50.0,
        "E": 90.0,
        "S": 220.0,
        "W": 270.0,
    }
    leg_order: List[str] = ["N", "NE", "E", "S", "W"]

    # CHANGED: helper for [0,1] normalization/clipping
    def _clip01(x: float) -> float:
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        return x

    def ang_diff_abs(a: float, b: float) -> float:
        d = (a - b) % 360.0
        return min(d, 360.0 - d)

    def _bearing_from_shape(
        shape: Sequence[Tuple[float, float]], *, toward_end: bool
    ) -> Optional[float]:
        # SUMO lane shapes are along direction of travel.
        if not shape or len(shape) < 2:
            return None
        (x1, y1), (x2, y2) = (
            (shape[-2], shape[-1]) if toward_end else (shape[0], shape[1])
        )
        dx, dy = (x2 - x1), (y2 - y1)
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return None
        # Bearing with 0°=North, 90°=East.
        return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

    def _assign_leg(
        bearing: Optional[float], *, max_err_deg: float = 90.0
    ) -> Optional[str]:
        if bearing is None:
            return None
        best_leg = min(
            leg_order, key=lambda l: ang_diff_abs(bearing, leg_bearing_deg[l])
        )
        if ang_diff_abs(bearing, leg_bearing_deg[best_leg]) > max_err_deg:
            return None
        return best_leg

    # --- Derive mappings once: inbound lanes per leg; edge->leg for BOTH inbound & outbound edges ---
    leg_in_lanes: Dict[str, List[str]] = {l: [] for l in leg_order}
    edge_leg_votes: Dict[str, Dict[str, int]] = {}  # edge_id -> {leg -> votes}
    inbound_edge_lane_counts: Dict[str, Dict[str, int]] = {
        l: {} for l in leg_order
    }  # leg -> edge -> count

    controlled_lanes = list(
        dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id))
    )
    try:
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    except Exception:
        controlled_links = None

    # (A) Use controlled LINKS to capture both incoming and outgoing edges with correct bearing direction.
    if controlled_links:
        for conn_list in controlled_links:
            if not conn_list:
                continue
            for conn in conn_list:
                # Typical conn: (inLane, outLane, viaLane) (or extended); index 0/1 are in/out lanes.
                try:
                    in_lane = conn[0]
                    out_lane = conn[1]
                except Exception:
                    continue

                # Incoming lane -> bearing toward junction (end of shape)
                if in_lane and not str(in_lane).startswith(":"):
                    try:
                        in_shape = traci.lane.getShape(in_lane)
                        in_bearing = _bearing_from_shape(in_shape, toward_end=True)
                        in_leg = _assign_leg(in_bearing)
                        if in_leg:
                            leg_in_lanes[in_leg].append(in_lane)
                            in_edge = str(traci.lane.getEdgeID(in_lane))
                            edge_leg_votes.setdefault(in_edge, {}).setdefault(in_leg, 0)
                            edge_leg_votes[in_edge][in_leg] += 1
                            inbound_edge_lane_counts[in_leg][in_edge] = (
                                inbound_edge_lane_counts[in_leg].get(in_edge, 0) + 1
                            )
                    except Exception:
                        pass

                # Outgoing lane -> bearing away from junction (start of shape)
                if out_lane and not str(out_lane).startswith(":"):
                    try:
                        out_shape = traci.lane.getShape(out_lane)
                        out_bearing = _bearing_from_shape(out_shape, toward_end=False)
                        out_leg = _assign_leg(out_bearing)
                        if out_leg:
                            out_edge = str(traci.lane.getEdgeID(out_lane))
                            edge_leg_votes.setdefault(out_edge, {}).setdefault(
                                out_leg, 0
                            )
                            edge_leg_votes[out_edge][out_leg] += 1
                    except Exception:
                        pass

    # (B) Fallback: if links were unavailable/partial, use controlled lanes as incoming-only sources.
    for ln in controlled_lanes:
        if not ln or str(ln).startswith(":"):
            continue
        # Avoid duplicate lane entries in leg_in_lanes if we already added via controlled_links.
        if any(ln in leg_in_lanes[l] for l in leg_order):
            continue
        try:
            shp = traci.lane.getShape(ln)
            b = _bearing_from_shape(shp, toward_end=True)
            leg = _assign_leg(b)
            if not leg:
                continue
            leg_in_lanes[leg].append(ln)
            e = str(traci.lane.getEdgeID(ln))
            edge_leg_votes.setdefault(e, {}).setdefault(leg, 0)
            edge_leg_votes[e][leg] += 1
            inbound_edge_lane_counts[leg][e] = (
                inbound_edge_lane_counts[leg].get(e, 0) + 1
            )
        except Exception:
            continue

    # Canonical edge->leg mapping (votes argmax)
    edge_to_leg: Dict[str, str] = {}
    for edge_id, votes in edge_leg_votes.items():
        if not votes:
            continue
        edge_to_leg[edge_id] = max(votes.items(), key=lambda kv: kv[1])[0]

    # Representative inbound edge per leg for spillback module (edge with most inbound lanes in this leg)
    inbound_edges_by_leg: Dict[str, Optional[str]] = {}
    for leg in leg_order:
        if inbound_edge_lane_counts[leg]:
            inbound_edges_by_leg[leg] = max(
                inbound_edge_lane_counts[leg].items(), key=lambda kv: kv[1]
            )[0]
        else:
            inbound_edges_by_leg[leg] = None

    # ------------------------
    # Expert1: spillback_adjusted_capacity_factor -> vector (N,NE,E,S,W)
    # ------------------------
    leg_factors: Dict[str, float] = {}

    for leg in leg_order:
        edge_id = inbound_edges_by_leg.get(leg)
        if not edge_id:
            leg_factors[leg] = 0.0
            continue

        # lane_ids = list(traci.edge.getLanes(edge_id))
        try:
            n_lanes = int(traci.edge.getLaneNumber(edge_id))
            lane_ids = [f"{edge_id}_{i}" for i in range(n_lanes)]
        except Exception:
            # Fallback: collect controlled lanes that belong to this edge.
            lane_ids = []
            for ln in controlled_lanes:
                if not ln or str(ln).startswith(":"):
                    continue
                try:
                    if str(traci.lane.getEdgeID(ln)) == edge_id:
                        lane_ids.append(str(ln))
                except Exception:
                    continue
        if not lane_ids:
            leg_factors[leg] = 0.0
            continue

        lane_factors: List[float] = []
        for ln in lane_ids:
            # 1) Spillback detection (jam length vs lane length)
            lane_len = float(traci.lane.getLength(ln))
            # jam_len = float(traci.lane.getJamLengthMeters(ln))
            # spillback = (lane_len > 0.0) and (jam_len >= spillback_jam_frac * lane_len)
            # CHANGED: TraCI may not expose getJamLengthMeters(); approximate "jam length"
            # using halting vehicles and a per-vehicle standstill gap (plus vehicle length).
            try:
                jam_len = float(traci.lane.getJamLengthMeters(ln))  # type: ignore[attr-defined]
            except AttributeError:
                # Approximation: jam_len ≈ (#halting vehicles) * (vehicle_length + standstill_gap)
                n_halt = float(traci.lane.getLastStepHaltingNumber(ln))
                avg_veh_len = 5.0  # meters (typical passenger car)
                standstill_gap = 2.5  # meters
                jam_len = n_halt * (avg_veh_len + standstill_gap)
            spillback = (lane_len > 0.0) and (jam_len >= spillback_jam_frac * lane_len)
            if spillback:
                lane_factors.append(0.0)
                continue

            # 2) Receiving blockage detection via outgoing lanes connected from this inbound lane
            blocked_recv = False
            try:
                links = traci.lane.getLinks(
                    ln
                )  # tuples; first element is outgoing lane id
            except Exception:
                links = []

            out_lanes: List[str] = []
            for link in links:
                if not link:
                    continue
                out_ln = link[0]
                if not out_ln or str(out_ln).startswith(":"):
                    continue
                out_lanes.append(out_ln)

            for out_ln in out_lanes:
                occ_pct = float(traci.lane.getLastStepOccupancy(out_ln))
                occ = occ_pct / 100.0 if occ_pct > 1.0 else occ_pct
                spd = float(traci.lane.getLastStepMeanSpeed(out_ln))
                if (occ >= recv_occ_thresh) and (spd <= recv_speed_thresh_mps):
                    blocked_recv = True
                    break

            lane_factor = 1.0 if not blocked_recv else max(0.0, 1.0 - blocked_penalty)
            lane_factors.append(lane_factor)

        leg_factor = sum(lane_factors) / max(1, len(lane_factors))
        leg_factors[leg] = float(max(0.0, min(1.0, leg_factor)))

    # Optional EMA smoothing (user-managed cache)
    if cache is not None:
        prev = cache.get("prev_leg_factors")
        if isinstance(prev, dict) and prev:
            smoothed: Dict[str, float] = {}
            for leg in leg_order:
                p = float(prev.get(leg, leg_factors[leg]))
                c = leg_factors[leg]
                smoothed[leg] = float(
                    max(0.0, min(1.0, ema_alpha * c + (1.0 - ema_alpha) * p))
                )
            leg_factors = smoothed
        cache["prev_leg_factors"] = dict(leg_factors)

    spillback_vec = [leg_factors[l] for l in leg_order]

    # ------------------------
    # Expert2: permissive_left_turn_gap_risk -> 10D vector
    # ------------------------
    def signed_diff_deg(a: float, b: float) -> float:
        d = (a - b) % 360.0
        return d - 360.0 if d > 180.0 else d

    def classify_turn(in_leg: str, out_leg: str) -> str:
        incoming_heading = (leg_bearing_deg[in_leg] + 180.0) % 360.0  # toward junction
        outgoing_heading = leg_bearing_deg[out_leg] % 360.0  # away from junction
        sd = signed_diff_deg(outgoing_heading, incoming_heading)
        if -45.0 <= sd <= 45.0:
            return "T"
        if 45.0 < sd < 135.0:
            return "R"
        if -135.0 < sd < -45.0:
            return "L"
        return "U"

    def is_green(sig_state_char: str) -> bool:
        return sig_state_char in ("G", "g")

    # Opposing-leg mapping (closest to 180° apart; accept if within 70°)
    opposite_leg: Dict[str, str] = {}
    for leg in leg_order:
        best_other, best_score = None, 1e9
        for other in leg_order:
            if other == leg:
                continue
            score = abs(
                ang_diff_abs(leg_bearing_deg[leg], leg_bearing_deg[other]) - 180.0
            )
            if score < best_score:
                best_other, best_score = other, score
        if best_other is not None and best_score <= 70.0:
            opposite_leg[leg] = best_other

    cand_vehicle_ids: set[str] = set()
    for ln in set(controlled_lanes):
        try:
            for vid in traci.lane.getLastStepVehicleIDs(ln):
                cand_vehicle_ids.add(str(vid))
        except Exception:
            continue

    left_by_leg: Dict[str, List[Tuple[float, str]]] = {leg: [] for leg in leg_order}
    thru_by_leg: Dict[str, List[Tuple[float, str]]] = {leg: [] for leg in leg_order}

    for vid in cand_vehicle_ids:
        next_tls = traci.vehicle.getNextTLS(vid)
        if not next_tls:
            continue
        tls0, _, dist_m, sig_state = next_tls[0]
        if tls0 != tls_id or float(dist_m) > approach_dist_m:
            continue

        in_edge = str(traci.vehicle.getRoadID(vid))
        if in_edge not in edge_to_leg:
            continue
        in_leg = edge_to_leg[in_edge]

        route = traci.vehicle.getRoute(vid)
        r_idx = traci.vehicle.getRouteIndex(vid)
        if r_idx is None or r_idx < 0 or r_idx >= len(route):
            continue
        out_edge = route[r_idx + 1] if (r_idx + 1) < len(route) else None
        out_leg = edge_to_leg.get(str(out_edge), None) if out_edge is not None else None
        if out_leg is None:
            continue

        turn = classify_turn(in_leg, out_leg)
        speed = float(traci.vehicle.getSpeed(vid))
        tti_s = float(dist_m) / max(speed, float(min_speed_mps_gap))

        if turn == "L":
            left_by_leg[in_leg].append((tti_s, str(sig_state)))
        elif turn == "T":
            thru_by_leg[in_leg].append((tti_s, str(sig_state)))

    gap_vec: List[float] = []
    no_pair_min_gap = approach_dist_m / min_speed_mps_gap

    for leg in leg_order:
        opp = opposite_leg.get(leg, None)
        if opp is None:
            # gap_vec.extend([0.0, float(no_pair_min_gap)])
            # CHANGED: normalize minGap to [0,1]
            gap_vec.extend([0.0, 1.0])
            continue

        risk = 0.0
        min_gap = float(no_pair_min_gap)
        left_list = [tti for (tti, st) in left_by_leg[leg] if is_green(st)]
        thru_list = [tti for (tti, st) in thru_by_leg[opp] if is_green(st)]

        for t_left in left_list:
            for t_thru in thru_list:
                gap = abs(t_left - t_thru)
                if gap <= delta_max_s:
                    risk += math.exp(-gap / max(tau_s, 1e-6))
                    if gap < min_gap:
                        min_gap = gap

        # gap_vec.extend([float(risk), float(min_gap)])
        # CHANGED: normalize expert2 outputs to [0,1]
        # risk is unbounded (sum of exp terms); map with a soft saturation.
        risk_norm = 0.0 if risk <= 0.0 else (risk / (risk + 1.0))
        min_gap_norm = _clip01(
            min_gap / no_pair_min_gap
        )  # 0..1 (0=very small gap, 1=no conflict)
        gap_vec.extend([float(risk_norm), float(min_gap_norm)])

    # ------------------------
    # Expert3: tsp_priority_budget_pressure -> 6D vector
    # ------------------------
    lane_count_by_leg: Dict[str, int] = {"N": 1, "NE": 2, "E": 3, "S": 3, "W": 3}

    def _startswith_any(s: str, prefixes: Sequence[str]) -> bool:
        sl = s.lower()
        return any(sl.startswith(p.lower()) for p in prefixes)

    def _get_current_logic(tl_id: str) -> Any:
        try:
            prog_id = traci.trafficlight.getProgram(tl_id)
        except Exception:
            prog_id = None
        try:
            logics = traci.trafficlight.getAllProgramLogics(tl_id)
        except Exception:
            logics = []
        if prog_id is not None:
            for lg in logics:
                if getattr(lg, "programID", None) == prog_id:
                    return lg
        return logics[0] if logics else None

    def _time_to_green_seconds(
        logic: Any, curr_phase_idx: int, rem_curr_phase_s: float, tls_index: int
    ) -> float:
        if logic is None or not getattr(logic, "phases", None):
            return float("inf")
        phases = logic.phases
        n = len(phases)

        curr_state = getattr(phases[curr_phase_idx], "state", "")
        if 0 <= tls_index < len(curr_state) and curr_state[tls_index] in ("G", "g"):
            return 0.0

        t = max(0.0, rem_curr_phase_s)
        for k in range(1, 2 * n + 1):
            idx = (curr_phase_idx + k) % n
            st = getattr(phases[idx], "state", "")
            if 0 <= tls_index < len(st) and st[tls_index] in ("G", "g"):
                return t
            dur = getattr(phases[idx], "duration", 0.0)
            t += float(dur) if dur is not None else 0.0
        return float("inf")

    now = float(traci.simulation.getTime())
    next_switch = float(traci.trafficlight.getNextSwitch(tls_id))
    rem_curr_phase_s = max(0.0, next_switch - now)
    curr_phase_idx = int(traci.trafficlight.getPhase(tls_id))
    logic = _get_current_logic(tls_id)

    veh_ids: set[str] = set()
    for lane_id in traci.trafficlight.getControlledLanes(tls_id):
        try:
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                veh_ids.add(str(vid))
        except Exception:
            continue

    bus_need_by_leg: Dict[str, float] = {leg: 0.0 for leg in leg_order}
    emergency_need_total = 0.0

    for vid in veh_ids:
        next_tls = traci.vehicle.getNextTLS(vid)
        if not next_tls:
            continue

        entry = None
        for e in next_tls:
            if e[0] == tls_id:
                entry = e
                break
        if entry is None:
            continue

        _, tls_index, dist_m, sig_state = entry
        dist_m = float(dist_m)
        if dist_m < 0:
            continue

        speed = max(float(traci.vehicle.getSpeed(vid)), float(min_speed_mps_tsp))
        eta_s = dist_m / speed
        if eta_s > lookahead_s:
            continue

        type_id = str(traci.vehicle.getTypeID(vid))
        is_emergency = _startswith_any(type_id, emergency_type_prefixes)
        is_bus = _startswith_any(type_id, bus_type_prefixes)
        if not (is_emergency or is_bus):
            continue

        edge_id = str(traci.vehicle.getRoadID(vid))
        leg = edge_to_leg.get(edge_id, "UNK")

        lane_cnt = lane_count_by_leg.get(leg, 2)
        lane_factor = 1.0 + 0.2 * max(
            0, 3 - lane_cnt
        )  # N(1)->1.4, NE(2)->1.2, others(3)->1.0

        sig_state = str(sig_state) if sig_state is not None else ""
        if sig_state in ("G", "g"):
            ext_need = max(0.0, eta_s - rem_curr_phase_s)
            need_s = min(ext_need, per_vehicle_max_ext_s)
        else:
            t_green = _time_to_green_seconds(
                logic, curr_phase_idx, rem_curr_phase_s, int(tls_index)
            )
            if t_green == float("inf"):
                continue
            eg_need = max(0.0, t_green - eta_s)
            need_s = min(eg_need, per_vehicle_max_eg_s)

        need_s *= lane_factor
        if is_emergency:
            emergency_need_total += need_s
        else:
            if leg in bus_need_by_leg:
                bus_need_by_leg[leg] += need_s

    available_budget = max(
        0.0, max_priority_budget_s - emergency_budget_weight * emergency_need_total
    )

    def _pressure(need: float) -> float:
        if need <= 0.0:
            return 0.0
        if available_budget <= 1e-6:
            return float(cap_pressure)
        return float(min(cap_pressure, need / available_budget))

    # tsp_vec = [_pressure(bus_need_by_leg[leg]) for leg in leg_order]
    # tsp_vec.append(_pressure(sum(bus_need_by_leg.values())))  # p_ALL
    # CHANGED: normalize expert3 outputs to [0,1] by dividing by cap_pressure
    _tsp_den = cap_pressure if cap_pressure > 0.0 else 1.0
    tsp_vec = [_clip01(_pressure(bus_need_by_leg[leg]) / _tsp_den) for leg in leg_order]
    tsp_vec.append(
        _clip01(_pressure(sum(bus_need_by_leg.values())) / _tsp_den)
    )  # p_ALL

    # ------------------------
    # Expert4: green_window_overlap_score -> 6D vector
    # ------------------------
    # Build per-second phase state strings across horizon
    t_now = float(traci.simulation.getTime())
    next_switch = float(traci.trafficlight.getNextSwitch(tls_id))
    time_left_cur = max(0.0, next_switch - t_now)
    cur_phase_idx = int(traci.trafficlight.getPhase(tls_id))

    logic2 = _get_current_logic(tls_id)
    phases = getattr(logic2, "phases", None) if logic2 is not None else None
    if not phases:
        gwos_vec = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        steps = max(1, int(horizon_s // dt_s))
        state_t: List[str] = []

        idx = cur_phase_idx
        rem = time_left_cur
        if idx < 0 or idx >= len(phases):
            idx = 0
            rem = float(getattr(phases[idx], "duration", 0.0))

        def _append_state(phase_state: str, dur_s: float) -> None:
            k = int(max(0.0, dur_s) // dt_s)
            for _ in range(k):
                if len(state_t) >= steps:
                    return
                state_t.append(phase_state)

        _append_state(getattr(phases[idx], "state"), rem)
        idx = (idx + 1) % len(phases)
        while len(state_t) < steps:
            dur = float(getattr(phases[idx], "duration", 0.0))
            _append_state(getattr(phases[idx], "state"), dur)
            idx = (idx + 1) % len(phases)

        if not controlled_links or not state_t:
            gwos_vec = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            # inLane -> signal indices affecting it
            inlane_to_sigidx: Dict[str, List[int]] = {}
            for sig_i, conn_list in enumerate(controlled_links):
                if not conn_list:
                    continue
                for conn in conn_list:
                    try:
                        in_lane = conn[0]
                    except Exception:
                        continue
                    if in_lane:
                        inlane_to_sigidx.setdefault(str(in_lane), []).append(sig_i)

            num_sig = max(1, len(state_t[0]))
            green_by_sigidx = [[False] * steps for _ in range(num_sig)]
            for t in range(steps):
                st = state_t[t]
                m = min(num_sig, len(st))
                for i in range(m):
                    if st[i] in ("G", "g"):
                        green_by_sigidx[i][t] = True

            def _inlane_green_series(in_lane: str) -> List[bool]:
                idxs = inlane_to_sigidx.get(in_lane, [])
                if not idxs:
                    return [False] * steps
                series = [False] * steps
                for si in idxs:
                    if si < 0 or si >= num_sig:
                        continue
                    g = green_by_sigidx[si]
                    for t in range(steps):
                        series[t] = series[t] or g[t]
                return series

            def _eta_to_stopline_s(veh_id: str, lane_id: str) -> Optional[float]:
                try:
                    v = float(traci.vehicle.getSpeed(veh_id))
                    x = float(traci.vehicle.getLanePosition(veh_id))
                    L = float(traci.lane.getLength(lane_id))
                except Exception:
                    return None
                dist_to_stop = max(0.0, L - x)
                if v >= min_moving_speed_mps:
                    return dist_to_stop / max(v, 1e-3)
                return startup_lost_time_s + dist_to_stop / max(
                    approach_speed_mps, 1e-3
                )

            leg_scores: Dict[str, float] = {}
            leg_weights: Dict[str, int] = {}

            for leg in leg_order:
                lanes = list(leg_in_lanes.get(leg, []))
                if not lanes:
                    leg_scores[leg] = 0.0
                    leg_weights[leg] = 0
                    continue

                num_hits = 0
                num_total = 0
                for ln in lanes:
                    green_series = _inlane_green_series(str(ln))
                    try:
                        vehs = traci.lane.getLastStepVehicleIDs(ln)
                    except Exception:
                        vehs = []
                    for vid in vehs:
                        eta = _eta_to_stopline_s(str(vid), str(ln))
                        if eta is None or eta < 0:
                            continue
                        t_idx = int(eta // dt_s)
                        if 0 <= t_idx < steps:
                            num_total += 1
                            if green_series[t_idx]:
                                num_hits += 1

                leg_weights[leg] = num_total
                leg_scores[leg] = (num_hits / num_total) if num_total > 0 else 0.0

            total = sum(leg_weights.values())
            global_mean = (
                sum(leg_scores[l] * leg_weights[l] for l in leg_order) / total
                if total > 0
                else 0.0
            )

            gwos_vec = [
                float(leg_scores["N"]),
                float(leg_scores["NE"]),
                float(leg_scores["E"]),
                float(leg_scores["S"]),
                float(leg_scores["W"]),
                float(global_mean),
            ]

    # Final concatenation into a single vector-like output
    ret = [float(x) for x in (spillback_vec + gap_vec + tsp_vec + gwos_vec)]
    return ret
