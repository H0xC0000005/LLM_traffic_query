from __future__ import annotations

import math
import re
from statistics import median
from typing import Dict, List, Optional, Tuple


def tsc_isolated_intersection_feature_vector(
    tls_id: str,
    *,
    intersection_encoding: str = "{N@0^(-5,+5):L|T|TR; E@90^(-1,+1):L|T|TR; S@190^(-5,+5):L|T|TR;W@270^(-1,+1):L|T|TR}",
    phase_params: Optional[Dict[int, Tuple[float, float]]] = None,
    junction_id: Optional[str] = None,
    horizon_s: float = 60.0,
    moving_v_mps: float = 2.0,
    stop_v_mps: float = 0.3,  # kept for parity with expert code; not directly used in PAG logic
    sat_headway_s: float = 2.0,
    detection_zone_m: float = 10.0,
    # discharge-capacity signature params
    ema_alpha_capacity: float = 0.25,
    depart_pos_threshold_m: float = 5.0,
    min_headways_for_update: int = 3,
    # spillback/blocking params
    speed_stop_thresh_mps: float = 0.10,
    stopline_window_m: float = 5.0,
    ema_alpha_blocking: float = 0.15,
    queue_midpoint: float = 0.80,
    queue_softness: float = 0.10,
    # gap-out / max-out params
    cache: dict | None = None,
    **kwargs,
) -> List[float]:
    """
    Combines four expert feature extractors into one vector: (1) lane-level predicted-arrivals-on-green (PAG)
    ratios over a short horizon, (2) per-leg discharge-capacity efficiency from stopline departures during
    green with EMA updates, (3) per-leg (gap-out pressure, max-out proximity) for the active phase from
    stopline actuation timing, and (4) per-leg spillback/blocking risk from queue fill and green-but-blocked
    detection with EMA. Uses only the provided external cache (if any).

    Returns: list[float] = [PAG_per_lane..., cap_sat_eff_per_leg, cap_startup_eff_per_leg, gap_per_leg,
                           maxout_per_leg, spillback_risk_per_leg]
    """
    # Prefer libsumo (drop-in replacement for traci); fallback only if unavailable.
    try:
        import libsumo as traci  # type: ignore
    except Exception:  # pragma: no cover
        import traci  # type: ignore

    # --------------------------
    # External cache namespace
    # --------------------------
    if cache is None:
        cache = {}
    root = cache.setdefault("tsc_isolated_intersection_feature_vector", {})
    tls_root = root.setdefault("by_tls", {}).setdefault(tls_id, {})

    # --------------------------
    # Helpers
    # --------------------------
    def _clip01(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))

    def _circ_diff(a: float, b: float) -> float:
        d = abs((a - b) % 360.0)
        return min(d, 360.0 - d)

    def _bearing_from_shape(shape: List[Tuple[float, float]]) -> Optional[float]:
        # Heading of the last segment toward the junction. SUMO coords: x east, y north.
        if not shape or len(shape) < 2:
            return None
        (x1, y1), (x2, y2) = shape[-2], shape[-1]
        dx, dy = (x2 - x1), (y2 - y1)
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return None
        angle_from_x_ccw = math.degrees(math.atan2(dy, dx))  # 0=east, 90=north
        bearing = (90.0 - angle_from_x_ccw) % 360.0  # 0=north, 90=east (clockwise)
        return bearing

    def _lane_index_guess(lane_id: str) -> int:
        # SUMO lane IDs are often "edge_0", "edge_1", ... (0 is typically rightmost).
        m = re.search(r"_(-?\d+)$", lane_id)
        return int(m.group(1)) if m else 0

    def _parse_intersection_encoding(enc: str) -> List[dict]:
        """
        Returns legs in *encoding order* with fields:
          name, bearing(float), lane_tokens(list[str]), offsets(tuple[float,float]), no_out(bool)
        """
        enc = re.sub(r"#.*", "", enc).strip()
        s = enc
        if s.startswith("{") and s.endswith("}"):
            s = s[1:-1]
        parts = [p.strip() for p in s.split(";") if p.strip()]
        legs: List[dict] = []

        pat = re.compile(
            r"^(?P<name>[A-Za-z]+)"
            r"(?:@(?P<bearing>\d+(?:\.\d+)?))?"
            r"(?:\^\(\s*(?P<din>[+\-]?\d+(?:\.\d+)?)\s*(?:,\s*(?P<dout>[+\-]?\d+(?:\.\d+)?))?\s*\))?"
            r"\s*:\s*(?P<lanes>.+)$"
        )

        for raw in parts:
            m = pat.match(raw)
            if not m:
                continue
            name = m.group("name")
            bearing = float(m.group("bearing")) % 360.0 if m.group("bearing") is not None else 0.0
            din = float(m.group("din")) if m.group("din") is not None else 0.0
            dout = float(m.group("dout")) if m.group("dout") is not None else din
            lanes_part = (m.group("lanes") or "").strip()

            no_out = lanes_part.endswith("x")
            if no_out:
                lanes_part = lanes_part[:-1].strip()

            lane_tokens = [t.strip() for t in lanes_part.split("|") if t.strip()]

            legs.append(
                {
                    "name": name,
                    "bearing": bearing,
                    "offsets": (din, dout),
                    "lane_tokens": lane_tokens,
                    "no_out": no_out,
                }
            )
        return legs

    def _get_or_build_leg_specs() -> List[dict]:
        key = ("parsed_encoding", intersection_encoding)
        enc_cache = root.setdefault("encoding_cache", {})
        if key in enc_cache:
            return enc_cache[key]
        legs = _parse_intersection_encoding(intersection_encoding)
        enc_cache[key] = legs
        return legs

    def _controlled_in_lanes_and_indices() -> Tuple[List[str], Dict[str, List[int]], str]:
        """
        Returns:
          incoming_lanes_unique,
          lane_to_signal_indices (incoming_lane -> [signal_index,...]),
          state_string
        """
        state = traci.trafficlight.getRedYellowGreenState(tls_id)
        controlled = traci.trafficlight.getControlledLinks(tls_id)

        lane_to_signal_indices: Dict[str, List[int]] = {}
        incoming_set = set()
        for idx, link_group in enumerate(controlled):
            for in_lane, _out_lane, _via_lane in link_group:
                incoming_set.add(in_lane)
                lane_to_signal_indices.setdefault(in_lane, []).append(idx)

        return sorted(incoming_set), lane_to_signal_indices, state

    def _infer_leg_in_lanes(legs: List[dict]) -> Dict[str, List[str]]:
        """
        Infer mapping {leg_name -> inbound_laneIDs} by matching approach bearing of each controlled incoming lane
        to the closest encoding leg bearing. Lanes within each leg are ordered left-to-right (leftmost first).
        """
        # Cache lane mapping per (tls_id, intersection_encoding)
        map_key = ("leg_in_lanes", intersection_encoding, junction_id)
        if map_key in tls_root:
            return tls_root[map_key]

        leg_names = [L["name"] for L in legs]
        bearings = {L["name"]: float(L["bearing"]) for L in legs}

        try:
            incoming_lanes, _lane_to_sig_idx, _state = _controlled_in_lanes_and_indices()
        except Exception:
            # Best-effort fallback: empty mapping with known legs.
            out = {ln: [] for ln in leg_names}
            tls_root[map_key] = out
            return out

        # Optional junction filtering: keep only lanes whose edge ends at this junction.
        if junction_id is not None:
            filtered = []
            for lane_id in incoming_lanes:
                try:
                    edge_id = traci.lane.getEdgeID(lane_id)
                    if traci.edge.getToJunction(edge_id) == junction_id:
                        filtered.append(lane_id)
                except Exception:
                    filtered.append(lane_id)
            incoming_lanes = filtered

        # Assign lane -> closest leg by bearing.
        leg_to_lanes: Dict[str, List[str]] = {ln: [] for ln in leg_names}
        for lane_id in incoming_lanes:
            try:
                shape = traci.lane.getShape(lane_id)
                b = _bearing_from_shape(shape)
                if b is None:
                    continue
            except Exception:
                continue

            best_leg = min(leg_names, key=lambda ln: _circ_diff(b, bearings.get(ln, 0.0)))
            best_d = _circ_diff(b, bearings.get(best_leg, 0.0))
            # Keep a plausibility gate, but be lenient for skewed/odd approaches.
            if best_d <= 80.0:
                leg_to_lanes[best_leg].append(lane_id)

        # Order lanes within each leg left-to-right using lane index heuristic (higher index = more left).
        for ln, lanes in leg_to_lanes.items():

            def _k(lid: str) -> Tuple[str, int, str]:
                try:
                    eid = traci.lane.getEdgeID(lid)
                except Exception:
                    eid = ""
                idx = _lane_index_guess(lid)
                return (eid, -idx, lid)

            leg_to_lanes[ln] = sorted(set(lanes), key=_k)

        tls_root[map_key] = leg_to_lanes
        return leg_to_lanes

    # --------------------------
    # Parse encoding + infer inbound lanes
    # --------------------------
    legs_spec = _get_or_build_leg_specs()
    if not legs_spec:
        return []

    leg_order_encoding = [L["name"] for L in legs_spec]
    leg_in_lanes = _infer_leg_in_lanes(legs_spec)

    # --------------------------
    # (1) Expert: lane_pag_surface_ratio (per lane, legs sorted clockwise by bearing)
    # --------------------------
    def _lane_pag_surface_ratio() -> List[float]:
        # Sort legs clockwise by bearing (expert’s algorithm)
        legs_clockwise = sorted(legs_spec, key=lambda d: float(d["bearing"]))
        # If TLS lookup fails, return neutral for expected number of inbound lanes in encoding.
        try:
            _ = traci.trafficlight.getRedYellowGreenState(tls_id)
        except Exception:
            total_lanes = sum(len(L["lane_tokens"]) for L in legs_clockwise)
            return [0.5] * total_lanes

        # Build per-link green schedule over horizon (expert’s algorithm)
        sim_t = float(traci.simulation.getTime())
        next_sw = float(traci.trafficlight.getNextSwitch(tls_id))
        remaining = max(0.0, next_sw - sim_t)
        current_phase_idx = int(traci.trafficlight.getPhase(tls_id))
        current_state = traci.trafficlight.getRedYellowGreenState(tls_id)

        try:
            program_id = traci.trafficlight.getProgram(tls_id)
            logics = traci.trafficlight.getAllProgramLogics(tls_id)
        except Exception:
            program_id = None
            logics = []

        logic = None
        for lg in logics:
            if getattr(lg, "programID", None) == program_id:
                logic = lg
                break
        if logic is None and logics:
            logic = logics[0]
        phases = getattr(logic, "phases", []) if logic is not None else []
        nph = len(phases)

        intervals: List[Tuple[float, float, str]] = []
        intervals.append((0.0, min(horizon_s, remaining), current_state))

        t = remaining
        idx = (current_phase_idx + 1) % nph if nph > 0 else 0
        while t < horizon_s and nph > 0:
            ph = phases[idx]
            dur = float(getattr(ph, "duration", 0.0))
            if dur <= 0.0:
                mind = getattr(ph, "minDur", None)
                maxd = getattr(ph, "maxDur", None)
                if mind is not None and maxd is not None and float(maxd) > 0:
                    dur = 0.5 * (float(mind) + float(maxd))
                else:
                    dur = 5.0
            st = getattr(ph, "state", None) or current_state
            end = min(horizon_s, t + dur)
            intervals.append((t, end, st))
            t = end
            idx = (idx + 1) % nph

        link_count = len(current_state)
        green_intervals: Dict[int, List[Tuple[float, float, bool]]] = {li: [] for li in range(link_count)}
        for a, b, st in intervals:
            if b <= a:
                continue
            for li, ch in enumerate(st[:link_count]):
                green_intervals[li].append((a, b, ch in ("G", "g")))

        def is_green_at(li: int, tau: float) -> bool:
            for a, b, g in green_intervals.get(li, []):
                if a <= tau < b:
                    return g
            return False

        def time_to_next_green(li: int) -> float:
            for a, b, g in green_intervals.get(li, []):
                if g and b > 0:
                    return max(0.0, a)
            return horizon_s

        out: List[float] = []

        for leg in legs_clockwise:
            leg_name = leg["name"]
            lane_tokens = list(leg["lane_tokens"])
            lane_ids = list(leg_in_lanes.get(leg_name, []))

            # Align lengths (expert’s algorithm)
            m = min(len(lane_ids), len(lane_tokens))
            lane_ids = lane_ids[:m]
            lane_tokens = lane_tokens[:m]  # kept for parity; not used beyond alignment

            for lane_id in lane_ids:
                try:
                    veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                except Exception:
                    veh_ids = []

                veh_info: List[Tuple[str, int, float, float, float]] = []
                for vid in veh_ids:
                    try:
                        nxt = traci.vehicle.getNextTLS(vid)
                    except Exception:
                        continue

                    li = None
                    dist_tls = None
                    for rec in nxt:
                        if rec[0] == tls_id:
                            li = int(rec[1])
                            dist_tls = float(rec[2])
                            break
                    if li is None or dist_tls is None:
                        continue

                    try:
                        spd = float(traci.vehicle.getSpeed(vid))
                    except Exception:
                        spd = 0.0

                    # Queue ordering: distance to stop line (lane_length - position)
                    try:
                        lane_len = float(traci.lane.getLength(lane_id))
                        pos = float(traci.vehicle.getLanePosition(vid))
                        dist_stop = max(0.0, lane_len - pos)
                    except Exception:
                        dist_stop = dist_tls

                    veh_info.append((vid, li, dist_tls, dist_stop, spd))

                if not veh_info:
                    out.append(0.5)
                    continue

                by_link: Dict[int, List[Tuple[float, str, float, float]]] = {}
                for vid, li, dist_tls, dist_stop, spd in veh_info:
                    by_link.setdefault(li, []).append((dist_stop, vid, dist_tls, spd))

                rank_map: Dict[str, int] = {}
                for li, lst in by_link.items():
                    lst.sort(key=lambda x: x[0])
                    for r, (_, vid, _, _) in enumerate(lst):
                        rank_map[vid] = r

                green_cnt = 0
                total_cnt = 0
                for vid, li, dist_tls, _dist_stop, spd in veh_info:
                    if spd >= moving_v_mps:
                        tau = dist_tls / max(spd, 1e-3)
                    else:
                        tau = time_to_next_green(li) + float(rank_map.get(vid, 0)) * sat_headway_s

                    if tau > horizon_s:
                        continue
                    total_cnt += 1
                    if is_green_at(li, tau):
                        green_cnt += 1

                out.append((green_cnt / total_cnt) if total_cnt > 0 else 0.5)

        return [_clip01(v) for v in out]

    pag_vec = _lane_pag_surface_ratio()

    # --------------------------
    # (2) Expert: discharge_capacity_signature (per leg in encoding order, uses EMA in external cache)
    # --------------------------
    def _discharge_capacity_signature() -> List[float]:
        cap_cache = tls_root.setdefault("discharge_capacity_signature", {})
        lane_state: Dict[str, dict] = cap_cache.setdefault("lane_state", {})

        # Lanes-by-leg mapping (cached; respects encoding leg order)
        lanes_by_leg = cap_cache.setdefault("lanes_by_leg", {})
        if cap_cache.get("leg_order") != leg_order_encoding or not isinstance(lanes_by_leg, dict):
            lanes_by_leg = {ln: list(leg_in_lanes.get(ln, [])) for ln in leg_order_encoding}
            cap_cache["lanes_by_leg"] = lanes_by_leg
            cap_cache["leg_order"] = list(leg_order_encoding)

        all_tracked_lanes = sorted({lid for ln in leg_order_encoding for lid in lanes_by_leg.get(ln, [])})

        for lane_id in all_tracked_lanes:
            if lane_id not in lane_state:
                lane_state[lane_id] = {
                    "prev_green": False,
                    "prev_lane_green": False,
                    "burst_times": [],
                    "h_hat": 2.0,
                    "l_hat": 2.0,
                    "prev_veh_pos": {},
                }

        # Determine green by lane (current step) from state + controlled links
        try:
            state_str = traci.trafficlight.getRedYellowGreenState(tls_id)
            controlled_links = traci.trafficlight.getControlledLinks(tls_id)
        except Exception:
            return [0.5] * (2 * len(leg_order_encoding))

        lane_green_now: Dict[str, bool] = {lane_id: False for lane_id in all_tracked_lanes}
        for i, sig in enumerate(state_str):
            if sig not in ("G", "g"):
                continue
            if i >= len(controlled_links):
                break
            for in_lane, _out_lane, _via_lane in controlled_links[i]:
                if in_lane in lane_green_now:
                    lane_green_now[in_lane] = True

        sim_t = float(traci.simulation.getTime())

        # Departure detection + green-burst updates
        for lane_id in all_tracked_lanes:
            st = lane_state[lane_id]
            prev_pos: Dict[str, float] = st["prev_veh_pos"]

            try:
                curr_ids = list(traci.lane.getLastStepVehicleIDs(lane_id))
            except Exception:
                curr_ids = []
            curr_set = set(curr_ids)

            curr_pos: Dict[str, float] = {}
            for vid in curr_ids:
                try:
                    curr_pos[vid] = float(traci.vehicle.getLanePosition(vid))
                except Exception:
                    pass

            departed = [vid for vid in prev_pos.keys() if vid not in curr_set]

            # Attribute departure to last-step green and near-stopline positions.
            if st["prev_lane_green"] and departed:
                try:
                    lane_len = float(traci.lane.getLength(lane_id))
                except Exception:
                    lane_len = None
                if lane_len is not None:
                    for vid in departed:
                        if prev_pos.get(vid, -1e9) >= lane_len - float(depart_pos_threshold_m):
                            st["burst_times"].append(sim_t)

            g_now = bool(lane_green_now.get(lane_id, False))
            g_prev = bool(st["prev_green"])

            if g_now and not g_prev:
                st["burst_times"] = []

            if (not g_now) and g_prev:
                times = sorted(st["burst_times"])
                if len(times) >= 2:
                    headways = [times[i] - times[i - 1] for i in range(1, len(times))]
                    if len(headways) >= int(min_headways_for_update):
                        tail = headways[2:] if len(headways) >= 5 else headways[1:] if len(headways) >= 3 else headways
                        h_sat = float(median(tail)) if tail else float(median(headways))
                        h_sat = max(0.8, min(4.5, h_sat))

                        k = min(2, len(headways))
                        l_start = max(0.0, float(sum(headways[:k]) - k * h_sat))
                        l_start = min(6.0, l_start)

                        st["h_hat"] = (1.0 - ema_alpha_capacity) * float(st["h_hat"]) + ema_alpha_capacity * h_sat
                        st["l_hat"] = (1.0 - ema_alpha_capacity) * float(st["l_hat"]) + ema_alpha_capacity * l_start

                st["burst_times"] = []

            st["prev_green"] = g_now
            st["prev_lane_green"] = g_now
            st["prev_veh_pos"] = curr_pos

        out: List[float] = []
        for leg in leg_order_encoding:
            lanes = lanes_by_leg.get(leg, [])
            if not lanes:
                out.extend([0.5, 0.5])
                continue

            h_vals = [float(lane_state[lid]["h_hat"]) for lid in lanes if lid in lane_state]
            l_vals = [float(lane_state[lid]["l_hat"]) for lid in lanes if lid in lane_state]
            if not h_vals or not l_vals:
                out.extend([0.5, 0.5])
                continue

            h_hat = sum(h_vals) / len(h_vals)
            l_hat = sum(l_vals) / len(l_vals)

            s_per_lane = 3600.0 / max(0.8, h_hat)
            sat_eff = _clip01((s_per_lane - 1200.0) / 1200.0)
            startup_eff = 1.0 - _clip01(l_hat / 4.0)

            out.extend([sat_eff, startup_eff])

        return out

    cap_vec = _discharge_capacity_signature()

    # --------------------------
    # (3) Expert: gapout_maxout_feature_vector (per leg in encoding order, uses external cache)
    # --------------------------
    def _gapout_maxout_feature_vector() -> List[float]:
        if phase_params is None:
            local_phase_params: Dict[int, Tuple[float, float]] = {}
        else:
            local_phase_params = dict(phase_params)

        gom_cache = tls_root.setdefault("gapout_maxout_feature_vector", {})
        last_phase = gom_cache.get("last_phase", None)
        phase_start_time = gom_cache.get("phase_start_time", None)
        last_actuation_by_leg = gom_cache.setdefault("last_actuation_by_leg", {})

        now = float(traci.simulation.getTime())
        try:
            cur_phase = int(traci.trafficlight.getPhase(tls_id))
        except Exception:
            return [0.5 for _ in range(2 * len(leg_order_encoding))]

        default_passage, _default_maxg = local_phase_params.get(cur_phase, (2.5, 30.0))
        for leg in leg_order_encoding:
            if leg not in last_actuation_by_leg:
                last_actuation_by_leg[leg] = now - 0.5 * float(default_passage)

        if last_phase is None or phase_start_time is None:
            gom_cache["last_phase"] = cur_phase
            gom_cache["phase_start_time"] = now
            last_phase = cur_phase
            phase_start_time = now
        elif cur_phase != last_phase:
            gom_cache["last_phase"] = cur_phase
            gom_cache["phase_start_time"] = now
            last_phase = cur_phase
            phase_start_time = now

        elapsed_green = max(0.0, now - float(phase_start_time))

        try:
            state = traci.trafficlight.getRedYellowGreenState(tls_id)
            controlled_links = traci.trafficlight.getControlledLinks(tls_id)
        except Exception:
            return [0.5 for _ in range(2 * len(leg_order_encoding))]

        lane_to_leg: Dict[str, str] = {}
        for leg, lanes in leg_in_lanes.items():
            for ln in lanes:
                lane_to_leg[ln] = leg

        green_legs = set()
        n = min(len(state), len(controlled_links))
        for i in range(n):
            if state[i] not in ("G", "g"):
                continue
            for in_lane, _out_lane, _via_lane in controlled_links[i]:
                leg = lane_to_leg.get(in_lane, None)
                if leg is not None:
                    green_legs.add(leg)

        # Update last actuation times by leg: actuated if any vehicle is within detection_zone_m of stopline.
        for leg in leg_order_encoding:
            lanes = leg_in_lanes.get(leg, [])
            actuated = False
            for lane_id in lanes:
                try:
                    veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                    if not veh_ids:
                        continue
                    lane_len = float(traci.lane.getLength(lane_id))
                    for vid in veh_ids:
                        pos = float(traci.vehicle.getLanePosition(vid))
                        dist_to_stop = lane_len - pos
                        if dist_to_stop <= detection_zone_m:
                            actuated = True
                            break
                    if actuated:
                        break
                except Exception:
                    continue
            if actuated:
                last_actuation_by_leg[leg] = now

        passage_time_s, max_green_s = local_phase_params.get(cur_phase, (2.5, 30.0))
        passage_time_s = max(0.5, float(passage_time_s))
        max_green_s = max(1.0, float(max_green_s))

        out: List[float] = []
        for leg in leg_order_encoding:
            if leg in green_legs:
                headway = max(0.0, now - float(last_actuation_by_leg.get(leg, now)))
                gap_ratio = _clip01(headway / passage_time_s)
                maxout_ratio = _clip01(elapsed_green / max_green_s)
            else:
                gap_ratio = 0.0
                maxout_ratio = 0.0
            out.extend([gap_ratio, maxout_ratio])

        return out

    gom_vec = _gapout_maxout_feature_vector()

    # --------------------------
    # (4) Expert: spillback_blocking_risk_per_leg (per leg in encoding order, uses external cache)
    # --------------------------
    def _spillback_blocking_risk_per_leg() -> List[float]:
        def _sigmoid(x: float) -> float:
            if x >= 0:
                z = math.exp(-x)
                return 1.0 / (1.0 + z)
            z = math.exp(x)
            return z / (1.0 + z)

        legs = legs_spec
        leg_names_in_order = [L["name"] for L in legs]
        leg_bearings = {L["name"]: float(L["bearing"]) for L in legs}

        try:
            incoming_lanes, lane_to_signal_indices, state = _controlled_in_lanes_and_indices()
        except Exception:
            return [0.0 for _ in leg_names_in_order]

        if not incoming_lanes:
            return [0.0 for _ in leg_names_in_order]

        # Map each incoming lane to closest leg by bearing (expert’s algorithm)
        leg_to_lanes: Dict[str, List[str]] = {ln: [] for ln in leg_names_in_order}
        for lane_id in incoming_lanes:
            try:
                shape = traci.lane.getShape(lane_id)
                b = _bearing_from_shape(shape)
                if b is None:
                    continue
            except Exception:
                continue

            best_leg = min(leg_names_in_order, key=lambda ln: _circ_diff(b, leg_bearings.get(ln, 0.0)))
            leg_to_lanes[best_leg].append(lane_id)

        sb_cache = tls_root.setdefault("spillback_blocking_risk_per_leg", {})
        block_ema_by_leg = sb_cache.setdefault("block_ema_by_leg", {})

        risks: List[float] = []
        for leg in leg_names_in_order:
            lanes = leg_to_lanes.get(leg, [])
            if not lanes:
                risks.append(0.15)
                continue

            lane_queue_fill_ratios: List[float] = []
            lane_blocked_when_green: List[float] = []

            for lane_id in lanes:
                try:
                    lane_len = float(traci.lane.getLength(lane_id))
                except Exception:
                    lane_len = 0.0
                if lane_len <= 1e-3:
                    continue

                try:
                    veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                except Exception:
                    veh_ids = []

                queue_extent_m = 0.0
                stopline_blocked = 0.0

                for vid in veh_ids:
                    try:
                        v_speed = float(traci.vehicle.getSpeed(vid))
                        v_pos = float(traci.vehicle.getLanePosition(vid))
                    except Exception:
                        continue

                    dist_to_stopline = max(0.0, lane_len - v_pos)
                    if v_speed <= speed_stop_thresh_mps:
                        queue_extent_m = max(queue_extent_m, dist_to_stopline)
                        if dist_to_stopline <= stopline_window_m:
                            stopline_blocked = 1.0

                queue_fill = min(1.0, queue_extent_m / lane_len)
                lane_queue_fill_ratios.append(queue_fill)

                green_now = 0.0
                for si in lane_to_signal_indices.get(lane_id, []):
                    if 0 <= si < len(state) and state[si] in ("g", "G"):
                        green_now = 1.0
                        break

                lane_blocked_when_green.append(green_now * stopline_blocked)

            if not lane_queue_fill_ratios:
                risks.append(0.15)
                continue

            leg_queue_fill = max(lane_queue_fill_ratios)
            x = (leg_queue_fill - queue_midpoint) / max(queue_softness, 1e-6)
            queue_risk = _sigmoid(x)

            block_inst = float(sum(lane_blocked_when_green) / max(1, len(lane_blocked_when_green)))
            prev = float(block_ema_by_leg.get(leg, block_inst))
            block_ema = (1.0 - ema_alpha_blocking) * prev + ema_alpha_blocking * block_inst
            block_ema_by_leg[leg] = block_ema

            combined = 0.65 * queue_risk + 0.35 * block_ema
            risks.append(_clip01(combined))

        return risks

    spill_vec = _spillback_blocking_risk_per_leg()

    # --------------------------
    # Combined vector
    # --------------------------
    # Preserve each expert’s internal ordering; concatenate in a fixed block order.
    return [float(x) for x in (pag_vec + cap_vec + gom_vec + spill_vec)]
