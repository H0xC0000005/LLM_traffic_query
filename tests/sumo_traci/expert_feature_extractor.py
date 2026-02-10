def tsc_isolated_intersection_feature_vector(
    tls_id: str,
    *,
    intersection_encoding: str = "{N@7^(+1,+1):LT|TR; E@90^(-1,+3):L60|T|T|T|R60; S@190^(+1,+1):LT|TR;W@270^(-1,+3):L60|T|T|T|R60}",
    # -------- expert1: green band reliability --------
    horizon_s: float = 120.0,
    approach_dist_m: float = 250.0,
    cv: float = 0.15,
    min_sigma_s: float = 1.0,
    sat_headway_s: float = 2.1,
    start_lost_s: float = 2.0,
    queue_speed_thresh_mps: float = 0.2,
    smoothing_n: float = 2.0,
    # -------- expert2: permissive-left exposure --------
    zone_distance_m: float = 80.0,
    tau_s: float = 4.0,
    volume_scale_k: float = 3.0,
    v_min_mps: float = 0.5,
    p_left_given_LT: float = 0.35,
    p_through_given_TR: float = 0.70,
    # -------- expert3: spillback time-to-storage --------
    t_max_s: float = 120.0,
    avg_standstill_spacing_m: float = 7.5,
    v_blocked_mps: float = 1.0,
    # (kept for API parity with the expert module; the expert implementation does not use it as a hard counter)
    blocked_persist_steps: int = 5,
    # -------- expert4: blockage-adjusted turnbay capacity --------
    alpha: float = 0.35,
    jam_spacing_m: float = 7.5,
    w_interference: float = 0.7,
) -> list[float]:
    """
    Combine four expert feature modules into a single numeric vector for a single isolated intersection:
    green-band reliability (per leg + global), permissive-left conflict exposure (per leg),
    spillback time-to-storage urgency (N/E/S/W × 3), and EWMA-smoothed blockage risk (N/E/S/W).
    Only input is tls_id (and optional encoding/parameters). Output is a flat list[float] in a fixed order.
    """
    import math
    import re
    import libsumo as traci

    # ----------------------------
    # Unified encoding parser (merged from expert modules)
    # ----------------------------
    def _parse_intersection_encoding(
        enc: str,
    ) -> tuple[
        list[str],
        dict[str, float],
        dict[str, tuple[float, float]],
        dict[str, list[dict]],
    ]:
        """
        Returns:
          - leg_order: legs in appearance order in the encoding string
          - bearing_by_leg: leg -> bearing degrees (float)
          - offsets_by_leg: leg -> (din, dout) (float, float)
          - lanespecs_by_leg: leg -> list of lane specs (left->right in encoding)
                spec fields:
                  perm (str), moves (set[str]), pocket_len (float|None)
        """
        s = enc.strip()
        if s.startswith("{"):
            s = s[1:]
        if s.endswith("}"):
            s = s[:-1]
        parts = [p.strip() for p in s.split(";") if p.strip()]

        leg_order: list[str] = []
        bearing_by_leg: dict[str, float] = {}
        offsets_by_leg: dict[str, tuple[float, float]] = {}
        lanespecs_by_leg: dict[str, list[dict]] = {}

        for part in parts:
            if ":" not in part:
                continue
            head, lane_str = part.split(":", 1)
            head = head.strip()
            lane_str = lane_str.strip()

            # remove outbound 'x' marker (ignored by experts here)
            if lane_str.endswith("x"):
                lane_str = lane_str[:-1].strip()

            # leg name is prefix before @ or ^
            leg_name = re.split(r"[@^]", head, maxsplit=1)[0].strip()
            if leg_name and leg_name not in leg_order:
                leg_order.append(leg_name)

            # bearing
            m_b = re.search(r"@(\d+)", head)
            bearing = float(m_b.group(1)) if m_b else 0.0
            bearing_by_leg[leg_name] = bearing

            # offsets ^(din[,dout])
            din, dout = 0.0, 0.0
            m_off = re.search(r"\^\(([^)]*)\)", head)
            if m_off:
                vals = [v.strip() for v in m_off.group(1).split(",") if v.strip()]
                try:
                    din = float(vals[0]) if len(vals) >= 1 else 0.0
                    dout = float(vals[1]) if len(vals) >= 2 else din
                except Exception:
                    din, dout = 0.0, 0.0
            offsets_by_leg[leg_name] = (din, dout)

            lane_tokens = [t.strip() for t in lane_str.split("|") if t.strip()]
            specs: list[dict] = []
            for tok in lane_tokens:
                # support L60, R60, LT, TR, T, etc.
                m = re.fullmatch(r"([A-Za-z]+)(\d+)?", tok)
                if not m:
                    perm = tok.upper()
                    pocket_len = None
                else:
                    perm = m.group(1).upper()
                    pocket_len = float(m.group(2)) if m.group(2) else None
                specs.append(
                    {"perm": perm, "moves": set(perm), "pocket_len": pocket_len}
                )
            lanespecs_by_leg[leg_name] = specs

        return leg_order, bearing_by_leg, offsets_by_leg, lanespecs_by_leg

    def _angular_diff_deg(a: float, b: float) -> float:
        d = (a - b) % 360.0
        return min(d, 360.0 - d)

    # ----------------------------
    # Infer inbound lane IDs per leg from TLS (left->right ordering at stopline)
    # ----------------------------
    def _infer_lane_ids_by_leg(
        tls: str,
        leg_order: list[str],
        bearing_by_leg: dict[str, float],
        lanespecs_by_leg: dict[str, list[dict]],
    ) -> dict[str, list[str]]:
        # Collect from-lanes controlled by the TLS.
        from_lanes: list[str] = []
        try:
            controlled = traci.trafficlight.getControlledLinks(tls)
            for links in controlled:
                for link in links:
                    if not isinstance(link, (list, tuple)) or len(link) < 1:
                        continue
                    from_lane = link[0]
                    if isinstance(from_lane, (list, tuple)) and len(from_lane) >= 1:
                        from_lane = from_lane[0]
                    if isinstance(from_lane, str):
                        from_lanes.append(from_lane)
        except Exception:
            from_lanes = []

        if not from_lanes:
            try:
                from_lanes = list(traci.trafficlight.getControlledLanes(tls))
            except Exception:
                from_lanes = []

        # Unique while preserving order
        seen: set[str] = set()
        uniq: list[str] = []
        for ln in from_lanes:
            if ln not in seen:
                seen.add(ln)
                uniq.append(ln)

        # Inbound bearing targets: travel direction toward intersection is (bearing + 180) mod 360.
        inbound_bear: dict[str, float] = {
            leg: (bearing_by_leg.get(leg, 0.0) + 180.0) % 360.0 for leg in leg_order
        }

        # Assign each lane to closest leg by approach bearing.
        per_leg_entries: dict[str, list[tuple[str, tuple[float, float]]]] = {
            leg: [] for leg in leg_order
        }
        for lane_id in uniq:
            try:
                shape = traci.lane.getShape(lane_id)
                if not shape or len(shape) < 2:
                    continue
                (x0, y0), (x1, y1) = shape[-2], shape[-1]
                dx, dy = (x1 - x0), (y1 - y0)
                if abs(dx) + abs(dy) < 1e-9:
                    continue

                # Bearing with North=0, clockwise positive (atan2(x,y)).
                lane_bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

                best_leg, best_d = leg_order[0], 1e9
                for leg in leg_order:
                    d = _angular_diff_deg(lane_bearing, inbound_bear[leg])
                    if d < best_d:
                        best_leg, best_d = leg, d

                per_leg_entries[best_leg].append((lane_id, (float(x1), float(y1))))
            except Exception:
                continue

        # Order lanes left->right for each leg using lateral coordinate at stopline w.r.t inbound direction.
        lane_ids_by_leg: dict[str, list[str]] = {}
        for leg in leg_order:
            entries = per_leg_entries.get(leg, [])
            if not entries:
                lane_ids_by_leg[leg] = []
                continue

            mx = sum(p[0] for _, p in entries) / len(entries)
            my = sum(p[1] for _, p in entries) / len(entries)

            th = math.radians(inbound_bear[leg])
            v = (math.sin(th), math.cos(th))  # unit direction of travel

            scored: list[tuple[float, str]] = []
            for lane_id, (x, y) in entries:
                rx, ry = (x - mx), (y - my)
                lateral = v[0] * ry - v[1] * rx  # >0 means left of travel
                scored.append((lateral, lane_id))

            scored.sort(key=lambda t: t[0], reverse=True)  # left -> right
            ordered = [lid for _, lid in scored]

            # Trim to the number of encoded inbound lanes (ensures alignment with lane specs)
            expected = len(lanespecs_by_leg.get(leg, []))
            if expected > 0:
                ordered = ordered[:expected]

            lane_ids_by_leg[leg] = ordered

        return lane_ids_by_leg

    # ----------------------------
    # Expert1: green-band reliability (preserved, global weighted by inbound lane counts)
    # ----------------------------
    def _build_timeline(tls: str, horizon: float) -> list[tuple[float, float, str]]:
        t0 = float(traci.simulation.getTime())
        state_now = traci.trafficlight.getRedYellowGreenState(tls)
        phase_idx = int(traci.trafficlight.getPhase(tls))
        next_sw = float(traci.trafficlight.getNextSwitch(tls))
        rem = max(0.0, next_sw - t0)

        logics = traci.trafficlight.getAllProgramLogics(tls)
        if logics and getattr(logics[0], "phases", None):
            phases = logics[0].phases
            states = [str(ph.state) for ph in phases]
            durs = [float(ph.duration) for ph in phases]
            if phase_idx < 0 or phase_idx >= len(states):
                phase_idx = 0
        else:
            return [(0.0, float(horizon), state_now)]

        timeline: list[tuple[float, float, str]] = []
        t = 0.0
        idx = phase_idx

        first_len = rem if rem > 0.0 else max(0.1, durs[idx])
        first_end = min(float(horizon), t + first_len)
        timeline.append((t, first_end, state_now))
        t = first_end
        idx = (idx + 1) % len(states)

        while t < float(horizon):
            dur = max(0.1, durs[idx])
            end = min(float(horizon), t + dur)
            timeline.append((t, end, states[idx]))
            t = end
            idx = (idx + 1) % len(states)

        return timeline

    def _merge_intervals(
        intervals: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        for a, b in intervals[1:]:
            la, lb = merged[-1]
            if a <= lb + 1e-6:
                merged[-1] = (la, max(lb, b))
            else:
                merged.append((a, b))
        return merged

    def _green_intervals_for_index(
        timeline: list[tuple[float, float, str]], tls_index: int
    ) -> list[tuple[float, float]]:
        intervals: list[tuple[float, float]] = []
        for a, b, st in timeline:
            if 0 <= tls_index < len(st) and st[tls_index] in ("G", "g"):
                intervals.append((a, b))
        return _merge_intervals(intervals)

    _SQRT2 = math.sqrt(2.0)

    def _phi(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / _SQRT2))

    def _prob_in_intervals(
        mu: float, sigma: float, intervals: list[tuple[float, float]]
    ) -> float:
        if sigma <= 0.0 or not intervals:
            return 0.0
        p = 0.0
        for a, b in intervals:
            p += _phi((b - mu) / sigma) - _phi((a - mu) / sigma)
        return max(0.0, min(1.0, p))

    def _green_band_reliability(
        leg_order: list[str],
        lane_ids_by_leg: dict[str, list[str]],
        lanespecs_by_leg: dict[str, list[dict]],
    ) -> tuple[list[str], list[float]]:
        # Leg order is taken from the encoding appearance (expert behavior); global is weighted by inbound lane counts.
        lane_counts = {leg: len(lanespecs_by_leg.get(leg, [])) for leg in leg_order}

        timeline = _build_timeline(tls_id, horizon_s)
        green_cache: dict[int, list[tuple[float, float]]] = {}
        next_green_cache: dict[int, float] = {}

        leg_vehicle_probs: dict[str, list[float]] = {leg: [] for leg in leg_order}

        for leg in leg_order:
            for lane_id in lane_ids_by_leg.get(leg, []):
                try:
                    veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                except Exception:
                    continue
                if not veh_ids:
                    continue

                per_lane: list[tuple[str, int, float, float]] = (
                    []
                )  # (veh_id, tls_index, dist, speed)
                stopped: list[tuple[float, str, int]] = []  # (dist, veh_id, tls_index)

                for vid in veh_ids:
                    try:
                        spd = float(traci.vehicle.getSpeed(vid))
                        next_tls = traci.vehicle.getNextTLS(vid)
                    except Exception:
                        continue

                    tls_index = None
                    dist = None
                    for entry in next_tls:
                        if str(entry[0]) == tls_id:
                            tls_index = int(entry[1])
                            dist = float(entry[2])
                            break

                    if tls_index is None or dist is None:
                        continue
                    if dist < 0.0 or dist > float(approach_dist_m):
                        continue

                    per_lane.append((vid, tls_index, dist, spd))
                    if spd <= float(queue_speed_thresh_mps):
                        stopped.append((dist, vid, tls_index))

                if not per_lane:
                    continue

                stopped.sort(key=lambda x: x[0])
                rank_map = {vid: r for r, (_, vid, _) in enumerate(stopped)}

                for vid, tls_index, dist, spd in per_lane:
                    if tls_index not in green_cache:
                        gi = _green_intervals_for_index(timeline, tls_index)
                        green_cache[tls_index] = gi
                        next_green_cache[tls_index] = (
                            gi[0][0] if gi else float(horizon_s)
                        )

                    intervals = green_cache[tls_index]
                    t_next_green = next_green_cache[tls_index]

                    v_eff = max(0.1, spd)
                    t_kin = float(dist) / v_eff

                    if vid in rank_map:
                        r = rank_map[vid]
                        mu = (
                            max(t_kin, t_next_green)
                            + float(start_lost_s)
                            + float(r) * float(sat_headway_s)
                        )
                    else:
                        mu = (
                            t_kin
                            if t_kin >= t_next_green
                            else (t_next_green + float(start_lost_s))
                        )

                    sigma = max(float(min_sigma_s), mu * float(cv))
                    p_green = _prob_in_intervals(mu, sigma, intervals)
                    leg_vehicle_probs[leg].append(p_green)

        per_leg: list[float] = []
        for leg in leg_order:
            probs = leg_vehicle_probs.get(leg, [])
            n = len(probs)
            score = (sum(probs) + float(smoothing_n) * 0.5) / (n + float(smoothing_n))
            per_leg.append(max(0.0, min(1.0, float(score))))

        total_w = 0.0
        total_s = 0.0
        for leg, sc in zip(leg_order, per_leg):
            w = float(max(1, lane_counts.get(leg, 1)))
            total_w += w
            total_s += w * sc
        global_sc = max(0.0, min(1.0, total_s / total_w if total_w > 0 else 0.5))

        return leg_order, per_leg + [global_sc]

    # ----------------------------
    # Expert2: permissive-left conflict exposure (soft intent weights; direction index fix)
    # ----------------------------
    def _edge_from_lane(lane_id: str) -> str:
        return lane_id.rsplit("_", 1)[0] if "_" in lane_id else lane_id

    def _lane_index_suffix(lane_id: str):
        try:
            return int(lane_id.rsplit("_", 1)[1])
        except Exception:
            return None

    def _soft_weights_from_perm(perm: str) -> tuple[float, float]:
        P = perm.upper()
        hasL, hasT, hasR = ("L" in P), ("T" in P), ("R" in P)
        if hasL and not hasT and not hasR:
            return 1.0, 0.0
        if hasT and not hasL and not hasR:
            return 0.0, 1.0
        if hasL and hasT and not hasR:  # LT
            return float(p_left_given_LT), float(1.0 - p_left_given_LT)
        if hasT and hasR and not hasL:  # TR
            return 0.0, float(p_through_given_TR)
        if hasL and hasT and hasR:  # LTR
            return 0.25, 0.50
        return 0.0, 0.0

    def _infer_move_from_route_and_links(veh_id: str, lane_id: str) -> str:
        """
        Return one of {'l','s','r','t','?'}.
        Uses vehicle route + lane links. Direction for extended links may appear at index 6.
        """
        try:
            route = traci.vehicle.getRoute(veh_id)
            ridx = traci.vehicle.getRouteIndex(veh_id)
            if not route or ridx is None or ridx < 0 or ridx >= len(route) - 1:
                return "?"
            next_edge = route[ridx + 1]
            links = traci.lane.getLinks(lane_id)  # extended in many builds

            for link in links:
                out_lane = link[0] if len(link) >= 1 else None
                # expert fix: prefer index 6 when available
                direction = (
                    link[6] if len(link) >= 7 else (link[5] if len(link) >= 6 else None)
                )
                if out_lane and _edge_from_lane(out_lane) == next_edge and direction:
                    return str(direction)[0].lower()

            if len(links) == 1:
                direction = (
                    links[0][6]
                    if len(links[0]) >= 7
                    else (links[0][5] if len(links[0]) >= 6 else None)
                )
                if direction:
                    return str(direction)[0].lower()
        except Exception:
            pass
        return "?"

    def _compute_opposites(
        leg_order: list[str], bearing_by_leg: dict[str, float]
    ) -> dict[str, str]:
        opposites: dict[str, str] = {}
        for ln in leg_order:
            target = (bearing_by_leg[ln] + 180.0) % 360.0
            best, best_d = None, 1e9
            for other in leg_order:
                if other == ln:
                    continue
                d = _angular_diff_deg(target, bearing_by_leg[other])
                if d < best_d:
                    best, best_d = other, d
            if best is not None:
                opposites[ln] = best
        return opposites

    def _permissive_left_exposure(
        intersection_encoding: str,
        leg_in_lanes: dict[str, list[str]],
        leg_order: list[str],
        bearing_by_leg: dict[str, float],
        lanespecs_by_leg: dict[str, list[dict]],
    ) -> list[float]:
        opposites = _compute_opposites(leg_order, bearing_by_leg)

        lt_tts_w: dict[str, list[tuple[float, float]]] = {
            leg: [] for leg in leg_in_lanes.keys()
        }
        th_tts_w: dict[str, list[tuple[float, float]]] = {
            leg: [] for leg in leg_in_lanes.keys()
        }

        for leg, lane_ids in leg_in_lanes.items():
            perm_list_raw = [
                {"perm": s["perm"], "dedicated_m": s.get("pocket_len", None)}
                for s in lanespecs_by_leg.get(leg, [])
            ]

            # Expert2 alignment: if lane IDs look like SUMO right-to-left order (0..n-1), reverse the encoding perms.
            perm_list = perm_list_raw
            idxs = [_lane_index_suffix(lid) for lid in lane_ids]
            if len(perm_list_raw) == len(lane_ids) and all(i is not None for i in idxs):
                if idxs == sorted(idxs):
                    perm_list = list(reversed(perm_list_raw))

            # Expand zone if any lane that permits left has a dedicated bay length (e.g., L60).
            dedicated_max_lt = 0.0
            for li in perm_list:
                if li.get("dedicated_m") is not None and ("L" in li.get("perm", "")):
                    dedicated_max_lt = max(dedicated_max_lt, float(li["dedicated_m"]))

            leg_zone = (
                max(zone_distance_m, dedicated_max_lt + 20.0)
                if dedicated_max_lt > 0
                else zone_distance_m
            )

            for idx, lane_id in enumerate(lane_ids):
                fallback_perm = perm_list[idx]["perm"] if idx < len(perm_list) else "?"

                try:
                    lane_len = float(traci.lane.getLength(lane_id))
                    veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                except Exception:
                    continue

                for vid in veh_ids:
                    try:
                        pos = float(traci.vehicle.getLanePosition(vid))
                        remain = max(lane_len - pos, 0.0)
                        if remain > leg_zone:
                            continue

                        speed = float(traci.vehicle.getSpeed(vid))
                        tts = remain / max(speed, v_min_mps)

                        mv = _infer_move_from_route_and_links(vid, lane_id)
                        if mv == "l":
                            lt_tts_w[leg].append((tts, 1.0))
                        elif mv == "s":
                            th_tts_w[leg].append((tts, 1.0))
                        else:
                            wL, wT = _soft_weights_from_perm(fallback_perm)
                            if wL > 0.0:
                                lt_tts_w[leg].append((tts, wL))
                            if wT > 0.0:
                                th_tts_w[leg].append((tts, wT))
                    except Exception:
                        continue

        exposure_by_leg: dict[str, float] = {}
        for leg in leg_in_lanes.keys():
            opp = opposites.get(leg, None)
            if opp is None:
                exposure_by_leg[leg] = 0.0
                continue

            n_lt = sum(w for _, w in lt_tts_w.get(leg, []))
            n_opp = sum(w for _, w in th_tts_w.get(opp, []))

            if n_lt <= 1e-9 or n_opp <= 1e-9:
                exposure_by_leg[leg] = 0.0
                continue

            t_lt = min(tts for tts, w in lt_tts_w[leg] if w > 0.0)
            t_op = min(tts for tts, w in th_tts_w[opp] if w > 0.0)

            volume_pressure = 1.0 - math.exp(
                -(n_lt * n_opp) / max(volume_scale_k, 1e-6)
            )
            alignment = math.exp(-abs(t_lt - t_op) / max(tau_s, 1e-6))
            score = volume_pressure * (0.5 + 0.5 * alignment)
            exposure_by_leg[leg] = max(0.0, min(1.0, float(score)))

        # Return in the same leg order as other per-leg features.
        return [float(exposure_by_leg.get(leg, 0.0)) for leg in leg_order]

    # ----------------------------
    # Expert3: spillback time-to-storage features (soft aggregation; EMA growth; graded blocked)
    # ----------------------------
    def _spillback_time_to_storage_features(
        intersection_encoding: str,
        lane_ids_by_leg: dict[str, list[str]],
        tls: str,
        lanespecs_by_leg: dict[str, list[dict]],
    ) -> list[float]:
        # Persistent cache as function attribute (preserves expert semantics).
        if not hasattr(tsc_isolated_intersection_feature_vector, "_spill_cache"):
            tsc_isolated_intersection_feature_vector._spill_cache = {}
        cache: dict[str, float] = tsc_isolated_intersection_feature_vector._spill_cache  # type: ignore[attr-defined]

        # Build fromLane -> signal indices map (optional, but we provide tls_id so blocked can be gated when possible)
        lane_to_sigidx: dict[str, list[int]] = {}
        state = ""
        try:
            state = traci.trafficlight.getRedYellowGreenState(tls)
            controlled = traci.trafficlight.getControlledLinks(tls)
            for i, links in enumerate(controlled):
                for link in links:
                    if isinstance(link, (list, tuple)) and len(link) >= 1:
                        from_lane = link[0]
                        if isinstance(from_lane, (list, tuple)) and len(from_lane) >= 1:
                            from_lane = from_lane[0]
                        if isinstance(from_lane, str):
                            lane_to_sigidx.setdefault(from_lane, []).append(i)
        except Exception:
            lane_to_sigidx = {}
            state = ""

        # timestep (seconds)
        try:
            dt_raw = float(traci.simulation.getDeltaT())
            dt_s = dt_raw / 1000.0 if dt_raw > 50.0 else dt_raw
            dt_s = max(0.1, dt_s)
        except Exception:
            dt_s = 1.0

        def _clip01(x: float) -> float:
            return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)

        def _is_lane_green(lane_id: str) -> bool:
            for i in lane_to_sigidx.get(lane_id, []):
                if i < len(state) and state[i] in ("G", "g"):
                    return True
            return False

        beta_fill = 6.0
        tau_tts_local = 35.0
        alpha_g = 0.30
        alpha_b = 0.20

        def _log_norm(x: float, xmax: float = 1.5) -> float:
            x = max(0.0, min(xmax, x))
            return _clip01(math.log1p(beta_fill * x) / math.log1p(beta_fill * xmax))

        def _veh_in_storage(lane_id: str, lane_len: float, storage_m: float) -> int:
            try:
                vids = traci.lane.getLastStepVehicleIDs(lane_id)
            except Exception:
                return 0
            if storage_m >= lane_len - 0.5:
                return len(vids)
            cnt = 0
            for vid in vids:
                try:
                    pos = float(traci.vehicle.getLanePosition(vid))
                    dist_to_stop = lane_len - pos
                    if dist_to_stop <= storage_m + 1e-6:
                        cnt += 1
                except Exception:
                    continue
            return cnt

        out: list[float] = []
        for leg in ["N", "E", "S", "W"]:
            specs = lanespecs_by_leg.get(leg, [])
            lane_ids = lane_ids_by_leg.get(leg, [])
            k = min(len(specs), len(lane_ids))
            if k == 0:
                out.extend([0.0, 0.0, 0.0])
                continue

            sum_w = 0.0
            sum_fill = 0.0
            sum_tts = 0.0
            sum_block = 0.0

            for i in range(k):
                lane_id = lane_ids[i]
                pocket_m = specs[i].get("pocket_len", None)

                try:
                    lane_len = float(traci.lane.getLength(lane_id))
                except Exception:
                    lane_len = 200.0

                storage_m = (
                    lane_len if pocket_m is None else min(lane_len, float(pocket_m))
                )
                storage_m = max(1.0, storage_m)
                storage_veh = max(1.0, storage_m / max(1e-3, avg_standstill_spacing_m))

                n_in_store = float(_veh_in_storage(lane_id, lane_len, storage_m))
                fill_raw = n_in_store / storage_veh
                fill = _log_norm(fill_raw, xmax=1.5)

                prev_n = float(cache.get(f"n:{lane_id}", n_in_store))
                inst_growth = max(0.0, (n_in_store - prev_n) / dt_s)
                cache[f"n:{lane_id}"] = n_in_store

                g_ema = float(cache.get(f"gema:{lane_id}", inst_growth))
                g_ema = (1.0 - alpha_g) * g_ema + alpha_g * inst_growth
                cache[f"gema:{lane_id}"] = g_ema

                remaining = max(0.0, storage_veh - n_in_store)
                if g_ema <= 1e-3:
                    tts_score = 0.0
                else:
                    tts = remaining / g_ema
                    tts_score = _clip01(
                        math.exp(-min(tts, t_max_s) / max(1e-3, tau_tts_local))
                    )

                b_ema = float(cache.get(f"bema:{lane_id}", 0.0))
                try:
                    mean_v = float(traci.lane.getLastStepMeanSpeed(lane_id))
                except Exception:
                    mean_v = 0.0
                try:
                    occ = float(traci.lane.getLastStepOccupancy(lane_id)) / 100.0
                except Exception:
                    occ = 0.0

                slow_factor = _clip01(
                    (v_blocked_mps - mean_v) / max(1e-3, v_blocked_mps)
                )

                has_map = lane_id in lane_to_sigidx
                if has_map and _is_lane_green(lane_id):
                    inst_block = _clip01(occ) * slow_factor * _clip01(fill_raw)
                    b_ema = (1.0 - alpha_b) * b_ema + alpha_b * inst_block
                elif not has_map:
                    inst_block = 0.5 * _clip01(occ) * slow_factor * _clip01(fill_raw)
                    b_ema = (1.0 - alpha_b) * b_ema + alpha_b * inst_block
                else:
                    b_ema = (1.0 - alpha_b) * b_ema

                cache[f"bema:{lane_id}"] = b_ema
                blocked = _clip01(b_ema)

                is_pocket = 1.0 if pocket_m is not None else 0.0
                w = (1.0 / math.sqrt(max(1.0, storage_m))) * (
                    1.25 if is_pocket > 0.5 else 1.0
                )

                sum_w += w
                sum_fill += w * fill
                sum_tts += w * tts_score
                sum_block += w * blocked

            if sum_w <= 1e-9:
                out.extend([0.0, 0.0, 0.0])
            else:
                out.extend(
                    [
                        _clip01(sum_fill / sum_w),
                        _clip01(sum_tts / sum_w),
                        _clip01(sum_block / sum_w),
                    ]
                )

        return out

    # ----------------------------
    # Expert4: blockage-adjusted turnbay capacity state (multi-pocket + EWMA)
    # ----------------------------
    def _blockage_adjusted_turnbay_capacity_state(
        intersection_encoding: str,
        lane_ids_by_leg: dict[str, list[str]],
        lanespecs_by_leg: dict[str, list[dict]],
    ) -> list[float]:
        # Persistent EWMA cache (preserves expert semantics).
        if not hasattr(tsc_isolated_intersection_feature_vector, "_block_ewma_cache"):
            tsc_isolated_intersection_feature_vector._block_ewma_cache = {}
        ewma_cache: dict[str, float] = tsc_isolated_intersection_feature_vector._block_ewma_cache  # type: ignore[attr-defined]

        def _lane_jam_ratio(lane_id: str) -> tuple[float, int]:
            try:
                lane_len = float(traci.lane.getLength(lane_id))
            except Exception:
                return 0.0, 0
            try:
                halt = int(traci.lane.getLastStepHaltingNumber(lane_id))
            except Exception:
                halt = 0

            jam_m = None
            for attr in ("getLastStepJamLengthMeters", "getJamLengthMeters"):
                try:
                    jam_m = float(getattr(traci.lane, attr)(lane_id))
                    break
                except Exception:
                    jam_m = None
            if jam_m is None:
                jam_m = min(lane_len, halt * jam_spacing_m)

            jam_ratio = (
                0.0 if lane_len <= 1e-6 else max(0.0, min(1.0, jam_m / lane_len))
            )
            return jam_ratio, halt

        risks: list[float] = []
        for leg in ["N", "E", "S", "W"]:
            lane_ids = lane_ids_by_leg.get(leg, [])
            lane_specs = lanespecs_by_leg.get(leg, [])
            n = min(len(lane_ids), len(lane_specs))
            if n == 0:
                raw_risk = 0.0
            else:
                jam_ratios: list[float] = []
                halts: list[int] = []
                for i in range(n):
                    r, h = _lane_jam_ratio(lane_ids[i])
                    jam_ratios.append(r)
                    halts.append(h)

                total_halt = sum(halts)

                # Identify all dedicated pocket lanes (supports multiple pockets).
                pocket_items: list[tuple[int, float]] = []
                for i in range(n):
                    moves = lane_specs[i]["moves"]
                    pl = lane_specs[i].get("pocket_len", None)
                    if pl is not None and moves in ({"L"}, {"R"}):
                        pocket_items.append((i, float(pl)))
                pocket_idxs = {i for i, _ in pocket_items}

                # Spillback proxy: exclude pocket-only lanes when possible.
                spillback_candidates = [
                    jam_ratios[i] for i in range(n) if i not in pocket_idxs
                ]
                spillback = (
                    max(spillback_candidates)
                    if spillback_candidates
                    else (max(jam_ratios) if jam_ratios else 0.0)
                )

                # Interference proxy: worst pocket overflow; else shared left-capable lane blocking.
                interference = 0.0
                if pocket_items:
                    overflows: list[float] = []
                    for idx, pocket_len in pocket_items:
                        if pocket_len <= 0.0:
                            continue
                        try:
                            lane_len = float(traci.lane.getLength(lane_ids[idx]))
                            jam_m = jam_ratios[idx] * lane_len
                        except Exception:
                            jam_m = halts[idx] * jam_spacing_m
                        overflow = max(0.0, jam_m - pocket_len) / pocket_len
                        overflows.append(max(0.0, min(1.0, overflow)))
                    interference = max(overflows) if overflows else 0.0
                else:
                    left_capable = [
                        i for i in range(n) if "L" in lane_specs[i]["moves"]
                    ]
                    if left_capable and total_halt > 0:
                        i_star = max(left_capable, key=lambda i: jam_ratios[i])
                        share = halts[i_star] / total_halt
                        interference = max(0.0, min(1.0, share * jam_ratios[i_star]))

                w_int = max(0.0, min(1.0, w_interference))
                raw_risk = max(
                    0.0, min(1.0, w_int * interference + (1.0 - w_int) * spillback)
                )

            a = max(0.0, min(1.0, alpha))
            prev = ewma_cache.get(leg, raw_risk)
            smoothed = (1.0 - a) * prev + a * raw_risk
            ewma_cache[leg] = smoothed
            risks.append(float(smoothed))

        return risks

    # ----------------------------
    # Assemble final vector (fixed, model-friendly order)
    # ----------------------------
    leg_order, bearing_by_leg, _, lanespecs_by_leg = _parse_intersection_encoding(
        intersection_encoding
    )

    # For this task batch, we keep canonical per-leg ordering in the output to match expert modules.
    canonical = ["N", "E", "S", "W"]
    if set(leg_order) == set(canonical):
        leg_order = canonical

    lane_ids_by_leg = _infer_lane_ids_by_leg(
        tls_id, leg_order, bearing_by_leg, lanespecs_by_leg
    )

    _, green_vec = _green_band_reliability(
        leg_order, lane_ids_by_leg, lanespecs_by_leg
    )  # [N,E,S,W,global]
    exposure_vec = _permissive_left_exposure(
        intersection_encoding,
        lane_ids_by_leg,
        leg_order,
        bearing_by_leg,
        lanespecs_by_leg,
    )  # [N,E,S,W]
    spillback_vec = _spillback_time_to_storage_features(
        intersection_encoding, lane_ids_by_leg, tls_id, lanespecs_by_leg
    )  # [N_fill,N_tts,N_blk,...,W_blk]
    blockage_vec = _blockage_adjusted_turnbay_capacity_state(
        intersection_encoding, lane_ids_by_leg, lanespecs_by_leg
    )  # [N,E,S,W]

    # Final vector layout:
    #  - 5: green reliability [N,E,S,W,global]
    #  - 4: permissive-left exposure [N,E,S,W]
    #  - 12: spillback [N_fill,N_tts,N_block, E_fill,..., W_block]
    #  - 4: blockage risk [N,E,S,W]
    return (
        list(map(float, green_vec))
        + list(map(float, exposure_vec))
        + list(map(float, spillback_vec))
        + list(map(float, blockage_vec))
    )
