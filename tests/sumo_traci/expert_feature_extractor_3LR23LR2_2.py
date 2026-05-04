def tsc_isolated_intersection_feature_vector(
    tls_id: str,
    *,
    cache: dict | None = None,
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
    # -------- expert2: permissive-left conflict exposure --------
    zone_distance_m: float = 80.0,
    tau_s: float = 4.0,
    volume_scale_k: float = 3.0,
    v_min_mps: float = 0.5,
    p_left_given_LT: float = 0.35,
    p_through_given_TR: float = 0.70,
    # -------- expert3: spillback / gating / actionability --------
    t_max_s: float = 120.0,
    avg_standstill_spacing_m: float = 7.5,
    v_blocked_mps: float = 1.0,
    blocked_persist_steps: int = 5,  # kept for API parity; expert3 no longer uses a hard counter
    # -------- expert4: blockage-adjusted turnbay capacity --------
    alpha: float = 0.35,
    jam_spacing_m: float = 7.5,
    w_interference: float = 0.7,
    **kwargs,
) -> list[float]:
    """
    Combine four expert feature modules for a single isolated intersection into one flat numeric vector:
    green-band reliability (per leg + global), permissive-left conflict pressure/confidence/criticality,
    spillback service/gating/freshness/actionability, and EWMA blockage risk. Infers inbound lane mapping
    from tls_id and geometry at runtime. `cache` holds temporal expert state and is owned/reset by caller.

    Output layout (fixed for N/E/S/W scenario):
      [green_N, green_E, green_S, green_W, green_global,
       pl_N_eff, pl_N_conf, pl_N_crit, ..., pl_W_eff, pl_W_conf, pl_W_crit,
       N_svc, N_gate, N_gfresh, N_act, ..., W_svc, W_gate, W_gfresh, W_act,
       block_N, block_E, block_S, block_W]
    """
    import math
    import re
    import libsumo as traci

    # ----------------------------
    # Cache root (caller-managed lifetime/reset)
    # ----------------------------
    if cache is None:
        # Optional fallback for convenience only; caller-provided cache is recommended.
        if not hasattr(tsc_isolated_intersection_feature_vector, "_cache"):
            tsc_isolated_intersection_feature_vector._cache = {}
        cache = tsc_isolated_intersection_feature_vector._cache  # type: ignore[attr-defined]

    root = cache.setdefault("_tsc_iso_feat", {})
    # Namespace by TLS + encoding to avoid accidental cross-intersection state mixing.
    ns_key = (str(tls_id), str(intersection_encoding))
    ns = root.setdefault(ns_key, {})
    state_cache = ns.setdefault("state", {})  # temporal expert states
    geom_cache = ns.setdefault("geom", {})  # parsed geometry + lane map cache

    # ----------------------------
    # Unified parsing / geometry helpers
    # ----------------------------
    def _parse_intersection_encoding(enc: str):
        """
        Returns:
          leg_order_appearance: list[str]
          bearing_by_leg: dict[str,float]
          offsets_by_leg: dict[str,tuple[float,float]]
          lanespecs_by_leg: dict[str,list[dict]]  # each: perm, moves, pocket_len
        """
        s = enc.strip()
        if s.startswith("{"):
            s = s[1:]
        if s.endswith("}"):
            s = s[:-1]

        parts = [p.strip() for p in s.split(";") if p.strip()]
        leg_order_appearance: list[str] = []
        bearing_by_leg: dict[str, float] = {}
        offsets_by_leg: dict[str, tuple[float, float]] = {}
        lanespecs_by_leg: dict[str, list[dict]] = {}

        for part in parts:
            if ":" not in part:
                continue
            head, lane_str = part.split(":", 1)
            head = head.strip()
            lane_str = lane_str.strip()

            # strip outbound "x" marker
            if lane_str.endswith("x"):
                lane_str = lane_str[:-1].strip()

            leg_name = re.split(r"[@^]", head, maxsplit=1)[0].strip()
            if leg_name not in leg_order_appearance:
                leg_order_appearance.append(leg_name)

            m_b = re.search(r"@(\d+)", head)
            bearing_by_leg[leg_name] = float(m_b.group(1)) if m_b else 0.0

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
                m = re.fullmatch(r"([A-Za-z]+)(\d+)?", tok)
                if m:
                    perm = m.group(1).upper()
                    pocket_len = float(m.group(2)) if m.group(2) else None
                else:
                    perm = tok.upper()
                    pocket_len = None
                specs.append(
                    {
                        "perm": perm,
                        "moves": set(perm),
                        "pocket_len": pocket_len,
                    }
                )
            lanespecs_by_leg[leg_name] = specs

        return leg_order_appearance, bearing_by_leg, offsets_by_leg, lanespecs_by_leg

    def _angular_diff_deg(a: float, b: float) -> float:
        d = (a - b) % 360.0
        return min(d, 360.0 - d)

    def _canonical_or_bearing_order(legs: list[str], bearing_by_leg: dict[str, float]) -> list[str]:
        canonical = ["N", "E", "S", "W"]
        if set(legs) == set(canonical):
            return canonical
        return sorted(legs, key=lambda k: (bearing_by_leg.get(k, 1e9), k))

    def _get_tls_signal_map(tls: str):
        """
        Returns:
          state (str),
          lane_to_sigidx (dict[str,list[int]]),
          controlled_lane_unique_order (list[str])
        """
        lane_to_sigidx: dict[str, list[int]] = {}
        controlled_unique: list[str] = []
        seen: set[str] = set()
        state = ""
        try:
            state = traci.trafficlight.getRedYellowGreenState(tls)
        except Exception:
            state = ""

        try:
            controlled = traci.trafficlight.getControlledLinks(tls)
            for i, links in enumerate(controlled):
                for link in links:
                    if not isinstance(link, (list, tuple)) or len(link) < 1:
                        continue
                    from_lane = link[0]
                    if isinstance(from_lane, (list, tuple)) and len(from_lane) >= 1:
                        from_lane = from_lane[0]
                    if isinstance(from_lane, str):
                        lane_to_sigidx.setdefault(from_lane, []).append(i)
                        if from_lane not in seen:
                            seen.add(from_lane)
                            controlled_unique.append(from_lane)
        except Exception:
            try:
                for ln in traci.trafficlight.getControlledLanes(tls):
                    if ln not in seen:
                        seen.add(ln)
                        controlled_unique.append(ln)
                    lane_to_sigidx.setdefault(ln, [])
            except Exception:
                pass

        return state, lane_to_sigidx, controlled_unique

    def _infer_lane_ids_by_leg(
        tls: str,
        leg_order_out: list[str],
        bearing_by_leg: dict[str, float],
        lanespecs_by_leg: dict[str, list[dict]],
    ) -> dict[str, list[str]]:
        """
        Infer inbound lanes controlled by TLS and order them left->right at the stopline for each leg.
        """
        _, _, uniq_from_lanes = _get_tls_signal_map(tls)

        # Precompute inbound approach bearing targets.
        inbound_bear = {leg: (bearing_by_leg.get(leg, 0.0) + 180.0) % 360.0 for leg in leg_order_out}

        # Assign each lane to closest leg by approach bearing.
        per_leg_entries: dict[str, list[tuple[str, tuple[float, float]]]] = {leg: [] for leg in leg_order_out}

        for lane_id in uniq_from_lanes:
            try:
                shape = traci.lane.getShape(lane_id)
                if not shape or len(shape) < 2:
                    continue
                (x0, y0), (x1, y1) = shape[-2], shape[-1]
                dx, dy = (x1 - x0), (y1 - y0)
                if abs(dx) + abs(dy) < 1e-9:
                    continue

                # North=0, clockwise-positive bearing.
                lane_bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

                best_leg = leg_order_out[0]
                best_d = 1e9
                for leg in leg_order_out:
                    d = _angular_diff_deg(lane_bearing, inbound_bear[leg])
                    if d < best_d:
                        best_leg, best_d = leg, d

                per_leg_entries[best_leg].append((lane_id, (float(x1), float(y1))))
            except Exception:
                continue

        # Sort lanes by lateral position at stopline relative to inbound travel direction.
        lane_ids_by_leg: dict[str, list[str]] = {}
        for leg in leg_order_out:
            entries = per_leg_entries.get(leg, [])
            if not entries:
                lane_ids_by_leg[leg] = []
                continue

            mx = sum(p[0] for _, p in entries) / len(entries)
            my = sum(p[1] for _, p in entries) / len(entries)

            th = math.radians(inbound_bear[leg])
            v = (math.sin(th), math.cos(th))  # inbound direction unit vector

            scored: list[tuple[float, str]] = []
            for lid, (x, y) in entries:
                rx, ry = (x - mx), (y - my)
                lateral = v[0] * ry - v[1] * rx  # >0 => left of travel direction
                scored.append((lateral, lid))

            scored.sort(key=lambda t: t[0], reverse=True)
            ordered = [lid for _, lid in scored]

            # Align count with encoding lane count.
            expected = len(lanespecs_by_leg.get(leg, []))
            if expected > 0:
                ordered = ordered[:expected]

            lane_ids_by_leg[leg] = ordered

        return lane_ids_by_leg

    def _edge_from_lane(lane_id: str) -> str:
        return lane_id.rsplit("_", 1)[0] if "_" in lane_id else lane_id

    def _lane_index_suffix(lane_id: str):
        try:
            return int(lane_id.rsplit("_", 1)[1])
        except Exception:
            return None

    # ----------------------------
    # Parse geometry (cached by encoding) + infer lane map (cached softly)
    # ----------------------------
    if geom_cache.get("enc_key") != intersection_encoding:
        leg_order_appearance, bearing_by_leg, offsets_by_leg, lanespecs_by_leg = _parse_intersection_encoding(
            intersection_encoding
        )
        geom_cache["enc_key"] = intersection_encoding
        geom_cache["parsed"] = (leg_order_appearance, bearing_by_leg, offsets_by_leg, lanespecs_by_leg)
        geom_cache.pop("lane_ids_by_leg", None)
    else:
        leg_order_appearance, bearing_by_leg, offsets_by_leg, lanespecs_by_leg = geom_cache["parsed"]

    leg_order = _canonical_or_bearing_order(list(leg_order_appearance), bearing_by_leg)

    # Recompute lane map each call (robust to dynamic TLS reprogramming); cheap enough for one junction.
    lane_ids_by_leg = _infer_lane_ids_by_leg(tls_id, leg_order, bearing_by_leg, lanespecs_by_leg)
    geom_cache["lane_ids_by_leg"] = lane_ids_by_leg

    # Build TLS signal mapping once and reuse across experts.
    tls_state, lane_to_sigidx, _ = _get_tls_signal_map(tls_id)

    # ----------------------------
    # Expert1: Green-band reliability (dict -> vector)
    # ----------------------------
    def _green_band_reliability() -> list[float]:
        def _build_timeline(tls: str, horizon: float):
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

            timeline = []
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

        def _merge_intervals(intervals):
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

        def _green_intervals_for_index(timeline, tls_index: int):
            intervals = []
            for a, b, st in timeline:
                if 0 <= tls_index < len(st) and st[tls_index] in ("G", "g"):
                    intervals.append((a, b))
            return _merge_intervals(intervals)

        _SQRT2 = math.sqrt(2.0)

        def _phi(z: float) -> float:
            return 0.5 * (1.0 + math.erf(z / _SQRT2))

        def _prob_in_intervals(mu: float, sigma: float, intervals):
            if sigma <= 0.0 or not intervals:
                return 0.0
            p = 0.0
            for a, b in intervals:
                p += _phi((b - mu) / sigma) - _phi((a - mu) / sigma)
            return max(0.0, min(1.0, p))

        timeline = _build_timeline(tls_id, horizon_s)
        green_cache_local: dict[int, list[tuple[float, float]]] = {}
        next_green_cache: dict[int, float] = {}

        lane_counts = {leg: len(lanespecs_by_leg.get(leg, [])) for leg in leg_order}
        leg_vehicle_probs: dict[str, list[float]] = {leg: [] for leg in leg_order}

        for leg in leg_order:
            for lane_id in lane_ids_by_leg.get(leg, []):
                try:
                    veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                except Exception:
                    continue
                if not veh_ids:
                    continue

                per_lane = []  # (veh_id, tls_index, dist, speed)
                stopped = []  # (dist, veh_id, tls_index)

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
                    if tls_index not in green_cache_local:
                        gi = _green_intervals_for_index(timeline, tls_index)
                        green_cache_local[tls_index] = gi
                        next_green_cache[tls_index] = gi[0][0] if gi else float(horizon_s)

                    intervals = green_cache_local[tls_index]
                    t_next_green = next_green_cache[tls_index]

                    v_eff = max(0.1, spd)
                    t_kin = float(dist) / v_eff

                    if vid in rank_map:
                        r = rank_map[vid]
                        mu = max(t_kin, t_next_green) + float(start_lost_s) + float(r) * float(sat_headway_s)
                    else:
                        mu = t_kin if t_kin >= t_next_green else (t_next_green + float(start_lost_s))

                    sigma = max(float(min_sigma_s), mu * float(cv))
                    p_green = _prob_in_intervals(mu, sigma, intervals)
                    leg_vehicle_probs[leg].append(p_green)

        per_leg_scores: list[float] = []
        for leg in leg_order:
            probs = leg_vehicle_probs.get(leg, [])
            n = len(probs)
            score = (sum(probs) + float(smoothing_n) * 0.5) / (n + float(smoothing_n))
            per_leg_scores.append(max(0.0, min(1.0, float(score))))

        # Lane-count-weighted global score (important for asymmetric E/W in this scenario).
        total_w = 0.0
        total_s = 0.0
        for leg, sc in zip(leg_order, per_leg_scores):
            w = float(max(1, lane_counts.get(leg, 1)))
            total_w += w
            total_s += w * sc
        global_score = max(0.0, min(1.0, total_s / total_w if total_w > 0 else 0.5))

        return per_leg_scores + [global_score]

    # ----------------------------
    # Expert2: permissive-left conflict exposure with confidence + criticality
    # (caller-managed cache; no expert-side auto-reset)
    # ----------------------------
    def _permissive_left_conflict_exposure() -> list[float]:
        pl_cache = state_cache.setdefault("expert2_plx", {})
        fast_cache: dict[str, float] = pl_cache.setdefault("fast", {})
        slow_cache: dict[str, float] = pl_cache.setdefault("slow", {})
        crit_cache: dict[str, float] = pl_cache.setdefault("crit", {})

        def _clamp01(x: float) -> float:
            return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

        def _compute_opposites() -> dict[str, str]:
            opposites: dict[str, str] = {}
            for ln in leg_order:
                target = (bearing_by_leg.get(ln, 0.0) + 180.0) % 360.0
                best, best_d = None, 1e9
                for other in leg_order:
                    if other == ln:
                        continue
                    d = _angular_diff_deg(target, bearing_by_leg.get(other, 0.0))
                    if d < best_d:
                        best, best_d = other, d
                if best is not None:
                    opposites[ln] = best
            return opposites

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
            Return one of {'l','s','r','t','?'} using route + lane links.
            For extended links, direction is typically at index 6.
            """
            try:
                route = traci.vehicle.getRoute(veh_id)
                ridx = traci.vehicle.getRouteIndex(veh_id)
                if not route or ridx is None or ridx < 0 or ridx >= len(route) - 1:
                    return "?"
                next_edge = route[ridx + 1]
                links = traci.lane.getLinks(lane_id)

                for link in links:
                    out_lane = link[0] if len(link) >= 1 else None
                    direction = link[6] if len(link) >= 7 else (link[5] if len(link) >= 6 else None)
                    if out_lane and _edge_from_lane(out_lane) == next_edge and direction:
                        return str(direction)[0].lower()

                if len(links) == 1:
                    direction = links[0][6] if len(links[0]) >= 7 else (links[0][5] if len(links[0]) >= 6 else None)
                    if direction:
                        return str(direction)[0].lower()
            except Exception:
                pass
            return "?"

        # dt for EWMA smoothing
        try:
            dt_raw = float(traci.simulation.getDeltaT())
            dt_s = dt_raw / 1000.0 if dt_raw > 50.0 else dt_raw
            dt_s = max(0.1, dt_s)
        except Exception:
            dt_s = 1.0

        # Optional phase overlap proxy (use actual TLS mapping when available).
        lane_green: dict[str, float] = {}
        if lane_to_sigidx and tls_state:
            for lane_id, idxs in lane_to_sigidx.items():
                is_g = 1.0 if any((i < len(tls_state) and tls_state[i] in ("G", "g")) for i in idxs) else 0.0
                lane_green[lane_id] = is_g

        opposites = _compute_opposites()

        # Stable order: bearing ascending (clockwise from North), fallback by name (expert semantics).
        leg_order_pl = sorted(leg_order, key=lambda leg: (bearing_by_leg.get(leg, 1e9), leg))

        lt_tts_w: dict[str, list[tuple[float, float]]] = {leg: [] for leg in leg_order}
        th_tts_w: dict[str, list[tuple[float, float]]] = {leg: [] for leg in leg_order}
        lane_support: dict[str, dict[str, float]] = {
            leg: {"wL": 0.0, "wT": 0.0, "gL": 0.0, "gT": 0.0} for leg in leg_order
        }

        for leg in leg_order:
            lane_ids = lane_ids_by_leg.get(leg, [])
            perm_list_raw = [
                {"perm": s["perm"], "dedicated_m": s.get("pocket_len", None)} for s in lanespecs_by_leg.get(leg, [])
            ]
            perm_list = perm_list_raw

            # Expert2 robust alignment fallback (useful if caller lane order is suffix-ascending SUMO order).
            idxs = [_lane_index_suffix(lid) for lid in lane_ids]
            if len(perm_list_raw) == len(lane_ids) and all(i is not None for i in idxs) and idxs == sorted(idxs):
                perm_list = list(reversed(perm_list_raw))

            # Extend zone based on left-permitting bay lengths.
            dedicated_max_lt = 0.0
            for li in perm_list:
                if li.get("dedicated_m") is not None and ("L" in li.get("perm", "")):
                    dedicated_max_lt = max(dedicated_max_lt, float(li["dedicated_m"]))
            leg_zone = max(zone_distance_m, dedicated_max_lt + 20.0) if dedicated_max_lt > 0 else zone_distance_m

            for idx, lane_id in enumerate(lane_ids):
                fallback_perm = perm_list[idx]["perm"] if idx < len(perm_list) else "?"

                # Lane-level support weights used in phase overlap confidence.
                wL_lane, wT_lane = _soft_weights_from_perm(fallback_perm)
                g = lane_green.get(lane_id, 1.0 if not lane_green else 0.0)
                lane_support[leg]["wL"] += wL_lane
                lane_support[leg]["wT"] += wT_lane
                lane_support[leg]["gL"] += g * wL_lane
                lane_support[leg]["gT"] += g * wT_lane

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

        # Multi-timescale EWMA
        hl_fast = 2.0
        hl_slow = 8.0
        a_fast = 1.0 - math.exp(-dt_s / max(hl_fast, 1e-6))
        a_slow = 1.0 - math.exp(-dt_s / max(hl_slow, 1e-6))

        # Confidence + persistent criticality
        s0 = 0.25
        N0 = 6.0
        Cmin = 0.20
        th_on = 0.55
        th_off = 0.35
        up_rate = 0.50
        down_rate = 0.25

        out: list[float] = []

        for leg in leg_order_pl:
            opp = opposites.get(leg, None)

            n_lt = sum(w for _, w in lt_tts_w.get(leg, []))
            n_opp = 0.0
            t_lt = min((tts for tts, w in lt_tts_w.get(leg, []) if w > 0.0), default=None)
            t_op = None

            if opp is not None:
                n_opp = sum(w for _, w in th_tts_w.get(opp, []))
                t_op = min((tts for tts, w in th_tts_w.get(opp, []) if w > 0.0), default=None)

            # Instantaneous pressure
            if n_lt <= 1e-9 or n_opp <= 1e-9 or t_lt is None or t_op is None:
                pressure_obs = 0.0
            else:
                volume_pressure = 1.0 - math.exp(-(n_lt * n_opp) / max(volume_scale_k, 1e-6))
                alignment = math.exp(-abs(t_lt - t_op) / max(tau_s, 1e-6))
                pressure_obs = _clamp01(volume_pressure * (0.5 + 0.5 * alignment))

            # EWMA states (persist across steps via caller cache)
            pf = float(fast_cache.get(leg, pressure_obs))
            ps = float(slow_cache.get(leg, pressure_obs))
            pf = (1.0 - a_fast) * pf + a_fast * pressure_obs
            ps = (1.0 - a_slow) * ps + a_slow * pressure_obs
            fast_cache[leg] = pf
            slow_cache[leg] = ps

            # Phase-overlap proxy: LT lane availability vs opposing-through availability
            denomL = lane_support[leg]["wL"]
            greenL = (lane_support[leg]["gL"] / denomL) if denomL > 1e-9 else 0.0

            if opp is not None:
                denomT = lane_support.get(opp, {}).get("wT", 0.0)
                gT = lane_support.get(opp, {}).get("gT", 0.0)
                greenOppT = (gT / denomT) if denomT > 1e-9 else 0.0
            else:
                greenOppT = 0.0

            phase_overlap = math.sqrt(_clamp01(greenL) * _clamp01(greenOppT)) if lane_green else 1.0

            evidence = _clamp01((n_lt + n_opp) / max(N0, 1e-6))
            stability = 1.0 - _clamp01(abs(pf - ps) / max(s0, 1e-6))
            confidence = _clamp01(evidence * stability * (0.5 + 0.5 * phase_overlap))

            # Leaky integrator with hysteresis for persistent criticality
            crit = float(crit_cache.get(leg, 0.0))
            if ps > th_on and confidence >= Cmin:
                crit = min(1.0, crit + up_rate * dt_s)
            elif ps < th_off or confidence < (0.5 * Cmin):
                crit = max(0.0, crit - down_rate * dt_s)
            crit_cache[leg] = crit

            effective_pressure = _clamp01(ps * confidence)
            out.extend([effective_pressure, confidence, _clamp01(crit)])

        return out

    # ----------------------------
    # Expert3: spillback service/gating/freshness/actionability (caller-managed cache)
    # ----------------------------
    def _spillback_time_to_storage_features() -> list[float]:
        sp_cache = state_cache.setdefault("expert3_spill", {})

        try:
            now_s = float(traci.simulation.getTime())
        except Exception:
            now_s = float(sp_cache.get("_now_fallback", 0.0) + 1.0)
            sp_cache["_now_fallback"] = now_s

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
                if i < len(tls_state) and tls_state[i] in ("G", "g"):
                    return True
            return False

        # Smooth compression to reduce zero-inflation and spikes
        beta = 6.0

        def _log_norm(x: float, xmax: float = 1.5) -> float:
            x = max(0.0, min(xmax, x))
            return _clip01(math.log1p(beta * x) / math.log1p(beta * xmax))

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

        a_fast = 0.35
        a_slow = 0.08
        tau_fresh = min(40.0, max(10.0, 0.35 * float(t_max_s)))
        gate_evidence_thr = 0.18

        out: list[float] = []
        for leg in ["N", "E", "S", "W"]:
            specs = lanespecs_by_leg.get(leg, [])
            lane_ids = lane_ids_by_leg.get(leg, [])
            k = min(len(specs), len(lane_ids))
            if k == 0:
                out.extend([0.0, 0.0, 0.0, 0.0])
                continue

            sum_w = 0.0
            sum_svc = 0.0
            sum_gate = 0.0

            last_gate_key = f"last_gate:{leg}"
            last_gate_seen = float(sp_cache.get(last_gate_key, -1e9))
            max_inst_gate = 0.0

            for i in range(k):
                lane_id = lane_ids[i]
                pocket_m = specs[i].get("pocket_len", None)

                try:
                    lane_len = float(traci.lane.getLength(lane_id))
                except Exception:
                    lane_len = 200.0

                storage_m = lane_len if pocket_m is None else min(lane_len, float(pocket_m))
                storage_m = max(1.0, storage_m)
                storage_veh = max(1.0, storage_m / max(1e-3, avg_standstill_spacing_m))

                n_in_store = float(_veh_in_storage(lane_id, lane_len, storage_m))
                fill_raw = n_in_store / storage_veh
                fill = _log_norm(fill_raw, xmax=1.5)

                # Growth of vehicles in storage zone (veh/s), lightly smoothed
                prev_n = float(sp_cache.get(f"n:{lane_id}", n_in_store))
                inst_growth = max(0.0, (n_in_store - prev_n) / dt_s)
                sp_cache[f"n:{lane_id}"] = n_in_store

                g_ema = float(sp_cache.get(f"g:{lane_id}", inst_growth))
                g_ema = (1.0 - 0.25) * g_ema + 0.25 * inst_growth
                sp_cache[f"g:{lane_id}"] = g_ema
                growth = _clip01(g_ema / 1.2)

                # Service evidence: pocket overflow/interference + approach storage pressure
                pocket_flag = 1.0 if pocket_m is not None else 0.0
                overflow = _clip01((fill_raw - 0.75) / 0.60)
                inst_service = _clip01(
                    (0.55 * fill + 0.25 * overflow + 0.20 * growth)
                    if pocket_flag > 0.5
                    else (0.75 * fill + 0.25 * growth)
                )

                sf_key, ss_key = f"sf:{lane_id}", f"ss:{lane_id}"
                svc_fast = float(sp_cache.get(sf_key, inst_service))
                svc_slow = float(sp_cache.get(ss_key, inst_service))
                svc_fast = (1.0 - a_fast) * svc_fast + a_fast * inst_service
                svc_slow = (1.0 - a_slow) * svc_slow + a_slow * inst_service
                sp_cache[sf_key] = svc_fast
                sp_cache[ss_key] = svc_slow
                svc_lane = _clip01(0.70 * svc_fast + 0.30 * svc_slow)

                # Gating evidence: receiving blockage / wasted-green proxy
                try:
                    occ = float(traci.lane.getLastStepOccupancy(lane_id)) / 100.0
                except Exception:
                    occ = 0.0
                try:
                    mean_v = float(traci.lane.getLastStepMeanSpeed(lane_id))
                except Exception:
                    mean_v = 0.0

                slow_factor = _clip01((v_blocked_mps - mean_v) / max(1e-3, v_blocked_mps))

                if not tls_state:
                    green_gate = 0.0
                else:
                    has_map = lane_id in lane_to_sigidx
                    if has_map:
                        green_gate = 1.0 if _is_lane_green(lane_id) else 0.0
                    else:
                        green_gate = 0.5  # conservative proxy if lane->signal mapping is missing

                inst_gate = _clip01(green_gate * _clip01(occ) * slow_factor * _clip01(fill_raw))
                max_inst_gate = max(max_inst_gate, inst_gate)

                gf_key, gs_key = f"gf:{lane_id}", f"gs:{lane_id}"
                gate_fast = float(sp_cache.get(gf_key, inst_gate))
                gate_slow = float(sp_cache.get(gs_key, inst_gate))
                gate_fast = (1.0 - a_fast) * gate_fast + a_fast * inst_gate
                gate_slow = (1.0 - a_slow) * gate_slow + a_slow * inst_gate
                sp_cache[gf_key] = gate_fast
                sp_cache[gs_key] = gate_slow
                gate_lane = _clip01(0.70 * gate_fast + 0.30 * gate_slow)

                # Aggregate all lanes softly (avoid dead hard-critical-lane features)
                w = (1.25 if pocket_flag > 0.5 else 1.0) * (1.0 / math.sqrt(max(1.0, storage_m)))
                sum_w += w
                sum_svc += w * svc_lane
                sum_gate += w * gate_lane

            if sum_w <= 1e-9:
                svc_leg = gate_leg = gfresh = act_leg = 0.0
            else:
                svc_leg = _clip01(sum_svc / sum_w)
                gate_leg = _clip01(sum_gate / sum_w)

                # Freshness tracks how recently gating was supported by current evidence
                if max_inst_gate >= gate_evidence_thr:
                    last_gate_seen = now_s
                    sp_cache[last_gate_key] = last_gate_seen

                gfresh = _clip01(math.exp(-(now_s - last_gate_seen) / max(1e-3, tau_fresh)))

                # Actionability downweights stale gating
                gate_eff = _clip01(gate_leg * (0.50 + 0.50 * gfresh))
                act_leg = _clip01(svc_leg * (1.0 - gate_eff))

            out.extend([svc_leg, gate_leg, gfresh, act_leg])

        return out

    # ----------------------------
    # Expert4: blockage-adjusted turnbay capacity risk (caller-managed EWMA cache)
    # ----------------------------
    def _blockage_adjusted_turnbay_capacity_state() -> list[float]:
        block_cache: dict[str, float] = state_cache.setdefault("expert4_block", {})

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

            jam_ratio = 0.0 if lane_len <= 1e-6 else max(0.0, min(1.0, jam_m / lane_len))
            return jam_ratio, halt

        out: list[float] = []

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

                # Pocket lanes may exist on both sides (e.g., L60 and R60)
                pocket_items: list[tuple[int, float]] = []
                for i in range(n):
                    moves = lane_specs[i]["moves"]
                    pl = lane_specs[i]["pocket_len"]
                    if pl is not None and moves in ({"L"}, {"R"}):
                        pocket_items.append((i, float(pl)))

                pocket_idxs = {i for i, _ in pocket_items}

                # Spillback should reflect general lanes if possible
                spillback_candidates = [jam_ratios[i] for i in range(n) if i not in pocket_idxs]
                spillback = (
                    max(spillback_candidates) if spillback_candidates else (max(jam_ratios) if jam_ratios else 0.0)
                )

                # Interference via worst pocket overflow, else shared left-capable lane blocking
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
                    left_capable = [i for i in range(n) if "L" in lane_specs[i]["moves"]]
                    if left_capable and total_halt > 0:
                        i_star = max(left_capable, key=lambda i: jam_ratios[i])
                        share = halts[i_star] / total_halt
                        interference = max(0.0, min(1.0, share * jam_ratios[i_star]))

                w_int = max(0.0, min(1.0, w_interference))
                raw_risk = max(0.0, min(1.0, w_int * interference + (1.0 - w_int) * spillback))

            a = max(0.0, min(1.0, alpha))
            prev = float(block_cache.get(leg, raw_risk))
            smoothed = (1.0 - a) * prev + a * raw_risk
            block_cache[leg] = smoothed
            out.append(float(smoothed))

        return out

    # ----------------------------
    # Compute all expert blocks and concatenate
    # ----------------------------
    green_vec = _green_band_reliability()  # 5 = [N,E,S,W,global]
    pl_vec = _permissive_left_conflict_exposure()  # 12 = 4 legs * [eff,conf,crit]
    spill_vec = _spillback_time_to_storage_features()  # 16 = 4 legs * [svc,gate,gfresh,act]
    block_vec = _blockage_adjusted_turnbay_capacity_state()  # 4 = [N,E,S,W]

    return (
        list(map(float, green_vec))
        + list(map(float, pl_vec))
        + list(map(float, spill_vec))
        + list(map(float, block_vec))
    )
