def tsc_isolated_intersection_feature_vector(
    tls_id: str,
    *,
    cache: dict | None = None,
    # intersection_encoding: str = "{N@7^(+1,+1):LT|TR; E@90^(-1,+3):L60|T|T|T|R60; S@190^(+1,+1):LT|TR;W@270^(-1,+3):L60|T|T|T|R60}",
    intersection_encoding: str = "{N@8^(0,+1):L60|T|TR; E@90:LT|TR; S@197^(0,+1):L60|T|TR;W@270:LT|TR}",
    # -------- expert1: green-band reliability --------
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
    # -------- expert3: spillback / gating / freshness / actionability --------
    t_max_s: float = 120.0,
    avg_standstill_spacing_m: float = 7.5,
    v_blocked_mps: float = 1.0,
    blocked_persist_steps: int = 5,  # kept for API compatibility
    # -------- expert4: blockage-adjusted turn-bay capacity --------
    alpha: float = 0.35,
    jam_spacing_m: float = 7.5,
    w_interference: float = 0.7,
) -> list[float]:
    """
    Build a flat numeric feature vector for an isolated SUMO/libsumo intersection by combining four expert modules:
    green-band reliability, permissive-left conflict exposure, spillback/gating/actionability, and blockage-adjusted
    turn-bay risk. Supports arbitrary leg names/counts via the compact intersection encoding. All persistent state is
    stored in `cache` (caller-managed); no expert-side auto-reset heuristics are used.

    Parameters
    ----------
    tls_id : str
        SUMO/libsumo traffic light ID.
    cache : dict | None
        Mutable cache for per-step persistent feature state. If None, a temporary local cache is used.
    intersection_encoding : str
        Compact geometry/lane-permission encoding.
    ... (remaining expert tuning parameters)
        Kept configurable and compatible with peer modules.

    Returns
    -------
    list[float]
        Flat numeric vector in this dynamic layout (L = number of legs in parsed leg order):
          - Green reliability:          L + 1     [green_<leg>..., green_global]
          - Perm-left exposure state:   3L        [(eff, conf, crit) per leg]
          - Spill/gate/actionability:   4L        [(svc, gate, gfresh, act) per leg]
          - Blockage risk:              L         [block_<leg>...]
        Total size = 9L + 1.
    """
    import math
    import re
    import libsumo as traci

    if cache is None:
        cache = {}

    # Namespace all internal state under one root to avoid collisions with caller keys.
    root = cache.setdefault("_tsc_iso_feat", {})

    # ----------------------------
    # Unified compact-encoding parser (shared by all experts)
    # ----------------------------
    def _parse_intersection_encoding(enc: str) -> list[dict]:
        """
        Returns a list of legs in appearance order:
          {
            "name": str,
            "bearing": float,          # [0,359], defaults 0
            "offsets": (din,dout),     # floats, defaults (0,0)
            "lane_specs": [            # left->right in encoding
                {"perm": str, "moves": set[str], "pocket_len": float|None}
            ]
          }
        """
        s = enc.strip()
        if s.startswith("{"):
            s = s[1:]
        if s.endswith("}"):
            s = s[:-1]
        parts = [p.strip() for p in s.split(";") if p.strip()]

        legs: list[dict] = []
        for part in parts:
            if ":" not in part:
                continue
            head, lane_str = part.split(":", 1)
            head = head.strip()
            lane_str = lane_str.strip()

            # Optional outbound marker x is irrelevant to inbound feature extraction here.
            if lane_str.endswith("x"):
                lane_str = lane_str[:-1].strip()

            leg_name = re.split(r"[@^]", head, maxsplit=1)[0].strip()

            m_b = re.search(r"@(\d+)", head)
            bearing = float(m_b.group(1)) if m_b else 0.0

            din, dout = 0.0, 0.0
            m_off = re.search(r"\^\(([^)]*)\)", head)
            if m_off:
                vals = [v.strip() for v in m_off.group(1).split(",") if v.strip()]
                try:
                    din = float(vals[0]) if len(vals) >= 1 else 0.0
                    dout = float(vals[1]) if len(vals) >= 2 else din
                except Exception:
                    din, dout = 0.0, 0.0

            lane_specs: list[dict] = []
            for tok in [t.strip() for t in lane_str.split("|") if t.strip()]:
                m = re.fullmatch(r"([A-Za-z]+)(\d+)?", tok)
                if not m:
                    perm = tok.upper()
                    pocket_len = None
                else:
                    perm = m.group(1).upper()
                    pocket_len = float(m.group(2)) if m.group(2) else None
                lane_specs.append(
                    {
                        "perm": perm,
                        "moves": set(perm),
                        "pocket_len": pocket_len,
                    }
                )

            legs.append(
                {
                    "name": leg_name,
                    "bearing": bearing,
                    "offsets": (din, dout),
                    "lane_specs": lane_specs,
                }
            )

        return legs

    def _angular_diff_deg(a: float, b: float) -> float:
        d = (a - b) % 360.0
        return min(d, 360.0 - d)

    def _clip01(x: float) -> float:
        return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)

    # ----------------------------
    # Determine a canonical leg order for THIS encoding (dynamic, arbitrary names supported)
    # ----------------------------
    def _canonical_leg_order(legs: list[dict]) -> list[str]:
        # Sort by bearing clockwise; ties resolved by appearance order.
        indexed = [(i, leg["name"], float(leg["bearing"])) for i, leg in enumerate(legs)]
        indexed.sort(key=lambda t: (t[2] % 360.0, t[0]))
        return [name for _, name, _ in indexed]

    # ----------------------------
    # Infer controlled inbound lanes by leg, ordered left->right at the stop line
    # ----------------------------
    def _infer_lane_ids_by_leg(
        tls: str,
        legs: list[dict],
        leg_order: list[str],
    ) -> dict[str, list[str]]:
        bearing_by_leg = {leg["name"]: float(leg["bearing"]) for leg in legs}
        lane_specs_by_leg = {leg["name"]: leg["lane_specs"] for leg in legs}

        # Collect unique "from" lanes from controlled links (more robust than controlledLanes alone).
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

        seen = set()
        uniq_from_lanes: list[str] = []
        for ln in from_lanes:
            if ln not in seen:
                seen.add(ln)
                uniq_from_lanes.append(ln)

        # Inbound travel direction toward intersection is opposite of outbound leg bearing.
        inbound_bearing = {leg: (bearing_by_leg.get(leg, 0.0) + 180.0) % 360.0 for leg in leg_order}
        per_leg_entries: dict[str, list[tuple[str, tuple[float, float]]]] = {leg: [] for leg in leg_order}

        for lane_id in uniq_from_lanes:
            try:
                shape = traci.lane.getShape(lane_id)
                if not shape or len(shape) < 2:
                    continue
                (x0, y0), (x1, y1) = shape[-2], shape[-1]
                dx, dy = (x1 - x0), (y1 - y0)
                if abs(dx) + abs(dy) < 1e-9:
                    continue

                # Bearing convention: North=0, clockwise positive.
                lane_bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

                best_leg, best_d = leg_order[0], 1e9
                for leg in leg_order:
                    d = _angular_diff_deg(lane_bearing, inbound_bearing[leg])
                    if d < best_d:
                        best_leg, best_d = leg, d

                per_leg_entries[best_leg].append((lane_id, (float(x1), float(y1))))
            except Exception:
                continue

        # Sort lanes left->right in inbound travel frame at stop line.
        lane_ids_by_leg: dict[str, list[str]] = {leg: [] for leg in leg_order}
        for leg in leg_order:
            entries = per_leg_entries.get(leg, [])
            if not entries:
                continue

            mx = sum(p[0] for _, p in entries) / len(entries)
            my = sum(p[1] for _, p in entries) / len(entries)

            th = math.radians(inbound_bearing[leg])
            v = (math.sin(th), math.cos(th))  # unit inbound direction

            scored: list[tuple[float, str]] = []
            for lane_id, (x, y) in entries:
                rx, ry = (x - mx), (y - my)
                lateral = v[0] * ry - v[1] * rx  # >0 => driver's left
                scored.append((lateral, lane_id))

            scored.sort(key=lambda t: t[0], reverse=True)  # left -> right
            ordered = [lid for _, lid in scored]

            # Trim to encoded inbound lane count to preserve lane-spec alignment.
            expected = len(lane_specs_by_leg.get(leg, []))
            if expected > 0:
                ordered = ordered[:expected]

            lane_ids_by_leg[leg] = ordered

        return lane_ids_by_leg

    # ----------------------------
    # Shared TLS link helpers (used by expert2 + expert3)
    # ----------------------------
    def _build_lane_signal_index_map(tls: str) -> tuple[dict[str, list[int]], str]:
        lane_to_sigidx: dict[str, list[int]] = {}
        state = ""
        try:
            state = traci.trafficlight.getRedYellowGreenState(tls)
            controlled = traci.trafficlight.getControlledLinks(tls)
            for sig_idx, links in enumerate(controlled):
                for link in links:
                    if not isinstance(link, (list, tuple)) or len(link) < 1:
                        continue
                    from_lane = link[0]
                    if isinstance(from_lane, (list, tuple)) and len(from_lane) >= 1:
                        from_lane = from_lane[0]
                    if isinstance(from_lane, str):
                        lane_to_sigidx.setdefault(from_lane, []).append(sig_idx)
        except Exception:
            lane_to_sigidx = {}
            state = ""
        return lane_to_sigidx, state

    def _is_lane_green(lane_id: str, lane_to_sigidx: dict[str, list[int]], state: str) -> bool:
        for i in lane_to_sigidx.get(lane_id, []):
            if 0 <= i < len(state) and state[i] in ("G", "g"):
                return True
        return False

    def _edge_from_lane(lane_id: str) -> str:
        return lane_id.rsplit("_", 1)[0] if "_" in lane_id else lane_id

    def _lane_index_suffix(lane_id: str):
        try:
            return int(lane_id.rsplit("_", 1)[1])
        except Exception:
            return None

    # ----------------------------
    # Geometry cache (refresh when tls_id or encoding changes)
    # ----------------------------
    geom_sig = (str(tls_id), str(intersection_encoding))
    if root.get("geom_sig") != geom_sig:
        legs = _parse_intersection_encoding(intersection_encoding)
        leg_order = _canonical_leg_order(legs)
        lane_ids_by_leg = _infer_lane_ids_by_leg(tls_id, legs, leg_order)

        # Replace geometry cache atomically.
        root["geom_sig"] = geom_sig
        root["legs"] = legs
        root["leg_order"] = leg_order
        root["lane_ids_by_leg"] = lane_ids_by_leg

        # Geometry changed => clear expert persistent caches to avoid stale state contamination.
        root["_expert2"] = {}
        root["_expert3"] = {}
        root["_expert4"] = {}
    else:
        legs = root["legs"]
        leg_order = root["leg_order"]
        lane_ids_by_leg = root["lane_ids_by_leg"]

    bearing_by_leg = {leg["name"]: float(leg["bearing"]) for leg in legs}
    lane_specs_by_leg = {leg["name"]: leg["lane_specs"] for leg in legs}
    lane_counts_by_leg = {leg["name"]: len(leg["lane_specs"]) for leg in legs}

    # ----------------------------
    # Expert1: Green-band reliability (dynamic per-leg + global)
    # ----------------------------
    def _green_band_reliability() -> tuple[list[float], list[str]]:
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

        def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
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

        SQRT2 = math.sqrt(2.0)

        def _phi(z: float) -> float:
            return 0.5 * (1.0 + math.erf(z / SQRT2))

        def _prob_in_intervals(mu: float, sigma: float, intervals: list[tuple[float, float]]) -> float:
            if sigma <= 0.0 or not intervals:
                return 0.0
            p = 0.0
            for a, b in intervals:
                p += _phi((b - mu) / sigma) - _phi((a - mu) / sigma)
            return max(0.0, min(1.0, p))

        timeline = _build_timeline(tls_id, horizon_s)
        green_cache_local: dict[int, list[tuple[float, float]]] = {}
        next_green_cache_local: dict[int, float] = {}

        leg_vehicle_probs: dict[str, list[float]] = {leg: [] for leg in leg_order}

        for leg in leg_order:
            for lane_id in lane_ids_by_leg.get(leg, []):
                try:
                    veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                except Exception:
                    continue
                if not veh_ids:
                    continue

                per_lane: list[tuple[str, int, float, float]] = []  # (veh_id, tls_index, dist, speed)
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
                    if tls_index not in green_cache_local:
                        gi = _green_intervals_for_index(timeline, tls_index)
                        green_cache_local[tls_index] = gi
                        next_green_cache_local[tls_index] = gi[0][0] if gi else float(horizon_s)

                    intervals = green_cache_local[tls_index]
                    t_next_green = next_green_cache_local[tls_index]

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

        # Weighted global score (weight by encoded inbound lane count).
        total_w = 0.0
        total_s = 0.0
        for leg, sc in zip(leg_order, per_leg_scores):
            w = float(max(1, lane_counts_by_leg.get(leg, 1)))
            total_w += w
            total_s += w * sc
        global_score = max(0.0, min(1.0, total_s / total_w if total_w > 0 else 0.5))

        names = [f"green/{leg}" for leg in leg_order] + ["green/global"]
        return per_leg_scores + [global_score], names

    # ----------------------------
    # Expert2: Permissive-left conflict exposure (effective, confidence, criticality per leg)
    # ----------------------------
    def _permissive_left_conflict_exposure(state_cache: dict) -> tuple[list[float], list[str]]:
        def _compute_opposites() -> dict[str, str]:
            opposites: dict[str, str] = {}
            for leg in leg_order:
                target = (bearing_by_leg.get(leg, 0.0) + 180.0) % 360.0
                best, best_d = None, 1e9
                for other in leg_order:
                    if other == leg:
                        continue
                    d = _angular_diff_deg(target, bearing_by_leg.get(other, 0.0))
                    if d < best_d:
                        best, best_d = other, d
                if best is not None:
                    opposites[leg] = best
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
            # Returns one of {'l','s','r','t','?'}.
            try:
                route = traci.vehicle.getRoute(veh_id)
                ridx = traci.vehicle.getRouteIndex(veh_id)
                if not route or ridx is None or ridx < 0 or ridx >= len(route) - 1:
                    return "?"
                next_edge = route[ridx + 1]
                links = traci.lane.getLinks(lane_id)
                for link in links:
                    out_lane = link[0] if len(link) >= 1 else None
                    # libsumo extended link direction commonly at index 6; fallback to 5.
                    direction = None
                    if len(link) >= 7:
                        direction = link[6]
                    elif len(link) >= 6:
                        direction = link[5]
                    if out_lane and _edge_from_lane(out_lane) == next_edge and direction:
                        return str(direction)[0].lower()
                if len(links) == 1:
                    l0 = links[0]
                    direction = l0[6] if len(l0) >= 7 else (l0[5] if len(l0) >= 6 else None)
                    if direction:
                        return str(direction)[0].lower()
            except Exception:
                pass
            return "?"

        fast_state = state_cache.setdefault("fast", {})
        slow_state = state_cache.setdefault("slow", {})
        crit_state = state_cache.setdefault("crit", {})

        try:
            dt_raw = float(traci.simulation.getDeltaT())
            dt_s = dt_raw / 1000.0 if dt_raw > 50.0 else dt_raw
            dt_s = max(0.1, dt_s)
        except Exception:
            dt_s = 1.0

        lane_to_sigidx, tls_state = _build_lane_signal_index_map(tls_id)
        lane_green_known = bool(tls_state)

        # Lane green weights for a phase-overlap proxy
        lane_support = {leg: {"wL": 0.0, "wT": 0.0, "gL": 0.0, "gT": 0.0} for leg in leg_order}

        opposites = _compute_opposites()

        # Weighted LT and opposing-through arrivals near conflict zone.
        lt_tts_w: dict[str, list[tuple[float, float]]] = {leg: [] for leg in leg_order}
        th_tts_w: dict[str, list[tuple[float, float]]] = {leg: [] for leg in leg_order}

        for leg in leg_order:
            lane_ids = lane_ids_by_leg.get(leg, [])
            specs_raw = lane_specs_by_leg.get(leg, [])

            # Align lane-spec ordering to SUMO lane suffix convention when visible.
            specs = specs_raw
            idxs = [_lane_index_suffix(lid) for lid in lane_ids]
            if len(specs_raw) == len(lane_ids) and all(i is not None for i in idxs):
                if idxs == sorted(idxs):
                    # Many SUMO nets use increasing suffix = right->left; encoding is left->right.
                    specs = list(reversed(specs_raw))

            # Expand observation zone using any lane that permits left and has a coded pocket length.
            dedicated_max_lt = 0.0
            for spec in specs:
                if spec.get("pocket_len") is not None and ("L" in spec.get("perm", "")):
                    dedicated_max_lt = max(dedicated_max_lt, float(spec["pocket_len"]))
            leg_zone = max(zone_distance_m, dedicated_max_lt + 20.0) if dedicated_max_lt > 0 else zone_distance_m

            for idx, lane_id in enumerate(lane_ids):
                fallback_perm = specs[idx]["perm"] if idx < len(specs) else "?"

                wL_lane, wT_lane = _soft_weights_from_perm(fallback_perm)
                g = 1.0 if _is_lane_green(lane_id, lane_to_sigidx, tls_state) else (0.0 if lane_green_known else 1.0)
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
                        tts = remain / max(float(v_min_mps), speed)

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

        # Multi-timescale smoothing + confidence + leaky criticality
        hl_fast, hl_slow = 2.0, 8.0
        a_fast = 1.0 - math.exp(-dt_s / max(1e-6, hl_fast))
        a_slow = 1.0 - math.exp(-dt_s / max(1e-6, hl_slow))

        s0 = 0.25  # stability scale
        N0 = 6.0  # evidence scale
        Cmin = 0.20
        th_on, th_off = 0.55, 0.35
        up_rate, down_rate = 0.50, 0.25  # per second

        out: list[float] = []
        names: list[str] = []
        for leg in leg_order:
            opp = opposites.get(leg, None)

            n_lt = sum(w for _, w in lt_tts_w.get(leg, []))
            n_opp = 0.0
            t_lt = None
            t_op = None

            if n_lt > 1e-9:
                t_lt = min(tts for tts, w in lt_tts_w[leg] if w > 0.0)

            if opp is not None:
                n_opp = sum(w for _, w in th_tts_w.get(opp, []))
                if n_opp > 1e-9:
                    t_op = min(tts for tts, w in th_tts_w[opp] if w > 0.0)

            if n_lt <= 1e-9 or n_opp <= 1e-9 or t_lt is None or t_op is None:
                pressure_obs = 0.0
            else:
                volume_pressure = 1.0 - math.exp(-(n_lt * n_opp) / max(1e-6, volume_scale_k))
                alignment = math.exp(-abs(t_lt - t_op) / max(1e-6, tau_s))
                pressure_obs = _clip01(volume_pressure * (0.5 + 0.5 * alignment))

            # EWMAs
            pf = float(fast_state.get(leg, pressure_obs))
            ps = float(slow_state.get(leg, pressure_obs))
            pf = (1.0 - a_fast) * pf + a_fast * pressure_obs
            ps = (1.0 - a_slow) * ps + a_slow * pressure_obs
            fast_state[leg] = pf
            slow_state[leg] = ps

            # Phase overlap proxy (LT green on this leg AND opposing-through green on opposite leg).
            if lane_green_known:
                denomL = lane_support[leg]["wL"]
                greenL = (lane_support[leg]["gL"] / denomL) if denomL > 1e-9 else 0.0
                if opp is not None:
                    denomT = lane_support[opp]["wT"]
                    greenOppT = (lane_support[opp]["gT"] / denomT) if denomT > 1e-9 else 0.0
                else:
                    greenOppT = 0.0
                phase_overlap = math.sqrt(_clip01(greenL) * _clip01(greenOppT))
            else:
                phase_overlap = 1.0  # no TLS mapping available; avoid suppressing by missing info

            evidence = _clip01((n_lt + n_opp) / max(1e-6, N0))
            stability = 1.0 - _clip01(abs(pf - ps) / max(1e-6, s0))
            confidence = _clip01(evidence * stability * (0.5 + 0.5 * phase_overlap))

            crit = float(crit_state.get(leg, 0.0))
            if ps > th_on and confidence >= Cmin:
                crit = min(1.0, crit + up_rate * dt_s)
            elif ps < th_off or confidence < (0.5 * Cmin):
                crit = max(0.0, crit - down_rate * dt_s)
            crit_state[leg] = crit

            effective_pressure = _clip01(ps * confidence)

            out.extend([effective_pressure, confidence, _clip01(crit)])
            names.extend([f"permleft/{leg}/eff", f"permleft/{leg}/conf", f"permleft/{leg}/crit"])

        return out, names

    # ----------------------------
    # Expert3: Spillback / gating / freshness / actionability (dynamic per-leg)
    # ----------------------------
    def _spillback_gating_actionability(state_cache: dict) -> tuple[list[float], list[str]]:
        lane_to_sigidx, tls_state = _build_lane_signal_index_map(tls_id)
        tls_known = bool(tls_state)

        try:
            now_s = float(traci.simulation.getTime())
        except Exception:
            now_s = float(state_cache.get("_now_fallback", 0.0) + 1.0)
            state_cache["_now_fallback"] = now_s

        try:
            dt_raw = float(traci.simulation.getDeltaT())
            dt_s = dt_raw / 1000.0 if dt_raw > 50.0 else dt_raw
            dt_s = max(0.1, dt_s)
        except Exception:
            dt_s = 1.0

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
        names: list[str] = []

        for leg in leg_order:
            specs = lane_specs_by_leg.get(leg, [])
            lane_ids = lane_ids_by_leg.get(leg, [])
            k = min(len(specs), len(lane_ids))
            if k == 0:
                out.extend([0.0, 0.0, 0.0, 0.0])
                names.extend(
                    [
                        f"spill/{leg}/svc",
                        f"spill/{leg}/gate",
                        f"spill/{leg}/gfresh",
                        f"spill/{leg}/act",
                    ]
                )
                continue

            sum_w = 0.0
            sum_svc = 0.0
            sum_gate = 0.0
            max_inst_gate = 0.0

            last_gate_key = f"last_gate:{leg}"
            last_gate_seen = float(state_cache.get(last_gate_key, -1e9))

            for i in range(k):
                lane_id = lane_ids[i]
                pocket_m = specs[i].get("pocket_len", None)

                try:
                    lane_len = float(traci.lane.getLength(lane_id))
                except Exception:
                    lane_len = 200.0

                storage_m = lane_len if pocket_m is None else min(lane_len, float(pocket_m))
                storage_m = max(1.0, storage_m)
                storage_veh = max(1.0, storage_m / max(1e-3, float(avg_standstill_spacing_m)))

                # Vehicles currently inside the finite storage region (lane-end or pocket region).
                n_in_store = float(_veh_in_storage(lane_id, lane_len, storage_m))
                fill_raw = n_in_store / storage_veh
                fill = _log_norm(fill_raw, xmax=1.5)

                # EMA growth of occupancy inside storage region.
                prev_n = float(state_cache.get(f"n:{lane_id}", n_in_store))
                inst_growth = max(0.0, (n_in_store - prev_n) / dt_s)
                state_cache[f"n:{lane_id}"] = n_in_store

                g_ema = float(state_cache.get(f"g:{lane_id}", inst_growth))
                g_ema = (1.0 - 0.25) * g_ema + 0.25 * inst_growth
                state_cache[f"g:{lane_id}"] = g_ema
                growth = _clip01(g_ema / 1.2)

                # "Service urgency" proxy (expert3 semantics).
                pocket_flag = 1.0 if pocket_m is not None else 0.0
                overflow = _clip01((fill_raw - 0.75) / 0.60)
                if pocket_flag > 0.5:
                    inst_service = _clip01(0.55 * fill + 0.25 * overflow + 0.20 * growth)
                else:
                    inst_service = _clip01(0.75 * fill + 0.25 * growth)

                sf_key, ss_key = f"sf:{lane_id}", f"ss:{lane_id}"
                svc_fast = float(state_cache.get(sf_key, inst_service))
                svc_slow = float(state_cache.get(ss_key, inst_service))
                svc_fast = (1.0 - a_fast) * svc_fast + a_fast * inst_service
                svc_slow = (1.0 - a_slow) * svc_slow + a_slow * inst_service
                state_cache[sf_key] = svc_fast
                state_cache[ss_key] = svc_slow
                svc_lane = _clip01(0.70 * svc_fast + 0.30 * svc_slow)

                # "Gating/blockage under green" proxy.
                try:
                    occ = float(traci.lane.getLastStepOccupancy(lane_id)) / 100.0
                except Exception:
                    occ = 0.0
                try:
                    mean_v = float(traci.lane.getLastStepMeanSpeed(lane_id))
                except Exception:
                    mean_v = 0.0
                slow_factor = _clip01((float(v_blocked_mps) - mean_v) / max(1e-3, float(v_blocked_mps)))

                if tls_known:
                    if lane_id in lane_to_sigidx:
                        green_gate = 1.0 if _is_lane_green(lane_id, lane_to_sigidx, tls_state) else 0.0
                    else:
                        green_gate = 0.5  # conservative fallback if lane-link mapping misses this lane
                else:
                    green_gate = 0.0  # no TLS mapping => no positive evidence for green-gating

                inst_gate = _clip01(green_gate * _clip01(occ) * slow_factor * _clip01(fill_raw))
                max_inst_gate = max(max_inst_gate, inst_gate)

                gf_key, gs_key = f"gf:{lane_id}", f"gs:{lane_id}"
                gate_fast = float(state_cache.get(gf_key, inst_gate))
                gate_slow = float(state_cache.get(gs_key, inst_gate))
                gate_fast = (1.0 - a_fast) * gate_fast + a_fast * inst_gate
                gate_slow = (1.0 - a_slow) * gate_slow + a_slow * inst_gate
                state_cache[gf_key] = gate_fast
                state_cache[gs_key] = gate_slow
                gate_lane = _clip01(0.70 * gate_fast + 0.30 * gate_slow)

                # Dense weighted aggregation across lanes; short pockets slightly upweighted.
                w = (1.25 if pocket_flag > 0.5 else 1.0) * (1.0 / math.sqrt(max(1.0, storage_m)))
                sum_w += w
                sum_svc += w * svc_lane
                sum_gate += w * gate_lane

            if sum_w <= 1e-9:
                svc_leg = gate_leg = gfresh = act_leg = 0.0
            else:
                svc_leg = _clip01(sum_svc / sum_w)
                gate_leg = _clip01(sum_gate / sum_w)

                # Track freshness of recent gating evidence.
                if max_inst_gate >= gate_evidence_thr:
                    last_gate_seen = now_s
                    state_cache[last_gate_key] = last_gate_seen

                gfresh = _clip01(math.exp(-(now_s - last_gate_seen) / max(1e-3, tau_fresh)))

                # Downweight stale gating to avoid permanently suppressing service urgency.
                gate_eff = _clip01(gate_leg * (0.50 + 0.50 * gfresh))
                act_leg = _clip01(svc_leg * (1.0 - gate_eff))

            out.extend([svc_leg, gate_leg, gfresh, act_leg])
            names.extend(
                [
                    f"spill/{leg}/svc",
                    f"spill/{leg}/gate",
                    f"spill/{leg}/gfresh",
                    f"spill/{leg}/act",
                ]
            )

        return out, names

    # ----------------------------
    # Expert4: Blockage-adjusted turn-bay capacity risk (dynamic per-leg)
    # ----------------------------
    def _blockage_adjusted_turnbay_capacity(state_cache: dict) -> tuple[list[float], list[str]]:
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
                jam_m = min(lane_len, halt * float(jam_spacing_m))

            jam_ratio = 0.0 if lane_len <= 1e-6 else max(0.0, min(1.0, jam_m / lane_len))
            return jam_ratio, halt

        out: list[float] = []
        names: list[str] = []

        a = max(0.0, min(1.0, float(alpha)))
        w_int = max(0.0, min(1.0, float(w_interference)))

        for leg in leg_order:
            lane_ids = lane_ids_by_leg.get(leg, [])
            lane_specs = lane_specs_by_leg.get(leg, [])
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

                # Support multiple dedicated pockets on a leg (e.g., L-pocket + R-pocket).
                pocket_items: list[tuple[int, float]] = []
                for i in range(n):
                    moves = lane_specs[i]["moves"]
                    pl = lane_specs[i].get("pocket_len", None)
                    if pl is not None and moves in ({"L"}, {"R"}):
                        pocket_items.append((i, float(pl)))

                pocket_idxs = {i for i, _ in pocket_items}

                # Spillback proxy should represent through/general lanes when possible.
                spill_cands = [jam_ratios[i] for i in range(n) if i not in pocket_idxs]
                spillback = max(spill_cands) if spill_cands else (max(jam_ratios) if jam_ratios else 0.0)

                # Interference proxy: pocket overflow if pockets exist, else left-capable shared-lane blocking.
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
                            jam_m = halts[idx] * float(jam_spacing_m)
                        overflow = max(0.0, jam_m - pocket_len) / pocket_len
                        overflows.append(max(0.0, min(1.0, overflow)))
                    interference = max(overflows) if overflows else 0.0
                else:
                    left_capable = [i for i in range(n) if "L" in lane_specs[i]["moves"]]
                    if left_capable and total_halt > 0:
                        i_star = max(left_capable, key=lambda i: jam_ratios[i])
                        share = halts[i_star] / total_halt
                        interference = max(0.0, min(1.0, share * jam_ratios[i_star]))

                raw_risk = max(0.0, min(1.0, w_int * interference + (1.0 - w_int) * spillback))

            prev = float(state_cache.get(leg, raw_risk))
            smoothed = (1.0 - a) * prev + a * raw_risk
            state_cache[leg] = smoothed

            out.append(float(smoothed))
            names.append(f"block/{leg}")

        return out, names

    # ----------------------------
    # Run all experts and assemble final vector
    # ----------------------------
    e2_cache = root.setdefault("_expert2", {})
    e3_cache = root.setdefault("_expert3", {})
    e4_cache = root.setdefault("_expert4", {})

    green_vec, green_names = _green_band_reliability()
    perm_vec, perm_names = _permissive_left_conflict_exposure(e2_cache)
    spill_vec, spill_names = _spillback_gating_actionability(e3_cache)
    block_vec, block_names = _blockage_adjusted_turnbay_capacity(e4_cache)

    features = (
        list(map(float, green_vec))
        + list(map(float, perm_vec))
        + list(map(float, spill_vec))
        + list(map(float, block_vec))
    )

    # Side-channel metadata for diagnostics / tensorboard global reports (not part of feature vector).
    root["_last_feature_names"] = green_names + perm_names + spill_names + block_names
    root["_last_feature_blocks"] = {
        "green": green_vec,
        "permleft": perm_vec,
        "spill": spill_vec,
        "blockage": block_vec,
        "leg_order": list(leg_order),
    }

    return features
