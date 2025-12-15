from __future__ import annotations
import math, random, datetime
from typing import Dict, List, Tuple, Any, Optional

from ROAD_CONFIG import DEFAULT_ROAD_CONFIG, DEFAULT_GLOBAL_CONSTANTS


# -----------------------------
# Utilities
# -----------------------------
def kmh_to_mps(v_kmh: float) -> float: return v_kmh / 3.6
def mps_to_kmh(v_mps: float) -> float: return v_mps * 3.6
def clamp(x: float, lo: float, hi: float) -> float: return max(lo, min(hi, x))

def weighted_choice(weights: Dict[str, float], rng: random.Random) -> str:
    total = sum(max(0.0, w) for w in weights.values())
    if total <= 0: return rng.choice(list(weights.keys()))
    r, cum = rng.random()*total, 0.0
    for k, w in weights.items():
        cum += max(0.0, w)
        if r <= cum: return k
    return list(weights.keys())[-1]


# -----------------------------
# Safe gap physics (shared by generator & your validator)
# -----------------------------
def compute_safe_gap_b2b_m(
    v_back_mps: float,
    v_front_mps: float,
    back_type: str,
    front_type: str,
    constants: Dict[str, Any],
    gap_scale: float = 1.0
) -> float:
    vt = constants["vehicle_types"]
    safety_buffer = constants.get("safety_buffer_m", 1.0)
    min_gap = vt[back_type]["min_gap_m"]
    t_react = vt[back_type]["reaction_time_s"]
    a_back  = max(0.1, vt[back_type]["max_decel_mps2"])
    a_front = max(0.1, vt[front_type]["max_decel_mps2"])

    physical_gap = v_back_mps*t_react + (v_back_mps**2)/(2*a_back) - (v_front_mps**2)/(2*a_front)
    base = max(min_gap, physical_gap + safety_buffer)
    return max(0.0, base * gap_scale)

# =============================
#  NEW: safety sweep & reporting
# =============================
def collect_violations(items: List[Tuple[float, Dict[str, Any]]],
                       phase: str, const: Dict[str, Any]) -> List[Dict[str, Any]]:
    """items sorted by s. Returns list of violating pairs (bumper gap < required)."""
    gap_scale = const["phase_params"][phase]["gap_scale"]
    out = []
    items_sorted = sorted(items, key=lambda t: t[0])
    for i in range(len(items_sorted)-1):
        s_b, back = items_sorted[i]
        s_f, front = items_sorted[i+1]
        gap = (s_f - front["length_m"]/2.0) - (s_b + back["length_m"]/2.0)
        req = compute_safe_gap_b2b_m(
            kmh_to_mps(back["speed_kmh"]), kmh_to_mps(front["speed_kmh"]),
            back["type"], front["type"], const, gap_scale
        )
        if gap + 1e-6 < req:
            out.append({
                "back_id": back["id"], "front_id": front["id"],
                "gap_m": round(gap,2), "req_m": round(req,2)
            })
    return out

def sweep_enforce_safety(items: List[Tuple[float, Dict[str, Any]]],
                         phase: str,
                         const: Dict[str, Any],
                         window_m: float,
                         mode: str = "shift_front",   # "shift_front" | "prune_front"
                         eps: float = 0.2) -> Tuple[List[Tuple[float, Dict[str, Any]]], Dict[str,int]]:
    """
    Ensure consecutive pairs satisfy safe gap. If not:
      - shift_front: move the front forward by (deficit + eps); drop if out of window.
      - prune_front: drop the front vehicle.
    Returns (new_items, stats).
    """
    stats = {"shifted": 0, "pruned": 0}
    gap_scale = const["phase_params"][phase]["gap_scale"]
    items = sorted(items, key=lambda t: t[0])
    i = 1
    while i < len(items):
        s_b, back = items[i-1]
        s_f, front = items[i]
        gap = (s_f - front["length_m"]/2.0) - (s_b + back["length_m"]/2.0)
        req = compute_safe_gap_b2b_m(
            kmh_to_mps(back["speed_kmh"]), kmh_to_mps(front["speed_kmh"]),
            back["type"], front["type"], const, gap_scale
        )
        deficit = req - gap
        if deficit <= 0.0 + 1e-6:
            i += 1
            continue
        if mode == "shift_front":
            s_new = s_f + deficit + eps
            if abs(s_new) > window_m:
                items.pop(i); stats["pruned"] += 1
                continue
            front["coord"][1] = round(s_new, 2)
            items[i] = (s_new, front)
            stats["shifted"] += 1
            # re-check this front vs. its next neighbor
            i += 1
        else:  # prune_front
            items.pop(i)
            stats["pruned"] += 1
            # do not advance i; same 'back' compares with the next 'front'
    return items, stats

# =============================
#  SCENE GENERATOR (integrated)
# =============================
def generate_traffic(
    spec: Dict[str, Any],
    road_config: Dict[str, Any] = None,
    constants: Dict[str, Any] = None,
    seed: Optional[int] = 17
) -> Dict[str, Any]:
    """
    Generate an ego-centric traffic snapshot with *safety-aware* spacing.
    IMPORTANT: Distances in output are center-to-center from the ego (s=0).
    Safe distances are enforced/built using bumper-to-bumper math.

    --- SPEC KEYS ---
    phase: "free"|"sync"|"jam"
    window_m: float
    n_vehicles: int
    avg_speed_kmh_by_type: dict(type->avg speed)
    type_mix: dict(type->weight) for driving lanes
    ego: {"lane": int, "speed_kmh": float, "type": str}
    allow_emergency_lane: bool
    lane_blacklist: list[int]

    # NEW: safety handles
    safety_enforce: {
        "enabled": True,
        "mode": "shift_front",    # or "prune_front"
        "eps_m": 0.2
    }
    """
    
    rng = random.Random(seed)
    road = (road_config or DEFAULT_ROAD_CONFIG).copy()
    const = (constants or DEFAULT_GLOBAL_CONSTANTS).copy()
    vt = const["vehicle_types"]

    phase = spec["phase"]
    phase_params = const["phase_params"][phase]
    window_m = float(spec["window_m"])
    n_target = int(spec["n_vehicles"])
    lane_activity_prob = phase_params.get("lane_activity_prob", {})   # >>> NEW

    type_mix = spec.get("type_mix") or {"sedan": 0.5, "suv": 0.25, "truck": 0.2, "van": 0.03, "bus": 0.01, "motorcycle": 0.01}
    avg_speed_kmh_by_type = spec["avg_speed_kmh_by_type"]
    ego     = spec.get("ego") or {"lane": 2, "speed_kmh": 100.0, "type": "sedan"}
    allow_emergency_lane = bool(spec.get("allow_emergency_lane", False))
    lane_blacklist = set(spec.get("lane_blacklist", []))

    # NEW: safety handles (with defaults)
    # -----------------------------------
    safety_cfg = spec.get("safety_enforce", {})
    safety_enabled = bool(safety_cfg.get("enabled", True))             # NEW
    safety_mode    = str(safety_cfg.get("mode", "shift_front"))        # NEW
    safety_eps     = float(safety_cfg.get("eps_m", 0.2))               # NEW

    assert ego["type"] in vt
    assert ego["lane"] in [l["id"] for l in road["lanes"]]
    assert window_m > 0 and n_target >= 0

    # Helpers
    # -----------------------------------
    def lane_by_id(lane_id: int) -> Dict[str, Any]:
        for l in road["lanes"]:
            if l["id"] == lane_id: return l
        raise KeyError(lane_id)

    def lane_is_driving(lane_id: int) -> bool:
        return lane_by_id(lane_id)["type"] == "driving"
    
    def lane_seed_type(lane_id: int) -> str:
        return "maintenance" if lane_by_id(lane_id)["type"] == "emergency" else ego["type"]

    def lane_seed_speed_kmh(lane_id: int) -> float:
        lane = lane_by_id(lane_id)
        limit = lane["speed_limit_kmh"]
        if lane["type"] == "emergency":
            cap = min(limit, phase_params.get("emergency_speed_cap_kmh", limit))
            base = spec["avg_speed_kmh_by_type"].get("maintenance", cap)
            return clamp(base, 0.0, cap)
        # driving lanes
        if lane_id == ego["lane"]:
            return float(ego["speed_kmh"])
        bias = phase_params["lane_speed_bias"].get(lane_id, 1.0)
        base = spec["avg_speed_kmh_by_type"].get("sedan", limit) * bias
        return clamp(base, 0.0, limit)


    # Candidate type weights per lane
    lane_type_weights: Dict[int, Dict[str, float]] = {}
    for l in road["lanes"]:
        if l["id"] in lane_blacklist: continue
        if l["type"] == "driving":
            lane_type_weights[l["id"]] = type_mix.copy()
        else:
            # emergency lane: prefer spec override, else phase default
            lane_type_weights[l["id"]] = spec.get("emergency_lane_mix_override", 
                                                phase_params["emergency_lane_mix"]).copy()  # >>> CHANGED
    # Lane set to fill
    lane_ids = sorted([lid for lid in lane_type_weights.keys()
                       if allow_emergency_lane or lane_is_driving(lid)])
    if const.get("allow_random_lane_skips", True):
        rng.shuffle(lane_ids)

    # Speed sampler
    speed_jitter = const.get("speed_jitter_kmh", 3.0)
    follow_delta_max = phase_params["follow_speed_delta_max_kmh"]
    gap_scale = phase_params["gap_scale"]

    def sample_speed_kmh(vtype: str, lane_id: int) -> float:
        lane = lane_by_id(lane_id)
        limit = lane["speed_limit_kmh"]
        base_avg = avg_speed_kmh_by_type.get(vtype, 80.0)
        jitter = const.get("speed_jitter_kmh", 3.0)

        if lane["type"] == "emergency":                                    # >>> NEW: dedicated policy
            cap = min(limit, phase_params.get("emergency_speed_cap_kmh", limit))
            v = base_avg + rng.uniform(-0.5*jitter, 0.5*jitter)            # small jitter around your avg
            v = clamp(v, 0.0, cap)
            # separate stopped fraction for emergency lane
            if rng.random() < phase_params.get("emergency_stopped_fraction",
                                            phase_params.get("stopped_fraction", 0.0)):
                v = 0.0
            return v

        # driving lanes (unchanged logic, but kept robust)
        bias = phase_params["lane_speed_bias"].get(lane_id, 1.0)
        if phase == "jam":
            base = min(base_avg, 20.0 + 10.0 * rng.random())
        elif phase == "sync":
            base = min(base_avg, limit * 0.9)
        else:
            base = min(base_avg, limit * 1.0)
        v = base * bias + rng.uniform(-jitter, jitter)
        v = clamp(v, 0.0, limit)  # hard speed limit obeyed
        if rng.random() < phase_params["stopped_fraction"]:
            v = 0.0
        return v

    # Placement structures
    ego_len = vt["ego"]["length_m"]
    ego_speed_mps = kmh_to_mps(ego["speed_kmh"])

    placed_by_lane: Dict[int, List[Tuple[float, Dict[str, Any]]]] = {lid: [] for lid in lane_ids}

    # ==========================================
    #  NEW A: stateful per-lane placement
    # ==========================================
    lane_state: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for lid in lane_ids:
        seed_t  = lane_seed_type(lid)            # ← 'maintenance' for emergency lane
        seed_vk = lane_seed_speed_kmh(lid)
        seed_vm = kmh_to_mps(seed_vk)
        seed_len = vt[seed_t]["length_m"]
        lane_state[lid] = {
            "front": {"s": 0.0, "type": seed_t, "speed_mps": seed_vm, "len": seed_len},
            "back":  {"s": 0.0, "type": seed_t, "speed_mps": seed_vm, "len": seed_len},
        }


    def place_one(lane_id: int, direction: int) -> bool:
        """
        Place exactly one vehicle in lane_id in the given direction:
          direction=+1 → ahead of ego chain
          direction=-1 → behind ego chain
        Enforces following speed delta and safe b2b gap relative to the chain's last vehicle.
        """
        # >>> NEW: gate placement attempts by lane activity probability
        p = lane_activity_prob.get(lane_id, 1.0)
        if rng.random() > p:
            return False
        prev = lane_state[lane_id]["front" if direction == +1 else "back"]
        vtype = weighted_choice(lane_type_weights[lane_id], rng)
        # # Prefer proper emergency vehicles on emergency lane
        # if lane_by_id(lane_id)["type"] == "emergency" and vtype != "maintenance" and rng.random() < 0.9:
        #     vtype = "maintenance"

        length = vt[vtype]["length_m"]
        v_kmh = sample_speed_kmh(vtype, lane_id)
        v_mps = kmh_to_mps(v_kmh)

        # follow-speed delta rule (does NOT replace safe gap!)
        max_delta_mps = kmh_to_mps(follow_delta_max)
        if direction == +1:
            v_mps = max(v_mps, prev["speed_mps"] - max_delta_mps)  # leader not drastically slower
        else:
            v_mps = min(v_mps, prev["speed_mps"] + max_delta_mps)  # follower not drastically faster
        v_kmh = mps_to_kmh(v_mps)

        # safe b2b gap wrt prev roles
        if direction == +1:
            gap_b2b = compute_safe_gap_b2b_m(prev["speed_mps"], v_mps, prev["type"], vtype, const, gap_scale)
            center_spacing = prev["len"]/2.0 + gap_b2b + length/2.0
            s = prev["s"] + center_spacing
        else:
            gap_b2b = compute_safe_gap_b2b_m(v_mps, prev["speed_mps"], vtype, prev["type"], const, gap_scale)
            center_spacing = length/2.0 + gap_b2b + prev["len"]/2.0
            s = prev["s"] - center_spacing

        if abs(s) > window_m:
            return False

        veh = {
            "id": f"veh_{lane_id}_{len(placed_by_lane[lane_id])+1:04d}",
            "type": vtype,
            "lane": lane_id,
            "coord": [lane_id, round(s, 2)],
            "speed_kmh": round(v_kmh, 1),
            "length_m": round(length, 2),
        }
        placed_by_lane[lane_id].append((s, veh))
        # update chain ends
        lane_state[lane_id]["front" if direction == +1 else "back"] = {
            "s": s, "type": vtype, "speed_mps": v_mps, "len": length
        }
        return True

    # Fill by alternating lanes/directions until target or no more room
    directions = [+1, -1]
    max_cycles = max(30, n_target * 8)  # guardrail to avoid infinite loops
    total_placed_prev = -1
    for _ in range(max_cycles):
        for lid in lane_ids:
            for direction in directions:
                if sum(len(v) for v in placed_by_lane.values()) >= n_target:
                    break
                # some randomness to avoid rigid pattern
                if rng.random() < 0.65:
                    place_one(lid, direction)
        total_now = sum(len(v) for v in placed_by_lane.values())
        if total_now == total_placed_prev:
            break  # no progress this cycle
        total_placed_prev = total_now
        if total_now >= n_target:
            break

    # Flatten vehicles and (pre-sweep) collect violations per lane
    safety_report = {"violations_before": 0, "violations_after": None, "shifted": 0, "pruned": 0, "mode": None}
    if safety_enabled:
        all_violations_before = 0
        for lid in lane_ids:
            vios = collect_violations(placed_by_lane[lid], phase, const)
            all_violations_before += len(vios)
        safety_report["violations_before"] = all_violations_before

    # ==========================================
    #  NEW B: post-generation safety sweep
    # ==========================================
    if safety_enabled:
        safety_report["mode"] = safety_mode
        for lid in lane_ids:
            items, stats = sweep_enforce_safety(placed_by_lane[lid], phase, const, window_m,
                                                mode=safety_mode, eps=safety_eps)
            placed_by_lane[lid] = items
            safety_report["shifted"] += stats["shifted"]
            safety_report["pruned"]  += stats["pruned"]

        # recompute remaining violations
        all_violations_after = 0
        for lid in lane_ids:
            vios = collect_violations(placed_by_lane[lid], phase, const)
            all_violations_after += len(vios)
        safety_report["violations_after"] = all_violations_after

    # Gather, sort by |s| and trim to (up to) target count
    all_vehicles = []
    for lid, lst in placed_by_lane.items():
        all_vehicles.extend(lst)
    all_vehicles.sort(key=lambda x: (abs(x[0]), -1 if x[0] > 0 else 1))
    all_vehicles = all_vehicles[:n_target]
    vehicles_out = [v for _, v in all_vehicles]

    out = {
        "ego": {"lane": ego["lane"], "speed_kmh": float(ego["speed_kmh"]), "type": ego["type"]},
        "road": road,
        "meta": {
            "phase": phase,
            "window_m": float(window_m),
            "n_requested": n_target,
            "n_emitted": len(vehicles_out),
            "generated_at": datetime.datetime.now().isoformat() + "Z",
            "seed": seed,
            "notes": "Distances are center-to-center; safe distances enforced via stateful placement + safety sweep.",
            # NEW: surface safety info to help you debug/measure strictness
            "safety_report": safety_report,     # NEW
        },
        "vehicles": vehicles_out,
    }
    return out


spec = {
    "phase": "jam", # free | sync | jam
    "window_m": 300,
    "n_vehicles": 40,
    # "avg_speed_kmh_by_type": {
    #     "sedan":120,"suv":120,"truck":85,"bus":90,"van":120,"motorcycle":120,"maintenance":10
    # },
    # "avg_speed_kmh_by_type": {
    #     "sedan":110,"suv":110,"truck":80,"bus":80,"van":100,"motorcycle":120,"maintenance":10
    # },
    "avg_speed_kmh_by_type": {
        "sedan":80,"suv":80,"truck":50,"bus":60,"van":70,"motorcycle":80,"maintenance":10
    },
    "type_mix": {"sedan":0.4,"suv":0.2,"truck":0.2,"van":0.1,"bus":0.05,"motorcycle":0.05,"maintenance":0.00},
    "ego": {"lane":2,"speed_kmh":60.0,"type":"sedan"},
    "allow_emergency_lane": True,

    # NEW: safety enforcement (matches your validator)
    "safety_enforce": {
        "enabled": True,
        "mode": "shift_front",  # or "prune_front"
        "eps_m": 0.2
    }
}

snapshot = generate_traffic(spec, road_config=DEFAULT_ROAD_CONFIG, constants=DEFAULT_GLOBAL_CONSTANTS, seed=None)
print(snapshot["meta"]["safety_report"])
scene_name = "to_be_saved"
import yaml

with open(f"{scene_name}.yaml", "w") as f:
    yaml.safe_dump(snapshot, f, sort_keys=False, allow_unicode=True)
# with open("demo_scene3.json", "w") as f:
#     import json
#     json.dump(snapshot, f, indent=2)
# -> {'violations_before': X, 'violations_after': 0, 'shifted': Y, 'pruned': Z, 'mode': 'shift_front'}
