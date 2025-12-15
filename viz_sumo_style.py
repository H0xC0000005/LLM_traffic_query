
import json
import random
import yaml

import os, json, csv
from pathlib import Path
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from typing import Dict, Any, List, Tuple, Set, Optional

from ROAD_CONFIG import DEFAULT_ROAD_CONFIG, DEFAULT_GLOBAL_CONSTANTS

# with open("demo_scene3.json", "r") as f:
#     user_snapshot = json.load(f)
# with open("viz_sumo_style.py", "r", encoding="utf-8") as f:
#     user_snapshot = yaml.safe_load(f)

# ----- Appearance -----
DRIVING_BG = "#e9e9ea"
EMERGENCY_BG = "#f4dada"
LANE_SEP = "#bdbdbd"

TYPE_COLORS = {
    "sedan": "#1f77b4",
    "suv": "#2ca02c",
    "truck": "#ff7f0e",
    "bus": "#9467bd",
    "van": "#8c564b",
    "motorcycle": "#17becf",
    "maintenance": "#d62728",
    "ego": "#6b6b6b",
}

# ----- Safety constants (should match your generator) -----
VALIDATION_CONSTANTS = DEFAULT_GLOBAL_CONSTANTS

def _kmh_to_mps(v): return v/3.6

def _clamp(x, lo, hi):  # >>> NEW
    return max(lo, min(hi, x))

def _safe_gap_b2b(v_back_mps, v_front_mps, back_type, front_type, constants, gap_scale=1.0):
    vt = constants["vehicle_types"]
    safety_buffer = constants.get("safety_buffer_m", 1.0)
    min_gap = vt[back_type]["min_gap_m"]
    t_react = vt[back_type]["reaction_time_s"]
    a_back = max(0.1, vt[back_type]["max_decel_mps2"])
    a_front = max(0.1, vt[front_type]["max_decel_mps2"])
    physical_gap = v_back_mps*t_react + (v_back_mps**2)/(2*a_back) - (v_front_mps**2)/(2*a_front)
    base = max(min_gap, physical_gap + safety_buffer)
    return max(0.0, base * gap_scale)

def _ensure_parent_dir(path: Path):
    parent = path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

def draw_sumo_style_panels(
    snapshot: Dict[str, Any],
    validate: bool = True,
    figsize: Tuple[float,float] = (16,7),
    lane_height: float = 1.3,
    lane_vpad: float = 0.4,
    font_size: int = 10,
    vehicle_height_frac: float = 0.75,
    save_path: Optional[str] = None,
    violations_out: Optional[str] = None,   # path to write CSV/JSON of violations
    print_violations: bool = True,
    label_mode: str = "stagger",            # >>> NEW: "stagger" or "center"
    label_jitter_frac: float = 0.15,        # >>> NEW: vertical jitter as fraction of vehicle height
    label_x_jitter_m: float = 0.2,          # >>> NEW: small horizontal jitter in meters
    label_bbox: bool = True                 # >>> NEW: white background box behind text
) -> List[Dict[str, Any]]:
    """
    Renders a SUMO-style lane panel plot and optionally returns/saves a list of violations.
    A 'violation' is when bumper gap < required safe gap per physics + min gap + buffer.
    """
    road = snapshot["road"]
    meta = snapshot.get("meta", {})
    phase = meta.get("phase", "sync")
    window_m = float(meta.get("window_m", 200.0))
    base_seed = int(meta.get("seed", 0) if meta.get("seed", 0) is not None else 0) 

    # Lanes left->right by id, shown top->bottom
    lanes_sorted = sorted(road["lanes"], key=lambda d: d["id"])
    lane_ids = [l["id"] for l in lanes_sorted]
    lane_types = {l["id"]: l["type"] for l in lanes_sorted}

    nlanes = len(lane_ids)
    total_h = nlanes * (lane_height + lane_vpad) + lane_vpad

    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    ax.set_xlim(-window_m, window_m)
    ax.set_ylim(0, total_h)
    ax.set_xlabel("Longitudinal position s (m)    —    ego at 0", fontsize=font_size)
    ax.grid(axis='x', linestyle=':', linewidth=0.7, alpha=0.4)
    ax.set_yticks([])

    # Title
    vehicles = snapshot.get("vehicles", [])
    ax.set_title(f"Phase: {phase} | Lanes: {nlanes} | Vehicles: {len(vehicles)}",
                 fontsize=font_size+2, pad=12)

    # Ego reference line
    ax.axvline(0, color="#666666", linewidth=1.2, alpha=0.8)

    # Lane bands
    base = total_h - lane_vpad
    lane_centers_y = {}
    lane_bounds = {}  # >>> NEW: store (y0, y1) per lane for clamping
    for lid in lane_ids:
        base -= lane_height
        y0 = base
        bg = EMERGENCY_BG if lane_types[lid] == "emergency" else DRIVING_BG
        ax.add_patch(Rectangle((-window_m, y0), 2*window_m, lane_height, facecolor=bg, edgecolor="none", zorder=0))
        ax.hlines(y0, -window_m, window_m, colors=LANE_SEP, linewidth=1.2, zorder=1)
        ax.hlines(y0 + lane_height, -window_m, window_m, colors=LANE_SEP, linewidth=1.2, zorder=1)
        ax.text(-window_m + 2, y0 + lane_height*0.82, f"Lane {lid} ({lane_types[lid]})",
                ha="left", va="top", fontsize=font_size, color="#4a4a4a")
        lane_centers_y[lid] = y0 + lane_height/2.0
        lane_bounds[lid] = (y0, y0 + lane_height)   # >>> NEW
        base -= lane_vpad

    # Prepare collections for validation (include ego!)
    per_lane: Dict[int, List[Dict[str, Any]]] = {lid: [] for lid in lane_ids}
    vt_defaults = VALIDATION_CONSTANTS["vehicle_types"]
    gap_scale = VALIDATION_CONSTANTS["phase_params"].get(phase, {"gap_scale":1.0})["gap_scale"]

    # --------- label utilities (NEW) ----------
    # cycle of offsets (fractions of vehicle height) to stagger labels per lane
    _STAGGER = [0.0, +0.28, -0.28, +0.48, -0.48]     # >>> NEW
    lane_label_idx = {lid: 0 for lid in lane_ids}    # >>> NEW

    def _label_xy(lane_id: int, s: float, y_center: float, veh_h: float, vid: str) -> Tuple[float,float]:
        """Compute label (x,y) using stagger + jitter and clamp inside the lane band."""  # >>> NEW
        if label_mode == "center":
            y = y_center
            x = s
        else:
            k = lane_label_idx[lane_id] % len(_STAGGER)
            lane_label_idx[lane_id] += 1
            y_off = _STAGGER[k] * veh_h

            # deterministic jitter
            jseed = (base_seed * 73856093) ^ (hash(vid) & 0xFFFFFFFF) ^ (lane_id * 19349663)
            rng = random.Random(jseed)
            y_off += (rng.uniform(-1.0, 1.0) * label_jitter_frac * veh_h)
            x_off = rng.uniform(-1.0, 1.0) * label_x_jitter_m

            y = y_center + y_off
            x = s + x_off

        # clamp label Y to lane band with a small margin
        y0, y1 = lane_bounds[lane_id]
        margin = 0.10 * lane_height
        y = _clamp(y, y0 + margin, y1 - margin)
        return x, y
    
    txt_bbox = dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2") if label_bbox else None  # >>> NEW

    # Draw ego with speed
    ego = snapshot["ego"]
    ego_len = vt_defaults["ego"]["length_m"]
    ego_lane = int(ego["lane"])
    ego_speed = float(ego.get("speed_kmh", 0.0))
    ego_y = lane_centers_y[ego_lane]
    ego_h = lane_height * vehicle_height_frac
    ego_color = TYPE_COLORS["ego"]
    ego_rect = Rectangle((-ego_len/2.0, ego_y - ego_h/2.0), ego_len, ego_h,
                         facecolor=ego_color, edgecolor="black", linewidth=1.5, zorder=3)
    ax.add_patch(ego_rect)
    # ax.text(0, ego_y, f"EGO {int(round(ego_speed))} km/h", ha="center", va="center",
    #         fontsize=font_size, color="black", zorder=4)
    ex, ey = _label_xy(ego_lane, 0.0, ego_y, ego_h, "ego")  # >>> NEW (use same label logic)
    ax.text(ex, ey, f"EGO {int(round(ego_speed))} km/h",
            ha="center", va="center", fontsize=font_size, color="black",
            zorder=4, bbox=txt_bbox)                          # >>> CHANGED: bbox + new position

    per_lane[ego_lane].append({
        "id": "ego",
        "x": 0.0,
        "len": ego_len,
        "speed": ego_speed,
        "type": "ego",
        "rect": ego_rect
    })

    # Draw other vehicles
    legend_types: Set[str] = set()
    for v in vehicles:
        lane_id, s = v["coord"]
        y = lane_centers_y[lane_id]
        length = float(v.get("length_m", vt_defaults.get(v["type"], {"length_m":4.5})["length_m"]))
        height = lane_height * vehicle_height_frac
        color = TYPE_COLORS.get(v["type"], "#999999")
        rect = Rectangle((s - length/2.0, y - height/2.0), length, height,
                         facecolor=color, edgecolor="black", linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        # ax.text(s, y, f"{int(round(v['speed_kmh']))} km/h", ha="center", va="center",
        #         fontsize=font_size, color="black", zorder=4)
        # >>> NEW: compute label position (stagger + jitter + clamp) and draw with background
        lx, ly = _label_xy(lane_id, float(s), y, height, v.get("id", f"{lane_id}_{s}"))
        ax.text(lx, ly, f"{int(round(v['speed_kmh']))} km/h",
                ha="center", va="center", fontsize=font_size, color="black",
                zorder=4, bbox=txt_bbox)
        per_lane[lane_id].append({
            "id": v.get("id", "n/a"),
            "x": float(s),
            "len": length,
            "speed": float(v["speed_kmh"]),
            "type": v["type"],
            "rect": rect
        })
        legend_types.add(v["type"])

    # Validate: include ego in the ordering; mark outlines + collect reasons
    violations: List[Dict[str, Any]] = []
    if validate:
        for lid, lst in per_lane.items():
            if len(lst) < 2:
                continue
            lst_sorted = sorted(lst, key=lambda d: d["x"])
            for i in range(len(lst_sorted)-1):
                back, front = lst_sorted[i], lst_sorted[i+1]
                gap_b2b = (front["x"] - front["len"]/2.0) - (back["x"] + back["len"]/2.0)
                req = _safe_gap_b2b(
                    _kmh_to_mps(back["speed"]), _kmh_to_mps(front["speed"]),
                    back["type"], front["type"], VALIDATION_CONSTANTS, gap_scale
                )
                ok = (gap_b2b + 0.5) >= req
                if not ok:
                    # emphasize by red outline
                    for item in (back, front):
                        item["rect"].set_edgecolor("red")
                        item["rect"].set_linewidth(2.2)
                    # record reason
                    violations.append({
                        "lane": lid,
                        "back_id": back["id"], "back_type": back["type"],
                        "back_speed_kmh": round(back["speed"], 1), "back_s_m": round(back["x"], 2),
                        "front_id": front["id"], "front_type": front["type"],
                        "front_speed_kmh": round(front["speed"], 1), "front_s_m": round(front["x"], 2),
                        "gap_m": round(gap_b2b, 2), "required_m": round(req, 2),
                        "reason": f"gap {gap_b2b:.1f} < required {req:.1f} (phase={phase})"
                    })

    # Legend
    handles = [Patch(facecolor=TYPE_COLORS[t], edgecolor="black", label=t) for t in sorted(legend_types)]
    handles.append(Patch(facecolor="none", edgecolor="red", label="violation"))
    handles.append(Patch(facecolor=TYPE_COLORS["ego"], edgecolor="black", label="ego"))
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.9)

    plt.tight_layout()

    # --- Robust saving (Windows/macOS/Linux) ---
    if save_path:
        out_path = Path(save_path)
        _ensure_parent_dir(out_path)
        plt.savefig(str(out_path), bbox_inches="tight")

    # --- Optional: write violations as CSV or JSON ---
    if violations_out and violations:
        vpath = Path(violations_out)
        _ensure_parent_dir(vpath)
        if vpath.suffix.lower() == ".json":
            with open(vpath, "w", encoding="utf-8") as f:
                json.dump(violations, f, indent=2)
        else:
            # default CSV
            cols = ["lane","back_id","back_type","back_speed_kmh","back_s_m",
                    "front_id","front_type","front_speed_kmh","front_s_m",
                    "gap_m","required_m","reason"]
            with open(vpath, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(violations)

    # --- Optional: print a concise violation list to console ---
    if print_violations and violations:
        print("\nViolations (gap < required):")
        for v in violations:
            print(
                f"[Lane {v['lane']}] "
                f"back={v['back_id']}({v['back_type']}@{v['back_speed_kmh']} km/h @ s={v['back_s_m']} m)  ->  "
                f"front={v['front_id']}({v['front_type']}@{v['front_speed_kmh']} km/h @ s={v['front_s_m']} m)  |  "
                f"gap={v['gap_m']} m, req={v['required_m']} m"
            )

    plt.show()
    return violations


from pathlib import Path

with open("scenes/jam_2.yaml", "r", encoding="utf-8") as f:
    user_snapshot = yaml.safe_load(f)
# pick an output folder inside your repo
out_dir = Path(__file__).parent / "plots"
img_path = out_dir / "sumo_style_panels.png"
viol_path = out_dir / "violations.csv"   # or .json

violations = draw_sumo_style_panels(
    user_snapshot,                     # your generated dict
    validate=True,
    figsize=(15,6),
    lane_height=1.0,
    lane_vpad=0.4,
    font_size=7,
    save_path=str(img_path),           # folder is created if missing
    violations_out=str(viol_path),     # writes CSV/JSON with reasons
    print_violations=True
)

print(f"Saved image to: {img_path}")
print(f"Saved violations to: {viol_path} ({len(violations)} rows)")
