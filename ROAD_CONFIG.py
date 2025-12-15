from typing import Dict, Any, List, Optional, Tuple

# -----------------------------
# Configs (editable) 
# -----------------------------

DEFAULT_ROAD_CONFIG = {
    "driving_direction": "right",  # "right" means keep right, pass left
    "overtaking_rule": "keep_right_pass_left",
    "lanes": [
        {"id": 1, "type": "driving",   "speed_limit_kmh": 130},
        {"id": 2, "type": "driving",   "speed_limit_kmh": 110},
        {"id": 3, "type": "emergency", "speed_limit_kmh": 30},
    ],
}


DEFAULT_GLOBAL_CONSTANTS = {
    # Vehicle physical/behavioral parameters
    "vehicle_types": {
        "sedan":       {"length_m": 4.6,  "min_gap_m": 2.0, "max_decel_mps2": 9.0, "reaction_time_s": 0.6},
        "suv":         {"length_m": 4.9,  "min_gap_m": 2.5, "max_decel_mps2": 8.0, "reaction_time_s": 0.9},
        "truck":       {"length_m": 12.0, "min_gap_m": 4.0, "max_decel_mps2": 5.0, "reaction_time_s": 1.2},
        "bus":         {"length_m": 13.0, "min_gap_m": 4.0, "max_decel_mps2": 5.0, "reaction_time_s": 1.0},
        "van":         {"length_m": 5.2,  "min_gap_m": 2.5, "max_decel_mps2": 7.0, "reaction_time_s": 0.9},
        "motorcycle":  {"length_m": 2.2,  "min_gap_m": 1.5, "max_decel_mps2": 8.0, "reaction_time_s": 0.5},
        "maintenance": {"length_m": 6.0,  "min_gap_m": 3.0, "max_decel_mps2": 5.0, "reaction_time_s": 1.0},
        # Ego type (used for safe distance wrt ego). Tweak if your ego is a different size/dynamics.
        "ego":         {"length_m": 4.6,  "min_gap_m": 2.0, "max_decel_mps2": 9.0, "reaction_time_s": 0.6},
    },
    # Phase-level knobs
    "phase_params": {
        # Higher speeds, larger gaps, emergency lane essentially unused
        "free": {
            "follow_speed_delta_max_kmh": 5.0,   # rear car should not exceed front by more than this if following
            "gap_scale": 1.4,                    # multiply computed safe gap to be extra conservative
            "lane_speed_bias": {1: 1.08, 2: 1.00, 3: 0.15},  # per-lane multiplier on type avg
            "stopped_fraction": 0.00,
            "emergency_lane_mix": {"maintenance": 0.98, "truck": 0.02},
            # >>> NEW: per-lane attempt probability (0..1)
            "lane_activity_prob": {1:1.0, 2:1.0, 3:0.05},
            # >>> NEW: emergency-lane speed policy
            "emergency_speed_cap_kmh": 25,           # ≤ speed limit, hard cap for emergency lane
            "emergency_stopped_fraction": 0.20       # often stationary/slow work vehicles
        },
        # Moderate speeds, tighter but still safe gaps, small chance of very slow cars
        "sync": {
            "follow_speed_delta_max_kmh": 2.0,
            "gap_scale": 1.05,
            "lane_speed_bias": {1: 1.02, 2: 0.98, 3: 0.10},
            "stopped_fraction": 0.00,
            "emergency_lane_mix": {"maintenance": 0.95, "truck": 0.05},
            "lane_activity_prob": {1:1.0, 2:1.0, 3:0.1},
            "emergency_speed_cap_kmh": 25,           # ≤ speed limit, hard cap for emergency lane
            "emergency_stopped_fraction": 0.20       # often stationary/slow work vehicles
        },
        # Low speeds, many stopped vehicles, minimal gaps (still safe)
        "jam": {
            "follow_speed_delta_max_kmh": 1.0,
            "gap_scale": 1.0,
            "lane_speed_bias": {1: 0.85, 2: 0.85, 3: 0.08},
            "stopped_fraction": 0.30,
            "emergency_lane_mix": {"maintenance": 0.90, "truck": 0.10},
            "lane_activity_prob": {1:1.0, 2:1.0, 3:0.1},
            "emergency_speed_cap_kmh": 25,           # ≤ speed limit, hard cap for emergency lane
            "emergency_stopped_fraction": 0.20       # often stationary/slow work vehicles
        },
    },
    # General generation knobs
    "speed_jitter_kmh": 3.0,        # noise added to sampled speeds
    "oversample_factor": 1.5,       # try to sample this many more candidates then prune by window and count
    "safety_buffer_m": 1.0,         # extra buffer added to physical safe gap
    "allow_random_lane_skips": True # slightly randomize which lane we fill next
}


# DEFAULT_ROAD_CONFIG = {
#     "driving_direction": "right",
#     "overtaking_rule": "keep_right_pass_left",
#     "lanes": [
#         {"id": 1, "type": "driving",   "speed_limit_kmh": 130},
#         {"id": 2, "type": "driving",   "speed_limit_kmh": 110},
#         {"id": 3, "type": "emergency", "speed_limit_kmh": 30},
#     ],
# }

# DEFAULT_GLOBAL_CONSTANTS = {
#     "vehicle_types": {
#         "sedan":       {"length_m": 4.6,  "min_gap_m": 2.0, "max_decel_mps2": 7.0, "reaction_time_s": 1.2},
#         "suv":         {"length_m": 4.9,  "min_gap_m": 2.5, "max_decel_mps2": 6.5, "reaction_time_s": 1.3},
#         "truck":       {"length_m": 12.0, "min_gap_m": 4.0, "max_decel_mps2": 5.0, "reaction_time_s": 1.5},
#         "bus":         {"length_m": 13.0, "min_gap_m": 4.0, "max_decel_mps2": 5.0, "reaction_time_s": 1.5},
#         "van":         {"length_m": 5.2,  "min_gap_m": 2.5, "max_decel_mps2": 6.0, "reaction_time_s": 1.3},
#         "motorcycle":  {"length_m": 2.2,  "min_gap_m": 1.5, "max_decel_mps2": 7.5, "reaction_time_s": 1.0},
#         "maintenance": {"length_m": 6.0,  "min_gap_m": 3.0, "max_decel_mps2": 5.0, "reaction_time_s": 1.2},
#         "ego":         {"length_m": 4.6,  "min_gap_m": 2.0, "max_decel_mps2": 7.0, "reaction_time_s": 1.2},
#     },
#     "phase_params": {
#         "free": {"follow_speed_delta_max_kmh": 5.0, "gap_scale": 1.2, "lane_speed_bias": {1:1.08,2:1.00,3:0.15}, "stopped_fraction": 0.00, "emergency_lane_mix": {"maintenance": 0.98, "truck": 0.02}},
#         "sync": {"follow_speed_delta_max_kmh": 2.0, "gap_scale": 1.05, "lane_speed_bias": {1:1.02,2:0.98,3:0.10}, "stopped_fraction": 0.05, "emergency_lane_mix": {"maintenance": 0.95, "truck": 0.05}},
#         "jam":  {"follow_speed_delta_max_kmh": 1.0, "gap_scale": 1.00, "lane_speed_bias": {1:0.85,2:0.85,3:0.08}, "stopped_fraction": 0.35, "emergency_lane_mix": {"maintenance": 0.90, "truck": 0.10}},
#     },
#     "speed_jitter_kmh": 3.0,
#     "safety_buffer_m": 1.0,
#     "allow_random_lane_skips": True
# }