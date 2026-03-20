from __future__ import annotations

"""ADLight-inspired non-learnable semantic-pooling encoder.

This is intentionally *not* a faithful reproduction of ADLight's learned shared
convolution extractor. Instead, it keeps the two parts that fit a fixed PPO host
with low engineering cost:

1) movement-level padded representation
2) deterministic semantic pooling over movement slots

The output is a flat numeric vector suitable for the existing PPO pipeline.
All ADLight-specific helper logic is kept in this file.
"""

from dataclasses import dataclass
from math import atan2, degrees
from typing import Any, Iterable

import numpy as np
try:
    import libsumo as traci
except ModuleNotFoundError:  # pragma: no cover
    import traci

from controllers.base import get_controller_structure
from .base import append_signal_context, downstream_norm_for_lane, lane_value_map, scene_lane_ids


@dataclass(frozen=True)
class _MovementSlot:
    in_lane: str
    out_lane: str
    is_straight_like: bool
    approach_lane_count_norm: float


def _lane_heading_deg(shape: list[tuple[float, float]], *, use_tail: bool) -> float:
    if len(shape) < 2:
        return 0.0
    if use_tail:
        (x1, y1), (x2, y2) = shape[-2], shape[-1]
    else:
        (x1, y1), (x2, y2) = shape[0], shape[1]
    return float(degrees(atan2(y2 - y1, x2 - x1)))



def _signed_angle_diff_deg(a_deg: float, b_deg: float) -> float:
    d = float(b_deg) - float(a_deg)
    while d <= -180.0:
        d += 360.0
    while d > 180.0:
        d -= 360.0
    return d



def _is_straight_like_movement(in_lane: str, out_lane: str, *, straight_thresh_deg: float) -> bool:
    try:
        in_shape = traci.lane.getShape(str(in_lane))
        out_shape = traci.lane.getShape(str(out_lane))
        in_hd = _lane_heading_deg(in_shape, use_tail=True)
        out_hd = _lane_heading_deg(out_shape, use_tail=False)
        d = _signed_angle_diff_deg(in_hd, out_hd)
        return abs(float(d)) <= float(straight_thresh_deg)
    except Exception:
        # Fall back to a conservative default when geometry is unavailable.
        return False



def _approach_lane_count_norm_map(lane_ids: Iterable[str]) -> dict[str, float]:
    lane_ids = tuple(str(x) for x in lane_ids)
    edge_of: dict[str, str] = {}
    edge_count: dict[str, int] = {}
    for lane_id in lane_ids:
        try:
            edge_id = str(traci.lane.getEdgeID(lane_id))
        except Exception:
            edge_id = lane_id.rsplit("_", 1)[0] if "_" in lane_id else lane_id
        edge_of[lane_id] = edge_id
        edge_count[edge_id] = int(edge_count.get(edge_id, 0)) + 1

    max_count = max(1, max(edge_count.values(), default=1))
    return {
        lane_id: float(edge_count.get(edge_of[lane_id], 1)) / float(max_count)
        for lane_id in lane_ids
    }



def _canonical_movement_slots(
    tls_id: str,
    *,
    scene_stats: Any,
    cache: dict,
    max_movement_slots: int,
    min_major_green_s: float,
    straight_thresh_deg: float,
) -> tuple[list[_MovementSlot], np.ndarray]:
    """Build a fixed-size movement list and a validity mask.

    Movements are taken from the union of controller-structure controlled movements,
    sorted deterministically by `(in_lane, out_lane)`, then truncated/padded.
    """
    struct = get_controller_structure(tls_id, cache, min_major_green_s=float(min_major_green_s))
    seen: set[tuple[str, str]] = set()
    raw: list[tuple[str, str]] = []
    for action_idx in range(len(struct.action_to_phase)):
        for in_lane, out_lane in struct.action_to_movements.get(int(action_idx), ()):  # type: ignore[arg-type]
            mv = (str(in_lane), str(out_lane))
            if mv not in seen:
                seen.add(mv)
                raw.append(mv)
    raw.sort(key=lambda x: (x[0], x[1]))

    lane_ids = scene_lane_ids(scene_stats)
    approach_norm = _approach_lane_count_norm_map(lane_ids)

    valid_n = min(int(max_movement_slots), len(raw))
    slots: list[_MovementSlot] = []
    for i in range(valid_n):
        in_lane, out_lane = raw[i]
        slots.append(
            _MovementSlot(
                in_lane=in_lane,
                out_lane=out_lane,
                is_straight_like=_is_straight_like_movement(
                    in_lane,
                    out_lane,
                    straight_thresh_deg=float(straight_thresh_deg),
                ),
                approach_lane_count_norm=float(approach_norm.get(in_lane, 0.0)),
            )
        )

    valid_mask = np.zeros(int(max_movement_slots), dtype=np.float32)
    valid_mask[:valid_n] = 1.0
    while len(slots) < int(max_movement_slots):
        slots.append(
            _MovementSlot(
                in_lane="",
                out_lane="",
                is_straight_like=False,
                approach_lane_count_norm=0.0,
            )
        )
    return slots, valid_mask



def _movement_feature_matrix(
    *,
    slots: list[_MovementSlot],
    valid_mask: np.ndarray,
    count_map: dict[str, float],
    queue_map: dict[str, float],
    speed_map: dict[str, float],
    wait_map: dict[str, float],
    green_map: dict[str, float],
    veh_equiv_len_m: float,
    clip_occ: float,
) -> np.ndarray:
    """Build the ADLight-inspired padded movement feature matrix.

    Per-slot features:
      0 count_ratio_norm(upstream)
      1 queue_ratio_norm(upstream)
      2 speed_norm(upstream)
      3 wait_norm(upstream)
      4 downstream_count_ratio_norm(outgoing)
      5 is_green_now(incoming lane)
      6 is_straight_like
      7 normalized approach width
    """
    feat_dim = 8
    X = np.zeros((len(slots), feat_dim), dtype=np.float32)
    for i, slot in enumerate(slots):
        if valid_mask[i] <= 0.0 or not slot.in_lane:
            continue
        cnt = float(count_map.get(slot.in_lane, 0.0))
        que = float(queue_map.get(slot.in_lane, 0.0))
        spd = float(speed_map.get(slot.in_lane, 0.0))
        wai = float(wait_map.get(slot.in_lane, 0.0))
        dwn = float(
            downstream_norm_for_lane(
                slot.out_lane,
                veh_equiv_len_m=float(veh_equiv_len_m),
                clip_occ=float(clip_occ),
            )
        ) if slot.out_lane else 0.0
        grn = float(green_map.get(slot.in_lane, 0.0))
        X[i, :] = (
            cnt,
            que,
            spd,
            wai,
            dwn,
            grn,
            1.0 if slot.is_straight_like else 0.0,
            float(slot.approach_lane_count_norm),
        )
    return X



def _pool_group(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean+max pooling for one semantic group.

    Returns a flat vector `[mean(features), max(features)]`. Empty groups map to zeros.
    """
    X = np.asarray(X, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D movement-feature matrix")
    if mask.shape[0] != X.shape[0]:
        raise ValueError("mask length must match number of movement slots")

    active = mask > 0.5
    feat_dim = X.shape[1]
    if not np.any(active):
        return np.zeros(2 * feat_dim, dtype=np.float32)

    Xi = X[active]
    mean_v = np.mean(Xi, axis=0, dtype=np.float32)
    max_v = np.max(Xi, axis=0)
    return np.concatenate([mean_v.astype(np.float32), max_v.astype(np.float32)], axis=0)



def adlight_state_encoding(
    tls_id: str,
    *,
    scene_stats: Any,
    cache: dict | None = None,
    max_movement_slots: int = 8,
    min_major_green_s: float = 5.0,
    straight_thresh_deg: float = 35.0,
    veh_equiv_len_m: float = 7.5,
    clip_occ: float = 1.0,
    include_green_red_groups: bool = True,
    include_straight_turn_groups: bool = True,
    include_signal_context: bool = True,
    include_current_phase_onehot: bool = True,
    include_major_since_served: bool = True,
    **kwargs,
) -> np.ndarray:
    """ADLight-inspired semantic-pooled movement encoder.

    Pipeline:
      1) Build a fixed-size padded movement tensor from controller movements.
      2) Apply deterministic semantic pooling instead of learned 1D convolution.
      3) Return a flat vector suitable for the existing PPO backbone.

    The semantic groups are:
      - all valid movements (always included)
      - green movements / red movements (optional)
      - straight-like movements / turning-like movements (optional)
    """
    if cache is None:
        cache = {}
    if scene_stats is None:
        raise AssertionError("scene_stats must not be None")

    count_map = lane_value_map(scene_stats, "count_ratio_norm")
    queue_map = lane_value_map(scene_stats, "queue_ratio_norm")
    speed_map = lane_value_map(scene_stats, "speed_norm")
    wait_map = lane_value_map(scene_stats, "wait_norm")
    green_map = lane_value_map(scene_stats, "is_green")

    slots, valid_mask = _canonical_movement_slots(
        tls_id,
        scene_stats=scene_stats,
        cache=cache,
        max_movement_slots=int(max_movement_slots),
        min_major_green_s=float(min_major_green_s),
        straight_thresh_deg=float(straight_thresh_deg),
    )

    X = _movement_feature_matrix(
        slots=slots,
        valid_mask=valid_mask,
        count_map=count_map,
        queue_map=queue_map,
        speed_map=speed_map,
        wait_map=wait_map,
        green_map=green_map,
        veh_equiv_len_m=float(veh_equiv_len_m),
        clip_occ=float(clip_occ),
    )

    features: list[float] = []

    # Always pool over all valid movements.
    features.extend(_pool_group(X, valid_mask).tolist())

    if include_green_red_groups:
        green_mask = valid_mask * (X[:, 5] > 0.5).astype(np.float32)
        red_mask = valid_mask * (X[:, 5] <= 0.5).astype(np.float32)
        features.extend(_pool_group(X, green_mask).tolist())
        features.extend(_pool_group(X, red_mask).tolist())

    if include_straight_turn_groups:
        straight_mask = valid_mask * (X[:, 6] > 0.5).astype(np.float32)
        turn_mask = valid_mask * (X[:, 6] <= 0.5).astype(np.float32)
        features.extend(_pool_group(X, straight_mask).tolist())
        features.extend(_pool_group(X, turn_mask).tolist())

    # Lightweight occupancy of semantic groups; helps distinguish 'all-zero because empty group'
    # from 'all-zero because features happen to be tiny'.
    valid_count = float(np.sum(valid_mask))
    green_count = float(np.sum(valid_mask * (X[:, 5] > 0.5).astype(np.float32)))
    straight_count = float(np.sum(valid_mask * (X[:, 6] > 0.5).astype(np.float32)))
    denom = max(1.0, valid_count)
    features.extend([
        valid_count / float(max(1, int(max_movement_slots))),
        green_count / denom,
        straight_count / denom,
    ])

    if include_signal_context:
        features.extend(
            append_signal_context(
                scene_stats,
                include_current_phase_onehot=bool(include_current_phase_onehot),
                include_major_since_served=bool(include_major_since_served),
            )
        )

    return np.asarray(features, dtype=np.float32)



def adlight_state_encoder(
    tls_id: str,
    *,
    scene_stats: Any,
    cache: dict | None = None,
    **kwargs,
) -> np.ndarray:
    return adlight_state_encoding(
        tls_id=tls_id,
        scene_stats=scene_stats,
        cache=cache,
        **kwargs,
    )
