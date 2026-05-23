from __future__ import annotations

"""Pressure-state encoding for fixed-backbone PPO comparisons.

This is a low-cost, literature-inspired substitution rather than a full reproduction of
PressLight/Efficient-XLight. The state focuses on major-action signed pressures computed from
portable ``scene_stats`` and appends the same signal-context terms used by the current baseline
(current phase one-hot and major-green starvation features) when available.
"""

from typing import Any

import numpy as np

from .base import append_signal_context, phase_signed_pressures


def pressure_state_encoder(
    tls_id: str,
    *,
    scene_stats: Any,
    cache: dict | None = None,
    upstream_key: str = "count_ratio_norm",
    veh_equiv_len_m: float = 7.5,
    clip_occ: float = 1.0,
    min_major_green_s: float = 5.0,
    include_signal_context: bool = True,
    normalize_by_movement_count: bool = True,
    use_abs_pressure: bool = False,
    pressure_scale: float = 1.0,
    soft_sat_pressure: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    if cache is None:
        cache = {}
    if scene_stats is None:
        raise AssertionError("scene_stats must not be None")

    phase_p, action_to_movements = phase_signed_pressures(
        tls_id,
        scene_stats=scene_stats,
        cache=cache,
        upstream_key=upstream_key,
        veh_equiv_len_m=float(veh_equiv_len_m),
        clip_occ=float(clip_occ),
        min_major_green_s=float(min_major_green_s),
    )

    vals = phase_p.astype(np.float32, copy=True)
    if normalize_by_movement_count:
        for i in range(vals.shape[0]):
            m = max(1, len(action_to_movements.get(int(i), ())))
            vals[i] = float(vals[i]) / float(m)
    if use_abs_pressure:
        vals = np.abs(vals)
    if soft_sat_pressure:
        scale = max(1e-6, float(pressure_scale))
        vals = vals / scale
        vals = np.tanh(vals).astype(np.float32)

    features: list[float] = vals.reshape(-1).tolist()
    if include_signal_context:
        features.extend(
            append_signal_context(
                scene_stats,
                include_current_phase_onehot=True,
                include_major_since_served=True,
            )
        )
    return np.asarray(features, dtype=np.float32)
