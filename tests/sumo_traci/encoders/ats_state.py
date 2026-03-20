from __future__ import annotations

"""ATS-style encoding for fixed-backbone PPO comparisons.

This is a low-cost adaptation of the "pressure + running vehicles" idea from ATS / Advanced-XLight,
implemented as a flat vector suitable for the existing PPO pipeline. It is not a full reproduction
of the original method's complete training setup; it is a controlled encoder substitution.
"""

from typing import Any

import numpy as np

from .base import append_signal_context, phase_effective_running_vehicles, phase_signed_pressures


def ats_state_encoder(
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
    speed_gate_ratio: float = 0.15,
    use_abs_pressure: bool = False,
    soft_sat_pressure: bool = False,
    pressure_scale: float = 1.0,
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
    if normalize_by_movement_count:
        phase_p = phase_p.astype(np.float32, copy=True)
        for i in range(phase_p.shape[0]):
            m = max(1, len(action_to_movements.get(int(i), ())))
            phase_p[i] = float(phase_p[i]) / float(m)
    if use_abs_pressure:
        phase_p = np.abs(phase_p)
    if soft_sat_pressure:
        scale = max(1e-6, float(pressure_scale))
        phase_p = np.tanh((phase_p / scale).astype(np.float32)).astype(np.float32)

    erv = phase_effective_running_vehicles(
        tls_id,
        scene_stats=scene_stats,
        cache=cache,
        speed_gate_ratio=float(speed_gate_ratio),
        min_major_green_s=float(min_major_green_s),
    )

    features: list[float] = []
    features.extend(np.asarray(phase_p, dtype=np.float32).reshape(-1).tolist())
    features.extend(np.asarray(erv, dtype=np.float32).reshape(-1).tolist())
    if include_signal_context:
        features.extend(
            append_signal_context(
                scene_stats,
                include_current_phase_onehot=True,
                include_major_since_served=True,
            )
        )
    return np.asarray(features, dtype=np.float32)
