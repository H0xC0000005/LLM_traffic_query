from __future__ import annotations

from typing import Any

import numpy as np

from .base import scene_per_lane


def queue_reward(
    *,
    tls_id: str,
    sim_time: float,
    scene_stats: Any,
    cache: dict | None = None,
    gamma_dt: float | None = None,
    init_only: bool = False,
    mode: str = "softmax",
    scene_key: str = "queue_ratio_norm",
    reduce: str = "hybrid",
    power: float = 1.0,
    scale: float = 1.0,
    softmax_beta: float = 5.0,
    clip_nonnegative: bool = True,
    **_: Any,
) -> float:
    """Queue-family reward baseline over portable scene statistics.

    Parameters other than runtime fields are bound once through ``resolve_reward``.
    """
    _ = tls_id, sim_time, cache, gamma_dt
    if init_only:
        return 0.0

    mode_key = str(mode).strip().lower()
    reduce_key = str(reduce).strip().lower()
    p = float(power)
    if p < 1.0:
        raise ValueError("power must be >= 1.0")
    s = float(scale) if float(scale) > 0.0 else 1.0
    beta = float(softmax_beta)

    qs = scene_per_lane(scene_stats, scene_key).astype(np.float64, copy=False).reshape(-1)
    if clip_nonnegative:
        qs = np.maximum(qs, 0.0)
    qs = qs / s
    if qs.size == 0:
        return 0.0

    if mode_key == "avg":
        if reduce_key == "mean":
            penalty = float(np.mean(qs))
        elif reduce_key == "sum":
            penalty = float(np.sum(qs))
        elif reduce_key == "hybrid":
            penalty = 0.5 * float(np.mean(qs)) + 0.5 * float(np.max(qs))
        else:
            raise ValueError("reduce must be one of {'mean', 'sum', 'hybrid'}")
        return -float(penalty)

    if mode_key != "softmax":
        raise ValueError("mode must be one of {'softmax', 'avg'}")
    if beta <= 0.0:
        raise ValueError("softmax_beta must be > 0.0")

    logits = beta * qs
    logits = logits - float(np.max(logits))
    exps = np.exp(logits)
    weights = exps / float(np.sum(exps) + 1e-12)
    penalty = float(np.sum(weights * (qs ** p)))
    return -float(penalty)
