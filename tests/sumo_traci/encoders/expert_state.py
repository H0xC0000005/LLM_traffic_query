# encoders/expert_state.py
from __future__ import annotations

import numpy as np

from expert_feature_extractor import tsc_isolated_intersection_feature_vector


def expert_feature_encoder(
    tls_id: str,
    *,
    scene_stats,
    cache: dict | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Addon encoder wrapper for the existing expert feature extractor.
    scene_stats is accepted for protocol compatibility but not used here.
    """
    v = tsc_isolated_intersection_feature_vector(
        tls_id,
        cache=cache,
        **kwargs,
    )
    return np.asarray(v, dtype=np.float32)
