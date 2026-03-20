# encoders/bounded_v2.py
from __future__ import annotations

import numpy as np

from scene_encoder import encode_tsc_state_vector_bounded_v2


def bounded_v2_encoder(
    tls_id: str,
    *,
    scene_stats,
    cache: dict | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Unified-registry adapter for the baseline bounded_v2 core encoder.
    """
    return np.asarray(
        encode_tsc_state_vector_bounded_v2(
            tls_id=tls_id,
            scene_stats=scene_stats,
            cache=cache,
            **kwargs,
        ),
        dtype=np.float32,
    ).reshape(-1)
