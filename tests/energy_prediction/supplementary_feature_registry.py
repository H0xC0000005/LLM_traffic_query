from __future__ import annotations

from typing import Callable, Dict, List

import pandas as pd

from supplementary_feature_expert import extract_expert_supplementary_features

# from supplementary_feature_rank5 import extract_rank5_mma_features
from supplementary_feature_rank2 import extract_rank2_weather_stats_features
from supplementary_feature_rank25 import extract_rank25_missingness_features
from supplementary_feature_rank4_revised import extract_rank4_features
from supplementary_feature_rank5_revised import extract_rank5_mma_features


SupplementaryExtractor = Callable[..., pd.DataFrame]


SUPPLEMENTARY_EXTRACTORS: Dict[str, SupplementaryExtractor] = {
    "expert": extract_expert_supplementary_features,
    "rank5": extract_rank5_mma_features,
    "rank2": extract_rank2_weather_stats_features,
    "rank25": extract_rank25_missingness_features,
    "rank4": extract_rank4_features,
}


def list_supplementary_extractors(include_none: bool = True) -> List[str]:
    names = sorted(SUPPLEMENTARY_EXTRACTORS)
    if include_none:
        return ["none", *names]
    return names


def get_supplementary_extractor(name: str) -> SupplementaryExtractor:
    key = normalize_supplementary_encoder_name(name)
    if key == "none":
        raise KeyError("'none' does not have an extractor function.")
    try:
        return SUPPLEMENTARY_EXTRACTORS[key]
    except KeyError as exc:
        valid = ", ".join(list_supplementary_extractors(include_none=True))
        raise KeyError(f"Unknown supplementary encoder '{name}'. Valid values: {valid}") from exc


def normalize_supplementary_encoder_name(name: str | None) -> str:
    if name is None:
        return "none"
    key = str(name).strip().lower()
    aliases = {
        "": "none",
        "baseline": "none",
        "no": "none",
        "false": "none",
        "expert_features": "expert",
        "rank5": "rank5",
        "mma": "rank5",
    }
    return aliases.get(key, key)
