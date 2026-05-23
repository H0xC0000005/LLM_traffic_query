# encoder_registry.py
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Iterable, Any


@dataclass(frozen=True)
class _EncoderEntry:
    target: str
    roles: frozenset[str]


_ENCODER_REGISTRY: dict[str, _EncoderEntry] = {
    "bounded_v2": _EncoderEntry(
        target="encoders.bounded_v2:bounded_v2_encoder",
        roles=frozenset({"core"}),
    ),
    "pressure_state": _EncoderEntry(
        target="encoders.pressure_state:pressure_state_encoder",
        roles=frozenset({"core", "addon"}),
    ),
    "ats": _EncoderEntry(
        target="encoders.ats_state:ats_state_encoder",
        roles=frozenset({"core", "addon"}),
    ),
    "expert": _EncoderEntry(
        target="encoders.expert_state:expert_feature_encoder",
        roles=frozenset({"addon"}),
    ),
    "frap_state": _EncoderEntry(
        target="encoders.frap_state:frap_state_encoder",
        roles=frozenset({"core", "addon"}),
    ),
    "adlight_state": _EncoderEntry(
        target="encoders.adlight_state:adlight_state_encoder",
        roles=frozenset({"core", "addon"}),
    ),
}


def _load_symbol(target: str):
    mod_name, attr_name = target.split(":", 1)
    mod = import_module(mod_name)
    return getattr(mod, attr_name)


def available_encoder_names() -> list[str]:
    return sorted(_ENCODER_REGISTRY.keys())


def available_encoder_names_for_role(role: str) -> list[str]:
    return sorted(name for name, entry in _ENCODER_REGISTRY.items() if role in entry.roles)


def resolve_encoder(name: str, **bound_kwargs) -> Callable[..., Any]:
    """
    Resolve an encoder name to a callable. Bound kwargs are partially applied.
    """
    try:
        entry = _ENCODER_REGISTRY[name]
    except KeyError as e:
        raise KeyError(f"Unknown encoder '{name}'. Available: {', '.join(available_encoder_names())}") from e

    fn = _load_symbol(entry.target)

    if not bound_kwargs:
        return fn

    def _bound_encoder(*args, **kwargs):
        merged = dict(bound_kwargs)
        merged.update(kwargs)
        return fn(*args, **merged)

    _bound_encoder.__name__ = getattr(fn, "__name__", f"{name}_encoder")
    return _bound_encoder


def get_encoder_roles(name: str) -> frozenset[str]:
    return _ENCODER_REGISTRY[name].roles
