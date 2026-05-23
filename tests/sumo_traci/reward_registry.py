from __future__ import annotations

"""Runtime reward resolver for TSC experiments.

This module keeps the training loop independent from concrete reward implementation
names and import locations. Rewards are selected by a user-facing ``reward_name`` and
resolved to a callable once at startup.

Design notes
------------
- The registry stores import paths as strings, so literature rewards can live outside
  ``utility.py``.
- ``resolve_reward(...)`` returns the constructed callable directly.
- The returned callable accepts a uniform runtime interface used by the trainer:

    reward_fn(
        tls_id=..., sim_time=..., scene_stats=..., cache=..., gamma_dt=...,
        init_only=False, **runtime_overrides,
    ) -> float

The wrapper binds experiment-level configuration once at startup, filters unsupported
kwargs based on the underlying function signature, and optionally applies reward-
specific warmup overrides when ``init_only=True``.
"""

from dataclasses import dataclass, field
import importlib
import inspect
from typing import Any, Callable


RewardCallable = Callable[..., float]


@dataclass(frozen=True)
class _RewardEntry:
    target: str
    init_overrides: dict[str, Any] = field(default_factory=dict)


# Extend this registry as reward modules are added.
_REWARD_REGISTRY: dict[str, _RewardEntry] = {
    # Existing utility-backed rewards.
    "avg_queue": _RewardEntry("utility:reward_avg_queue_from_encoded_state"),
    "top2_queue": _RewardEntry("utility:reward_top2_queue_from_encoded_state"),
    "softmax_queue": _RewardEntry("utility:reward_softmax_queue_from_encoded_state"),
    "wait_barrier": _RewardEntry("utility:reward_softmax_wait_barrier_from_encoded_state"),
    "throughput": _RewardEntry("utility:reward_throughput_per_second_on_decision"),
    "throughput_plus_softmax_queue": _RewardEntry("utility:reward_throughput_plus_softmax_queue"),
    "unbiased_simple_v1": _RewardEntry(
        "utility:reward_throughput_plus_softmax_queue_plus_softmax_wait_barrier_right_endpoint_v1",
        init_overrides={
            "w_throughput": 0.0,
            "w_queue": 0.0,
            "w_wait_barrier": 0.0,
        },
    ),
    "universal_v2": _RewardEntry(
        "utility:reward_throughput_plus_softmax_queue_deltaq_plus_softmax_wait_barrier_v2",
        init_overrides={
            "w_throughput": 0.0,
            "w_queue": 0.0,
            "w_delta_queue": 0.0,
            "w_wait_barrier": 0.0,
            "w_queue_zone": 0.0,
        },
    ),
    # New modular literature-style reward baselines.
    "queue": _RewardEntry("rewards.queue_reward:queue_reward"),
    "pressure": _RewardEntry("rewards.pressure_reward:pressure_reward"),
}


def available_reward_names() -> list[str]:
    return sorted(_REWARD_REGISTRY.keys())


def _load_target(path: str) -> RewardCallable:
    module_name, sep, attr_name = path.partition(":")
    if not sep:
        raise ValueError(f"reward target must look like 'module:function', got {path!r}")
    module = importlib.import_module(module_name)
    fn = getattr(module, attr_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"reward target {path!r} did not resolve to a callable")
    return fn


def _accepted_kwargs(fn: Callable[..., Any]) -> tuple[set[str] | None, bool]:
    sig = inspect.signature(fn)
    accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_varkw:
        return None, True
    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return accepted, False


def _filter_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    accepted, accepts_varkw = _accepted_kwargs(fn)
    if accepts_varkw or accepted is None:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in accepted}


def resolve_reward(reward_name: str, **bound_kwargs: Any) -> RewardCallable:
    """Resolve a user-facing reward name to a runtime callable."""
    try:
        entry = _REWARD_REGISTRY[reward_name]
    except KeyError as exc:
        raise ValueError(f"Unknown reward {reward_name!r}. Available: {', '.join(available_reward_names())}") from exc

    impl = _load_target(entry.target)
    bound_filtered = _filter_kwargs(impl, dict(bound_kwargs))

    def reward_fn(*, init_only: bool = False, **runtime_kwargs: Any) -> float:
        if init_only and not entry.init_overrides:
            return 0.0

        call_kwargs = dict(bound_filtered)
        call_kwargs.update(runtime_kwargs)
        if init_only and entry.init_overrides:
            call_kwargs.update(entry.init_overrides)

        call_kwargs = _filter_kwargs(impl, call_kwargs)
        return float(impl(**call_kwargs))

    reward_fn.__name__ = f"resolved_reward__{reward_name}"
    return reward_fn
