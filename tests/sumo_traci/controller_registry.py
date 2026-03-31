from __future__ import annotations

import importlib
from typing import Any, Callable


CONTROLLER_REGISTRY: dict[str, str] = {
    "fully_actuated": "controllers.fully_actuated:FullyActuatedController",
    "max_pressure": "controllers.max_pressure:MaxPressureController",
    "fixed_time": "controllers.fixed_time:WebsterController",
    "webster": "controllers.fixed_time:WebsterController",
}


def _resolve_symbol(target: str) -> Any:
    module_name, symbol_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def list_controllers() -> list[str]:
    return sorted(CONTROLLER_REGISTRY.keys())


def make_controller(name: str, /, **kwargs: Any):
    """
    Construct one controller backend from the registry.

    Example
    -------
    ``make_controller('fully_actuated', min_green_s=8.0, max_green_s=35.0)``
    """
    key = str(name).strip().lower()
    if key not in CONTROLLER_REGISTRY:
        known = ", ".join(list_controllers())
        raise ValueError(f"unknown controller: {name!r}. known controllers: {known}")
    cls = _resolve_symbol(CONTROLLER_REGISTRY[key])
    return cls(**kwargs)
