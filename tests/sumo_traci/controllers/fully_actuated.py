from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import (
    BaseSignalController,
    ControlDecision,
    current_major_action_index,
    get_controller_structure,
    lane_value_map,
)


@dataclass
class FullyActuatedController(BaseSignalController):
    """
    Gap-out / max-green style macro controller over major phases.

    This is a lightweight "fully-actuated-style" baseline tailored to the existing
    major-green abstraction in the PPO pipeline:
      - action = choose one major green
      - hold time can vary per decision
      - continue current green while demand remains and max-green is not exceeded
      - otherwise switch to the competing phase with the highest demand

    The implementation is intentionally simple so it can fit the current evaluation
    loop with minimal changes.
    """

    min_green_s: float = 8.0
    max_green_s: float = 35.0
    extension_s: float = 5.0
    min_major_green_s: float = 5.0
    demand_key: str = "queue_ratio_norm"
    gap_out_threshold: float = 0.05
    switch_hysteresis: float = 0.02
    min_switch_demand: float = 0.01
    aggregate: str = "sum"
    _state: dict[str, dict[str, float | int]] = field(default_factory=dict, init=False)

    def reset(self, *, tls_id: str, cache: dict | None = None) -> None:
        self._state.pop(str(tls_id), None)

    def _phase_demand(self, action_idx: int, *, lane_demands: dict[str, float], cache: dict, tls_id: str) -> float:
        struct = get_controller_structure(
            tls_id, cache, min_major_green_s=float(self.min_major_green_s)
        )
        served_lanes = struct.action_to_in_lanes.get(int(action_idx), ())
        vals = [max(0.0, float(lane_demands.get(ln, 0.0))) for ln in served_lanes]
        if not vals:
            return 0.0
        if self.aggregate == "max":
            return float(max(vals))
        return float(sum(vals))

    def choose_action(
        self,
        tls_id: str,
        *,
        scene_stats: Any,
        sim_time: float,
        cache: dict,
    ) -> ControlDecision:
        tls_id = str(tls_id)
        struct = get_controller_structure(
            tls_id, cache, min_major_green_s=float(self.min_major_green_s)
        )
        current_action = current_major_action_index(
            tls_id, cache, min_major_green_s=float(self.min_major_green_s)
        )

        st = self._state.get(tls_id)
        if st is None or int(st.get("current_action", -1)) != int(current_action):
            st = {
                "current_action": int(current_action),
                "action_start_time": float(sim_time),
            }
            self._state[tls_id] = st

        elapsed = max(0.0, float(sim_time) - float(st["action_start_time"]))
        lane_demands = lane_value_map(scene_stats, self.demand_key)
        phase_demands = [
            self._phase_demand(a, lane_demands=lane_demands, cache=cache, tls_id=tls_id)
            for a in range(len(struct.action_to_phase))
        ]

        cur_demand = float(phase_demands[current_action])
        best_other_action = int(current_action)
        best_other_demand = -1.0
        for a, dem in enumerate(phase_demands):
            if int(a) == int(current_action):
                continue
            if float(dem) > float(best_other_demand):
                best_other_demand = float(dem)
                best_other_action = int(a)

        must_hold = elapsed < float(self.min_green_s)
        must_switch = elapsed >= float(self.max_green_s)
        gap_out = cur_demand <= float(self.gap_out_threshold)
        clear_competitor = best_other_demand > max(
            float(self.min_switch_demand), cur_demand + float(self.switch_hysteresis)
        )

        if must_hold:
            hold = max(0.1, min(float(self.extension_s), float(self.min_green_s) - elapsed))
            return ControlDecision(
                action=int(current_action),
                hold_s=float(hold),
                info={
                    "controller": "fully_actuated",
                    "reason": "min_green",
                    "elapsed": float(elapsed),
                    "cur_demand": float(cur_demand),
                },
            )

        if must_switch or (gap_out and clear_competitor):
            chosen = int(best_other_action) if best_other_demand > 0.0 else int(current_action)
            if chosen != int(current_action):
                self._state[tls_id] = {
                    "current_action": int(chosen),
                    "action_start_time": float(sim_time),
                }
                return ControlDecision(
                    action=int(chosen),
                    hold_s=float(self.min_green_s),
                    info={
                        "controller": "fully_actuated",
                        "reason": "max_green" if must_switch else "gap_out",
                        "elapsed": float(elapsed),
                        "cur_demand": float(cur_demand),
                        "next_demand": float(best_other_demand),
                    },
                )

        hold = max(0.1, min(float(self.extension_s), float(self.max_green_s) - elapsed))
        return ControlDecision(
            action=int(current_action),
            hold_s=float(hold),
            info={
                "controller": "fully_actuated",
                "reason": "extend",
                "elapsed": float(elapsed),
                "cur_demand": float(cur_demand),
                "best_other_demand": float(best_other_demand),
            },
        )
