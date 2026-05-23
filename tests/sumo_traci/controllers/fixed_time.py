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
from utility import get_tls_phase_plan


@dataclass
class WebsterController(BaseSignalController):
    """
    Webster-style fixed-time controller over the existing major-green action space.

    Notes
    -----
    This is a *Webster-style* controller rather than a strict traffic-engineering
    implementation from detector counts and saturation flow tables. It uses the
    warmed-up portable scene snapshot to estimate a per-phase critical-ratio proxy,
    then computes a fixed cycle and green splits once per episode.

    It fits the current PPO/action abstraction:
      - action = choose one major green
      - hold_s = fixed green duration for that major action in the cycle
      - yellow/all-red auxiliary phases remain handled by the existing phase
        switching helpers in the evaluation/training loop
    """

    min_major_green_s: float = 5.0

    # Webster / fixed-time settings
    demand_key: str = "count_ratio_norm"
    cycle_min_s: float = 40.0
    cycle_max_s: float = 140.0
    startup_lost_per_phase_s: float = 2.0
    max_total_critical_ratio: float = 0.90
    critical_ratio_scale: float = 1.0
    align_first_phase_to_current: bool = True

    # State: one plan per TLS, rebuilt on reset
    _state: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def reset(self, *, tls_id: str, cache: dict | None = None) -> None:
        self._state.pop(str(tls_id), None)

    def _estimate_phase_critical_ratios(
        self,
        tls_id: str,
        *,
        scene_stats: Any,
        cache: dict,
    ) -> list[float]:
        """
        Build one critical-ratio proxy per major action.

        We use the maximum served-lane demand under each phase as a proxy for that
        phase's critical lane. Demand comes from the selected scene_stats key
        (default: normalized count).
        """
        struct = get_controller_structure(tls_id, cache, min_major_green_s=float(self.min_major_green_s))
        lane_demands = lane_value_map(scene_stats, self.demand_key)

        y_raw: list[float] = []
        n_actions = len(struct.action_to_phase)
        for action_idx in range(n_actions):
            served = struct.action_to_in_lanes.get(int(action_idx), ())
            vals = [max(0.0, float(lane_demands.get(ln, 0.0))) for ln in served]
            yi = max(vals) if vals else 0.0
            yi *= float(self.critical_ratio_scale)
            y_raw.append(float(max(0.0, yi)))

        total = float(sum(y_raw))
        if total > float(self.max_total_critical_ratio) and total > 1e-9:
            scale = float(self.max_total_critical_ratio) / total
            y = [float(v * scale) for v in y_raw]
        else:
            y = y_raw
        return y

    def _estimate_total_lost_time(
        self,
        tls_id: str,
        *,
        cache: dict,
    ) -> float:
        """
        Estimate total lost time per cycle.

        We combine:
          - actual auxiliary phase durations between major greens
          - startup lost time per major phase
        """
        plan = get_tls_phase_plan(tls_id, cache, min_major_green_s=float(self.min_major_green_s))
        major_greens = [int(x) for x in plan.major_greens]

        aux_lost = 0.0
        for mg in major_greens:
            for aux_idx in plan.aux_after_major.get(int(mg), []):
                aux_lost += float(plan.phase_duration[int(aux_idx)])

        startup_lost = float(self.startup_lost_per_phase_s) * float(len(major_greens))
        return float(aux_lost + startup_lost)

    def _build_plan(
        self,
        tls_id: str,
        *,
        scene_stats: Any,
        cache: dict,
    ) -> dict[str, Any]:
        struct = get_controller_structure(tls_id, cache, min_major_green_s=float(self.min_major_green_s))
        n_actions = len(struct.action_to_phase)
        if n_actions <= 0:
            raise RuntimeError(f"[{tls_id}] no major actions found for Webster controller")

        y = self._estimate_phase_critical_ratios(tls_id, scene_stats=scene_stats, cache=cache)
        Y = float(sum(y))
        L = float(self._estimate_total_lost_time(tls_id, cache=cache))

        # Webster cycle length
        if Y <= 1e-6:
            C = max(
                float(self.cycle_min_s),
                min(
                    float(self.cycle_max_s),
                    L + float(n_actions) * float(self.min_major_green_s),
                ),
            )
        else:
            C = (1.5 * L + 5.0) / max(1e-6, 1.0 - Y)
            C = max(float(self.cycle_min_s), min(float(self.cycle_max_s), float(C)))

        # Effective green time available for major greens
        G = max(float(n_actions) * float(self.min_major_green_s), float(C) - float(L))

        # Allocate green splits:
        #   base minimum green for every phase
        #   plus discretionary green proportional to y_i
        base_total = float(n_actions) * float(self.min_major_green_s)
        rem_green = max(0.0, float(G) - base_total)

        if Y <= 1e-6:
            discretionary = [float(rem_green) / float(n_actions)] * n_actions
        else:
            discretionary = [float(rem_green) * float(yi) / float(Y) for yi in y]

        green_times = [float(self.min_major_green_s) + float(discretionary[i]) for i in range(n_actions)]

        phase_order = list(range(n_actions))

        # Optional: align the cycle to the currently active major phase
        if bool(self.align_first_phase_to_current):
            current_action = current_major_action_index(tls_id, cache, min_major_green_s=float(self.min_major_green_s))
            if 0 <= int(current_action) < n_actions:
                k = int(current_action)
                phase_order = phase_order[k:] + phase_order[:k]
                green_times = green_times[k:] + green_times[:k]
                y = y[k:] + y[:k]

        return {
            "phase_order": phase_order,
            "green_times_s": green_times,
            "critical_ratios": y,
            "cycle_s": float(L + sum(green_times)),
            "lost_time_s": float(L),
            "cursor": 0,
        }

    def choose_action(
        self,
        tls_id: str,
        *,
        scene_stats: Any,
        sim_time: float,
        cache: dict,
    ) -> ControlDecision:
        tls_id = str(tls_id)
        _ = float(sim_time)  # kept for interface symmetry / future use

        st = self._state.get(tls_id)
        if st is None:
            st = self._build_plan(tls_id, scene_stats=scene_stats, cache=cache)
            self._state[tls_id] = st

        phase_order = st["phase_order"]
        green_times = st["green_times_s"]
        cursor = int(st["cursor"])

        action = int(phase_order[cursor])
        hold_s = float(green_times[cursor])

        st["cursor"] = int((cursor + 1) % len(phase_order))

        return ControlDecision(
            action=int(action),
            hold_s=float(hold_s),
            info={
                "controller": "webster",
                "cycle_s": float(st["cycle_s"]),
                "lost_time_s": float(st["lost_time_s"]),
                "critical_ratios": [float(x) for x in st["critical_ratios"]],
                "phase_order": [int(x) for x in phase_order],
                "green_times_s": [float(x) for x in green_times],
            },
        )
