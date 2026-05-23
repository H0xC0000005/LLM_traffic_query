from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import (
    BaseSignalController,
    ControlDecision,
    current_major_action_index,
    downstream_count_ratio_norm,
    get_controller_structure,
    lane_value_map,
)


@dataclass
class MaxPressureController(BaseSignalController):
    """
    Rule-based max-pressure controller over the existing major-green action space.

    Pressure for one major action is computed as the sum over all green movements
    under that major phase:

        upstream_demand(in_lane) - downstream_demand(out_lane)

    where upstream demand defaults to normalized incoming lane count and downstream
    demand is computed with the same soft-saturated count normalization used in the
    portable scene snapshot.
    """

    min_major_green_s: float = 5.0
    hold_s: float = 10.0
    upstream_key: str = "count_ratio_norm"
    veh_equiv_len_m: float = 7.5
    clip_occ: float = 1.0
    tie_break_current: bool = True

    def reset(self, *, tls_id: str, cache: dict | None = None) -> None:
        return None

    def choose_action(
        self,
        tls_id: str,
        *,
        scene_stats: Any,
        sim_time: float,
        cache: dict,
    ) -> ControlDecision:
        tls_id = str(tls_id)
        _ = float(sim_time)  # currently unused; kept for interface symmetry
        struct = get_controller_structure(
            tls_id, cache, min_major_green_s=float(self.min_major_green_s)
        )
        current_action = current_major_action_index(
            tls_id, cache, min_major_green_s=float(self.min_major_green_s)
        )

        upstream = lane_value_map(scene_stats, self.upstream_key)
        downstream_cache: dict[str, float] = {}
        pressures: list[float] = []
        for action_idx in range(len(struct.action_to_phase)):
            p = 0.0
            for in_lane, out_lane in struct.action_to_movements.get(int(action_idx), ()): 
                if out_lane not in downstream_cache:
                    downstream_cache[out_lane] = downstream_count_ratio_norm(
                        out_lane,
                        veh_equiv_len_m=float(self.veh_equiv_len_m),
                        clip_occ=float(self.clip_occ),
                    )
                p += float(upstream.get(in_lane, 0.0)) - float(downstream_cache[out_lane])
            pressures.append(float(p))

        best_pressure = max(pressures) if pressures else 0.0
        candidates = [i for i, p in enumerate(pressures) if abs(float(p) - float(best_pressure)) <= 1e-9]
        if self.tie_break_current and int(current_action) in candidates:
            chosen = int(current_action)
        else:
            chosen = int(candidates[0]) if candidates else int(current_action)

        return ControlDecision(
            action=int(chosen),
            hold_s=float(self.hold_s),
            info={
                "controller": "max_pressure",
                "pressure": {str(i): float(p) for i, p in enumerate(pressures)},
                "chosen_pressure": float(best_pressure),
            },
        )
