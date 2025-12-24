from __future__ import annotations

import os
import sys
import argparse
import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import traci
from sumolib import checkBinary

from dqn_agent import DQNAgent
from scene_encoder import encode_tsc_state_vector  # unchanged encoder


def ensure_sumo_tools_on_path() -> None:
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)


@dataclass
class TLSControllerState:
    next_decision_time: float = 0.0


def start_sumo(sumocfg: str, gui: bool, delay_ms: int) -> None:
    binary = checkBinary("sumo-gui" if gui else "sumo")
    cmd = [
        binary,
        "-c", sumocfg,
        "--start",
        "--no-step-log", "true",
        "--delay", str(delay_ms),
    ]
    traci.start(cmd)


def get_phase_count(tls_id: str) -> int:
    current_program = traci.trafficlight.getProgram(tls_id)
    logics = traci.trafficlight.getAllProgramLogics(tls_id)

    logic = None
    for lg in logics:
        try:
            if lg.getSubID() == current_program:
                logic = lg
                break
        except Exception:
            if getattr(lg, "programID", None) == current_program:
                logic = lg
                break
    if logic is None:
        logic = logics[0]

    try:
        phases = logic.getPhases()
    except Exception:
        phases = getattr(logic, "phases")
    return int(len(phases))


def run_dqn_tsc(
    sumocfg: str,
    *,
    gui: bool,
    max_time: float,
    seed: int,
    delay_ms: int,
    action_hold_s: float,
    epsilon: float,
    device: Optional[str] = None,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    start_sumo(sumocfg, gui=gui, delay_ms=delay_ms)

    tls_ids = list(traci.trafficlight.getIDList())
    if not tls_ids:
        traci.close()
        raise RuntimeError("No traffic lights found in this scenario.")

    tls_state: Dict[str, TLSControllerState] = {tls_id: TLSControllerState() for tls_id in tls_ids}
    encoder_cache_by_tls: Dict[str, dict] = {tls_id: {} for tls_id in tls_ids}
    agents: Dict[str, DQNAgent] = {}

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            sim_t = float(traci.simulation.getTime())
            if sim_t >= max_time:
                break

            for tls_id in tls_ids:
                st = tls_state[tls_id]
                if sim_t < st.next_decision_time:
                    continue

                state_vec = encode_tsc_state_vector(
                    tls_id,
                    moving_speed_threshold=0.1,
                    stopped_speed_threshold=0.1,
                    cache=encoder_cache_by_tls[tls_id],
                )

                if tls_id not in agents:
                    action_dim = get_phase_count(tls_id)
                    agents[tls_id] = DQNAgent(
                        state_dim=int(state_vec.shape[0]),
                        action_dim=action_dim,
                        seed=seed,
                        device=device,
                    )
                    print(f"[init] tls={tls_id} state_dim={state_vec.shape[0]} action_dim={action_dim}")

                action_phase = agents[tls_id].act(state_vec, epsilon=epsilon)

                traci.trafficlight.setPhase(tls_id, int(action_phase))
                traci.trafficlight.setPhaseDuration(tls_id, float(action_hold_s))

                st.next_decision_time = sim_t + float(action_hold_s)

                print(f"[t={sim_t:6.1f}] tls={tls_id} action(phase)={action_phase} hold={action_hold_s}s")

            traci.simulationStep()

    finally:
        traci.close()


if __name__ == "__main__":
    ensure_sumo_tools_on_path()

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--sumocfg", required=True)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--max-time", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--delay-ms", type=int, default=50)
    ap.add_argument("--hold", type=float, default=5.0)
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--device", type=str, default=None, help="e.g. cpu or cuda (optional)")
    args = ap.parse_args()

    run_dqn_tsc(
        args.sumocfg,
        gui=args.gui,
        max_time=args.max_time,
        seed=args.seed,
        delay_ms=args.delay_ms,
        action_hold_s=args.hold,
        epsilon=args.epsilon,
        device=args.device,
    )
