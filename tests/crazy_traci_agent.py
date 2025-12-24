# crazy_tsc_traci.py (updated to compute RL state features at each decision step)
#
# Assumption: encode_tsc_state_vector(...) exists unchanged (as you defined earlier),
# and is importable. Put it in encoder.py (same folder) or adjust the import below.

import os
import sys
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List

def ensure_sumo_tools_on_path():
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)

ensure_sumo_tools_on_path()

import traci  # noqa: E402
from sumolib import checkBinary  # noqa: E402

from sumo_traci.scene_encoder import encode_tsc_state_vector  # <-- your unchanged encoder function


@dataclass
class TLSControllerState:
    perm: List[int]
    k: int = 0
    next_decision_time: float = 0.0


def pick_logic_for_current_program(tls_id: str):
    logics = traci.trafficlight.getAllProgramLogics(tls_id)
    current_program = traci.trafficlight.getProgram(tls_id)
    for logic in logics:
        try:
            if logic.getSubID() == current_program:
                return logic
        except Exception:
            if getattr(logic, "programID", None) == current_program:
                return logic
    return logics[0]


def get_phase_count(tls_id: str) -> int:
    logic = pick_logic_for_current_program(tls_id)
    try:
        phases = logic.getPhases()
    except Exception:
        phases = getattr(logic, "phases")
    return len(phases)


def start_sumo(sumocfg: str, gui: bool, delay_ms: int):
    binary = checkBinary("sumo-gui" if gui else "sumo")
    cmd = [
        binary,
        "-c", sumocfg,
        "--start",
        "--quit-on-end", "true",
        "--delay", str(delay_ms),
        "--no-step-log", "true",
    ]
    traci.start(cmd)


def run_crazy_tsc(
    sumocfg: str,
    gui: bool,
    max_time: float,
    decision_interval: float,
    seed: int,
    delay_ms: int,
    print_state_preview: bool = True,
):
    random.seed(seed)
    start_sumo(sumocfg, gui=gui, delay_ms=delay_ms)

    tls_ids = list(traci.trafficlight.getIDList())
    if not tls_ids:
        traci.close()
        raise RuntimeError("No traffic lights found in this scenario.")

    # Initialize per-TLS “surprise plan”
    tls_state: Dict[str, TLSControllerState] = {}
    for tls_id in tls_ids:
        n = get_phase_count(tls_id)
        perm = list(range(n))
        random.shuffle(perm)
        tls_state[tls_id] = TLSControllerState(perm=perm)
        print(f"[init] tls={tls_id} phases={n} perm={perm}")

    # Per-TLS encoder cache (so lane ordering + time-in-phase tracking is stable)
    encoder_cache_by_tls: Dict[str, dict] = {tls_id: {} for tls_id in tls_ids}
    printed_dim_for_tls: set[str] = set()

    fib_durations = [1, 2, 3, 5, 8, 13]

    while traci.simulation.getMinExpectedNumber() > 0:
        sim_t = float(traci.simulation.getTime())
        if sim_t >= max_time:
            break

        for tls_id in tls_ids:
            st = tls_state[tls_id]

            # Only decide when the previous commanded duration has expired
            if sim_t < st.next_decision_time:
                continue

            # --------- compute RL features at decision time ----------
            state_vec = encode_tsc_state_vector(
                tls_id,
                moving_speed_threshold=0.1,
                stopped_speed_threshold=0.1,
                cache=encoder_cache_by_tls[tls_id],
            )

            if print_state_preview and tls_id not in printed_dim_for_tls:
                printed_dim_for_tls.add(tls_id)
                print(f"[state] tls={tls_id} dim={state_vec.shape[0]}")
                print(f"[state] tls={tls_id} feat={state_vec[:].tolist()}")

            # ----------------- Crazy policy (unchanged) -------------------
            n = len(st.perm)
            tick = int(sim_t)  # or keep your old tick formula if you prefer

            if int(sim_t) % 53 == 0:
                st.perm.reverse()
                st.k = (n - 1 - st.k)
            elif int(sim_t) % 37 == 0:
                st.k = (st.k + 3) % n
            elif int(sim_t) % 11 == 0:
                st.k = (st.k + 2) % n

            phase = st.perm[st.k]
            st.k = (st.k + 1) % n
            duration = fib_durations[tick % len(fib_durations)]

            traci.trafficlight.setPhase(tls_id, phase)
            # traci.trafficlight.setPhaseDuration(tls_id, float(duration))
            traci.trafficlight.setPhaseDuration(tls_id, 1e6) # override auto progression

            # NEW: lock out further decisions until this duration ends
            st.next_decision_time = sim_t + float(duration)

            print(f"[t={sim_t:6.1f}] tls={tls_id} -> phase={phase} for {duration}s")

        printed_dim_for_tls.clear()

        traci.simulationStep()

    traci.close()
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--sumocfg", required=True, help="Path to .sumocfg")
    ap.add_argument("--gui", action="store_true", help="Use sumo-gui (recommended for debugging)")
    ap.add_argument("--max-time", type=float, default=300.0, help="Stop after this simulation time (s)")
    ap.add_argument("--decision-interval", type=float, default=7.0, help="How often to change signals (s)")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    ap.add_argument("--delay-ms", type=int, default=50, help="GUI delay per step (ms)")
    ap.add_argument("--no-state-preview", action="store_true", help="Disable one-time state dim/preview printing")
    args = ap.parse_args()

    run_crazy_tsc(
        sumocfg=args.sumocfg,
        gui=args.gui,
        max_time=args.max_time,
        decision_interval=args.decision_interval,
        seed=args.seed,
        delay_ms=args.delay_ms,
        print_state_preview=(not args.no_state_preview),
    )
