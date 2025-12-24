from sumolib import checkBinary
import traci

def run_tsc(sumocfg_path: str, decision_interval_s: int = 5, max_steps: int = 3600):
    sumoCmd = [checkBinary("sumo-gui"), "-c", sumocfg_path, "--start"]
    traci.start(sumoCmd)

    tls_ids = traci.trafficlight.getIDList()
    if not tls_ids:
        traci.close()
        raise RuntimeError("No traffic lights found.")
    tls_id = tls_ids[0]

    # You must map these to your network’s real phase indices
    PHASE_A = 0
    PHASE_B = 2

    step = 0
    while step < max_steps and traci.simulation.getMinExpectedNumber() > 0:
        # ---- Observe (examples) ----
        # Option 1: detector-based (like the SUMO tutorial)
        # count = traci.inductionloop.getLastStepVehicleNumber("detectorID")

        # Option 2: queue estimate on controlled lanes
        lanes = set(traci.trafficlight.getControlledLanes(tls_id))
        queues = {ln: traci.lane.getLastStepHaltingNumber(ln) for ln in lanes}

        # ---- Decide & Act ----
        if step % decision_interval_s == 0:
            action = your_model_policy(queues)  # return PHASE_A or PHASE_B (or any valid index)
            traci.trafficlight.setPhase(tls_id, action)

        traci.simulationStep()
        step += 1

    traci.close()

def your_model_policy(queues: dict) -> int:
    # placeholder policy
    return 0
