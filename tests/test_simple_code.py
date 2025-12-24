import os, sys
import time
from sumolib import checkBinary
import traci

def start_sumo_gui(sumocfg_path: str):
    sumoBinary = checkBinary("sumo-gui")  # GUI for debugging
    sumoCmd = [
        sumoBinary,
        "-c", sumocfg_path,
        "--start",                 # auto-start in GUI
        "--quit-on-end", "true",   # close when done
    ]
    traci.start(sumoCmd)

start_sumo_gui("E:\Sumo\sumo_maps\simple_4leg_intersection\simple_single_intersection.sumocfg")
for _ in range(100):
    traci.simulationStep()
    time.sleep(0.1)  # slow down for visualization
traci.close()