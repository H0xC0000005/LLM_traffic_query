# run_dqn_tsc.py
from __future__ import annotations

from csv import writer
import os
import sys
import argparse
import random
from dataclasses import dataclass
from typing import Dict, Optional, Deque, Tuple
from collections import deque

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time

# Ensure SUMO tools are on path before importing traci/sumolib (works whether traci is pip-installed or from SUMO_HOME)
def ensure_sumo_tools_on_path() -> None:
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)

ensure_sumo_tools_on_path()

import traci  # noqa: E402
from sumolib import checkBinary  # noqa: E402

from dqn_agent import DQNAgent
from scene_encoder import encode_tsc_state_vector, encode_tsc_state_vector_bounded  # unchanged encoder
from utility import throughput_tracker_step
from utility import reward_throughput_plus_top2_queue


@dataclass
class TLSControllerState:
    next_decision_time: float = 0.0
    pending_state: Optional[np.ndarray] = None   # s
    pending_action: Optional[int] = None         # a
    log_step: int = 0   # NEW: per-TLS scalar step for reward logging


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buf)

    def add(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.buf.append((s, int(a), float(r), s2, 1.0 if done else 0.0))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.random.choice(len(self.buf), size=batch_size, replace=False)
        states = np.stack([self.buf[i][0] for i in idx]).astype(np.float32)
        actions = np.array([self.buf[i][1] for i in idx], dtype=np.int64)
        rewards = np.array([self.buf[i][2] for i in idx], dtype=np.float32)
        next_states = np.stack([self.buf[i][3] for i in idx]).astype(np.float32)
        dones = np.array([self.buf[i][4] for i in idx], dtype=np.float32)
        return states, actions, rewards, next_states, dones


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


def epsilon_schedule(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    # Keep eps_start until training actually begins (step==0), then decay.
    if step <= 0:
        return float(eps_start)
    if decay_steps <= 0:
        return float(eps_end)
    t = min(1.0, float(step) / float(decay_steps))
    return float(eps_start + t * (eps_end - eps_start))

def run_dqn_tsc(
    sumocfg: str,
    *,
    gui: bool,
    max_time: float,
    seed: int,
    delay_ms: int,
    action_hold_s: float,
    # OLD: epsilon: float,
    epsilon_start: float,
    epsilon_final: float,
    epsilon_decay_steps: int,
    device: Optional[str] = None,
    # ---- training hyperparams (minimal) ----
    gamma: float = 0.99,
    lr: float = 1e-3,
    buffer_capacity: int = 50_000,
    train_start: int = 2_000,
    batch_size: int = 64,
    train_freq: int = 1,                 # train every N transitions added
    target_update_every: int = 1_000,     # hard update steps
    tb_logdir: str = "runs",
    tb_run_name: Optional[str] = None,
    # ---- reward params (NEW) ----
    throughput_ref_veh_per_s: float = 0.30,
    queue_ref_veh: float = 15.0,
    w_throughput: float = 1.0,
    w_queue: float = 1.0,
    queue_power: float = 1.0,
    top2_w1: float = 0.7,
    top2_w2: float = 0.3,
    reward_clip_lo: float = -1.0,
    reward_clip_hi: float = 1.0,
) -> None:
    if tb_run_name is None:
        tb_run_name = f"sumo_dqn_seed{seed}_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join(tb_logdir, tb_run_name))
    random.seed(seed)
    np.random.seed(seed)

    start_sumo(sumocfg, gui=gui, delay_ms=delay_ms)
    traci.simulation.setScale(0.75)

    tls_ids = list(traci.trafficlight.getIDList())
    if not tls_ids:
        traci.close()
        raise RuntimeError("No traffic lights found in this scenario.")

    tls_state: Dict[str, TLSControllerState] = {tls_id: TLSControllerState() for tls_id in tls_ids}
    encoder_cache_by_tls: Dict[str, dict] = {tls_id: {} for tls_id in tls_ids}

    agents: Dict[str, DQNAgent] = {}
    buffers: Dict[str, ReplayBuffer] = {tls_id: ReplayBuffer(buffer_capacity) for tls_id in tls_ids}

    transitions_added = 0

    try:
        while True:
            sim_t = float(traci.simulation.getTime())
            if sim_t >= max_time:
                done_episode = True
            else:
                done_episode = (traci.simulation.getMinExpectedNumber() <= 0)

            # If episode is done, flush pending transitions and exit
            if done_episode:
                for tls_id in tls_ids:
                    st = tls_state[tls_id]
                    if st.pending_state is None or st.pending_action is None:
                        continue

                    # terminal_state = encode_tsc_state_vector(
                    #     tls_id,
                    #     moving_speed_threshold=0.1,
                    #     stopped_speed_threshold=0.1,
                    #     cache=encoder_cache_by_tls[tls_id],
                    # ).astype(np.float32)
                    terminal_state = encode_tsc_state_vector_bounded(
                        tls_id,
                        moving_speed_threshold=0.1,
                        stopped_speed_threshold=0.1,
                        cache=encoder_cache_by_tls[tls_id],
                    ).astype(np.float32)

                    # num_lanes = len(encoder_cache_by_tls[tls_id].get("lane_ids", []))
                    # r = reward_avg_queue_from_encoded_state(terminal_state, num_lanes=num_lanes)
                    # reward is throughput (veh/s) since last decision up to terminal sim_t
                    # r = reward_throughput_per_second_on_decision(
                    #     sim_time=sim_t,
                    #     cache=encoder_cache_by_tls[tls_id],
                    # )
                    num_lanes = len(encoder_cache_by_tls[tls_id].get("lane_ids", []))
                    if num_lanes <= 0:
                        raise RuntimeError(f"encoder cache missing lane_ids for tls={tls_id}")

                    r = reward_throughput_plus_top2_queue(
                        tls_id=tls_id,
                        sim_time=sim_t,
                        state_vec=terminal_state,
                        cache=encoder_cache_by_tls[tls_id],
                        num_lanes=num_lanes,
                        throughput_ref_veh_per_s=throughput_ref_veh_per_s,
                        queue_ref_veh=queue_ref_veh,
                        w_throughput=w_throughput,
                        w_queue=w_queue,
                        top2_weights=(top2_w1, top2_w2),
                        queue_power=queue_power,
                        reward_clip=(reward_clip_lo, reward_clip_hi),
                    )


                    buffers[tls_id].add(st.pending_state, st.pending_action, r, terminal_state, done=True)
                break

            # Normal step: decide per TLS when prior hold expires
            for tls_id in tls_ids:
                st = tls_state[tls_id]
                if sim_t < st.next_decision_time:
                    continue

                # Current observation (this will be s' for the previous action if one is pending)
                cur_state = encode_tsc_state_vector(
                    tls_id,
                    moving_speed_threshold=0.1,
                    stopped_speed_threshold=0.1,
                    cache=encoder_cache_by_tls[tls_id],
                ).astype(np.float32)

                # Lazy init agent once state_dim/action_dim known
                if tls_id not in agents:
                    action_dim = get_phase_count(tls_id)
                    agents[tls_id] = DQNAgent(
                        state_dim=int(cur_state.shape[0]),
                        action_dim=action_dim,
                        seed=seed,
                        device=device,
                        lr=lr,
                    )
                    print(f"[init] tls={tls_id} state_dim={cur_state.shape[0]} action_dim={action_dim}")

                # If we have a previous (s,a) pending, compute reward from current encoded state and store transition
                if st.pending_state is not None and st.pending_action is not None:
                    # num_lanes = len(encoder_cache_by_tls[tls_id].get("lane_ids", []))
                    # r = reward_avg_queue_from_encoded_state(cur_state, num_lanes=num_lanes)

                    # reward is throughput (veh/s) between last decision time and *now*
                    # r = reward_throughput_per_second_on_decision(
                    #     sim_time=sim_t,
                    #     cache=encoder_cache_by_tls[tls_id],
                    # )
                    num_lanes = len(encoder_cache_by_tls[tls_id].get("lane_ids", []))
                    if num_lanes <= 0:
                        raise RuntimeError(f"encoder cache missing lane_ids for tls={tls_id}")

                    r = reward_throughput_plus_top2_queue(
                        tls_id=tls_id,
                        sim_time=sim_t,
                        state_vec=cur_state,
                        cache=encoder_cache_by_tls[tls_id],
                        num_lanes=num_lanes,
                        throughput_ref_veh_per_s=throughput_ref_veh_per_s,
                        queue_ref_veh=queue_ref_veh,
                        w_throughput=w_throughput,
                        w_queue=w_queue,
                        top2_weights=(top2_w1, top2_w2),
                        queue_power=queue_power,
                        reward_clip=(reward_clip_lo, reward_clip_hi),
                    )

                    buffers[tls_id].add(st.pending_state, st.pending_action, r, cur_state, done=False)
                    transitions_added += 1

                    # tensorboard logging
                    st.log_step += 1
                    writer.add_scalar(f"{tls_id}/reward", float(r), st.log_step)
                    writer.add_scalar(f"{tls_id}/epsilon", float(eps), st.log_step)  # optional but useful

                    # Train (minimal): sample random minibatch from replay
                    if len(buffers[tls_id]) >= train_start and (transitions_added % train_freq == 0):
                        batch = buffers[tls_id].sample(batch_size)
                        loss = agents[tls_id].update(batch, gamma=gamma)

                        writer.add_scalar(f"{tls_id}/loss", float(loss), agents[tls_id].train_steps)

                        # Hard-update target network periodically
                        if agents[tls_id].train_steps % target_update_every == 0:
                            agents[tls_id].update_target()

                        if agents[tls_id].train_steps % 200 == 0:
                            print(f"[train] tls={tls_id} step={agents[tls_id].train_steps} loss={loss:.4f} r={r:.3f}")

                # Choose next action using current state
                eps = epsilon_schedule(
                    agents[tls_id].train_steps,
                    epsilon_start,
                    epsilon_final,
                    epsilon_decay_steps,
                )
                action_phase = agents[tls_id].act(cur_state, epsilon=eps)

                # Apply action
                traci.trafficlight.setPhase(tls_id, int(action_phase))

                # Keep SUMO from auto-advancing inside the hold window (controller decides when to switch)
                # Hold timing is controlled by next_decision_time below.
                traci.trafficlight.setPhaseDuration(tls_id, 1e6)

                st.next_decision_time = sim_t + float(action_hold_s)

                # Mark this (s,a) as pending until the next decision time, when we'll observe s' and compute reward
                st.pending_state = cur_state
                st.pending_action = int(action_phase)

                print(f"[t={sim_t:6.1f}] tls={tls_id} a(phase)={action_phase} hold={action_hold_s}s")

            traci.simulationStep()

            # Accumulate which vehicles newly ENTERED downstream lanes during this step
            for tls_id in tls_ids:
                throughput_tracker_step(tls_id, encoder_cache_by_tls[tls_id])
    except Exception as e:
        print(f"[error] Exception during simulation: {e}")
        raise
    finally:
        writer.flush()
        writer.close()
        traci.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--sumocfg", required=True)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--max-time", type=float, default=3000.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--delay-ms", type=int, default=50)
    ap.add_argument("--hold", type=float, default=5.0)

    # Exploration (untrained -> start high; later you can decay)
    ap.add_argument("--epsilon-start", type=float, default=1.0)
    ap.add_argument("--epsilon-final", type=float, default=0.1)
    ap.add_argument("--epsilon-decay", type=int, default=20000, help="decay steps measured in gradient updates")

    # Training knobs (minimal)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--buffer", type=int, default=50_000)
    ap.add_argument("--train-start", type=int, default=2_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--target-update", type=int, default=1_000)
    ap.add_argument("--device", type=str, default=None, help="e.g. cpu or cuda (optional)")
    ap.add_argument("--tb-logdir", type=str, default="runs", help="TensorBoard log directory")
    ap.add_argument("--tb-run-name", type=str, default=None, help="Optional run name (subfolder)")

    # combined reward options
    ap.add_argument("--thr-ref", type=float, default=0.30, help="Throughput normalization ref (veh/s)")
    ap.add_argument("--queue-ref", type=float, default=15.0, help="Queue normalization ref (veh)")
    ap.add_argument("--w-thr", type=float, default=1.0)
    ap.add_argument("--w-queue", type=float, default=1.0)
    ap.add_argument("--queue-power", type=float, default=1.0)
    ap.add_argument("--top2-w1", type=float, default=0.7)
    ap.add_argument("--top2-w2", type=float, default=0.3)
    ap.add_argument("--reward-clip-lo", type=float, default=-1.0)
    ap.add_argument("--reward-clip-hi", type=float, default=1.0)
    args = ap.parse_args()

    run_dqn_tsc(
        args.sumocfg,
        gui=args.gui,
        max_time=args.max_time,
        seed=args.seed,
        delay_ms=args.delay_ms,
        action_hold_s=args.hold,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        epsilon_decay_steps=args.epsilon_decay,
        device=args.device,
        gamma=args.gamma,
        lr=args.lr,
        buffer_capacity=args.buffer,
        train_start=args.train_start,
        batch_size=args.batch,
        target_update_every=args.target_update,
        tb_logdir=args.tb_logdir,
        tb_run_name=args.tb_run_name,

        throughput_ref_veh_per_s=args.thr_ref,
        queue_ref_veh=args.queue_ref,
        w_throughput=args.w_thr,
        w_queue=args.w_queue,
        queue_power=args.queue_power,
        top2_w1=args.top2_w1,
        top2_w2=args.top2_w2,
        reward_clip_lo=args.reward_clip_lo,
        reward_clip_hi=args.reward_clip_hi,
    )