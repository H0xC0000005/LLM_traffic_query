# run_dqn_tsc.py
from __future__ import annotations

import os
import sys
import argparse
import random
import time
from dataclasses import dataclass
from typing import Dict, Optional, Deque, Tuple
from collections import deque

import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Ensure SUMO tools are on path before importing traci/sumolib
def ensure_sumo_tools_on_path() -> None:
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)


ensure_sumo_tools_on_path()

import traci  # noqa: E402
from sumolib import checkBinary  # noqa: E402

from dqn_agent import DQNAgent
from scene_encoder import encode_tsc_state_vector  # unchanged encoder
from utility import throughput_tracker_step
from utility import reward_throughput_plus_top2_queue


@dataclass
class TLSControllerState:
    next_decision_time: float = 0.0
    pending_state: Optional[np.ndarray] = None   # s
    pending_action: Optional[int] = None         # a


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


def start_sumo(sumocfg: str, gui: bool, delay_ms: int, *, sumo_seed: int) -> None:
    binary = checkBinary("sumo-gui" if gui else "sumo")
    cmd = [
        binary,
        "-c", sumocfg,
        "--start",
        "--no-step-log", "true",
        "--delay", str(delay_ms),
        "--seed", str(sumo_seed),
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
    # decay by gradient update steps (agent.train_steps)
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
    max_time: float,                 # TOTAL sim seconds across episodes
    seed: int,
    delay_ms: int,
    action_hold_s: float,
    epsilon_start: float,
    epsilon_final: float,
    epsilon_decay_steps: int,
    device: Optional[str] = None,
    # ---- episodic control ----
    episode_len_s: float = 3600.0,   # per-episode sim seconds
    warmup_s: float = 300.0,         # per-episode warmup sim seconds (no control/no storage/no training)
    episodes: int = 0,               # 0 => run until max_time total; else run fixed count
    # ---- training hyperparams ----
    gamma: float = 0.99,
    lr: float = 1e-3,
    buffer_capacity: int = 50_000,
    train_start: int = 2_000,
    batch_size: int = 64,
    train_freq: int = 1,
    target_update_every: int = 1_000,
    tb_logdir: str = "runs",
    tb_run_name: Optional[str] = None,
    # ---- reward params ----
    throughput_ref_veh_per_s: float = 0.30,
    queue_ref_veh: float = 15.0,
    w_throughput: float = 1.0,
    w_queue: float = 1.0,
    queue_power: float = 1.0,
    top2_w1: float = 0.7,
    top2_w2: float = 0.3,
    reward_clip_lo: float = -1.0,
    reward_clip_hi: float = 1.0,
    # ---- traffic scaling (optional) ----
    traffic_scale: Optional[float] = 0.75,  # None to skip calling traci.simulation.setScale
) -> None:
    if tb_run_name is None:
        tb_run_name = f"sumo_dqn_seed{seed}_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join(tb_logdir, tb_run_name))

    random.seed(seed)
    np.random.seed(seed)

    # Keep agents/buffers across episodes
    agents: Dict[str, DQNAgent] = {}
    buffers: Dict[str, ReplayBuffer] = {}
    transitions_added = 0

    # For stable tensorboard x-axis (per TLS)
    transition_step_by_tls: Dict[str, int] = {}
    decision_step_by_tls: Dict[str, int] = {}

    total_elapsed = 0.0
    episode_idx = 0

    try:
        while True:
            if episodes > 0 and episode_idx >= episodes:
                break
            if total_elapsed >= max_time:
                break

            remaining = max_time - total_elapsed
            this_ep_limit = min(float(episode_len_s), float(max(0.0, remaining)))

            sumo_seed = int(seed + episode_idx)
            start_sumo(sumocfg, gui=gui, delay_ms=delay_ms, sumo_seed=sumo_seed)

            # Apply traffic scaling if desired (TraCI supports changing simulation scale)
            if traffic_scale is not None:
                try:
                    traci.simulation.setScale(float(traffic_scale))
                except Exception:
                    # ignore if unsupported in this SUMO build
                    pass

            tls_ids = list(traci.trafficlight.getIDList())
            if not tls_ids:
                traci.close()
                raise RuntimeError("No traffic lights found in this scenario.")

            # Per-episode controller state + encoder cache (reset each episode)
            tls_state: Dict[str, TLSControllerState] = {tls_id: TLSControllerState() for tls_id in tls_ids}
            encoder_cache_by_tls: Dict[str, dict] = {tls_id: {} for tls_id in tls_ids}

            # Ensure persistent buffer/log counters exist
            for tls_id in tls_ids:
                if tls_id not in buffers:
                    buffers[tls_id] = ReplayBuffer(buffer_capacity)
                transition_step_by_tls.setdefault(tls_id, 0)
                decision_step_by_tls.setdefault(tls_id, 0)

            # Episode loop
            try:
                while True:
                    sim_t = float(traci.simulation.getTime())

                    # episode end conditions
                    done_network = (traci.simulation.getMinExpectedNumber() <= 0)
                    done_time = (sim_t >= this_ep_limit)
                    if done_network or done_time:
                        # flush pending transitions as terminal
                        for tls_id in tls_ids:
                            st = tls_state[tls_id]
                            if st.pending_state is None or st.pending_action is None:
                                continue

                            terminal_state = encode_tsc_state_vector(
                                tls_id,
                                moving_speed_threshold=0.1,
                                stopped_speed_threshold=0.1,
                                cache=encoder_cache_by_tls[tls_id],
                            ).astype(np.float32)

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
                            transitions_added += 1
                            transition_step_by_tls[tls_id] += 1
                            writer.add_scalar(f"{tls_id}/reward", float(r), transition_step_by_tls[tls_id])

                            # clear
                            st.pending_state = None
                            st.pending_action = None
                        break

                    # Warm-up phase: do not control / do not store / do not train
                    if sim_t < float(warmup_s):
                        traci.simulationStep()
                        for tls_id in tls_ids:
                            throughput_tracker_step(tls_id, encoder_cache_by_tls[tls_id])
                        continue

                    # Control phase: per TLS act when hold expires
                    for tls_id in tls_ids:
                        st = tls_state[tls_id]
                        if sim_t < st.next_decision_time:
                            continue

                        cur_state = encode_tsc_state_vector(
                            tls_id,
                            moving_speed_threshold=0.1,
                            stopped_speed_threshold=0.1,
                            cache=encoder_cache_by_tls[tls_id],
                        ).astype(np.float32)

                        # Lazy init agent once dims known
                        if tls_id not in agents:
                            action_dim = get_phase_count(tls_id)
                            agents[tls_id] = DQNAgent(
                                state_dim=int(cur_state.shape[0]),
                                action_dim=int(action_dim),
                                seed=seed,
                                device=device,
                                lr=lr,
                            )
                            print(f"[init] tls={tls_id} state_dim={cur_state.shape[0]} action_dim={action_dim}")

                        # If previous (s,a) pending, close it with reward at current decision time
                        if st.pending_state is not None and st.pending_action is not None:
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
                            transition_step_by_tls[tls_id] += 1

                            writer.add_scalar(f"{tls_id}/reward", float(r), transition_step_by_tls[tls_id])
                            writer.add_scalar(f"{tls_id}/epsilon", float(last_eps), transition_step_by_tls[tls_id])

                            # Train online
                            if len(buffers[tls_id]) >= train_start and (transitions_added % train_freq == 0):
                                batch = buffers[tls_id].sample(batch_size)
                                loss = agents[tls_id].update(batch, gamma=gamma)
                                writer.add_scalar(f"{tls_id}/loss", float(loss), agents[tls_id].train_steps)

                                if agents[tls_id].train_steps % target_update_every == 0:
                                    agents[tls_id].update_target()

                                if agents[tls_id].train_steps % 200 == 0:
                                    print(
                                        f"[train] tls={tls_id} step={agents[tls_id].train_steps} "
                                        f"loss={loss:.4f} r={r:.3f}"
                                    )
                        else:
                            # First action in this episode after warmup:
                            # initialize throughput interval bookkeeping so warmup throughput doesn't leak into reward.
                            num_lanes = len(encoder_cache_by_tls[tls_id].get("lane_ids", []))
                            if num_lanes > 0:
                                _ = reward_throughput_plus_top2_queue(
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

                        # Choose action (epsilon-greedy)
                        eps = epsilon_schedule(
                            agents[tls_id].train_steps,
                            epsilon_start,
                            epsilon_final,
                            epsilon_decay_steps,
                        )
                        last_eps = eps  # used for logging when the transition is closed
                        action_phase = agents[tls_id].act(cur_state, epsilon=eps)

                        # Apply action, disable SUMO auto-advancing within hold window
                        traci.trafficlight.setPhase(tls_id, int(action_phase))
                        traci.trafficlight.setPhaseDuration(tls_id, 1e6)

                        st.next_decision_time = sim_t + float(action_hold_s)
                        st.pending_state = cur_state
                        st.pending_action = int(action_phase)

                        decision_step_by_tls[tls_id] += 1
                        print(f"[ep={episode_idx} t={sim_t:6.1f}] tls={tls_id} a(phase)={action_phase} hold={action_hold_s}s")

                    # Step simulation, then update throughput tracker once per sim step
                    traci.simulationStep()
                    for tls_id in tls_ids:
                        throughput_tracker_step(tls_id, encoder_cache_by_tls[tls_id])

            finally:
                # close SUMO for this episode
                try:
                    traci.close()
                except Exception:
                    pass

            # Update total elapsed by how long this episode ran
            # (simulation time restarts at 0 each episode)
            total_elapsed += float(min(this_ep_limit, float(episode_len_s)))
            episode_idx += 1

    finally:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--sumocfg", required=True)
    ap.add_argument("--gui", action="store_true")

    # total run length across episodes (seconds)
    ap.add_argument("--max-time", type=float, default=300.0)

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--delay-ms", type=int, default=50)
    ap.add_argument("--hold", type=float, default=5.0)

    # episodic reset controls
    ap.add_argument("--episode-len", type=float, default=3600.0)
    ap.add_argument("--warmup", type=float, default=300.0)
    ap.add_argument("--episodes", type=int, default=0, help="0 => run until --max-time, else fixed episode count")

    # exploration
    ap.add_argument("--epsilon-start", type=float, default=1.0)
    ap.add_argument("--epsilon-final", type=float, default=0.1)
    ap.add_argument("--epsilon-decay", type=int, default=20000, help="decay steps measured in gradient updates")

    # training knobs
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--buffer", type=int, default=50_000)
    ap.add_argument("--train-start", type=int, default=2_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--target-update", type=int, default=1_000)
    ap.add_argument("--device", type=str, default=None, help="e.g. cpu or cuda (optional)")
    ap.add_argument("--tb-logdir", type=str, default="runs")
    ap.add_argument("--tb-run-name", type=str, default=None)

    # combined reward options
    ap.add_argument("--thr-ref", type=float, default=0.30)
    ap.add_argument("--queue-ref", type=float, default=15.0)
    ap.add_argument("--w-thr", type=float, default=1.0)
    ap.add_argument("--w-queue", type=float, default=1.0)
    ap.add_argument("--queue-power", type=float, default=1.0)
    ap.add_argument("--top2-w1", type=float, default=0.7)
    ap.add_argument("--top2-w2", type=float, default=0.3)
    ap.add_argument("--reward-clip-lo", type=float, default=-1.0)
    ap.add_argument("--reward-clip-hi", type=float, default=1.0)

    # traffic scaling (optional)
    ap.add_argument("--traffic-scale", type=float, default=0.75, help="TraCI simulation.setScale (set <0 to disable)")

    args = ap.parse_args()

    traffic_scale = None if args.traffic_scale is None or float(args.traffic_scale) < 0.0 else float(args.traffic_scale)

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
        episode_len_s=args.episode_len,
        warmup_s=args.warmup,
        episodes=args.episodes,
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
        traffic_scale=traffic_scale,
    )
