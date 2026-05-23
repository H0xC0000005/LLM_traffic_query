from __future__ import annotations

import json
import os
from pathlib import Path
import sys

# Ensure SUMO tools are importable before importing traci/sumolib
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)

import argparse
import random
import time
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple, List

import numpy as np
import traci
from torch.utils.tensorboard import SummaryWriter
import torch

from dqn_agent import DQNAgent
from scene_encoder import encode_tsc_state_vector_bounded
from utility import *


@dataclass
class ReplayBuffer:
    """Replay buffer storing transitions with sim time to support span-based warm starts."""
    capacity: int
    data: Deque[Tuple[float, np.ndarray, int, float, np.ndarray, bool]] = field(default_factory=deque)

    def add(self, t: float, s: np.ndarray, a: int, r: float, s2: np.ndarray, *, done: bool) -> None:
        if len(self.data) >= self.capacity:
            self.data.popleft()
        # Store copies to avoid accidental mutation
        self.data.append((float(t), np.asarray(s, dtype=np.float32), int(a), float(r), np.asarray(s2, dtype=np.float32), bool(done)))

    def __len__(self) -> int:
        return len(self.data)

    def span_s(self) -> float:
        if len(self.data) < 2:
            return 0.0
        return float(self.data[-1][0] - self.data[0][0])

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size <= 0 or batch_size > len(self.data):
            raise ValueError("bad batch_size")
        idx = np.random.choice(len(self.data), size=batch_size, replace=False)
        s, a, r, s2, d = [], [], [], [], []
        for i in idx:
            _t, si, ai, ri, s2i, di = self.data[int(i)]
            s.append(si)
            a.append(ai)
            r.append(ri)
            s2.append(s2i)
            d.append(1.0 if di else 0.0)
        return (
            np.stack(s).astype(np.float32),
            np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32),
            np.stack(s2).astype(np.float32),
            np.asarray(d, dtype=np.float32),
        )




def epsilon_schedule(step: int, eps_start: float, eps_final: float, decay_steps: int) -> float:
    """Linear epsilon decay by gradient steps."""
    if decay_steps <= 0:
        return float(eps_final)
    frac = min(max(step, 0), decay_steps) / float(decay_steps)
    return float(eps_start + frac * (eps_final - eps_start))


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
    episodes: int,
    episode_len_s: float,
    warmup_s: float,
    seed: int,
    sumo_seed: int,
    delay_ms: int,
    action_hold_s: float,
    epsilon_start: float,
    epsilon_final: float,
    epsilon_decay_steps: int,
    device: Optional[str],
    gamma: float,
    lr: float,
    buffer_size: int,
    train_start_s: float,
    batch_size: int,
    train_freq: int,
    target_update_every_s: float,
    throughput_ref_veh_per_s: float,
    queue_ref_veh: float,
    w_throughput: float,
    w_queue: float,
    queue_power: float,
    top2_w1: float,
    top2_w2: float,
    reward_clip_lo: float,
    reward_clip_hi: float,
    traffic_scale: float,
    tb_logdir: str,
    save_dir: str = "dqn_tsc_models",
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    # TensorBoard x-axis steps must be monotonic across episodes
    tb_step_decision = defaultdict(int)    # per TLS: increments on each decision
    tb_step_transition = defaultdict(int)  # per TLS: increments on each stored transition
    tb_step_episode = defaultdict(int)     # per TLS: increments once per episode (optional)
    encoder_fn = encode_tsc_state_vector_bounded

    run_name = f"sumo_dqn_seed{seed}_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join(tb_logdir, run_name))

    buffers: Dict[str, ReplayBuffer] = {}
    agents: Dict[str, DQNAgent] = {}

    total_elapsed = 0.0
    for ep in range(int(episodes)):
        if total_elapsed >= max_time:
            break

        start_sumo(sumocfg, gui=gui, delay_ms=delay_ms, sumo_seed=sumo_seed + ep, traffic_scale=traffic_scale)
        tls_ids = list(traci.trafficlight.getIDList())
        if not tls_ids:
            traci.close()
            raise RuntimeError("No traffic lights found in this scenario.")

        tls_state: Dict[str, TLSControllerState] = {tls_id: TLSControllerState() for tls_id in tls_ids}
        encoder_cache_by_tls: Dict[str, dict] = {tls_id: {} for tls_id in tls_ids}

        transitions_added = 0
        in_control_phase = False
        ep_start_wall = time.time()

        try:
            while True:
                sim_t = float(traci.simulation.getTime())

                if sim_t >= float(episode_len_s):
                    done_episode = True
                else:
                    done_episode = (traci.simulation.getMinExpectedNumber() <= 0)

                if not in_control_phase and sim_t >= float(warmup_s):
                    in_control_phase = True

                # If episode is done, flush pending transitions and exit
                if done_episode:
                    if in_control_phase:
                        for tls_id in tls_ids:
                            st = tls_state[tls_id]
                            if not st.in_control_when_pending or st.pending_state is None or st.pending_action is None:
                                continue

                            terminal_state = encoder_fn(
                                tls_id, moving_speed_threshold=0.1, stopped_speed_threshold=0.1, cache=encoder_cache_by_tls[tls_id]
                            ).astype(np.float32)

                            num_lanes = len(encoder_cache_by_tls[tls_id].get("lane_ids", []))
                            num_lanes = max(1, num_lanes)

                            r = reward_throughput_plus_top2_queue(
                                tls_id=tls_id, sim_time=sim_t, state_vec=terminal_state, cache=encoder_cache_by_tls[tls_id],
                                num_lanes=num_lanes, throughput_ref_veh_per_s=throughput_ref_veh_per_s, queue_ref_veh=queue_ref_veh,
                                w_throughput=w_throughput, w_queue=w_queue, top2_weights=(top2_w1, top2_w2),
                                queue_power=queue_power, reward_clip=(reward_clip_lo, reward_clip_hi),
                            )

                            buffers[tls_id].add(sim_t, st.pending_state, st.pending_action, r, terminal_state, done=True)
                    break

                # Control phase: decide per TLS when prior hold expires
                if in_control_phase:
                    for tls_id in tls_ids:
                        st = tls_state[tls_id]
                        if sim_t < st.next_decision_time:
                            continue

                        cur_state = encoder_fn(
                            tls_id, moving_speed_threshold=0.1, stopped_speed_threshold=0.1, cache=encoder_cache_by_tls[tls_id]
                        ).astype(np.float32)

                        if tls_id not in agents:
                            action_dim = get_phase_count(tls_id)
                            agents[tls_id] = DQNAgent(
                                state_dim=int(cur_state.shape[0]), action_dim=action_dim, seed=seed, device=device, lr=lr
                            )
                            st.next_target_update_time = sim_t + float(target_update_every_s)
                            print(f"[init] tls={tls_id} state_dim={cur_state.shape[0]} action_dim={action_dim}")

                        if tls_id not in buffers:
                            buffers[tls_id] = ReplayBuffer(capacity=int(buffer_size))

                        # If we have a previous (s,a) pending, compute reward from current encoded state and store transition
                        if st.pending_state is not None and st.pending_action is not None and st.in_control_when_pending:
                            num_lanes = len(encoder_cache_by_tls[tls_id].get("lane_ids", []))
                            num_lanes = max(1, num_lanes)

                            r = reward_throughput_plus_top2_queue(
                                tls_id=tls_id, sim_time=sim_t, state_vec=cur_state, cache=encoder_cache_by_tls[tls_id],
                                num_lanes=num_lanes, throughput_ref_veh_per_s=throughput_ref_veh_per_s, queue_ref_veh=queue_ref_veh,
                                w_throughput=w_throughput, w_queue=w_queue, top2_weights=(top2_w1, top2_w2),
                                queue_power=queue_power, reward_clip=(reward_clip_lo, reward_clip_hi),
                            )

                            buffers[tls_id].add(sim_t, st.pending_state, st.pending_action, r, cur_state, done=False)
                            transitions_added += 1

                            tb_step_decision[tls_id] += 1  # monotonic across episodes
                            tb_step_transition[tls_id] += 1 
                            writer.add_scalar(f"{tls_id}/epsilon", float(st.pending_epsilon), tb_step_decision[tls_id])  # step never resets
                            writer.add_scalar(f"{tls_id}/reward", float(r), tb_step_transition[tls_id])  # step never resets
                            # writer.add_scalar(f"{tls_id}/reward", float(r), st.log_step)
                            # writer.add_scalar(f"{tls_id}/epsilon", float(st.pending_epsilon), st.log_step)

                            if buffers[tls_id].span_s() >= float(train_start_s) and len(buffers[tls_id]) >= batch_size:
                                if train_freq <= 1 or (transitions_added % train_freq == 0):
                                    batch = buffers[tls_id].sample(batch_size)
                                    loss = agents[tls_id].update(batch, gamma=gamma)
                                    writer.add_scalar(f"{tls_id}/loss", float(loss), agents[tls_id].train_steps)

                                    if sim_t >= st.next_target_update_time:
                                        agents[tls_id].update_target()
                                        st.next_target_update_time = sim_t + float(target_update_every_s)
                                if agents[tls_id].train_steps % 10 == 0:
                                    print(f"[train] tls={tls_id} step={agents[tls_id].train_steps} loss={loss:.4f} r={r:.3f}")

                        else:
                            # First controlled action: initialize throughput window (no transition stored)
                            num_lanes = len(encoder_cache_by_tls[tls_id].get("lane_ids", []))
                            num_lanes = max(1, num_lanes)
                            _ = reward_throughput_plus_top2_queue(
                                tls_id=tls_id, sim_time=sim_t, state_vec=cur_state, cache=encoder_cache_by_tls[tls_id],
                                num_lanes=num_lanes, throughput_ref_veh_per_s=throughput_ref_veh_per_s, queue_ref_veh=queue_ref_veh,
                                w_throughput=0.0, w_queue=0.0, top2_weights=(top2_w1, top2_w2), queue_power=queue_power,
                                reward_clip=(reward_clip_lo, reward_clip_hi),
                            )

                        # Choose next action
                        eps = epsilon_schedule(agents[tls_id].train_steps, epsilon_start, epsilon_final, epsilon_decay_steps)
                        action_phase = agents[tls_id].act(cur_state, epsilon=eps)

                        # Apply action
                        traci.trafficlight.setPhase(tls_id, int(action_phase))
                        traci.trafficlight.setPhaseDuration(tls_id, 1e6)  # prevent SUMO auto-advance

                        st.next_decision_time = sim_t + float(action_hold_s)
                        st.pending_state = cur_state
                        st.pending_action = int(action_phase)
                        st.pending_epsilon = float(eps)
                        st.in_control_when_pending = True

                        print(f"[ep={ep} t={sim_t:6.1f}] tls={tls_id} a(phase)={action_phase} hold={action_hold_s}s")

                # Advance SUMO by one simulation step
                traci.simulationStep()

                # Accumulate which vehicles newly ENTERED downstream lanes during this step
                for tls_id in tls_ids:
                    throughput_tracker_step(tls_id, encoder_cache_by_tls[tls_id])

            # end while

        finally:
            traci.close()
            ep_wall = time.time() - ep_start_wall
            total_elapsed += float(min(episode_len_s, max_time - total_elapsed))
            print(f"[episode] {ep+1}/{episodes} transitions={transitions_added} wall={ep_wall:.1f}s")

    writer.flush()
    writer.close()

    # ---- Save trained model(s) ----
    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    for tls_id, agent in agents.items():
        ckpt_path = save_root / f"{run_name}__{tls_id}.pt"
        meta = {
            "run_name": run_name,
            "tls_id": tls_id,
            "sumocfg": sumocfg,
            "seed": int(seed),
            "state_dim": int(agent.state_dim),
            "action_dim": int(agent.action_dim),
            "hidden_dim": int(agent.hidden_dim),
            "gamma": float(gamma),
            "lr": float(lr),
            "encoder": getattr(encoder_fn, "__name__", "<unknown>"),
            "train_steps": int(agent.train_steps),
            "saved_unix_time": float(time.time()),
        }

        torch.save({"meta": meta, "model_state_dict": agent.model.state_dict()}, ckpt_path)
        (ckpt_path.with_suffix(".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[save] tls={tls_id} -> {ckpt_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--sumocfg", required=True)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--max-time", type=float, default=300000.0)
    ap.add_argument("--episode-len", type=float, default=30000.0, help="sim seconds per episode")
    ap.add_argument("--warmup", type=float, default=200.0, help="warmup sim seconds (no control, no store)")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--sumo-seed", type=int, default=0)
    ap.add_argument("--delay-ms", type=int, default=10)
    ap.add_argument("--hold", type=float, default=10.0)
    ap.add_argument("--epsilon-start", type=float, default=1.0)
    ap.add_argument("--epsilon-final", type=float, default=0.1)
    ap.add_argument("--epsilon-decay", type=int, default=12000, help="decay steps in gradient updates")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--buffer", type=int, default=50000)
    ap.add_argument("--train-start", type=float, default=1500.0, help="required buffer span (sim seconds)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train-freq", type=int, default=1, help="train every N stored transitions")
    ap.add_argument("--target-update", type=float, default=1000.0, help="target sync period (sim seconds)")

    ap.add_argument("--thr-ref", type=float, default=0.30)
    ap.add_argument("--queue-ref", type=float, default=15.0)
    ap.add_argument("--w-thr", type=float, default=1.0)
    ap.add_argument("--w-queue", type=float, default=1.0)
    ap.add_argument("--queue-power", type=float, default=1.0)
    ap.add_argument("--top2-w1", type=float, default=0.7)
    ap.add_argument("--top2-w2", type=float, default=0.3)
    ap.add_argument("--reward-clip-lo", type=float, default=-5.0)
    ap.add_argument("--reward-clip-hi", type=float, default=2.0)

    ap.add_argument("--traffic-scale", type=float, default=1.0)
    ap.add_argument("--tb-logdir", type=str, default="tensorboard_logs")

    args = ap.parse_args()

    run_dqn_tsc(
        args.sumocfg,
        gui=args.gui,
        max_time=args.max_time,
        episodes=args.episodes,
        episode_len_s=args.episode_len,
        warmup_s=args.warmup,
        seed=args.seed,
        sumo_seed=args.sumo_seed,
        delay_ms=args.delay_ms,
        action_hold_s=args.hold,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        epsilon_decay_steps=args.epsilon_decay,
        device=args.device,
        gamma=args.gamma,
        lr=args.lr,
        buffer_size=args.buffer,
        train_start_s=args.train_start,
        batch_size=args.batch,
        train_freq=args.train_freq,
        target_update_every_s=args.target_update,
        throughput_ref_veh_per_s=args.thr_ref,
        queue_ref_veh=args.queue_ref,
        w_throughput=args.w_thr,
        w_queue=args.w_queue,
        queue_power=args.queue_power,
        top2_w1=args.top2_w1,
        top2_w2=args.top2_w2,
        reward_clip_lo=args.reward_clip_lo,
        reward_clip_hi=args.reward_clip_hi,
        traffic_scale=args.traffic_scale,
        tb_logdir=args.tb_logdir,
    )
