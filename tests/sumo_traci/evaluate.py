from __future__ import annotations

import os
import sys

# Ensure SUMO tools are importable before importing traci/sumolib
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import traci
from sumolib import checkBinary

from dqn_agent import DQN
from scene_encoder import encode_tsc_state_vector_bounded
from utility import *


@dataclass
class FixedScheduleState:
    idx: int = 0


@torch.no_grad()
def act_greedy(model: DQN, state_vec: np.ndarray, *, device: torch.device) -> int:
    x = torch.from_numpy(state_vec.astype(np.float32)).unsqueeze(0).to(device)
    q = model(x)
    return int(torch.argmax(q, dim=1).item())


def load_models(model_path: str) -> Tuple[Dict[str, DQN], Dict[str, dict]]:
    """Load one checkpoint file or a directory of .pt checkpoints."""
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(model_path)

    ckpt_files: List[Path]
    if p.is_dir():
        ckpt_files = sorted(p.glob("*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No .pt files found in directory: {model_path}")
    else:
        ckpt_files = [p]

    models: Dict[str, DQN] = {}
    metas: Dict[str, dict] = {}

    for f in ckpt_files:
        ckpt = torch.load(f, map_location="cpu")
        meta = dict(ckpt.get("meta", {}))
        tls_id = str(meta.get("tls_id") or f.stem)
        state_dim = int(meta.get("state_dim"))
        action_dim = int(meta.get("action_dim"))
        hidden_dim = int(meta.get("hidden_dim", 128))

        model = DQN(in_dim=state_dim, out_dim=action_dim, hidden_dim=hidden_dim)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        models[tls_id] = model
        metas[tls_id] = meta

    return models, metas


def fixed_schedule_next(fixed: FixedScheduleState, *, action_dim: int) -> Tuple[int, float]:
    phase, dur = FIXED_SCHEDULE[int(fixed.idx) % len(FIXED_SCHEDULE)]
    fixed.idx += 1
    return int(phase) % int(action_dim), float(dur)



def evaluate(
    sumocfg: str,
    *,
    policy: str,
    gui: bool,
    episode_len_s: float,
    warmup_s: float,
    hold_s: float,
    seed: int,
    sumo_seed: int,
    delay_ms: int,
    traffic_scale: float,
    # reward params
    throughput_ref_veh_per_s: float,
    queue_ref_veh: float,
    w_throughput: float,
    w_queue: float,
    queue_power: float,
    top2_w1: float,
    top2_w2: float,
    reward_clip_lo: float,
    reward_clip_hi: float,
    # models
    models_by_tls: Optional[Dict[str, DQN]] = None,
    meta_by_tls: Optional[Dict[str, dict]] = None,
) -> dict:
    """Run one SUMO episode and compute decision-interval rewards."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    encoder_fn = encode_tsc_state_vector_bounded

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if models_by_tls is not None:
        for m in models_by_tls.values():
            m.to(device)

    start_sumo(sumocfg, gui=gui, delay_ms=delay_ms, sumo_seed=sumo_seed, traffic_scale=traffic_scale)
    tls_ids = list(traci.trafficlight.getIDList())
    if not tls_ids:
        traci.close()
        raise RuntimeError("No traffic lights found in this scenario.")

    if policy == "dqn":
        if not models_by_tls:
            traci.close()
            raise ValueError("policy='dqn' requires --model")
        missing = [t for t in tls_ids if t not in models_by_tls]
        if missing:
            traci.close()
            raise ValueError(f"No checkpoint loaded for TLS IDs: {missing}")

    tls_state: Dict[str, TLSControllerState] = {tls_id: TLSControllerState() for tls_id in tls_ids}
    encoder_cache_by_tls: Dict[str, dict] = {tls_id: {} for tls_id in tls_ids}
    fixed_state_by_tls: Dict[str, FixedScheduleState] = {tls_id: FixedScheduleState() for tls_id in tls_ids}

    logs: List[dict] = []
    totals = {"reward_sum": 0.0, "reward_count": 0}
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

            # Episode end: flush last interval reward (if any) and exit
            if done_episode:
                if in_control_phase:
                    for tls_id in tls_ids:
                        st = tls_state[tls_id]
                        if not st.in_control_when_pending or st.pending_state is None or st.pending_action is None:
                            continue

                        terminal_state = encoder_fn(
                            tls_id,
                            moving_speed_threshold=0.1,
                            stopped_speed_threshold=0.1,
                            cache=encoder_cache_by_tls[tls_id],
                        ).astype(np.float32)

                        num_lanes = max(1, len(encoder_cache_by_tls[tls_id].get("lane_ids", [])))
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

                        logs.append(
                            {
                                "policy": policy,
                                "tls_id": tls_id,
                                "interval_end_t": sim_t,
                                "action": int(st.pending_action),
                                "reward": float(r),
                                "terminal": True,
                            }
                        )
                        totals["reward_sum"] += float(r)
                        totals["reward_count"] += 1
                break

            if in_control_phase:
                for tls_id in tls_ids:
                    st = tls_state[tls_id]
                    if sim_t < st.next_decision_time:
                        continue

                    cur_state = encoder_fn(
                        tls_id,
                        moving_speed_threshold=0.1,
                        stopped_speed_threshold=0.1,
                        cache=encoder_cache_by_tls[tls_id],
                    ).astype(np.float32)

                    action_dim = get_phase_count(tls_id)

                    # Close previous decision interval (s,a) -> reward based on current state
                    if st.pending_state is not None and st.pending_action is not None and st.in_control_when_pending:
                        num_lanes = max(1, len(encoder_cache_by_tls[tls_id].get("lane_ids", [])))
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

                        logs.append(
                            {
                                "policy": policy,
                                "tls_id": tls_id,
                                "interval_end_t": sim_t,
                                "action": int(st.pending_action),
                                "reward": float(r),
                                "terminal": False,
                            }
                        )
                        totals["reward_sum"] += float(r)
                        totals["reward_count"] += 1
                    else:
                        # First controlled action: initialize throughput window (no reward yet)
                        num_lanes = max(1, len(encoder_cache_by_tls[tls_id].get("lane_ids", [])))
                        _ = reward_throughput_plus_top2_queue(
                            tls_id=tls_id,
                            sim_time=sim_t,
                            state_vec=cur_state,
                            cache=encoder_cache_by_tls[tls_id],
                            num_lanes=num_lanes,
                            throughput_ref_veh_per_s=throughput_ref_veh_per_s,
                            queue_ref_veh=queue_ref_veh,
                            w_throughput=0.0,
                            w_queue=0.0,
                            top2_weights=(top2_w1, top2_w2),
                            queue_power=queue_power,
                            reward_clip=(reward_clip_lo, reward_clip_hi),
                        )

                    # Choose and apply next action
                    if policy == "dqn":
                        assert models_by_tls is not None
                        action_phase = act_greedy(models_by_tls[tls_id], cur_state, device=device)
                        hold_for = float(hold_s)
                    elif policy == "fixed":
                        action_phase, hold_for = fixed_schedule_next(fixed_state_by_tls[tls_id], action_dim=action_dim)
                    else:
                        raise ValueError("policy must be one of: dqn, fixed")

                    traci.trafficlight.setPhase(tls_id, int(action_phase))
                    traci.trafficlight.setPhaseDuration(tls_id, 1e6)  # prevent SUMO auto-advance

                    st.next_decision_time = sim_t + float(hold_for)
                    st.pending_state = cur_state
                    st.pending_action = int(action_phase)
                    st.in_control_when_pending = True

            # Advance simulation
            traci.simulationStep()
            for tls_id in tls_ids:
                throughput_tracker_step(tls_id, encoder_cache_by_tls[tls_id])

    finally:
        traci.close()

    wall_s = time.time() - ep_start_wall
    reward_mean = totals["reward_sum"] / max(1, int(totals["reward_count"]))

    return {
        "policy": policy,
        "episode_len_s": float(episode_len_s),
        "warmup_s": float(warmup_s),
        "hold_s": float(hold_s),
        "reward_sum": float(totals["reward_sum"]),
        "reward_count": int(totals["reward_count"]),
        "reward_mean": float(reward_mean),
        "wall_s": float(wall_s),
        "logs": logs,
        "fixed_schedule": FIXED_SCHEDULE if policy == "fixed" else None,
    }


def write_results(out_dir: Path, *, results: Dict[str, dict], args_dict: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "results.json").write_text(
        json.dumps({"args": args_dict, "results": results}, indent=2),
        encoding="utf-8",
    )

    # Decision logs as CSV (one row per decision interval)
    csv_path = out_dir / "decision_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["policy", "tls_id", "interval_end_t", "action", "reward", "terminal"],
        )
        w.writeheader()
        for res in results.values():
            for row in res.get("logs", []):
                w.writerow(
                    {
                        "policy": row.get("policy"),
                        "tls_id": row.get("tls_id"),
                        "interval_end_t": row.get("interval_end_t"),
                        "action": row.get("action"),
                        "reward": row.get("reward"),
                        "terminal": row.get("terminal"),
                    }
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--sumocfg", required=True)
    ap.add_argument("--model", type=str, default=None, help=".pt checkpoint or directory of .pt files (required for dqn)")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--episode-len", type=float, default=30000.0)
    ap.add_argument("--warmup", type=float, default=200.0)
    ap.add_argument("--hold", type=float, default=10.0, help="decision interval for DQN (fixed schedule uses its own durations)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--sumo-seed", type=int, default=0)
    ap.add_argument("--delay-ms", type=int, default=10)
    ap.add_argument("--traffic-scale", type=float, default=1.0)
    ap.add_argument("--out-dir", type=str, default="eval_results")

    # Reward params (must match training for apples-to-apples)
    ap.add_argument("--thr-ref", type=float, default=0.30)
    ap.add_argument("--queue-ref", type=float, default=15.0)
    ap.add_argument("--w-thr", type=float, default=1.0)
    ap.add_argument("--w-queue", type=float, default=1.0)
    ap.add_argument("--queue-power", type=float, default=1.0)
    ap.add_argument("--top2-w1", type=float, default=0.7)
    ap.add_argument("--top2-w2", type=float, default=0.3)
    ap.add_argument("--reward-clip-lo", type=float, default=-5.0)
    ap.add_argument("--reward-clip-hi", type=float, default=2.0)

    ap.add_argument(
        "--policies",
        type=str,
        default="dqn,fixed",
        help="comma-separated subset of: dqn,fixed",
    )

    args = ap.parse_args()
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    models_by_tls: Optional[Dict[str, DQN]] = None
    meta_by_tls: Optional[Dict[str, dict]] = None
    if "dqn" in policies:
        if args.model is None:
            raise ValueError("--model is required when evaluating policy 'dqn'")
        models_by_tls, meta_by_tls = load_models(args.model)

    results: Dict[str, dict] = {}
    for policy in policies:
        results[policy] = evaluate(
            args.sumocfg,
            policy=policy,
            gui=bool(args.gui),
            episode_len_s=float(args.episode_len),
            warmup_s=float(args.warmup),
            hold_s=float(args.hold),
            seed=int(args.seed),
            sumo_seed=int(args.sumo_seed),
            delay_ms=int(args.delay_ms),
            traffic_scale=float(args.traffic_scale),
            throughput_ref_veh_per_s=float(args.thr_ref),
            queue_ref_veh=float(args.queue_ref),
            w_throughput=float(args.w_thr),
            w_queue=float(args.w_queue),
            queue_power=float(args.queue_power),
            top2_w1=float(args.top2_w1),
            top2_w2=float(args.top2_w2),
            reward_clip_lo=float(args.reward_clip_lo),
            reward_clip_hi=float(args.reward_clip_hi),
            models_by_tls=models_by_tls,
            meta_by_tls=meta_by_tls,
        )

    run_dir = Path(args.out_dir) / f"eval_{int(time.time())}"
    write_results(run_dir, results=results, args_dict=vars(args))
    print(f"Wrote evaluation outputs to: {run_dir}")


if __name__ == "__main__":
    main()
