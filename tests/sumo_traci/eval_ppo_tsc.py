# [NEW FILE] eval_ppo_tsc.py
#
# Evaluate a saved PPO TSC policy checkpoint on SUMO/libsumo.
# Supports expert-feature ablation by zeroing the expert feature slice at inference time.

from __future__ import annotations

import os
import sys

# Ensure SUMO tools are importable before importing traci/sumolib
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import torch
import libsumo as traci

from ppo_agent import PPOAgent
from scene_encoder import (
    encode_tsc_state_vector_bounded,
    encode_tsc_state_vector_bounded_v2,
)
from expert_feature_extractor import tsc_isolated_intersection_feature_vector
from utility import (
    start_sumo,
    tls_major_action_dim,
    tls_action_to_major_phase,
    tls_build_switch_segments,
    tls_set_phase_frozen,
    tls_advance_pending_segments,
    throughput_tracker_step,
    reward_throughput_per_second_on_decision,
    reward_softmax_queue_from_encoded_state,
)


# ------------------------------
# Encoder selection (by meta["encoder"])
# ------------------------------
def _encode_core(
    tls_id: str,
    *,
    cache: Dict[str, Any],
    encoder_name: str,
) -> np.ndarray:
    # Keep this mapping small and explicit to avoid surprises.
    if encoder_name == "encode_tsc_state_vector_bounded_v2":
        return encode_tsc_state_vector_bounded_v2(
            tls_id,
            moving_speed_threshold=0.1,
            stopped_speed_threshold=0.1,
            cache=cache,
        ).astype(np.float32)

    # fallback: bounded (v1)
    return encode_tsc_state_vector_bounded(
        tls_id,
        moving_speed_threshold=0.1,
        stopped_speed_threshold=0.1,
        cache=cache,
    ).astype(np.float32)


def encode_state_for_policy(
    tls_id: str,
    *,
    cache_root: Dict[str, Any],
    meta_encoder_name: str,
    use_expert_features: bool,
    zero_expert_features: bool,
    # granular expert ablation
    zero_expert_dims: List[int],
    noise_expert_dims: List[int],
    noise_sigma: float,
) -> Tuple[np.ndarray, int]:
    """
    Returns:
      state_vec: np.float32 (state_dim,)
      core_dim:  int length of the core slice (used to locate expert slice)
    """
    if use_expert_features:
        core_cache = cache_root.setdefault("_enc_core", {})
        sem_cache = cache_root.setdefault("_enc_sem", {})

        # IMPORTANT: match your training combined encoder (core=v2 + expert)
        core = encode_tsc_state_vector_bounded_v2(
            tls_id,
            moving_speed_threshold=0.1,
            stopped_speed_threshold=0.1,
            cache=core_cache,
        ).astype(np.float32)
        core_dim = int(core.shape[0])

        if zero_expert_features:
            expert = np.zeros(
                (27,), dtype=np.float32
            )  # expert extractor output is length 27
        else:
            expert = np.asarray(
                tsc_isolated_intersection_feature_vector(tls_id, cache=sem_cache),
                dtype=np.float32,
            )

        # granular expert ablation
        if (not zero_expert_features) and expert.size > 0:
            for d in zero_expert_dims:
                if 0 <= d < expert.shape[0]:
                    expert[d] = 0.0

            if noise_sigma > 0 and len(noise_expert_dims) > 0:
                for d in noise_expert_dims:
                    if 0 <= d < expert.shape[0]:
                        expert[d] += np.random.normal(0.0, noise_sigma)

        return np.concatenate([core, expert], axis=0), core_dim

    # core-only encoders
    core_cache = cache_root
    core = _encode_core(tls_id, cache=core_cache, encoder_name=meta_encoder_name)
    return core, int(core.shape[0])


def get_num_lanes_from_cache(
    cache_root: Dict[str, Any], use_expert_features: bool
) -> int:
    if use_expert_features:
        lane_ids = cache_root.get("_enc_core", {}).get("lane_ids", [])
    else:
        lane_ids = cache_root.get("lane_ids", [])
    return max(1, len(lane_ids))


def clip_reward(x: float, lo: float, hi: float) -> float:
    if lo > hi:
        lo, hi = hi, lo
    return lo if x < lo else hi if x > hi else x


# ------------------------------
# Eval controller state (segments + previous decision bookkeeping)
# ------------------------------
@dataclass
class EvalTLSState:
    next_decision_time: float = 0.0
    action_start_time: float = 0.0

    # segments scheduler
    pending_segments: Deque[Tuple[int, float]] = field(default_factory=deque)
    segment_end_time: float = 0.0

    # previous decision (for interval reward closure)
    prev_state: Optional[np.ndarray] = None
    prev_action: Optional[int] = None
    prev_in_control: bool = False


# ------------------------------
# Main evaluation
# ------------------------------
def eval_one_checkpoint(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    # load metadata json (optional)
    meta_path = (
        Path(args.meta) if args.meta is not None else ckpt_path.with_suffix(".json")
    )
    if not meta_path.exists():
        raise FileNotFoundError(f"meta json not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # load checkpoint
    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"> Using device: {device}")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )

    tls_id = args.tls_id if args.tls_id is not None else str(meta.get("tls_id", ""))
    print(f"> Using tls_id: {tls_id}")
    if not tls_id:
        raise ValueError("tls_id not found in meta, and --tls-id not provided")

    sumocfg = args.sumocfg if args.sumocfg is not None else str(meta.get("sumocfg", ""))
    if not sumocfg:
        raise ValueError("sumocfg not found in meta, and --sumocfg not provided")

    # model hyperparams (must match)
    state_dim = int(meta["state_dim"])
    action_dim = int(meta["action_dim"])
    hidden_dim = int(meta.get("hidden_dim", 256))
    layer_count = int(meta.get("layer_count", 2))
    use_skip = bool(meta.get("use_skip", False))

    use_expert_features = bool(meta.get("use_expert_features", False))
    encoder_name = str(meta.get("encoder", "encode_tsc_state_vector_bounded_v2"))
    print(f"> Using encoder: {encoder_name}")
    print(f"> Using expert features: {use_expert_features}")

    # eval knobs
    episodes = int(args.episodes)
    episode_len_s = float(args.episode_len)
    warmup_s = float(args.warmup)
    action_hold_s = (
        float(args.hold)
        if args.hold is not None
        else float(meta.get("action_hold_s", 10.0))
    )

    sumo_seed_base = (
        int(args.sumo_seed)
        if args.sumo_seed is not None
        else int(meta.get("sumo_seed", 0))
    )
    traffic_scale = (
        float(args.traffic_scale)
        if args.traffic_scale is not None
        else float(meta.get("traffic_scale_mean", 1.0))
    )
    print(f"> Using traffic_scale: {traffic_scale}")

    # reward knobs (for decomposition)
    throughput_ref = float(args.thr_ref)
    queue_ref = float(args.queue_ref)
    w_thr = float(args.w_thr)
    w_q = float(args.w_queue)
    queue_power = float(args.queue_power)
    r_clip_lo = float(args.reward_clip_lo)
    r_clip_hi = float(args.reward_clip_hi)

    # ablation: only makes sense if checkpoint expects expert features
    zero_expert = bool(args.zero_expert)

    def _parse_dim_list(s: str) -> List[int]:
        s = (s or "").strip()
        if not s:
            return []
        return [int(x) for x in s.split(",") if x.strip()]

    zero_dims = _parse_dim_list(args.zero_expert_dims)
    noise_dims = _parse_dim_list(args.noise_expert_dims)
    noise_sigma = float(args.noise_sigma)

    # logging setup
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    tag = f"__{args.log_tag}" if args.log_tag else ""
    mode = (
        "zeroexp"
        if (args.zero_expert and bool(meta.get("use_expert_features", False)))
        else "normal"
    )
    print(f"> Eval mode: {mode}")
    base = f"eval_{meta.get('run_name','run')}__{tls_id}__{mode}{tag}__{ts}"
    jsonl_path = log_dir / f"{base}.jsonl"
    summary_path = log_dir / f"{base}_summary.json"
    jsonl_f = open(jsonl_path, "w", encoding="utf-8")
    header = {
        "type": "header",
        "checkpoint": str(ckpt_path),
        "meta": str(meta_path),
        "tls_id": tls_id,
        "sumocfg": sumocfg,
        "episodes": int(args.episodes),
        "deterministic": bool(args.deterministic),
        "use_expert_features_meta": bool(meta.get("use_expert_features", False)),
        "zero_expert_eval": bool(args.zero_expert),
        "traffic_scale": float(traffic_scale),
        "sumo_seed_base": int(sumo_seed_base),
    }
    header.update(
        {
            "zero_expert_dims": zero_dims,
            "noise_expert_dims": noise_dims,
            "noise_sigma": noise_sigma,
        }
    )
    jsonl_f.write(json.dumps(header, separators=(",", ":")) + "\n")
    jsonl_f.flush()

    # build agent and load weights
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        seed=int(meta.get("seed", 0)),
        hidden_dim=hidden_dim,
        n_layer=layer_count,
        use_skip=use_skip,
        actor_lr=float(
            meta.get("actor_lr", 3e-4)
        ),  # irrelevant for eval but required by class
        critic_lr=float(
            meta.get("critic_lr", 1e-3)
        ),  # irrelevant for eval but required by class
        device=device,
        clip_eps=float(meta.get("clip_eps", 0.2)),
        vf_clip_eps=float(meta.get("vf_clip_eps", 0.2)),
        epochs=int(meta.get("ppo_epochs", 4)),
        minibatch_size=int(meta.get("minibatch_size", 64)),
        gamma=float(meta.get("gamma", 0.99)),
        gae_lambda=float(meta.get("gae_lambda", 0.95)),
        vf_coef=float(meta.get("vf_coef", 0.5)),
        ent_coef=float(meta.get("ent_coef", 0.01)),
    )
    agent.model.load_state_dict(state_dict, strict=True)
    agent.model.eval()

    # summary accumulators
    ep_returns: List[float] = []
    ep_thr_norms: List[float] = []
    ep_q_rewards: List[float] = []

    for ep in range(episodes):
        action_counts = np.zeros((action_dim,), dtype=np.int32)
        switches = 0
        last_action = None
        # --- policy-probability stats (decision points) ---
        pi_sum = np.zeros((action_dim,), dtype=np.float64)
        pi_entropy_sum = 0.0
        pi_min_sum = 0.0
        pi_max_sum = 0.0
        n_decisions = 0
        # --- empirical action-distribution stats (over episode) ---
        a_interval_count = np.zeros((action_dim,), dtype=np.int32)
        a_reward_sum = np.zeros((action_dim,), dtype=np.float64)
        a_thr_sum = np.zeros((action_dim,), dtype=np.float64)
        a_q_sum = np.zeros((action_dim,), dtype=np.float64)

        start_sumo(
            sumocfg,
            gui=bool(args.gui),
            delay_ms=int(args.delay_ms),
            sumo_seed=sumo_seed_base + ep,
            traffic_scale=float(traffic_scale),
        )

        try:
            if tls_id not in traci.trafficlight.getIDList():
                raise RuntimeError(f"tls_id={tls_id} not found in scenario")

            # centralized per-episode reset
            cache_root: Dict[str, Any] = {}
            tls_state = EvalTLSState()

            # init phase-plan cache once (needed for major action mapping)
            # (this populates cache_root with TLS plan data via utility.get_tls_phase_plan)
            _ = tls_major_action_dim(tls_id, cache_root)

            # per-episode accumulators
            ret_sum = 0.0
            thr_norm_sum = 0.0
            q_reward_sum = 0.0
            n_intervals = 0

            while True:
                sim_t = float(traci.simulation.getTime())
                done_episode = (sim_t >= episode_len_s) or (
                    traci.simulation.getMinExpectedNumber() <= 0
                )
                in_control = sim_t >= warmup_s

                # keep throughput tracker up-to-date every sim step
                throughput_tracker_step(tls_id, cache_root)

                # advance aux segments if needed (only relevant after policy starts)
                if (
                    tls_state.segment_end_time > 0.0
                    and sim_t >= tls_state.segment_end_time
                ):
                    tls_state.segment_end_time = tls_advance_pending_segments(
                        tls_id=tls_id,
                        pending_segments=tls_state.pending_segments,
                        segment_end_time=tls_state.segment_end_time,
                        sim_t=sim_t,
                    )

                if done_episode:
                    # close last controlled interval if exists
                    if (
                        in_control
                        and tls_state.prev_in_control
                        and tls_state.prev_state is not None
                    ):
                        # terminal next-state for queue term
                        s_next, _core_dim = encode_state_for_policy(
                            tls_id,
                            cache_root=cache_root,
                            meta_encoder_name=encoder_name,
                            use_expert_features=use_expert_features,
                            zero_expert_features=zero_expert and use_expert_features,
                            zero_expert_dims=zero_dims,
                            noise_expert_dims=noise_dims,
                            noise_sigma=noise_sigma,
                        )
                        num_lanes = get_num_lanes_from_cache(
                            cache_root, use_expert_features
                        )

                        thr = reward_throughput_per_second_on_decision(
                            sim_time=sim_t, cache=cache_root
                        )
                        thr_norm = min(
                            1.0, max(0.0, float(thr) / max(1e-6, throughput_ref))
                        )

                        q_reward = reward_softmax_queue_from_encoded_state(
                            s_next,
                            num_lanes=num_lanes,
                            lane_block_size=4,
                            queue_offset_in_block=0,
                            power=queue_power,
                            scale=queue_ref,
                            softmax_beta=5.0,
                            clip_nonnegative=True,
                        )
                        r = float(w_thr) * thr_norm + float(w_q) * float(q_reward)
                        r = clip_reward(r, r_clip_lo, r_clip_hi)

                        ret_sum += float(r)
                        thr_norm_sum += float(thr_norm)
                        q_reward_sum += float(q_reward)
                        n_intervals += 1

                        # --- empirical action-distribution metrics update ---
                        if tls_state.prev_action is not None:
                            pa = int(tls_state.prev_action)
                            a_interval_count[pa] += 1
                            a_reward_sum[pa] += float(r)
                            a_thr_sum[pa] += float(thr_norm)
                            a_q_sum[pa] += float(q_reward)

                    break  # done

                # decision point
                if in_control and sim_t >= tls_state.next_decision_time:
                    # next-state (for closing previous interval + selecting new action)
                    s_cur, _core_dim = encode_state_for_policy(
                        tls_id,
                        cache_root=cache_root,
                        meta_encoder_name=encoder_name,
                        use_expert_features=use_expert_features,
                        zero_expert_features=zero_expert and use_expert_features,
                        zero_expert_dims=zero_dims,
                        noise_expert_dims=noise_dims,
                        noise_sigma=noise_sigma,
                    )
                    num_lanes = get_num_lanes_from_cache(
                        cache_root, use_expert_features
                    )

                    # close previous interval reward
                    if tls_state.prev_in_control and tls_state.prev_state is not None:
                        thr = reward_throughput_per_second_on_decision(
                            sim_time=sim_t, cache=cache_root
                        )
                        thr_norm = min(
                            1.0, max(0.0, float(thr) / max(1e-6, throughput_ref))
                        )

                        q_reward = reward_softmax_queue_from_encoded_state(
                            s_cur,
                            num_lanes=num_lanes,
                            lane_block_size=4,
                            queue_offset_in_block=0,
                            power=queue_power,
                            scale=queue_ref,
                            softmax_beta=5.0,
                            clip_nonnegative=True,
                        )
                        r = float(w_thr) * thr_norm + float(w_q) * float(q_reward)
                        r = clip_reward(r, r_clip_lo, r_clip_hi)

                        ret_sum += float(r)
                        thr_norm_sum += float(thr_norm)
                        q_reward_sum += float(q_reward)
                        n_intervals += 1

                        # --- empirical action-distribution metrics update ---
                        if tls_state.prev_action is not None:
                            pa = int(tls_state.prev_action)
                            a_interval_count[pa] += 1
                            a_reward_sum[pa] += float(r)
                            a_thr_sum[pa] += float(thr_norm)
                            a_q_sum[pa] += float(q_reward)

                    else:
                        # first controlled decision: initialize throughput window
                        _ = reward_throughput_per_second_on_decision(
                            sim_time=sim_t, cache=cache_root
                        )

                    # select action
                    if args.deterministic:
                        a = agent.act_greedy(s_cur)
                    else:
                        a, _logp, _v = agent.act(s_cur)

                    # --- policy distribution at this decision ---
                    logits, probs, _v = agent.forward_logits_value(
                        s_cur, return_probs=True, to_cpu=True
                    )
                    p = probs.numpy()[0]  # (action_dim,)
                    pi_sum += p
                    pi_entropy_sum += float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())
                    pi_min_sum += float(p.min())
                    pi_max_sum += float(p.max())
                    n_decisions += 1

                    action_counts[int(a)] += 1
                    if last_action is not None and int(a) != int(last_action):
                        switches += 1
                    last_action = int(a)

                    # map policy action -> major green phase index (SUMO phase id)
                    target_major_phase = tls_action_to_major_phase(
                        tls_id, cache_root, action=int(a)
                    )

                    # build segment list (aux phases then target major green)
                    segments = tls_build_switch_segments(
                        tls_id,
                        cache_root,
                        target_major_phase=int(target_major_phase),
                        hold_s=float(action_hold_s),
                        current_phase=int(traci.trafficlight.getPhase(tls_id)),
                    )

                    # play first immediately, queue the rest
                    first_phase, first_dur = segments[0]
                    tls_set_phase_frozen(tls_id, int(first_phase))

                    tls_state.pending_segments = deque(segments[1:])
                    tls_state.segment_end_time = sim_t + float(first_dur)
                    tls_state.next_decision_time = sim_t + float(
                        sum(d for _, d in segments)
                    )

                    # store for next interval closure
                    tls_state.prev_state = s_cur
                    tls_state.prev_action = int(a)
                    tls_state.prev_in_control = True
                    tls_state.action_start_time = sim_t

                traci.simulationStep()

            ep_returns.append(float(ret_sum))
            ep_thr_norms.append(float(thr_norm_sum / max(1, n_intervals)))
            ep_q_rewards.append(float(q_reward_sum / max(1, n_intervals)))

            controlled_time_s = max(1e-6, float(episode_len_s - warmup_s))
            switch_rate_per_min = float(switches) / (controlled_time_s / 60.0)

            rec = {
                "type": "episode",
                "ep": int(ep),
                "sumo_seed": int(sumo_seed_base + ep),
                "return_sum": float(ret_sum),
                "thr_norm_mean": float(thr_norm_sum / max(1, n_intervals)),
                "q_reward_mean": float(q_reward_sum / max(1, n_intervals)),
                "n_intervals": int(n_intervals),
                "action_counts": action_counts.tolist(),
                "switches": int(switches),
                "switch_rate_per_min": float(switch_rate_per_min),
            }
            mean_pi = (pi_sum / max(1, n_decisions)).tolist()
            rec.update(
                {
                    "n_decisions": int(n_decisions),
                    "mean_pi": mean_pi,
                    "pi_entropy_mean": float(pi_entropy_sum / max(1, n_decisions)),
                    "pi_min_mean": float(pi_min_sum / max(1, n_decisions)),
                    "pi_max_mean": float(pi_max_sum / max(1, n_decisions)),
                }
            )
            # --- empirical action-distribution metrics ---
            act_total = int(action_counts.sum())
            p_emp = action_counts.astype(np.float64) / max(1, act_total)
            emp_entropy = float(-(p_emp * np.log(np.clip(p_emp, 1e-12, 1.0))).sum())
            emp_neff = float(1.0 / np.sum(p_emp * p_emp))  # 1 / sum(p^2)
            min_action_frac = float(p_emp.min())
            rec.update(
                {
                    "emp_action_entropy": emp_entropy,
                    "emp_action_neff": emp_neff,
                    "min_action_frac": min_action_frac,
                }
            )

            # empirical action-distribution per-action means
            a_denom = np.maximum(1, a_interval_count).astype(np.float64)
            rec.update(
                {
                    "a_interval_count": a_interval_count.tolist(),
                    "a_reward_mean": (a_reward_sum / a_denom).tolist(),
                    "a_thr_norm_mean": (a_thr_sum / a_denom).tolist(),
                    "a_q_reward_mean": (a_q_sum / a_denom).tolist(),
                }
            )

            jsonl_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            jsonl_f.flush()
            print(
                f"[eval ep={ep}] return_sum={ret_sum:.4f} "
                f"thr_norm_mean={ep_thr_norms[-1]:.4f} "
                f"q_reward_mean={ep_q_rewards[-1]:.4f} "
                f"intervals={n_intervals}"
            )

        finally:
            try:
                traci.close()
            except Exception:
                pass

    # summary
    arr_r = np.asarray(ep_returns, dtype=np.float32)
    arr_thr = np.asarray(ep_thr_norms, dtype=np.float32)
    arr_q = np.asarray(ep_q_rewards, dtype=np.float32)

    summary = {
        "checkpoint": str(ckpt_path),
        "meta": str(meta_path),
        "tls_id": tls_id,
        "sumocfg": sumocfg,
        "use_expert_features_meta": bool(use_expert_features),
        "zero_expert_eval": bool(zero_expert and use_expert_features),
        "episodes": int(episodes),
        "traffic_scale": float(traffic_scale),
        "sumo_seed_base": int(sumo_seed_base),
        "return_sum_mean": float(arr_r.mean()),
        "return_sum_std": float(arr_r.std()),
        "thr_norm_mean_mean": float(arr_thr.mean()),
        "thr_norm_mean_std": float(arr_thr.std()),
        "q_reward_mean_mean": float(arr_q.mean()),
        "q_reward_mean_std": float(arr_q.std()),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    jsonl_f.close()

    print(f"\n[eval] wrote: {jsonl_path}")
    print(f"[eval] wrote: {summary_path}")

    print("\n=== Evaluation Summary ===")
    print(f"checkpoint: {ckpt_path}")
    print(f"meta:       {meta_path}")
    print(f"tls_id:     {tls_id}")
    print(f"sumocfg:    {sumocfg}")
    print(f"use_expert_features(meta): {use_expert_features}")
    print(f"zero_expert_features(eval): {zero_expert and use_expert_features}")
    print(f"episodes:   {episodes}")
    print(f"traffic_scale: {traffic_scale}")
    print(f"sumo_seed_base: {sumo_seed_base}")
    print(f"return_sum: mean={arr_r.mean():.4f} std={arr_r.std():.4f}")
    print(f"thr_norm_mean: mean={arr_thr.mean():.4f} std={arr_thr.std():.4f}")
    print(f"q_reward_mean: mean={arr_q.mean():.4f} std={arr_q.std():.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()

    # paths
    ap.add_argument(
        "--checkpoint", type=str, required=True, help="Path to .pt checkpoint"
    )
    ap.add_argument(
        "--meta",
        type=str,
        default=None,
        help="Path to meta .json (default: checkpoint with .json suffix)",
    )
    ap.add_argument(
        "--log-dir",
        type=str,
        default="eval_results",
        help="Directory to write eval logs (jsonl + summary).",
    )
    ap.add_argument(
        "--log-tag",
        type=str,
        default="",
        help="Optional tag appended to log filename (e.g. 'orange', 'grey', 'zeroexp').",
    )

    # SUMO runtime knobs
    ap.add_argument(
        "--sumocfg", type=str, default=None, help="Override sumocfg in meta"
    )
    ap.add_argument("--tls-id", type=str, default=None, help="Override tls_id in meta")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--delay-ms", type=int, default=1)

    # episode controls
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--episode-len", type=float, default=3600.0)
    ap.add_argument("--warmup", type=float, default=100.0)
    ap.add_argument(
        "--hold", type=float, default=None, help="Override action_hold_s in meta"
    )
    ap.add_argument("--sumo-seed", type=int, default=None)
    ap.add_argument("--traffic-scale", type=float, default=None)

    # action selection
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Use greedy action selection (recommended)",
    )

    # expert ablation
    ap.add_argument(
        "--zero-expert",
        action="store_true",
        help="If checkpoint uses expert features, zero the expert slice at inference time.",
    )
    # granular expert ablation
    ap.add_argument(
        "--zero-expert-dims",
        type=str,
        default="",
        help="Comma-separated expert dims to zero (e.g. '0,3,7')",
    )
    ap.add_argument(
        "--noise-expert-dims",
        type=str,
        default="",
        help="Comma-separated expert dims to add Gaussian noise",
    )
    ap.add_argument(
        "--noise-sigma",
        type=float,
        default=0.05,
        help="Stddev for noise applied to --noise-expert-dims",
    )

    # reward decomposition knobs (match training defaults unless overridden)
    ap.add_argument("--thr-ref", type=float, default=2.0)
    ap.add_argument("--queue-ref", type=float, default=1.0)
    ap.add_argument("--w-thr", type=float, default=1.0)
    ap.add_argument("--w-queue", type=float, default=1.0)
    ap.add_argument("--queue-power", type=float, default=1.0)
    ap.add_argument("--reward-clip-lo", type=float, default=-2.0)
    ap.add_argument("--reward-clip-hi", type=float, default=2.0)

    # torch device
    ap.add_argument("--device", type=str, default=None)

    args = ap.parse_args()
    if not args.deterministic:
        # default to deterministic unless user explicitly wants sampling
        args.deterministic = True

    eval_one_checkpoint(args)


if __name__ == "__main__":
    main()
