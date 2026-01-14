# [NEW FILE] run_ppo_tsc_merged.py

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
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import libsumo as traci
from sumolib import checkBinary
from torch.utils.tensorboard import SummaryWriter
import torch  # imported after numpy (Windows OpenMP duplicate init mitigation)

from ppo_agent import PPOAgent, RolloutBuffer
from utility import *
from scene_encoder import encode_tsc_state_vector_bounded
from expert_feature_extractor import *


# run-specfic encoding function, combining expert features and original scene encoder
def encode_tsc_state_vector_combined(
    tls_id: str, *, cache: Optional[dict] = None, **kwargs
) -> np.ndarray:
    """
    Returns concatenated features:
      [ encode_tsc_state_vector_bounded(...) , tsc_isolated_intersection_feature_vector(...) ]

    Uses namespaced caches to avoid key collisions:
      cache["_enc_core"] : for scenario encoder
      cache["_enc_sem"]  : for semantic extractor (EMA, trackers, etc.)
    """
    if cache is None:
        cache = {}

    core_cache = cache.setdefault("_enc_core", {})
    sem_cache = cache.setdefault("_enc_sem", {})

    v_core = encode_tsc_state_vector_bounded(tls_id, cache=core_cache, **kwargs)
    v_sem = tsc_isolated_intersection_feature_vector(tls_id, cache=sem_cache)

    return np.concatenate(
        [
            np.asarray(v_core, dtype=np.float32),
            np.asarray(v_sem, dtype=np.float32),
        ],
        axis=0,
    )


# --- [NEW] PPO rollout diagnostics logging -----------------------------------
def _to_np(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    try:
        return np.asarray(x)
    except Exception:
        return None


def _get_attr_any(obj, names):
    for n in names:
        if hasattr(obj, n):
            return _to_np(getattr(obj, n))
    return None


def _explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # EV = 1 - Var[y - yhat] / Var[y]
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    var_y = float(np.var(y_true))
    if var_y < 1e-12:
        return float("nan")
    return 1.0 - float(np.var(y_true - y_pred)) / var_y


def tb_log_rollout_diagnostics(
    writer: SummaryWriter, tls_id: str, step: int, buf
) -> None:
    """
    Logs rollout-level diagnostics to TensorBoard.
    Assumes buf.compute_gae() has been called so buf has returns/advantages populated.
    Works with common attribute names; adjust the name lists if your RolloutBuffer differs.
    """
    rets = _get_attr_any(buf, ["returns", "rets", "return_s"])
    advs = _get_attr_any(buf, ["advantages", "advs", "adv"])
    vpred = _get_attr_any(buf, ["values", "vpred", "value_preds", "value_pred"])
    durs = _get_attr_any(buf, ["durations_s", "duration_s", "durations", "dts"])

    if rets is not None:
        rets = rets.astype(np.float32).reshape(-1)
        writer.add_scalar(f"{tls_id}/rollout/return_mean", float(np.mean(rets)), step)
        writer.add_scalar(f"{tls_id}/rollout/return_std", float(np.std(rets)), step)

    if advs is not None:
        advs = advs.astype(np.float32).reshape(-1)
        writer.add_scalar(f"{tls_id}/rollout/adv_mean", float(np.mean(advs)), step)
        writer.add_scalar(f"{tls_id}/rollout/adv_std", float(np.std(advs)), step)

    if vpred is not None:
        vpred = vpred.astype(np.float32).reshape(-1)
        writer.add_scalar(f"{tls_id}/rollout/vpred_mean", float(np.mean(vpred)), step)
        writer.add_scalar(f"{tls_id}/rollout/vpred_std", float(np.std(vpred)), step)

    if (rets is not None) and (vpred is not None) and (rets.shape[0] == vpred.shape[0]):
        ev = _explained_variance(rets, vpred)
        writer.add_scalar(f"{tls_id}/rollout/explained_variance", float(ev), step)

    if durs is not None:
        durs = durs.astype(np.float32).reshape(-1)
        writer.add_scalar(f"{tls_id}/rollout/duration_mean", float(np.mean(durs)), step)
        writer.add_scalar(f"{tls_id}/rollout/duration_std", float(np.std(durs)), step)
        writer.add_scalar(f"{tls_id}/rollout/duration_min", float(np.min(durs)), step)
        writer.add_scalar(f"{tls_id}/rollout/duration_max", float(np.max(durs)), step)


# ---------------------------------------------------------------------------


def start_sumo(
    sumocfg: str, *, gui: bool, delay_ms: int, sumo_seed: int, traffic_scale: float
) -> None:
    binary = checkBinary("sumo-gui" if gui else "sumo")
    cmd = [
        binary,
        "-c",
        sumocfg,
        "--start",
        "--no-step-log",
        "true",
        "--delay",
        str(delay_ms),
        "--seed",
        str(int(sumo_seed)),
        "--scale",
        str(float(traffic_scale)),
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


# NEW (adds scheduling state for aux phases)
@dataclass
class PendingDecision:
    state: Optional[np.ndarray] = None
    action: Optional[int] = None
    logp: Optional[float] = None
    value: Optional[float] = None
    in_control: bool = False
    next_decision_time: float = 0.0
    action_start_time: float = 0.0

    # [NEW] queued segments after the currently playing one: (phase_idx, duration_s)
    pending_segments: Deque[Tuple[int, float]] = field(default_factory=deque)
    # [NEW] when the currently-playing segment ends
    segment_end_time: float = 0.0


def run_ppo_tsc(
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
    device: Optional[str],
    hidden_dim: int,
    n_layer: int,
    use_skip: bool,
    # lr: float,
    actor_lr: float,
    critic_lr: float,
    gamma: float,
    traffic_scale_mean: float,
    traffic_scale_std: float,
    tb_logdir: str,
    save_dir: str,
    # reward params (reuse your existing utility functions)
    throughput_ref_veh_per_s: float,
    queue_ref_veh: float,
    w_throughput: float,
    w_queue: float,
    queue_power: float,
    top2_w1: float,
    top2_w2: float,
    reward_clip_lo: float,
    reward_clip_hi: float,
    # PPO defaults (kept minimal)
    rollout_steps: int,
    ppo_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    vf_clip_eps: float,
    gae_lambda: float,
    ent_coef: float,
    vf_coef: float,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # encoder_fn = encode_tsc_state_vector_bounded
    encoder_fn = encode_tsc_state_vector_combined

    run_name = f"sumo_ppo_seed{seed}_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join(tb_logdir, run_name))

    agents: Dict[str, PPOAgent] = {}
    buffers: Dict[str, RolloutBuffer] = {}
    pending: Dict[str, PendingDecision] = {}
    encoder_cache: Dict[str, dict] = {}

    tb_step_decision = {}  # per TLS decision counter

    total_elapsed = 0.0
    for ep in range(int(episodes)):
        if total_elapsed >= float(max_time):
            break

        ep_wall_start = time.time()
        traffic_scale_sampled = random.gauss(
            mu=float(traffic_scale_mean), sigma=float(traffic_scale_std)
        )
        start_sumo(
            sumocfg,
            gui=gui,
            delay_ms=delay_ms,
            sumo_seed=sumo_seed + ep,
            traffic_scale=traffic_scale_sampled,  # use sampled scale per episode instead of fixed
        )
        try:
            tls_ids = list(traci.trafficlight.getIDList())
            if not tls_ids:
                raise RuntimeError("No traffic lights found in scenario")

            # init per-TLS structures lazily (reuse across episodes)
            for tls_id in tls_ids:
                if tls_id not in encoder_cache:
                    encoder_cache[tls_id] = {}
                if tls_id not in pending:
                    pending[tls_id] = PendingDecision()
                if tls_id not in tb_step_decision:
                    tb_step_decision[tls_id] = 0
                if tls_id not in agents:
                    s0 = encoder_fn(
                        tls_id,
                        moving_speed_threshold=0.1,
                        stopped_speed_threshold=0.1,
                        cache=encoder_cache[tls_id],
                    ).astype(np.float32)
                    state_dim = int(s0.shape[0])
                    # action_dim = int(get_phase_count(tls_id))
                    # NEW (major greens only)
                    action_dim = int(
                        tls_major_action_dim(tls_id, encoder_cache[tls_id])
                    )

                    agents[tls_id] = PPOAgent(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        seed=seed,
                        hidden_dim=hidden_dim,
                        n_layer=n_layer,
                        use_skip=use_skip,
                        # lr=lr,
                        actor_lr=actor_lr,
                        critic_lr=critic_lr,
                        device=device,
                        clip_eps=clip_eps,
                        vf_clip_eps=vf_clip_eps,
                        epochs=ppo_epochs,
                        minibatch_size=minibatch_size,
                        gamma=gamma,
                        gae_lambda=gae_lambda,
                        vf_coef=vf_coef,
                        ent_coef=ent_coef,
                    )
                    buffers[tls_id] = RolloutBuffer()

            # reset pending decisions each episode
            for tls_id in tls_ids:
                pending[tls_id] = PendingDecision()

            ep_reward_sum = {tls_id: 0.0 for tls_id in tls_ids}
            ep_reward_n = {tls_id: 0 for tls_id in tls_ids}

            while True:
                sim_t = float(traci.simulation.getTime())
                done_episode = (sim_t >= float(episode_len_s)) or (
                    traci.simulation.getMinExpectedNumber() <= 0
                )
                in_control = sim_t >= float(warmup_s)

                if done_episode:
                    # close last pending interval as terminal transition
                    if in_control:
                        for tls_id in tls_ids:
                            st = pending[tls_id]
                            st.segment_end_time = tls_advance_pending_segments(
                                tls_id=tls_id,
                                pending_segments=st.pending_segments,
                                segment_end_time=st.segment_end_time,
                                sim_t=sim_t,
                            )
                            if not (
                                st.in_control
                                and st.state is not None
                                and st.action is not None
                                and st.logp is not None
                                and st.value is not None
                            ):
                                continue

                            terminal_state = encoder_fn(
                                tls_id,
                                moving_speed_threshold=0.1,
                                stopped_speed_threshold=0.1,
                                cache=encoder_cache[tls_id],
                            ).astype(np.float32)

                            # OLD: only work with single scenario encoder
                            # num_lanes = max(
                            #     1, len(encoder_cache[tls_id].get("lane_ids", []))
                            # )
                            # NEW: combined encoder uses namespaced cache
                            num_lanes = max(
                                1,
                                len(
                                    encoder_cache[tls_id]
                                    .get("_enc_core", {})
                                    .get("lane_ids", [])
                                ),
                            )
                            r = reward_throughput_plus_softmax_queue(
                                tls_id=tls_id,
                                sim_time=sim_t,
                                state_vec=terminal_state,
                                cache=encoder_cache[tls_id],
                                num_lanes=num_lanes,
                                throughput_ref_veh_per_s=throughput_ref_veh_per_s,
                                queue_ref_veh=queue_ref_veh,
                                w_throughput=w_throughput,
                                w_queue=w_queue,
                                # top2_weights=(top2_w1, top2_w2),
                                queue_power=queue_power,
                                reward_clip=(reward_clip_lo, reward_clip_hi),
                            )

                            dt_interval = sim_t - float(st.action_start_time)
                            buffers[tls_id].add(
                                state=st.state,
                                action=st.action,
                                logp=st.logp,
                                value=st.value,
                                reward=r,
                                done=True,
                                duration_s=dt_interval,
                            )
                            ep_reward_sum[tls_id] += float(r)
                            ep_reward_n[tls_id] += 1

                            # final update on leftover rollout
                            buf = buffers[tls_id]
                            if len(buf) > 0:
                                buf.compute_gae(
                                    last_value=0.0,
                                    gamma=gamma,
                                    gae_lambda=gae_lambda,
                                    base_dt_s=float(action_hold_s),
                                )

                                step = tb_step_decision[tls_id]
                                tb_log_rollout_diagnostics(
                                    writer, tls_id, step, buf
                                )  # [NEW]
                                stats = agents[tls_id].update(buf)
                                buf.clear()

                                step = tb_step_decision[tls_id]
                                writer.add_scalar(
                                    f"{tls_id}/ppo/policy_loss",
                                    stats["policy_loss"],
                                    step,
                                )
                                writer.add_scalar(
                                    f"{tls_id}/ppo/value_loss",
                                    stats["value_loss"],
                                    step,
                                )
                                writer.add_scalar(
                                    f"{tls_id}/ppo/entropy", stats["entropy"], step
                                )
                                writer.add_scalar(
                                    f"{tls_id}/ppo/approx_kl", stats["approx_kl"], step
                                )
                                writer.add_scalar(
                                    f"{tls_id}/ppo/clip_frac", stats["clip_frac"], step
                                )
                    break

                if in_control:
                    for tls_id in tls_ids:
                        st = pending[tls_id]
                        if st.segment_end_time > 0.0 and sim_t >= st.segment_end_time:
                            st.segment_end_time = tls_advance_pending_segments(
                                tls_id=tls_id,
                                pending_segments=st.pending_segments,
                                segment_end_time=st.segment_end_time,
                                sim_t=sim_t,
                            )
                        if sim_t < st.next_decision_time:
                            continue

                        cur_state = encoder_fn(
                            tls_id,
                            moving_speed_threshold=0.1,
                            stopped_speed_threshold=0.1,
                            cache=encoder_cache[tls_id],
                        ).astype(np.float32)

                        # OLD: only work with single scenario encoder
                        # num_lanes = max(
                        #     1, len(encoder_cache[tls_id].get("lane_ids", []))
                        # )
                        # NEW: combined encoder uses namespaced cache
                        num_lanes = max(
                            1,
                            len(
                                encoder_cache[tls_id]
                                .get("_enc_core", {})
                                .get("lane_ids", [])
                            ),
                        )

                        # [NEW BLOCK] close previous interval and push to PPO rollout buffer
                        if (
                            st.in_control
                            and st.state is not None
                            and st.action is not None
                            and st.logp is not None
                            and st.value is not None
                        ):
                            r = reward_throughput_plus_softmax_queue(
                                tls_id=tls_id,
                                sim_time=sim_t,
                                state_vec=cur_state,
                                cache=encoder_cache[tls_id],
                                num_lanes=num_lanes,
                                throughput_ref_veh_per_s=throughput_ref_veh_per_s,
                                queue_ref_veh=queue_ref_veh,
                                w_throughput=w_throughput,
                                w_queue=w_queue,
                                # top2_weights=(top2_w1, top2_w2),
                                queue_power=queue_power,
                                reward_clip=(reward_clip_lo, reward_clip_hi),
                            )
                            dt_interval = sim_t - float(
                                st.action_start_time
                            )  # [NEW] real elapsed seconds for this transition
                            buffers[tls_id].add(
                                state=st.state,
                                action=st.action,
                                logp=st.logp,
                                value=st.value,
                                reward=r,
                                done=False,
                                duration_s=dt_interval,  # [NEW]
                            )
                            ep_reward_sum[tls_id] += float(r)
                            ep_reward_n[tls_id] += 1

                            step = tb_step_decision[tls_id]
                            writer.add_scalar(f"{tls_id}/train/reward", float(r), step)
                        else:
                            # first controlled action: initialize throughput window (no-op reward)
                            _ = reward_throughput_plus_softmax_queue(
                                tls_id=tls_id,
                                sim_time=sim_t,
                                state_vec=cur_state,
                                cache=encoder_cache[tls_id],
                                num_lanes=num_lanes,
                                throughput_ref_veh_per_s=throughput_ref_veh_per_s,
                                queue_ref_veh=queue_ref_veh,
                                w_throughput=0.0,
                                w_queue=0.0,
                                # top2_weights=(top2_w1, top2_w2),
                                queue_power=queue_power,
                                reward_clip=(reward_clip_lo, reward_clip_hi),
                            )

                        # NEW
                        a, logp, v = agents[tls_id].act(cur_state)

                        # [NEW] policy action indexes ONLY major greens
                        target_major_phase = tls_action_to_major_phase(
                            tls_id, encoder_cache[tls_id], action=int(a)
                        )

                        # [NEW] build segments: (aux phases after current major if switching) + (target green for hold_s)
                        segments = tls_build_switch_segments(
                            tls_id,
                            encoder_cache[tls_id],
                            target_major_phase=int(target_major_phase),
                            hold_s=float(action_hold_s),
                            current_phase=int(traci.trafficlight.getPhase(tls_id)),
                        )

                        # [NEW] play first segment now, queue the rest
                        first_phase, first_dur = segments[0]
                        tls_set_phase_frozen(tls_id, int(first_phase))

                        st.pending_segments = deque(segments[1:])
                        st.segment_end_time = sim_t + float(first_dur)

                        # [NEW] next decision after ALL segments (aux time NOT counted inside hold_s)
                        st.next_decision_time = sim_t + float(
                            sum(d for _, d in segments)
                        )

                        # unchanged bookkeeping
                        st.state = cur_state
                        st.action = int(a)  # store policy action (major-green index)
                        st.logp = float(logp)
                        st.value = float(v)
                        st.in_control = True
                        st.action_start_time = sim_t
                        tb_step_decision[tls_id] += 1

                        # [NEW BLOCK] update when buffer reaches rollout_steps
                        buf = buffers[tls_id]
                        if len(buf) >= int(rollout_steps):
                            buf.compute_gae(
                                last_value=float(v),
                                gamma=gamma,
                                gae_lambda=gae_lambda,
                                base_dt_s=float(
                                    action_hold_s
                                ),  # [NEW] interpret gamma as “per hold_s”
                            )

                            step = tb_step_decision[tls_id]
                            tb_log_rollout_diagnostics(
                                writer, tls_id, step, buf
                            )  # [NEW]
                            stats = agents[tls_id].update(buf)
                            buf.clear()

                            step = tb_step_decision[tls_id]
                            writer.add_scalar(
                                f"{tls_id}/ppo/policy_loss", stats["policy_loss"], step
                            )
                            writer.add_scalar(
                                f"{tls_id}/ppo/value_loss", stats["value_loss"], step
                            )
                            writer.add_scalar(
                                f"{tls_id}/ppo/entropy", stats["entropy"], step
                            )
                            writer.add_scalar(
                                f"{tls_id}/ppo/approx_kl", stats["approx_kl"], step
                            )
                            writer.add_scalar(
                                f"{tls_id}/ppo/clip_frac", stats["clip_frac"], step
                            )

                        # NEW: logging-friendly phase info
                        cur_phase_idx = int(
                            traci.trafficlight.getPhase(tls_id)
                        )  # actual SUMO phase executing now (may be aux)
                        cur_major_idx = int(
                            tls_current_major_phase(
                                tls_id,
                                encoder_cache[tls_id],
                                current_phase=cur_phase_idx,
                            )
                        )
                        tgt_major_idx = int(
                            target_major_phase
                        )  # SUMO phase index of the selected major green
                        print(
                            f"[ep={ep} t={sim_t:6.1f}] tls={tls_id} "
                            f"a(major_idx)={int(a)} "
                            f"cur_phase={cur_phase_idx} cur_major={cur_major_idx} "
                            f"tgt_major={tgt_major_idx} "
                            f"segments={[(p, round(d,1)) for (p,d) in segments]} "
                            f"hold={action_hold_s}s"
                        )

                traci.simulationStep()
                for tls_id in tls_ids:
                    throughput_tracker_step(tls_id, encoder_cache[tls_id])

            for tls_id in tls_ids:
                mean_r = ep_reward_sum[tls_id] / max(1, ep_reward_n[tls_id])
                writer.add_scalar(f"{tls_id}/episode/reward_mean", float(mean_r), ep)
                writer.add_scalar(
                    f"{tls_id}/episode/reward_sum", float(ep_reward_sum[tls_id]), ep
                )

        finally:
            try:
                traci.close()
            except Exception:
                pass

        ep_elapsed = time.time() - ep_wall_start
        total_elapsed += ep_elapsed
        writer.add_scalar("global/episode_wall_s", float(ep_elapsed), ep)
        writer.add_scalar("global/traffic_scale", float(traffic_scale_sampled), ep)

    writer.flush()
    writer.close()

    # Save trained model(s)
    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    for tls_id, agent in agents.items():
        ckpt_path = save_root / f"{run_name}__{tls_id}.pt"
        meta = {
            "run_name": run_name,
            "tls_id": tls_id,
            "sumocfg": sumocfg,
            "seed": int(seed),
            "sumo_seed": int(sumo_seed),
            "action_hold_s": float(action_hold_s),
            "state_dim": int(agent.state_dim),
            "action_dim": int(agent.action_dim),
            "hidden_dim": int(agent.hidden_dim),
            "layer_count": int(agent.n_layer),
            "use_skip": bool(agent.use_skip),
            "gamma": float(gamma),
            # "lr": float(lr),
            "actor_lr": float(actor_lr),
            "critic_lr": float(critic_lr),
            "clip_eps": float(clip_eps),
            "vf_clip_eps": float(vf_clip_eps),
            "vf_coef": float(vf_coef),
            "ent_coef": float(ent_coef),
            "gae_lambda": float(gae_lambda),
            "ppo_epochs": int(ppo_epochs),
            "minibatch_size": int(minibatch_size),
            "rollout_steps": int(rollout_steps),
            "encoder": getattr(encoder_fn, "__name__", "<unknown>"),
            "traffic_scale_mean": float(traffic_scale_mean),
            "traffic_scale_std": float(traffic_scale_std),
            "saved_unix_time": float(time.time()),
        }

        torch.save(
            {"meta": meta, "model_state_dict": agent.model.state_dict()}, ckpt_path
        )
        (ckpt_path.with_suffix(".json")).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"[save] tls={tls_id} -> {ckpt_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--sumocfg", required=True)

    # Minimal interface to run (others default)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--episode-len", type=float, default=30000.0)
    ap.add_argument("--warmup", type=float, default=200.0)
    ap.add_argument("--hold", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--sumo-seed", type=int, default=0)
    ap.add_argument("--delay-ms", type=int, default=1)
    ap.add_argument("--traffic-scale-mean", type=float, default=1.0)
    ap.add_argument("--traffic-scale-std", type=float, default=0.0)

    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--use-skip", action="store_true")
    # ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--actor-lr", type=float, default=3e-4)
    ap.add_argument("--critic-lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)

    # PPO optional knobs (defaults)
    ap.add_argument("--rollout-steps", type=int, default=256)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=64)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--vf-clip-eps", type=float, default=None)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)

    # Reward (defaults)
    ap.add_argument("--thr-ref", type=float, default=0.30)
    ap.add_argument("--queue-ref", type=float, default=15.0)
    ap.add_argument("--w-thr", type=float, default=1.0)
    ap.add_argument("--w-queue", type=float, default=1.0)
    ap.add_argument("--queue-power", type=float, default=1.0)
    ap.add_argument("--top2-w1", type=float, default=0.7)
    ap.add_argument("--top2-w2", type=float, default=0.3)
    ap.add_argument("--reward-clip-lo", type=float, default=-5.0)
    ap.add_argument("--reward-clip-hi", type=float, default=2.0)

    ap.add_argument("--tb-logdir", type=str, default="tensorboard_logs")
    ap.add_argument("--save-dir", type=str, default="saved_models_ppo")
    ap.add_argument("--max-time", type=float, default=1e18)

    args = ap.parse_args()

    run_ppo_tsc(
        args.sumocfg,
        gui=bool(args.gui),
        max_time=float(args.max_time),
        episodes=int(args.episodes),
        episode_len_s=float(args.episode_len),
        warmup_s=float(args.warmup),
        seed=int(args.seed),
        sumo_seed=int(args.sumo_seed),
        delay_ms=int(args.delay_ms),
        action_hold_s=float(args.hold),
        device=args.device,
        hidden_dim=int(args.hidden_dim),
        n_layer=int(args.n_layer),
        use_skip=bool(args.use_skip),
        # lr=float(args.lr),
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        gamma=float(args.gamma),
        traffic_scale_mean=float(args.traffic_scale_mean),
        traffic_scale_std=float(args.traffic_scale_std),
        tb_logdir=args.tb_logdir,
        save_dir=args.save_dir,
        throughput_ref_veh_per_s=float(args.thr_ref),
        queue_ref_veh=float(args.queue_ref),
        w_throughput=float(args.w_thr),
        w_queue=float(args.w_queue),
        queue_power=float(args.queue_power),
        top2_w1=float(args.top2_w1),
        top2_w2=float(args.top2_w2),
        reward_clip_lo=float(args.reward_clip_lo),
        reward_clip_hi=float(args.reward_clip_hi),
        rollout_steps=int(args.rollout_steps),
        ppo_epochs=int(args.ppo_epochs),
        minibatch_size=int(args.minibatch),
        clip_eps=float(args.clip_eps),
        vf_clip_eps=float(args.vf_clip_eps),
        gae_lambda=float(args.gae_lambda),
        ent_coef=float(args.ent_coef),
        vf_coef=float(args.vf_coef),
    )


if __name__ == "__main__":
    main()
