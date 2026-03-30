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
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import torch
import libsumo as traci

from ppo_agent import PPOAgent
from encoder_registry import available_encoder_names, resolve_encoder
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
from controller_registry import make_controller, list_controllers
from scene_encoder import collect_tsc_scene_snapshot


# ------------------------------
# Encoder selection (via registry + model meta)
# ------------------------------
def _encoder_name_is_none(name: str | None) -> bool:
    return str(name or "").strip().lower() in {"", "none", "null"}


def make_combined_encoder(core_encoder_fn, addon_encoder_fn=None):
    """Mirror training-time encoder composition and cache layout."""

    def encoder_fn(
        tls_id: str,
        *,
        scene_stats,
        cache: dict | None = None,
        **kwargs,
    ) -> np.ndarray:
        if cache is None:
            cache = {}

        core_cache = cache.setdefault("_enc_core", {})
        v_core = core_encoder_fn(
            tls_id=tls_id,
            scene_stats=scene_stats,
            cache=core_cache,
            **kwargs,
        )
        v_core = np.asarray(v_core, dtype=np.float32)
        core_cache["_last_v_core"] = v_core

        if addon_encoder_fn is None:
            return v_core

        addon_cache = cache.setdefault("_enc_addon", {})
        v_add = addon_encoder_fn(
            tls_id=tls_id,
            scene_stats=scene_stats,
            cache=addon_cache,
            **kwargs,
        )
        v_add = np.asarray(v_add, dtype=np.float32)
        addon_cache["_last_v_addon"] = v_add

        return np.concatenate([v_core, v_add], axis=0)

    return encoder_fn


def _apply_vector_ablation(
    x: np.ndarray,
    *,
    zero_all: bool,
    zero_dims: List[int],
    noise_dims: List[int],
    noise_sigma: float,
) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).reshape(-1).copy()
    if zero_all:
        out[:] = 0.0
        return out

    for d in zero_dims:
        if 0 <= d < out.shape[0]:
            out[d] = 0.0
    if noise_sigma > 0.0:
        for d in noise_dims:
            if 0 <= d < out.shape[0]:
                out[d] += np.random.normal(0.0, noise_sigma)
    return out


def encode_state_for_policy(
    tls_id: str,
    *,
    cache_root: Dict[str, Any],
    encoder_fn: Callable[..., np.ndarray],
    addon_encoder_name: str,
    zero_expert_features: bool,
    zero_expert_dims: List[int],
    noise_expert_dims: List[int],
    noise_sigma: float,
    expected_state_dim: Optional[int] = None,
    scene_stats: Any | None = None,
) -> np.ndarray:
    state = np.asarray(
        encoder_fn(
            tls_id,
            scene_stats=scene_stats,
            moving_speed_threshold=0.1,
            stopped_speed_threshold=0.1,
            cache=cache_root,
        ),
        dtype=np.float32,
    ).reshape(-1)

    wants_expert_ablation = bool(zero_expert_features or zero_expert_dims or noise_expert_dims)
    addon_name_norm = str(addon_encoder_name).strip().lower()
    if wants_expert_ablation:
        if addon_name_norm != "expert":
            raise ValueError(
                "Expert-feature ablation/noise is only supported when addon_encoder_name='expert'. "
                f"Got addon_encoder_name={addon_encoder_name!r}."
            )

        core = np.asarray(cache_root.get("_enc_core", {}).get("_last_v_core", []), dtype=np.float32).reshape(-1)
        addon = np.asarray(cache_root.get("_enc_addon", {}).get("_last_v_addon", []), dtype=np.float32).reshape(-1)
        if addon.size == 0:
            raise RuntimeError(
                "Expert addon encoder produced no cached addon vector; cannot apply expert ablation/noise."
            )

        addon = _apply_vector_ablation(
            addon,
            zero_all=bool(zero_expert_features),
            zero_dims=zero_expert_dims,
            noise_dims=noise_expert_dims,
            noise_sigma=float(noise_sigma),
        )
        state = np.concatenate([core, addon], axis=0).astype(np.float32)

    if expected_state_dim is not None and state.size != int(expected_state_dim):
        raise ValueError(f"Final state size mismatch: got {state.size}, expected {expected_state_dim}")

    return state


def get_num_lanes_from_cache(cache_root: Dict[str, Any]) -> int:
    lane_ids = cache_root.get("_enc_core", {}).get("lane_ids", [])
    if not lane_ids:
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


@dataclass
class EpisodeUniversalMetrics:
    """Per-episode accumulators for universal evaluation metrics.

    Definitions used here:
      - average queue: time-average of the per-step mean queued vehicle count across lanes
      - worst queue: time-average of the per-step maximum queued vehicle count across lanes
      - throughput: total vehicles first observed on downstream lanes during control
      - average waiting time: time-average of the per-step queue-weighted mean waiting
        time among currently stopped/queued vehicles
    """

    step_count: int = 0
    avg_queue_sum: float = 0.0
    worst_queue_sum: float = 0.0
    avg_waiting_time_sum: float = 0.0
    worst_queue_peak: float = 0.0
    throughput_total: float = 0.0

    def update(self, scene_stats: Any, *, throughput_delta: float = 0.0) -> None:
        per_lane = _scene_per_lane_map(scene_stats)
        q = np.asarray(per_lane.get("queue_count", []), dtype=np.float32).reshape(-1)
        w = np.asarray(per_lane.get("mean_wait_stopped_s", []), dtype=np.float32).reshape(-1)

        if q.size == 0:
            avg_queue = 0.0
            worst_queue = 0.0
            avg_waiting = 0.0
        else:
            avg_queue = float(q.mean())
            worst_queue = float(q.max())
            queued_total = float(q.sum())
            if queued_total > 1e-6 and w.size == q.size:
                avg_waiting = float(np.dot(w, q) / queued_total)
            else:
                avg_waiting = 0.0

        self.step_count += 1
        self.avg_queue_sum += avg_queue
        self.worst_queue_sum += worst_queue
        self.avg_waiting_time_sum += avg_waiting
        self.worst_queue_peak = max(self.worst_queue_peak, worst_queue)
        self.throughput_total += float(max(0.0, throughput_delta))

    def as_dict(self, *, controlled_time_s: float) -> dict[str, float]:
        denom = max(1, self.step_count)
        ct = max(1e-6, float(controlled_time_s))
        return {
            "avg_queue_mean": float(self.avg_queue_sum / denom),
            "worst_queue_mean": float(self.worst_queue_sum / denom),
            "worst_queue_peak": float(self.worst_queue_peak),
            "avg_waiting_time_s_mean": float(self.avg_waiting_time_sum / denom),
            "throughput_total": float(self.throughput_total),
            "throughput_veh_per_hour": float(self.throughput_total * 3600.0 / ct),
            "metric_step_count": int(self.step_count),
        }


def _scene_per_lane_map(scene_stats: Any) -> dict[str, Any]:
    if hasattr(scene_stats, "per_lane"):
        return getattr(scene_stats, "per_lane")
    if isinstance(scene_stats, dict) and "per_lane" in scene_stats:
        return scene_stats["per_lane"]
    raise TypeError("scene_stats must expose per_lane")


def _parse_dim_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip()]


def _load_meta(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_controller(args: argparse.Namespace):
    name = str(args.controller_name).strip().lower()
    if name == "ppo":
        return None
    if name == "fully_actuated":
        return make_controller(
            name,
            min_green_s=float(args.fa_min_green),
            max_green_s=float(args.fa_max_green),
            extension_s=float(args.fa_extension),
            min_major_green_s=float(args.fa_min_major_green),
            demand_key=str(args.fa_demand_key),
            gap_out_threshold=float(args.fa_gap_out_threshold),
            switch_hysteresis=float(args.fa_switch_hysteresis),
            min_switch_demand=float(args.fa_min_switch_demand),
            aggregate=str(args.fa_aggregate),
        )
    if name == "max_pressure":
        return make_controller(
            name,
            min_major_green_s=float(args.mp_min_major_green),
            hold_s=float(args.mp_hold),
            upstream_key=str(args.mp_upstream_key),
            veh_equiv_len_m=float(args.mp_veh_equiv_len),
            clip_occ=float(args.mp_clip_occ),
            tie_break_current=bool(args.mp_tie_break_current),
        )
    if name == "webster" or name == "fixed_time":
        controller = make_controller(
            "webster",
            min_major_green_s=5.0,
            demand_key="count_ratio_norm",
            cycle_min_s=40.0,
            cycle_max_s=140.0,
            startup_lost_per_phase_s=2.0,
        )
        return controller
    known = ", ".join(["ppo", *list_controllers()])
    raise ValueError(f"unknown controller_name={args.controller_name!r}. known: {known}")


def eval_one_checkpoint(args: argparse.Namespace) -> None:
    controller_name = str(args.controller_name).strip().lower()
    use_ppo = controller_name == "ppo"
    ckpt_path: Path | None = Path(args.checkpoint) if args.checkpoint else None
    meta_path: Path | None = (
        Path(args.meta) if args.meta is not None else (ckpt_path.with_suffix(".json") if ckpt_path else None)
    )

    if use_ppo:
        if ckpt_path is None or not ckpt_path.exists():
            print(f"get ckpt path from args: {ckpt_path}, meta: {meta_path}")
            raise FileNotFoundError("--checkpoint is required and must exist when --controller-name ppo")
        if meta_path is None or not meta_path.exists():
            print(f"get meta path from args: {meta_path}, or derived from ckpt: {meta_path}")
            raise FileNotFoundError("meta json is required for PPO evaluation")
    meta = _load_meta(meta_path)

    # load checkpoint/model only for PPO
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"> Using device: {device}")
    agent: PPOAgent | None = None
    state_dim: Optional[int] = None
    action_dim_meta: Optional[int] = None
    expert_feature_dim = 0
    layer_count = 2
    use_skip = False
    use_expert_features = False
    core_encoder_name = str(getattr(args, "core_encoder_name", "bounded_v2") or "bounded_v2").strip()
    deprecated_encoder_name = getattr(args, "encoder_name", None)
    if deprecated_encoder_name is not None:
        core_encoder_name = str(deprecated_encoder_name).strip()
    addon_encoder_name = str(getattr(args, "addon_encoder_name", "none") or "none").strip()

    if use_ppo:
        ckpt = torch.load(str(ckpt_path), map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        state_dim = int(meta["state_dim"])
        action_dim_meta = int(meta["action_dim"])
        hidden_dim = int(meta.get("hidden_dim", 256))
        expert_feature_dim = int(meta.get("expert_dim", 0))
        layer_count = int(meta.get("layer_count", 2))
        use_skip = bool(meta.get("use_skip", False))
        use_expert_features = bool(meta.get("use_expert_features", False))
        core_encoder_name = str(meta.get("core_encoder_name", "")).strip()
        if not core_encoder_name:
            raise KeyError(
                "meta json missing 'core_encoder_name'; PPO evaluation requires registry-based encoder metadata."
            )
        addon_encoder_name = str(meta.get("addon_encoder_name", "none") or "none").strip()

        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim_meta,
            seed=int(meta.get("seed", 0)),
            hidden_dim=hidden_dim,
            n_layer=layer_count,
            use_skip=use_skip,
            actor_lr=float(meta.get("actor_lr", 3e-4)),
            critic_lr=float(meta.get("critic_lr", 1e-3)),
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

    tls_id = args.tls_id if args.tls_id is not None else str(meta.get("tls_id", ""))
    print(f"> Using tls_id: {tls_id}")
    if not tls_id:
        raise ValueError("tls_id not found in meta, and --tls-id not provided")

    sumocfg = args.sumocfg if args.sumocfg is not None else str(meta.get("sumocfg", ""))
    if not sumocfg:
        raise ValueError("sumocfg not found in meta, and --sumocfg not provided")

    print(f"> Using controller: {controller_name}")
    print(f"> Using core encoder: {core_encoder_name}")
    print(f"> Using addon encoder: {addon_encoder_name}")
    print(f"> Using expert features(meta): {use_expert_features}")

    # eval knobs
    episodes = int(args.episodes)
    episode_len_s = float(args.episode_len)
    warmup_s = float(args.warmup)
    action_hold_s = float(args.hold) if args.hold is not None else float(meta.get("action_hold_s", 10.0))

    core_encoder_fn = resolve_encoder(
        core_encoder_name,
        min_major_green_s=action_hold_s,
    )
    addon_encoder_fn = None
    if not _encoder_name_is_none(addon_encoder_name):
        addon_encoder_fn = resolve_encoder(
            addon_encoder_name,
            min_major_green_s=action_hold_s,
        )
    policy_encoder_fn = make_combined_encoder(
        core_encoder_fn=core_encoder_fn,
        addon_encoder_fn=addon_encoder_fn,
    )
    zero_expert_active = bool(args.zero_expert) and str(addon_encoder_name).strip().lower() == "expert"

    sumo_seed_base = int(args.sumo_seed) if args.sumo_seed is not None else int(meta.get("sumo_seed", 0))
    traffic_scale = (
        float(args.traffic_scale) if args.traffic_scale is not None else float(meta.get("traffic_scale_mean", 1.0))
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
    # [NEW] base interval for right-endpoint interval scaling
    base_interval_s = max(1e-6, float(action_hold_s))

    # ablation: only makes sense if checkpoint expects expert features
    zero_expert = bool(args.zero_expert)
    zero_dims = _parse_dim_list(args.zero_expert_dims)
    noise_dims = _parse_dim_list(args.noise_expert_dims)
    noise_sigma = float(args.noise_sigma)

    # controller backend
    controller = _build_controller(args)

    # logging setup
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    tag = f"__{args.log_tag}" if args.log_tag else ""
    mode = "zeroexp" if zero_expert_active else controller_name
    run_name = str(meta.get("run_name", "run" if use_ppo else controller_name))
    base = f"eval_{run_name}__{tls_id}__{mode}{tag}__{ts}"
    jsonl_path = log_dir / f"{base}.jsonl"
    summary_path = log_dir / f"{base}_summary.json"
    jsonl_f = open(jsonl_path, "w", encoding="utf-8")
    header = {
        "type": "header",
        "controller_name": controller_name,
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "meta": str(meta_path) if meta_path else None,
        "tls_id": tls_id,
        "sumocfg": sumocfg,
        "episodes": int(args.episodes),
        "deterministic": bool(args.deterministic),
        "core_encoder_name": core_encoder_name,
        "addon_encoder_name": addon_encoder_name,
        "use_expert_features_meta": bool(use_expert_features),
        "zero_expert_eval": bool(zero_expert_active),
        "traffic_scale": float(traffic_scale),
        "sumo_seed_base": int(sumo_seed_base),
        "zero_expert_dims": zero_dims,
        "noise_expert_dims": noise_dims,
        "noise_sigma": noise_sigma,
    }
    jsonl_f.write(json.dumps(header, separators=(",", ":")) + "\n")
    jsonl_f.flush()

    # summary accumulators
    ep_returns: List[float] = []
    ep_thr_norms: List[float] = []
    ep_q_rewards: List[float] = []
    ep_avg_queue: List[float] = []
    ep_worst_queue: List[float] = []
    ep_worst_queue_peak: List[float] = []
    ep_throughput_total: List[float] = []
    ep_throughput_per_hour: List[float] = []
    ep_avg_waiting_time: List[float] = []

    for ep in range(episodes):
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

            cache_root: Dict[str, Any] = {}
            tls_state = EvalTLSState()

            action_dim = int(tls_major_action_dim(tls_id, cache_root))
            if action_dim_meta is not None and action_dim != int(action_dim_meta):
                raise ValueError(
                    f"Action dimension mismatch: scenario has {action_dim}, checkpoint expects {action_dim_meta}"
                )

            action_counts = np.zeros((action_dim,), dtype=np.int32)
            switches = 0
            last_action = None
            pi_sum = np.zeros((action_dim,), dtype=np.float64)
            pi_entropy_sum = 0.0
            pi_min_sum = 0.0
            pi_max_sum = 0.0
            n_decisions = 0
            a_interval_count = np.zeros((action_dim,), dtype=np.int32)
            a_reward_sum = np.zeros((action_dim,), dtype=np.float64)
            a_thr_sum = np.zeros((action_dim,), dtype=np.float64)
            a_q_sum = np.zeros((action_dim,), dtype=np.float64)

            if controller is not None:
                controller.reset(tls_id=tls_id, cache=cache_root)

            ret_sum = 0.0
            thr_norm_sum = 0.0
            q_reward_sum = 0.0
            n_intervals = 0
            universal_metrics = EpisodeUniversalMetrics()
            last_seen_total = 0

            while True:
                sim_t = float(traci.simulation.getTime())
                done_episode = (sim_t >= episode_len_s) or (traci.simulation.getMinExpectedNumber() <= 0)
                in_control = sim_t >= warmup_s

                throughput_tracker_step(tls_id, cache_root)
                seen_total = cache_root.get("_tp_seen_total", set())
                current_seen_total = len(seen_total) if isinstance(seen_total, set) else int(len(seen_total))
                throughput_delta_step = max(0, current_seen_total - int(last_seen_total))
                last_seen_total = int(current_seen_total)

                scene_stats = None
                if in_control:
                    scene_stats = collect_tsc_scene_snapshot(
                        tls_id,
                        cache=cache_root.setdefault("_scene", {}),
                    )
                    universal_metrics.update(scene_stats, throughput_delta=float(throughput_delta_step))

                if tls_state.segment_end_time > 0.0 and sim_t >= tls_state.segment_end_time:
                    tls_state.segment_end_time = tls_advance_pending_segments(
                        tls_id=tls_id,
                        pending_segments=tls_state.pending_segments,
                        segment_end_time=tls_state.segment_end_time,
                        sim_t=sim_t,
                    )

                if done_episode:
                    if tls_state.prev_in_control and tls_state.prev_state is not None:
                        assert scene_stats is not None
                        s_next = encode_state_for_policy(
                            tls_id,
                            cache_root=cache_root,
                            encoder_fn=policy_encoder_fn,
                            addon_encoder_name=addon_encoder_name,
                            zero_expert_features=zero_expert_active,
                            zero_expert_dims=zero_dims,
                            noise_expert_dims=noise_dims,
                            noise_sigma=noise_sigma,
                            expected_state_dim=state_dim,
                            scene_stats=scene_stats,
                        )

                        thr = reward_throughput_per_second_on_decision(sim_time=sim_t, cache=cache_root)
                        thr_norm = min(1.0, max(0.0, float(thr) / max(1e-6, throughput_ref)))

                        q_reward = reward_softmax_queue_from_encoded_state(
                            scene_stats=scene_stats,
                            power=queue_power,
                            scale=queue_ref,
                            softmax_beta=5.0,
                            clip_nonnegative=True,
                        )
                        # [NEW] right-endpoint interval scaling
                        dt_interval = max(1e-6, float(sim_t - tls_state.action_start_time))
                        interval_scale = float(dt_interval / base_interval_s)
                        thr_term = float(thr_norm) * interval_scale
                        q_term = float(q_reward) * interval_scale

                        # r = float(w_thr) * thr_norm + float(w_q) * float(q_reward)
                        r = float(w_thr) * thr_term + float(w_q) * q_term
                        r = clip_reward(r, r_clip_lo, r_clip_hi)

                        ret_sum += float(r)
                        thr_norm_sum += float(thr_norm)
                        q_reward_sum += float(q_reward)
                        n_intervals += 1

                        if tls_state.prev_action is not None:
                            pa = int(tls_state.prev_action)
                            a_interval_count[pa] += 1
                            a_reward_sum[pa] += float(r)
                            a_thr_sum[pa] += float(thr_norm)
                            a_q_sum[pa] += float(q_reward)
                    break

                if in_control and sim_t >= tls_state.next_decision_time:
                    assert scene_stats is not None
                    s_cur = encode_state_for_policy(
                        tls_id,
                        cache_root=cache_root,
                        encoder_fn=policy_encoder_fn,
                        addon_encoder_name=addon_encoder_name,
                        zero_expert_features=zero_expert_active,
                        zero_expert_dims=zero_dims,
                        noise_expert_dims=noise_dims,
                        noise_sigma=noise_sigma,
                        expected_state_dim=state_dim,
                        scene_stats=scene_stats,
                    )

                    if tls_state.prev_in_control and tls_state.prev_state is not None:
                        thr = reward_throughput_per_second_on_decision(sim_time=sim_t, cache=cache_root)
                        thr_norm = min(1.0, max(0.0, float(thr) / max(1e-6, throughput_ref)))

                        q_reward = reward_softmax_queue_from_encoded_state(
                            scene_stats=scene_stats,
                            power=queue_power,
                            scale=queue_ref,
                            softmax_beta=5.0,
                            clip_nonnegative=True,
                        )
                        # [NEW] right-endpoint interval scaling
                        dt_interval = max(1e-6, float(sim_t - tls_state.action_start_time))
                        interval_scale = float(dt_interval / base_interval_s)
                        thr_term = float(thr_norm) * interval_scale
                        q_term = float(q_reward) * interval_scale

                        # r = float(w_thr) * thr_norm + float(w_q) * float(q_reward)
                        r = float(w_thr) * thr_term + float(w_q) * q_term
                        r = clip_reward(r, r_clip_lo, r_clip_hi)

                        ret_sum += float(r)
                        thr_norm_sum += float(thr_norm)
                        q_reward_sum += float(q_reward)
                        n_intervals += 1

                        if tls_state.prev_action is not None:
                            pa = int(tls_state.prev_action)
                            a_interval_count[pa] += 1
                            a_reward_sum[pa] += float(r)
                            a_thr_sum[pa] += float(thr_norm)
                            a_q_sum[pa] += float(q_reward)
                    else:
                        _ = reward_throughput_per_second_on_decision(sim_time=sim_t, cache=cache_root)

                    if controller_name == "ppo":
                        assert agent is not None
                        if args.deterministic:
                            a = int(agent.act_greedy(s_cur))
                        else:
                            a, _logp, _v = agent.act(s_cur)
                            a = int(a)

                        logits, probs, _v = agent.forward_logits_value(s_cur, return_probs=True, to_cpu=True)
                        p = probs.numpy()[0]
                        pi_sum += p
                        pi_entropy_sum += float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())
                        pi_min_sum += float(p.min())
                        pi_max_sum += float(p.max())
                        n_decisions += 1
                        decision_hold_s: Optional[float] = None
                    else:
                        assert controller is not None
                        decision = controller.choose_action(
                            tls_id,
                            scene_stats=scene_stats,
                            sim_time=sim_t,
                            cache=cache_root,
                        )
                        a = int(decision.action)
                        decision_hold_s = decision.hold_s

                    action_counts[int(a)] += 1
                    if last_action is not None and int(a) != int(last_action):
                        switches += 1
                    last_action = int(a)

                    target_major_phase = tls_action_to_major_phase(tls_id, cache_root, action=int(a))
                    hold_for_decision = float(decision_hold_s) if decision_hold_s is not None else float(action_hold_s)
                    segments = tls_build_switch_segments(
                        tls_id,
                        cache_root,
                        target_major_phase=int(target_major_phase),
                        hold_s=float(hold_for_decision),
                        current_phase=int(traci.trafficlight.getPhase(tls_id)),
                    )

                    first_phase, first_dur = segments[0]
                    tls_set_phase_frozen(tls_id, int(first_phase))

                    tls_state.pending_segments = deque(segments[1:])
                    tls_state.segment_end_time = sim_t + float(first_dur)
                    tls_state.next_decision_time = sim_t + float(sum(d for _, d in segments))
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
            metric_rec = universal_metrics.as_dict(controlled_time_s=controlled_time_s)

            ep_avg_queue.append(float(metric_rec["avg_queue_mean"]))
            ep_worst_queue.append(float(metric_rec["worst_queue_mean"]))
            ep_worst_queue_peak.append(float(metric_rec["worst_queue_peak"]))
            ep_throughput_total.append(float(metric_rec["throughput_total"]))
            ep_throughput_per_hour.append(float(metric_rec["throughput_veh_per_hour"]))
            ep_avg_waiting_time.append(float(metric_rec["avg_waiting_time_s_mean"]))

            rec = {
                "type": "episode",
                "ep": int(ep),
                "sumo_seed": int(sumo_seed_base + ep),
                "controller_name": controller_name,
                "return_sum": float(ret_sum),
                "thr_norm_mean": float(thr_norm_sum / max(1, n_intervals)),
                "q_reward_mean": float(q_reward_sum / max(1, n_intervals)),
                "n_intervals": int(n_intervals),
                "action_counts": action_counts.tolist(),
                "switches": int(switches),
                "switch_rate_per_min": float(switch_rate_per_min),
            }
            # universal metrics
            rec.update(
                {
                    "avg_queue_mean": float(metric_rec["avg_queue_mean"]),
                    "worst_queue_mean": float(metric_rec["worst_queue_mean"]),
                    "worst_queue_peak": float(metric_rec["worst_queue_peak"]),
                    "throughput_total": float(metric_rec["throughput_total"]),
                    "throughput_veh_per_hour": float(metric_rec["throughput_veh_per_hour"]),
                    "avg_waiting_time_s_mean": float(metric_rec["avg_waiting_time_s_mean"]),
                    "metric_step_count": int(metric_rec["metric_step_count"]),
                }
            )
            if controller_name == "ppo":
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

            act_total = int(action_counts.sum())
            p_emp = action_counts.astype(np.float64) / max(1, act_total)
            emp_entropy = float(-(p_emp * np.log(np.clip(p_emp, 1e-12, 1.0))).sum())
            emp_neff = float(1.0 / np.sum(p_emp * p_emp)) if act_total > 0 else 0.0
            min_action_frac = float(p_emp.min()) if act_total > 0 else 0.0
            rec.update(
                {
                    "emp_action_entropy": emp_entropy,
                    "emp_action_neff": emp_neff,
                    "min_action_frac": min_action_frac,
                }
            )

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
                f"[eval ep={ep}] controller={controller_name} return_sum={ret_sum:.4f} "
                f"thr_norm_mean={ep_thr_norms[-1]:.4f} "
                f"q_reward_mean={ep_q_rewards[-1]:.4f} "
                f"avg_queue={ep_avg_queue[-1]:.3f} "
                f"worst_queue={ep_worst_queue[-1]:.3f} "
                f"throughput={ep_throughput_total[-1]:.1f} "
                f"avg_wait={ep_avg_waiting_time[-1]:.3f} "
                f"intervals={n_intervals}"
            )

        finally:
            try:
                traci.close()
            except Exception:
                pass

    arr_r = np.asarray(ep_returns, dtype=np.float32)
    arr_thr = np.asarray(ep_thr_norms, dtype=np.float32)
    arr_q = np.asarray(ep_q_rewards, dtype=np.float32)
    arr_avg_queue = np.asarray(ep_avg_queue, dtype=np.float32)
    arr_worst_queue = np.asarray(ep_worst_queue, dtype=np.float32)
    arr_worst_queue_peak = np.asarray(ep_worst_queue_peak, dtype=np.float32)
    arr_tp_total = np.asarray(ep_throughput_total, dtype=np.float32)
    arr_tp_hour = np.asarray(ep_throughput_per_hour, dtype=np.float32)
    arr_wait = np.asarray(ep_avg_waiting_time, dtype=np.float32)

    summary = {
        "controller_name": controller_name,
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "meta": str(meta_path) if meta_path else None,
        "tls_id": tls_id,
        "sumocfg": sumocfg,
        "core_encoder_name": core_encoder_name,
        "addon_encoder_name": addon_encoder_name,
        "use_expert_features_meta": bool(use_expert_features),
        "zero_expert_eval": bool(zero_expert_active),
        "episodes": int(episodes),
        "traffic_scale": float(traffic_scale),
        "sumo_seed_base": int(sumo_seed_base),
        "return_sum_mean": float(arr_r.mean()),
        "return_sum_std": float(arr_r.std()),
        "thr_norm_mean_mean": float(arr_thr.mean()),
        "thr_norm_mean_std": float(arr_thr.std()),
        "q_reward_mean_mean": float(arr_q.mean()),
        "q_reward_mean_std": float(arr_q.std()),
        "avg_queue_mean": float(arr_avg_queue.mean()),
        "avg_queue_std": float(arr_avg_queue.std()),
        "worst_queue_mean": float(arr_worst_queue.mean()),
        "worst_queue_std": float(arr_worst_queue.std()),
        "worst_queue_peak_mean": float(arr_worst_queue_peak.mean()),
        "worst_queue_peak_std": float(arr_worst_queue_peak.std()),
        "throughput_total_mean": float(arr_tp_total.mean()),
        "throughput_total_std": float(arr_tp_total.std()),
        "throughput_veh_per_hour_mean": float(arr_tp_hour.mean()),
        "throughput_veh_per_hour_std": float(arr_tp_hour.std()),
        "avg_waiting_time_s_mean": float(arr_wait.mean()),
        "avg_waiting_time_s_std": float(arr_wait.std()),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    jsonl_f.close()

    print(f"\n[eval] wrote: {jsonl_path}")
    print(f"[eval] wrote: {summary_path}")
    print("\n=== Evaluation Summary ===")
    print(f"controller: {controller_name}")
    print(f"checkpoint: {ckpt_path}")
    print(f"meta:       {meta_path}")
    print(f"tls_id:     {tls_id}")
    print(f"sumocfg:    {sumocfg}")
    print(f"core_encoder_name: {core_encoder_name}")
    print(f"addon_encoder_name: {addon_encoder_name}")
    print(f"use_expert_features(meta): {use_expert_features}")
    print(f"zero_expert_features(eval): {zero_expert_active}")
    print(f"episodes:   {episodes}")
    print(f"traffic_scale: {traffic_scale}")
    print(f"sumo_seed_base: {sumo_seed_base}")
    print(f"return_sum: mean={arr_r.mean():.4f} std={arr_r.std():.4f}")
    print(f"thr_norm_mean: mean={arr_thr.mean():.4f} std={arr_thr.std():.4f}")
    print(f"q_reward_mean: mean={arr_q.mean():.4f} std={arr_q.std():.4f}")
    print(f"avg_queue_mean: mean={arr_avg_queue.mean():.4f} std={arr_avg_queue.std():.4f}")
    print(f"worst_queue_mean: mean={arr_worst_queue.mean():.4f} std={arr_worst_queue.std():.4f}")
    print(f"worst_queue_peak: mean={arr_worst_queue_peak.mean():.4f} std={arr_worst_queue_peak.std():.4f}")
    print(f"throughput_total: mean={arr_tp_total.mean():.4f} std={arr_tp_total.std():.4f}")
    print(f"throughput_veh_per_hour: mean={arr_tp_hour.mean():.4f} std={arr_tp_hour.std():.4f}")
    print(f"avg_waiting_time_s_mean: mean={arr_wait.mean():.4f} std={arr_wait.std():.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()

    # backend selection
    ap.add_argument(
        "--controller-name",
        type=str,
        default="ppo",
        choices=["ppo", *list_controllers()],
        help="Controller backend to evaluate: PPO policy or rule-based baseline.",
    )

    # paths
    ap.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint (required for PPO)")
    ap.add_argument(
        "--meta",
        type=str,
        default=None,
        help="Path to meta .json (default for PPO: checkpoint with .json suffix)",
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
        help="Optional tag appended to log filename.",
    )

    # SUMO runtime knobs
    ap.add_argument("--sumocfg", type=str, default=None, help="Override sumocfg in meta or supply directly")
    ap.add_argument("--tls-id", type=str, default=None, help="Override tls_id in meta or supply directly")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--delay-ms", type=int, default=1)

    # episode controls
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--episode-len", type=float, default=3600.0)
    ap.add_argument("--warmup", type=float, default=100.0)
    ap.add_argument(
        "--hold", type=float, default=None, help="Override default hold_s for PPO and as fallback for controllers"
    )
    ap.add_argument("--sumo-seed", type=int, default=None)
    ap.add_argument("--traffic-scale", type=float, default=None)

    # action selection
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Use greedy action selection for PPO (recommended)",
    )

    # encoder override for non-PPO evaluation (PPO uses meta core/addon encoder names)
    ap.add_argument(
        "--core-encoder-name",
        type=str,
        default="bounded_v2",
        choices=available_encoder_names(),
        help="Core encoder registry name used when PPO meta is unavailable.",
    )
    ap.add_argument(
        "--addon-encoder-name",
        type=str,
        default="none",
        choices=["none", *available_encoder_names()],
        help="Optional addon encoder registry name used when PPO meta is unavailable.",
    )
    ap.add_argument(
        "--encoder-name",
        type=str,
        default=None,
        help="Deprecated alias for --core-encoder-name. Must still be a registry encoder name.",
    )

    # expert ablation (PPO only)
    ap.add_argument(
        "--zero-expert",
        action="store_true",
        help="If checkpoint uses expert features, zero the expert slice at inference time.",
    )
    ap.add_argument("--zero-expert-dims", type=str, default="", help="Comma-separated expert dims to zero")
    ap.add_argument(
        "--noise-expert-dims", type=str, default="", help="Comma-separated expert dims to add Gaussian noise"
    )
    ap.add_argument("--noise-sigma", type=float, default=0.05, help="Stddev for expert-feature noise")

    # reward decomposition knobs (common evaluation metric)
    ap.add_argument("--thr-ref", type=float, default=2.0)
    ap.add_argument("--queue-ref", type=float, default=1.0)
    ap.add_argument("--w-thr", type=float, default=1.0)
    ap.add_argument("--w-queue", type=float, default=1.0)
    ap.add_argument("--queue-power", type=float, default=1.0)
    ap.add_argument("--reward-clip-lo", type=float, default=-2.0)
    ap.add_argument("--reward-clip-hi", type=float, default=2.0)

    # Fully actuated controller knobs
    ap.add_argument("--fa-min-green", type=float, default=8.0)
    ap.add_argument("--fa-max-green", type=float, default=35.0)
    ap.add_argument("--fa-extension", type=float, default=5.0)
    ap.add_argument("--fa-min-major-green", type=float, default=5.0)
    ap.add_argument("--fa-demand-key", type=str, default="queue_ratio_norm")
    ap.add_argument("--fa-gap-out-threshold", type=float, default=0.05)
    ap.add_argument("--fa-switch-hysteresis", type=float, default=0.02)
    ap.add_argument("--fa-min-switch-demand", type=float, default=0.01)
    ap.add_argument("--fa-aggregate", type=str, default="sum", choices=["sum", "max"])

    # Max-pressure controller knobs
    ap.add_argument("--mp-min-major-green", type=float, default=5.0)
    ap.add_argument("--mp-hold", type=float, default=10.0)
    ap.add_argument("--mp-upstream-key", type=str, default="count_ratio_norm")
    ap.add_argument("--mp-veh-equiv-len", type=float, default=7.5)
    ap.add_argument("--mp-clip-occ", type=float, default=1.0)
    ap.add_argument("--mp-tie-break-current", action="store_true")

    # torch device
    ap.add_argument("--device", type=str, default=None)

    args = ap.parse_args()
    if args.controller_name == "ppo" and not args.deterministic:
        args.deterministic = True

    eval_one_checkpoint(args)


if __name__ == "__main__":
    main()
