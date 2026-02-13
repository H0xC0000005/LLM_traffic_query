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

import matplotlib

matplotlib.use("Agg")  # headless-safe for training machines
import matplotlib.pyplot as plt

from ppo_agent import PPOAgent, RolloutBuffer
from utility import *
from scene_encoder import (
    encode_tsc_state_vector_bounded_v2,
)
from expert_feature_extractor import *

"""
constants
"""
HEATMAP_EVERY = 2000  # e.g., every 10 update points


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

    v_core = encode_tsc_state_vector_bounded_v2(tls_id, cache=core_cache, **kwargs)
    # v_sem = tsc_isolated_intersection_feature_vector(tls_id, cache=sem_cache)
    v_sem = tsc_isolated_intersection_feature_vector(tls_id)
    sem_cache["_last_v_sem"] = v_sem  # <-- key line

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


def _next_heatmap_event(
    tls_id: str,
    event_counter: Dict[str, int],
    *,
    every_updates: int,
    log_first: bool = True,
) -> Tuple[bool, int]:
    """
    Returns (should_log_now, update_event_idx).
    update_event_idx increments ONLY when we are at a rollout-consume point.
    """
    idx = int(event_counter.get(tls_id, 0)) + 1
    event_counter[tls_id] = idx

    if log_first and idx == 1:
        return True, idx
    if every_updates <= 0:
        return False, idx
    return (idx % int(every_updates) == 0), idx


def _tb_add_corr_heatmap(
    writer: SummaryWriter,
    tag: str,
    mat: np.ndarray,
    step: int,
    *,
    xlabel: str,
    ylabel: str,
    title: Optional[str] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    """
    Logging-only helper:
      - mat can be 1D (will be shown as 1xD) or 2D
      - assumes correlation-like values in [-1, 1]
    """
    a = np.asarray(mat, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2 or a.size == 0:
        return
    if not np.isfinite(a).any():
        return

    h, w = a.shape

    # dynamic figure size: keep readable for ~100 dims
    fig_w = float(np.clip(6.0 + w * 0.08, 6.0, 24.0))
    fig_h = float(np.clip(2.5 + h * 0.08, 2.5, 18.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(
        a,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # sparse ticks to avoid clutter for large dims
    def _ticks(n: int, max_ticks: int = 30):
        if n <= 0:
            return np.array([], dtype=int)
        if n <= max_ticks:
            return np.arange(n, dtype=int)
        stride = int(np.ceil(n / max_ticks))
        return np.arange(0, n, stride, dtype=int)

    xt = _ticks(w, max_ticks=40)
    yt = _ticks(h, max_ticks=30)

    ax.set_xticks(xt)
    ax.set_yticks(yt)

    # labels are actual indices
    ax.set_xticklabels([str(int(i)) for i in xt], rotation=90, fontsize=7)
    ax.set_yticklabels([str(int(i)) for i in yt], fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("correlation", rotation=90)

    writer.add_figure(tag, fig, global_step=step)
    plt.close(fig)


def tb_log_rollout_diagnostics(
    writer: SummaryWriter,
    tls_id: str,
    step: int,
    buf,
    agent: PPOAgent,
    action_dim: int,
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
        # writer.add_scalar(f"{tls_id}/rollout/adv_std", float(np.std(advs)), step)
        acts = _get_attr_any(buf, ["actions", "acts", "action"])
        if acts is not None:
            acts = acts.astype(np.int64).reshape(-1)

            if acts.shape[0] == advs.shape[0]:
                n = float(len(acts))

                for a in range(int(action_dim)):
                    mask = acts == a
                    cnt = int(mask.sum())

                    # how often action was sampled in this rollout
                    # writer.add_scalar(f"{tls_id}/rollout/action_count/a{a}", cnt, step)
                    writer.add_scalar(
                        f"{tls_id}/rollout/action_frac/a{a}", cnt / max(1.0, n), step
                    )

                    # # advantage stats for that action
                    # if cnt > 0:
                    #     adv_a = advs[mask]
                    #     writer.add_scalar(
                    #         f"{tls_id}/rollout/adv_mean/a{a}",
                    #         float(np.mean(adv_a)),
                    #         step,
                    #     )
                    #     writer.add_scalar(
                    #         f"{tls_id}/rollout/adv_std/a{a}", float(np.std(adv_a)), step
                    #     )
                    #     writer.add_scalar(
                    #         f"{tls_id}/rollout/adv_pos_frac/a{a}",
                    #         float(np.mean(adv_a > 0.0)),
                    #         step,
                    #     )
                    # else:
                    #     # no samples => mean advantage is undefined; keep 0 but count=0 exposes it
                    #     writer.add_scalar(f"{tls_id}/rollout/adv_mean/a{a}", 0.0, step)
                    #     writer.add_scalar(f"{tls_id}/rollout/adv_std/a{a}", 0.0, step)
                    #     writer.add_scalar(
                    #         f"{tls_id}/rollout/adv_pos_frac/a{a}", 0.0, step
                    #     )

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

    # Action distribution diagnostics
    acts = _get_attr_any(buf, ["actions", "act", "action"])
    if acts is not None:
        acts = np.asarray(acts, dtype=np.int64).reshape(-1)
        counts = np.bincount(acts, minlength=int(action_dim)).astype(np.float64)
        p_emp = counts / max(1.0, float(counts.sum()))

        emp_entropy = float(-(p_emp * np.log(np.clip(p_emp, 1e-12, 1.0))).sum())
        emp_neff = float(1.0 / np.sum(p_emp * p_emp))

        writer.add_scalar(f"{tls_id}/rollout/emp_action_entropy", emp_entropy, step)
        writer.add_scalar(f"{tls_id}/rollout/emp_action_neff", emp_neff, step)
        writer.add_scalar(
            f"{tls_id}/rollout/emp_min_action_frac", float(p_emp.min()), step
        )
    # pi distribution diagnostics
    # states = _get_attr_any(buf, ["states", "obs", "observations"])
    # if states is not None:
    #     X = np.asarray(states, dtype=np.float32)
    #     logits, probs, _v = agent.forward_logits_value(
    #         X, return_probs=True, to_cpu=True
    #     )
    #     P = probs.numpy()  # (B, A)
    #     mean_pi = P.mean(axis=0)

    #     pi_entropy = float(-(mean_pi * np.log(np.clip(mean_pi, 1e-12, 1.0))).sum())
    #     pi_neff = float(1.0 / np.sum(mean_pi * mean_pi))

    #     writer.add_scalar(f"{tls_id}/rollout/pi_entropy", pi_entropy, step)
    #     writer.add_scalar(f"{tls_id}/rollout/pi_neff", pi_neff, step)
    #     writer.add_scalar(f"{tls_id}/rollout/pi_min", float(mean_pi.min()), step)
    #     writer.add_scalar(f"{tls_id}/rollout/pi_max", float(mean_pi.max()), step)

    #     for a in range(int(action_dim)):
    #         writer.add_scalar(f"{tls_id}/rollout/pi_mean/a{a}", float(mean_pi[a]), step)


# ---------------------------------------------------------------------------


# ===========================
# Proposal 1 logging only
# ===========================
def tb_log_proposal1_expert_adv_corr_rollout(
    writer: SummaryWriter,
    tls_id: str,
    step: int,
    buf: RolloutBuffer,
    sem_dim: int,
    tracker: Optional[RunningExpertAdvPearson],
    *,
    log_heatmap: bool = False,
    heatmap_step: Optional[int] = None,  # NEW
) -> None:
    rep = proposal1_expert_adv_corr_from_rollout(buf, sem_dim=sem_dim, tracker=tracker)
    if not rep.get("ok", False):
        return

    writer.add_scalar(
        f"{tls_id}/expert_quality/p1_adv_corr/mean_abs", float(rep["mean_abs"]), step
    )
    writer.add_scalar(
        f"{tls_id}/expert_quality/p1_adv_corr/max_abs", float(rep["max_abs"]), step
    )
    writer.add_scalar(
        f"{tls_id}/expert_quality/p1_adv_corr/frac_abs_gt_0p10",
        float(rep["frac_abs_gt_0p10"]),
        step,
    )

    corr = np.asarray(rep["corr"], dtype=np.float32)
    finite = np.isfinite(corr)
    if np.any(finite):
        writer.add_histogram(
            f"{tls_id}/expert_quality/p1_adv_corr/hist_abs", np.abs(corr[finite]), step
        )

    # NEW: heatmap (1 x D)
    if log_heatmap:
        _tb_add_corr_heatmap(
            writer=writer,
            tag=f"{tls_id}/expert_quality/p1_adv_corr/heatmap",
            mat=corr.reshape(1, -1),
            step=(int(heatmap_step) if heatmap_step is not None else step),  # NEW,
            xlabel="expert_dim",
            ylabel="advantage",
            title="corr(expert_dim, advantage)",
        )


def tb_log_proposal1_expert_adv_corr_final(
    writer: SummaryWriter,
    tls_id: str,
    step: int,
    tracker: Optional[RunningExpertAdvPearson],
) -> None:
    if tracker is None:
        return
    rep = tracker.finalize()
    corr = np.asarray(rep["corr"], dtype=np.float32)
    n = np.asarray(rep["n"], dtype=np.int64)

    writer.add_scalar(
        f"{tls_id}/expert_quality_final/p1_adv_corr/mean_abs",
        float(rep["mean_abs"]),
        step,
    )
    writer.add_scalar(
        f"{tls_id}/expert_quality_final/p1_adv_corr/max_abs",
        float(rep["max_abs"]),
        step,
    )
    writer.add_scalar(
        f"{tls_id}/expert_quality_final/p1_adv_corr/frac_abs_gt_0p10",
        float(rep["frac_abs_gt_0p10"]),
        step,
    )

    finite = np.isfinite(corr)
    if np.any(finite):
        writer.add_histogram(
            f"{tls_id}/expert_quality_final/p1_adv_corr/hist_abs",
            np.abs(corr[finite]),
            step,
        )

    # NEW: final heatmap
    _tb_add_corr_heatmap(
        writer=writer,
        tag=f"{tls_id}/expert_quality_final/p1_adv_corr/heatmap",
        mat=corr.reshape(1, -1),
        step=step,
        xlabel="expert_dim",
        ylabel="advantage",
        title="Proposal 1 FINAL: corr(expert_dim, advantage)",
    )

    # keep your text table too if useful
    lines = ["|expert_dim|corr_to_adv|n|", "|---:|---:|---:|"]
    for i in range(corr.shape[0]):
        c = float(corr[i]) if np.isfinite(corr[i]) else float("nan")
        lines.append(f"|{i}|{c:.6f}|{int(n[i])}|")
    writer.add_text(
        f"{tls_id}/expert_quality_final/p1_adv_corr/report", "\n".join(lines), step
    )


# ===========================
# Proposal 2 logging only
# ===========================
def tb_log_proposal2_expert_core_xcorr_rollout(
    writer: SummaryWriter,
    tls_id: str,
    step: int,
    buf: RolloutBuffer,
    sem_dim: int,
    tracker: Optional[RunningExpertCoreCrossCorr],
    *,
    log_heatmap: bool = False,
    heatmap_step: Optional[int] = None,  # NEW
) -> None:
    rep = proposal2_expert_core_xcorr_from_rollout(
        buf, sem_dim=sem_dim, tracker=tracker
    )
    if not rep.get("ok", False):
        return

    writer.add_scalar(
        f"{tls_id}/expert_quality/p2_core_xcorr/mean_abs", float(rep["mean_abs"]), step
    )
    writer.add_scalar(
        f"{tls_id}/expert_quality/p2_core_xcorr/p95_abs", float(rep["p95_abs"]), step
    )
    writer.add_scalar(
        f"{tls_id}/expert_quality/p2_core_xcorr/frac_abs_gt_0p30",
        float(rep["frac_abs_gt_0p30"]),
        step,
    )
    writer.add_scalar(
        f"{tls_id}/expert_quality/p2_core_xcorr/sem_max_abs_mean",
        float(rep["sem_max_abs_mean"]),
        step,
    )

    c = np.asarray(rep["corr"], dtype=np.float32)
    finite = np.isfinite(c)
    if np.any(finite):
        writer.add_histogram(
            f"{tls_id}/expert_quality/p2_core_xcorr/hist_abs", np.abs(c[finite]), step
        )

    # NEW: heatmap (D_sem x D_core)
    if log_heatmap:
        _tb_add_corr_heatmap(
            writer=writer,
            tag=f"{tls_id}/expert_quality/p2_core_xcorr/heatmap",
            mat=c,
            step=(int(heatmap_step) if heatmap_step is not None else step),
            xlabel="core_dim",
            ylabel="expert_dim",
            title="corr(expert_dim, core_dim)",
        )


def tb_log_proposal2_expert_core_xcorr_final(
    writer: SummaryWriter,
    tls_id: str,
    step: int,
    tracker: Optional[RunningExpertCoreCrossCorr],
) -> None:
    if tracker is None:
        return
    rep = tracker.finalize()
    c = np.asarray(rep["corr"], dtype=np.float32)

    writer.add_scalar(
        f"{tls_id}/expert_quality_final/p2_core_xcorr/mean_abs",
        float(rep["mean_abs"]),
        step,
    )
    writer.add_scalar(
        f"{tls_id}/expert_quality_final/p2_core_xcorr/p95_abs",
        float(rep["p95_abs"]),
        step,
    )
    writer.add_scalar(
        f"{tls_id}/expert_quality_final/p2_core_xcorr/frac_abs_gt_0p30",
        float(rep["frac_abs_gt_0p30"]),
        step,
    )
    writer.add_scalar(
        f"{tls_id}/expert_quality_final/p2_core_xcorr/sem_max_abs_mean",
        float(rep["sem_max_abs_mean"]),
        step,
    )

    finite = np.isfinite(c)
    if np.any(finite):
        writer.add_histogram(
            f"{tls_id}/expert_quality_final/p2_core_xcorr/hist_abs",
            np.abs(c[finite]),
            step,
        )

    # NEW: final heatmap
    _tb_add_corr_heatmap(
        writer=writer,
        tag=f"{tls_id}/expert_quality_final/p2_core_xcorr/heatmap",
        mat=c,
        step=step,
        xlabel="core_dim",
        ylabel="expert_dim",
        title="Proposal 2 FINAL: corr(expert_dim, core_dim)",
    )

    # keep top-k text if wanted
    top_pairs = proposal2_topk_abs_pairs(c, k=20)
    lines = ["|rank|expert_dim|core_dim|corr|abs_corr|", "|---:|---:|---:|---:|---:|"]
    for rk, (i, j, v) in enumerate(top_pairs, start=1):
        lines.append(f"|{rk}|{i}|{j}|{v:+.6f}|{abs(v):.6f}|")
    writer.add_text(
        f"{tls_id}/expert_quality_final/p2_core_xcorr/top_pairs", "\n".join(lines), step
    )


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


pcnt = 0


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
    w_delta_queue: float,
    w_wait: float,
    w_queue_zone: float,
    wait_ref_s: float,
    wait_barrier_start_s: float,
    softmax_wait_beta: float,
    softmax_queue_beta: float,
    queue_power: float,
    top2_w1: float,
    top2_w2: float,
    reward_clip_lo: float,
    reward_clip_hi: float,
    # [NEW] potential-based shaping against major-phase starvation
    w_starve_potential: float,
    starve_softmax_beta: float,
    starve_power: float,
    # PPO defaults (kept minimal)
    rollout_steps: int,
    ppo_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    vf_clip_eps: float,
    gae_lambda: float,
    ent_coef: float,
    vf_coef: float,
    # -------------------------------
    # [NEW] Training-scheme stabilizers
    # -------------------------------
    ent_coef_end: Optional[float],
    ent_coef_decay_updates: int,
    explore_alpha_start: float,
    explore_alpha_end: float,
    explore_alpha_decay_updates: int,
    target_kl: Optional[float],
    adv_clip: Optional[float],
    # control flags & tags
    use_expert_features: bool = False,
    log_tag: str = "",
) -> None:
    global pcnt
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_expert_features:
        encoder_fn = encode_tsc_state_vector_combined
        print("[run_ppo_tsc] Using combined encoder with expert features.")
        time.sleep(3)  # allow user to see the print
    else:
        encoder_fn = encode_tsc_state_vector_bounded_v2
        print("[run_ppo_tsc] Using core scenario encoder only.")
        time.sleep(3)  # allow user to see the print
    expert_stats: dict[RunningFeatureStats] = {}  # tls_id -> RunningFeatureStats
    expert_names = {}  # tls_id -> list[str] (optional)

    # NEW: expert feature report specific trackers and dims
    expert_sem_dim: Dict[str, int] = {}
    expert_adv_corr_trackers: Dict[str, RunningExpertAdvPearson] = {}
    expert_core_xcorr_trackers: Dict[str, RunningExpertCoreCrossCorr] = {}

    run_name = (
        f"sumo_ppo_seed{seed}_{(log_tag + '_' if log_tag else '')}{int(time.time())}"
    )
    writer = SummaryWriter(log_dir=os.path.join(tb_logdir, run_name))

    agents: Dict[str, PPOAgent] = {}
    buffers: Dict[str, RolloutBuffer] = {}
    pending: Dict[str, PendingDecision] = {}
    encoder_cache: Dict[str, dict] = {}

    tb_step_decision = {}  # per TLS decision counter
    kl_early_stop_cum = {}
    expert_heatmap_event_idx: Dict[str, int] = {}
    HEATMAP_EVERY_UPDATES = 15  # cadence in rollout-consume events, not env step
    HEATMAP_LOG_FIRST = True

    total_elapsed = 0.0
    for ep in range(int(episodes)):
        if total_elapsed >= float(max_time):
            break

        # single reset authority (per-episode)
        # Clears: core encoder cache, expert/EMA cache (if combined), throughput tracker state, etc.
        encoder_cache.clear()
        pending.clear()

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
                if tls_id not in kl_early_stop_cum:
                    kl_early_stop_cum[tls_id] = 0
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
                    if use_expert_features:
                        v_sem0 = (
                            encoder_cache[tls_id]
                            .get("_enc_sem", {})
                            .get("_last_v_sem", None)
                        )
                        if v_sem0 is not None and tls_id not in expert_stats:
                            sem_dim = int(np.asarray(v_sem0).shape[0])
                            expert_stats[tls_id] = RunningFeatureStats(
                                sem_dim, eps=1e-3, reservoir_k=2048, bounded_01=True
                            )

                            # NEW: init proposal trackers
                            expert_sem_dim[tls_id] = sem_dim
                            core_dim = int(max(0, state_dim - sem_dim))
                            expert_adv_corr_trackers[tls_id] = RunningExpertAdvPearson(
                                sem_dim=sem_dim
                            )
                            expert_core_xcorr_trackers[tls_id] = (
                                RunningExpertCoreCrossCorr(
                                    sem_dim=sem_dim, core_dim=core_dim
                                )
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
                        # training stabilizers
                        ent_coef_end=ent_coef_end,
                        ent_coef_decay_updates=ent_coef_decay_updates,
                        explore_alpha_start=explore_alpha_start,
                        explore_alpha_end=explore_alpha_end,
                        explore_alpha_decay_updates=explore_alpha_decay_updates,
                        target_kl=target_kl,
                        adv_clip=adv_clip,
                    )
                    buffers[tls_id] = RolloutBuffer()

            # reset pending decisions each episode
            for tls_id in tls_ids:
                pending[tls_id] = PendingDecision()

            ep_reward_sum = {tls_id: 0.0 for tls_id in tls_ids}
            ep_reward_n = {tls_id: 0 for tls_id in tls_ids}
            # --- [NEW] deadlock detector: early-stop episodes that become fully gridlocked ---
            # Hardcoded thresholds; intended to catch "no movement for long time while queues are stopped".
            deadlock_flag = False
            deadlock_t = 0.0
            deadlock_tls: Optional[str] = None
            deadlock_reason: str = ""
            deadlock_last_seen_n = {tls_id: 0 for tls_id in tls_ids}
            # Start the "no-flow clock" from warmup to avoid false triggers before control starts.
            deadlock_last_flow_t = {tls_id: float(warmup_s) for tls_id in tls_ids}
            # Cache controlled lanes once (fallback if encoder lane_ids are missing).
            deadlock_ctrl_lanes = {
                tls_id: sorted(set(traci.trafficlight.getControlledLanes(tls_id)))
                for tls_id in tls_ids
            }
            # ------------------------------------------------------------------------------

            # --- wait barrier logging (per-episode accumulators) ---
            ep_waitbar_sum = {
                tls_id: 0.0 for tls_id in tls_ids
            }  # raw wait reward (<=0)
            ep_waitbar_n = {tls_id: 0 for tls_id in tls_ids}
            ep_waitbar_active = {
                tls_id: 0 for tls_id in tls_ids
            }  # count of steps where penalty fires
            ep_waitbar_min = {
                tls_id: 0.0 for tls_id in tls_ids
            }  # most negative value in episode

            while True:
                sim_t = float(traci.simulation.getTime())
                done_episode = (sim_t >= float(episode_len_s)) or (
                    traci.simulation.getMinExpectedNumber() <= 0
                )
                in_control = sim_t >= float(warmup_s)

                # [NEW] deadlock can force an early episode termination
                if deadlock_flag:
                    done_episode = True
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
                                wait_ref_s=wait_ref_s,
                            ).astype(np.float32)

                            # OLD: only work with single scenario encoder
                            # num_lanes = max(
                            #     1, len(encoder_cache[tls_id].get("lane_ids", []))
                            # )
                            # NEW: combined encoder uses namespaced cache
                            # num_lanes = max(
                            #     1,
                            #     len(
                            #         encoder_cache[tls_id]
                            #         .get("_enc_core", {})
                            #         .get("lane_ids", [])
                            #     ),
                            # )
                            lane_ids = (
                                encoder_cache[tls_id]
                                .get("_enc_core", {})
                                .get("lane_ids", [])
                                if use_expert_features
                                else encoder_cache[tls_id].get("lane_ids", [])
                            )
                            num_lanes = max(1, len(lane_ids))

                            dt_interval = sim_t - float(st.action_start_time)
                            gamma_dt = float(gamma) ** (
                                dt_interval / max(1e-6, float(action_hold_s))
                            )
                            r = reward_throughput_plus_softmax_queue_deltaq_plus_softmax_wait_barrier_v2(
                                tls_id=tls_id,
                                sim_time=sim_t,
                                state_vec=terminal_state,
                                cache=encoder_cache[tls_id],
                                num_lanes=num_lanes,
                                throughput_ref_veh_per_s=throughput_ref_veh_per_s,
                                queue_ref_veh=queue_ref_veh,
                                # [NEW] wait barrier params
                                wait_ref_s=wait_ref_s,
                                wait_barrier_start_s=wait_barrier_start_s,
                                softmax_wait_beta=softmax_wait_beta,
                                # [NEW] weights for new reward terms
                                w_throughput=w_throughput,
                                w_queue=w_queue,
                                w_delta_queue=w_delta_queue,
                                w_wait_barrier=w_wait,
                                w_queue_zone=w_queue_zone,
                                # [NEW] starvation shaping
                                w_starve_potential=w_starve_potential,
                                starve_softmax_beta=starve_softmax_beta,
                                starve_power=starve_power,
                                gamma_dt=gamma_dt,
                                # unchanged queue config
                                queue_power=queue_power,
                                softmax_queue_beta=softmax_queue_beta,
                            )

                            # [IMPORTANT] your v2 reward fn DOES NOT take reward_clip=...
                            # keep clipping outside (same behavior as before)
                            r = float(np.clip(r, reward_clip_lo, reward_clip_hi))
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
                                    writer,
                                    tls_id,
                                    step,
                                    buf,
                                    agent=agents[tls_id],
                                    action_dim=agents[tls_id].action_dim,
                                )
                                if use_expert_features and tls_id in expert_sem_dim:
                                    sem_dim = int(expert_sem_dim[tls_id])

                                    log_hm, hm_event_idx = _next_heatmap_event(
                                        tls_id,
                                        expert_heatmap_event_idx,
                                        every_updates=HEATMAP_EVERY_UPDATES,
                                        log_first=HEATMAP_LOG_FIRST,
                                    )

                                    tb_log_proposal1_expert_adv_corr_rollout(
                                        writer=writer,
                                        tls_id=tls_id,
                                        step=step,  # keep normal scalar x-axis for non-figure metrics
                                        buf=buf,
                                        sem_dim=sem_dim,
                                        tracker=expert_adv_corr_trackers.get(tls_id),
                                        log_heatmap=log_hm,
                                        heatmap_step=hm_event_idx,  # figure x-axis in update-event time
                                    )

                                    tb_log_proposal2_expert_core_xcorr_rollout(
                                        writer=writer,
                                        tls_id=tls_id,
                                        step=step,
                                        buf=buf,
                                        sem_dim=sem_dim,
                                        tracker=expert_core_xcorr_trackers.get(tls_id),
                                        log_heatmap=log_hm,
                                        heatmap_step=hm_event_idx,
                                    )

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
                                kl_stop = int(stats.get("kl_early_stop", 0.0) > 0.5)
                                kl_early_stop_cum[tls_id] += kl_stop
                                writer.add_scalar(
                                    f"{tls_id}/ppo/kl_early_stop", kl_stop, step
                                )
                                writer.add_scalar(
                                    f"{tls_id}/ppo/kl_early_stop_cum",
                                    kl_early_stop_cum[tls_id],
                                    step,
                                )
                                writer.add_scalar(
                                    f"{tls_id}/ppo/updates_done",
                                    stats.get("updates_done", 0.0),
                                    step,
                                )
                                writer.add_scalar(
                                    f"{tls_id}/ppo/ent_coef",
                                    stats.get("ent_coef", 0.0),
                                    step,
                                )
                                writer.add_scalar(
                                    f"{tls_id}/ppo/explore_alpha",
                                    stats.get("explore_alpha", 0.0),
                                    step,
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
                            wait_ref_s=wait_ref_s,
                        ).astype(np.float32)

                        if use_expert_features:
                            v_sem = (
                                encoder_cache[tls_id]
                                .get("_enc_sem", {})
                                .get("_last_v_sem", None)
                            )
                            if v_sem is not None:
                                expert_stats[tls_id].update(v_sem)

                        lane_ids = (
                            encoder_cache[tls_id]
                            .get("_enc_core", {})
                            .get("lane_ids", [])
                            if use_expert_features
                            else encoder_cache[tls_id].get("lane_ids", [])
                        )
                        num_lanes = max(1, len(lane_ids))

                        # [NEW BLOCK] close previous interval and push to PPO rollout buffer
                        if (
                            st.in_control
                            and st.state is not None
                            and st.action is not None
                            and st.logp is not None
                            and st.value is not None
                        ):
                            dt_interval = sim_t - float(st.action_start_time)
                            gamma_dt = float(gamma) ** (
                                dt_interval / max(1e-6, float(action_hold_s))
                            )
                            r = reward_throughput_plus_softmax_queue_deltaq_plus_softmax_wait_barrier_v2(
                                tls_id=tls_id,
                                sim_time=sim_t,
                                state_vec=cur_state,
                                cache=encoder_cache[tls_id],
                                num_lanes=num_lanes,
                                throughput_ref_veh_per_s=throughput_ref_veh_per_s,
                                queue_ref_veh=queue_ref_veh,
                                # [NEW] wait barrier params
                                wait_ref_s=wait_ref_s,
                                wait_barrier_start_s=wait_barrier_start_s,
                                softmax_wait_beta=softmax_wait_beta,
                                # [NEW] weights for new reward terms
                                w_throughput=w_throughput,
                                w_queue=w_queue,
                                w_delta_queue=w_delta_queue,
                                w_wait_barrier=w_wait,
                                w_queue_zone=w_queue_zone,
                                # [NEW] starvation shaping
                                w_starve_potential=w_starve_potential,
                                starve_softmax_beta=starve_softmax_beta,
                                starve_power=starve_power,
                                gamma_dt=gamma_dt,
                                # unchanged queue config
                                queue_power=queue_power,
                                softmax_queue_beta=softmax_queue_beta,  # keep default, or expose as arg if you want
                            )
                            r = float(np.clip(r, reward_clip_lo, reward_clip_hi))
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
                            # --- wait barrier term (raw <= 0) ---
                            waitbar_raw = reward_softmax_wait_barrier_from_encoded_state(
                                cur_state,  # IMPORTANT: use the same state_vec you used for reward
                                num_lanes=num_lanes,
                                wait_ref_s=wait_ref_s,
                                softmax_beta=softmax_wait_beta,
                                barrier_start_s=wait_barrier_start_s,
                                barrier_power=1.0,  # match your reward config unless you exposed it
                                wait_is_encoded=True,  # because encode_tsc_state_vector_bounded_v2 encodes wait
                            )

                            # episode aggregation
                            wb = float(waitbar_raw)
                            ep_waitbar_sum[tls_id] += wb
                            ep_waitbar_n[tls_id] += 1
                            if wb < 0.0:
                                ep_waitbar_active[tls_id] += 1
                                ep_waitbar_min[tls_id] = min(ep_waitbar_min[tls_id], wb)

                            # per-decision scalars (optional but useful)
                            writer.add_scalar(f"{tls_id}/train/waitbar_raw", wb, step)
                        else:
                            _ = reward_throughput_plus_softmax_queue_deltaq_plus_softmax_wait_barrier_v2(
                                tls_id=tls_id,
                                sim_time=sim_t,
                                state_vec=cur_state,
                                cache=encoder_cache[tls_id],
                                num_lanes=num_lanes,
                                throughput_ref_veh_per_s=throughput_ref_veh_per_s,
                                queue_ref_veh=queue_ref_veh,
                                # [NEW] wait barrier params (still update caches)
                                wait_ref_s=wait_ref_s,
                                wait_barrier_start_s=wait_barrier_start_s,
                                softmax_wait_beta=softmax_wait_beta,
                                # [NEW] all weights zero => no-op reward but caches get initialized
                                w_throughput=0.0,
                                w_queue=0.0,
                                w_delta_queue=0.0,
                                w_wait_barrier=0.0,
                                w_queue_zone=0.0,
                                # [NEW] init starvation cache too (still no-op because weights are 0)
                                w_starve_potential=w_starve_potential,
                                starve_softmax_beta=starve_softmax_beta,
                                starve_power=starve_power,
                                gamma_dt=1.0,
                                # unchanged queue config
                                queue_power=queue_power,
                                softmax_queue_beta=softmax_queue_beta,
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
                                writer,
                                tls_id,
                                step,
                                buf,
                                agent=agents[tls_id],
                                action_dim=agents[tls_id].action_dim,
                            )
                            if use_expert_features and tls_id in expert_sem_dim:
                                sem_dim = int(expert_sem_dim[tls_id])

                                log_hm, hm_event_idx = _next_heatmap_event(
                                    tls_id,
                                    expert_heatmap_event_idx,
                                    every_updates=HEATMAP_EVERY_UPDATES,
                                    log_first=HEATMAP_LOG_FIRST,
                                )
                                tb_log_proposal1_expert_adv_corr_rollout(
                                    writer=writer,
                                    tls_id=tls_id,
                                    step=step,  # keep normal scalar x-axis for non-figure metrics
                                    buf=buf,
                                    sem_dim=sem_dim,
                                    tracker=expert_adv_corr_trackers.get(tls_id),
                                    log_heatmap=log_hm,
                                    heatmap_step=hm_event_idx,  # figure x-axis in update-event time
                                )

                                tb_log_proposal2_expert_core_xcorr_rollout(
                                    writer=writer,
                                    tls_id=tls_id,
                                    step=step,
                                    buf=buf,
                                    sem_dim=sem_dim,
                                    tracker=expert_core_xcorr_trackers.get(tls_id),
                                    log_heatmap=log_hm,
                                    heatmap_step=hm_event_idx,
                                )
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

                        if pcnt % 100 == 0:
                            print(
                                f"[ep={ep} t={sim_t:6.1f}] tls={tls_id} "
                                f"a(major_idx)={int(a)} "
                                f"cur_phase={cur_phase_idx} cur_major={cur_major_idx} "
                                f"tgt_major={tgt_major_idx} "
                                f"segments={[(p, round(d,1)) for (p,d) in segments]} "
                                f"hold={action_hold_s}s"
                            )
                        pcnt += 1

                traci.simulationStep()
                sim_t_next = float(traci.simulation.getTime())
                for tls_id in tls_ids:
                    throughput_tracker_step(tls_id, encoder_cache[tls_id])

                # --- [NEW] deadlock detection (checked after stepping & throughput update) ---
                # Detect: no new downstream entries for a long time + almost all vehicles stopped.
                if (not deadlock_flag) and (sim_t_next >= float(warmup_s) + 120.0):
                    if traci.simulation.getMinExpectedNumber() > 0:
                        for tls_id in tls_ids:
                            cache = encoder_cache[tls_id]

                            # Flow proxy: total # unique vehicles that ever entered downstream lanes.
                            seen_total = cache.get("_tp_seen_total", None)
                            seen_n = (
                                len(seen_total) if isinstance(seen_total, set) else 0
                            )
                            if seen_n > int(deadlock_last_seen_n[tls_id]):
                                deadlock_last_seen_n[tls_id] = int(seen_n)
                                deadlock_last_flow_t[tls_id] = float(sim_t_next)

                            noflow_s = float(sim_t_next) - float(
                                deadlock_last_flow_t[tls_id]
                            )
                            if noflow_s < 150.0:
                                continue

                            # Use encoder lane_ids when available; fall back to controlled lanes.
                            lane_ids = (
                                cache.get("_enc_core", {}).get("lane_ids", [])
                                if use_expert_features
                                else cache.get("lane_ids", [])
                            )
                            if not lane_ids:
                                lane_ids = deadlock_ctrl_lanes.get(tls_id, [])
                            # (lane_ids can contain duplicates depending on SUMO, make unique)
                            lane_ids = list(dict.fromkeys(lane_ids))

                            tot_veh = 0
                            tot_halt = 0
                            tot_speed = 0.0
                            for ln in lane_ids:
                                nveh = int(traci.lane.getLastStepVehicleNumber(ln))
                                if nveh <= 0:
                                    continue
                                tot_veh += nveh
                                tot_halt += int(traci.lane.getLastStepHaltingNumber(ln))
                                vln = float(traci.lane.getLastStepMeanSpeed(ln))
                                if vln > 0.0:
                                    tot_speed += vln * float(nveh)

                            if tot_veh < 8:
                                continue

                            mean_speed = tot_speed / max(1.0, float(tot_veh))
                            stop_ratio = float(tot_halt) / max(1.0, float(tot_veh))

                            # "Gridlock-like": almost everyone stopped + no flow for long time.
                            if (stop_ratio >= 0.93) and (mean_speed <= 0.15):
                                deadlock_flag = True
                                deadlock_t = float(sim_t_next)
                                deadlock_tls = str(tls_id)
                                deadlock_reason = (
                                    f"noflow_s={noflow_s:.1f} tot_veh={tot_veh} "
                                    f"stop_ratio={stop_ratio:.2f} mean_speed={mean_speed:.2f}"
                                )
                                print(
                                    f"[ep={ep}] DEADLOCK detected at t={deadlock_t:.1f} tls={deadlock_tls} "
                                    f"({deadlock_reason}) -> early stop episode"
                                )
                                break

                if deadlock_flag:
                    # Jump to the episode-closing branch immediately (no extra simulationStep).
                    continue
                # ------------------------------------------------------------------------------

            for tls_id in tls_ids:
                mean_r = ep_reward_sum[tls_id] / max(1, ep_reward_n[tls_id])
                writer.add_scalar(f"{tls_id}/episode/reward_mean", float(mean_r), ep)
                writer.add_scalar(
                    f"{tls_id}/episode/reward_sum", float(ep_reward_sum[tls_id]), ep
                )
                # --- episode summary for wait barrier ---
                wb_n = max(1, ep_waitbar_n[tls_id])
                wb_mean_raw = ep_waitbar_sum[tls_id] / wb_n  # <= 0
                wb_active_frac = ep_waitbar_active[tls_id] / wb_n

                writer.add_scalar(
                    f"{tls_id}/episode/waitbar_mean_raw", float(wb_mean_raw), ep
                )
                writer.add_scalar(
                    f"{tls_id}/episode/waitbar_sum_raw",
                    float(ep_waitbar_sum[tls_id]),
                    ep,
                )
                writer.add_scalar(
                    f"{tls_id}/episode/waitbar_active_frac", float(wb_active_frac), ep
                )
                writer.add_scalar(
                    f"{tls_id}/episode/waitbar_min_raw",
                    float(ep_waitbar_min[tls_id]),
                    ep,
                )
                # deadlock episode flag
                writer.add_scalar(
                    f"{tls_id}/episode/deadlock",
                    float(1.0 if deadlock_flag else 0.0),
                    ep,
                )
                if deadlock_flag and (deadlock_tls is not None):
                    writer.add_scalar(
                        f"{tls_id}/episode/deadlock_is_owner",
                        float(1.0 if str(deadlock_tls) == str(tls_id) else 0.0),
                        ep,
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
        # --- [NEW] episode-level deadlock logging ---
        writer.add_scalar("global/deadlock", float(1.0 if deadlock_flag else 0.0), ep)
        if deadlock_flag:
            writer.add_scalar("global/deadlock_time_s", float(deadlock_t), ep)
        # -----------------------------------------------

    for tls_id, st in expert_stats.items():
        rep = st.finalize()
        step = int(tb_step_decision.get(tls_id, 0))

        # small set of scalars (avoid TB clutter)
        writer.add_scalar(
            f"{tls_id}/expert_features/mean_std", float(np.mean(rep["std"])), step
        )
        writer.add_scalar(
            f"{tls_id}/expert_features/max_std", float(np.max(rep["std"])), step
        )
        writer.add_scalar(
            f"{tls_id}/expert_features/mean_dead_frac",
            float(np.mean(rep["frac_abs_lt_eps"])),
            step,
        )
        writer.add_scalar(
            f"{tls_id}/expert_features/max_dead_frac",
            float(np.max(rep["frac_abs_lt_eps"])),
            step,
        )
        writer.add_scalar(f"{tls_id}/expert_features/n_samples", float(rep["n"]), step)

        # detailed per-dim report as text (markdown table)
        lines = [
            "|idx|mean|std|min|max|p5|p50|p95|dead_frac|nan|inf|",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for i in range(st.dim):
            lines.append(
                f"|{i}|{rep['mean'][i]:.4g}|{rep['std'][i]:.4g}|{rep['min'][i]:.4g}|{rep['max'][i]:.4g}"
                f"|{rep.get('p5',[0]*st.dim)[i]:.4g}|{rep.get('p50',[0]*st.dim)[i]:.4g}|{rep.get('p95',[0]*st.dim)[i]:.4g}"
                f"|{rep['frac_abs_lt_eps'][i]:.3f}|{int(rep['nan'][i])}|{int(rep['inf'][i])}|"
            )
        writer.add_text(f"{tls_id}/expert_features/report", "\n".join(lines), step)

    # Final Proposal 1 report
    tb_log_proposal1_expert_adv_corr_final(
        writer=writer,
        tls_id=tls_id,
        step=step,
        tracker=expert_adv_corr_trackers.get(tls_id),
    )

    # Final Proposal 2 report
    tb_log_proposal2_expert_core_xcorr_final(
        writer=writer,
        tls_id=tls_id,
        step=step,
        tracker=expert_core_xcorr_trackers.get(tls_id),
    )

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
            "ent_coef_end": float(ent_coef_end),
            "explore_alpha_start": float(explore_alpha_start),
            "explore_alpha_end": float(explore_alpha_end),
            "explore_alpha_decay_updates": int(explore_alpha_decay_updates),
            "target_kl": float(target_kl),
            "adv_clip": float(adv_clip) if adv_clip is not None else None,
            "gae_lambda": float(gae_lambda),
            "ppo_epochs": int(ppo_epochs),
            "minibatch_size": int(minibatch_size),
            "rollout_steps": int(rollout_steps),
            "encoder": getattr(encoder_fn, "__name__", "<unknown>"),
            "traffic_scale_mean": float(traffic_scale_mean),
            "traffic_scale_std": float(traffic_scale_std),
            "saved_unix_time": float(time.time()),
            "use_expert_features": bool(use_expert_features),
            "weighted_reward_config": {
                "w_throughput": float(w_throughput),
                "w_queue": float(w_queue),
                "w_delta_queue": float(w_delta_queue),
                "w_wait": float(w_wait),
            },
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
    # -------------------------------
    # [NEW] Training-scheme stabilizers
    # -------------------------------
    ap.add_argument("--ent-coef-end", type=float, default=None)
    ap.add_argument("--ent-coef-decay-updates", type=int, default=0)

    # Uniform ridge exploration: μ = (1-α)π + αU
    ap.add_argument("--explore-alpha-start", type=float, default=0.0)
    ap.add_argument("--explore-alpha-end", type=float, default=0.0)
    ap.add_argument("--explore-alpha-decay-updates", type=int, default=0)

    # PPO update safety knobs
    ap.add_argument("--target-kl", type=float, default=None)
    ap.add_argument("--adv-clip", type=float, default=None)
    # -------------------------------
    ap.add_argument("--vf-coef", type=float, default=0.5)

    # Reward (defaults)
    ap.add_argument("--thr-ref", type=float, default=0.30)
    ap.add_argument("--queue-ref", type=float, default=15.0)
    ap.add_argument("--w-thr", type=float, default=1.0)
    ap.add_argument("--w-queue", type=float, default=1.0)
    ap.add_argument("--w-delta-queue", type=float, default=0.5)
    ap.add_argument("--w-wait", type=float, default=0.5)
    ap.add_argument("--wait-ref", type=float, default=60.0)
    ap.add_argument("--w-queue-zone", type=float, default=0.3)
    ap.add_argument("--wait-barrier-start", type=float, default=30.0)
    ap.add_argument("--softmax-wait-beta", type=float, default=10.0)
    ap.add_argument("--softmax-queue-beta", type=float, default=4.0)
    # [NEW] potential-based shaping against major-phase starvation
    ap.add_argument("--w-starve-potential", type=float, default=0.0)
    ap.add_argument("--starve-softmax-beta", type=float, default=10.0)
    ap.add_argument("--starve-power", type=float, default=1.0)
    ap.add_argument("--queue-power", type=float, default=1.0)
    ap.add_argument("--top2-w1", type=float, default=0.7)
    ap.add_argument("--top2-w2", type=float, default=0.3)
    ap.add_argument("--reward-clip-lo", type=float, default=-5.0)
    ap.add_argument("--reward-clip-hi", type=float, default=2.0)

    ap.add_argument("--tb-logdir", type=str, default="tensorboard_logs")
    ap.add_argument("--save-dir", type=str, default="saved_models_ppo")
    ap.add_argument("--max-time", type=float, default=1e18)

    # control flags
    ap.add_argument(
        "--use-expert-features",
        action="store_true",
        help="Use combined encoder (core scene encoder + expert_feature_extractor). Default: off.",
    )
    ap.add_argument("--log-tag", type=str, default="")

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
        w_delta_queue=float(args.w_delta_queue),
        w_wait=float(args.w_wait),
        w_queue_zone=float(args.w_queue_zone),
        wait_ref_s=float(args.wait_ref),
        wait_barrier_start_s=float(args.wait_barrier_start),
        softmax_wait_beta=float(args.softmax_wait_beta),
        softmax_queue_beta=float(args.softmax_queue_beta),
        # [NEW] starvation shaping args
        w_starve_potential=float(args.w_starve_potential),
        starve_softmax_beta=float(args.starve_softmax_beta),
        starve_power=float(args.starve_power),
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
        # [NEW] Training-scheme stabilizers
        ent_coef_end=args.ent_coef_end,
        ent_coef_decay_updates=int(args.ent_coef_decay_updates),
        explore_alpha_start=float(args.explore_alpha_start),
        explore_alpha_end=float(args.explore_alpha_end),
        explore_alpha_decay_updates=int(args.explore_alpha_decay_updates),
        target_kl=args.target_kl,
        adv_clip=args.adv_clip,
        # control flags
        use_expert_features=bool(args.use_expert_features),
        log_tag=str(args.log_tag),
    )


if __name__ == "__main__":
    main()
