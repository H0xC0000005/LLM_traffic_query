# utility.py

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple
from dataclasses import dataclass
from sumolib import checkBinary
import libsumo as traci

"""
universal helpers
"""


def _soft_sat(x: float, *, sat: float = 1.0) -> float:
    """Soft saturating map ~identity near 0, asymptote to `sat` for large x."""
    x = float(x)
    if x <= 0.0:
        return 0.0
    sat = max(1e-6, float(sat))
    y = sat * (1.0 - float(np.exp(-x / sat)))
    if y < 0.0:
        y = 0.0
    if y > sat:
        y = sat
    return float(y)


"""
essential data structures 
"""


class RunningFeatureStats:
    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1e-3,
        reservoir_k: int = 2048,
        bounded_01: bool = True,
    ):
        self.dim = int(dim)
        self.eps = float(eps)
        self.bounded_01 = bool(bounded_01)

        self.n = 0
        self.nan = np.zeros(self.dim, dtype=np.int64)
        self.inf = np.zeros(self.dim, dtype=np.int64)

        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.M2 = np.zeros(self.dim, dtype=np.float64)
        self.minv = np.full(self.dim, np.inf, dtype=np.float64)
        self.maxv = np.full(self.dim, -np.inf, dtype=np.float64)

        self.frac_abs_lt_eps_cnt = np.zeros(self.dim, dtype=np.int64)
        self.frac_lt_eps_cnt = np.zeros(self.dim, dtype=np.int64)
        self.frac_gt_1m_eps_cnt = np.zeros(self.dim, dtype=np.int64)

        # reservoir sampling (approx quantiles)
        self.k = int(reservoir_k)
        self.res = np.empty((self.k, self.dim), dtype=np.float32)
        self.res_n = 0

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.dim:
            return  # or raise

        self.n += 1

        is_nan = np.isnan(x)
        is_inf = np.isinf(x)
        self.nan += is_nan.astype(np.int64)
        self.inf += is_inf.astype(np.int64)

        # ignore non-finite for numeric moments/min/max
        xf = x.copy()
        xf[~np.isfinite(xf)] = 0.0

        # Welford
        delta = xf - self.mean
        self.mean += delta / self.n
        delta2 = xf - self.mean
        self.M2 += delta * delta2

        self.minv = np.minimum(self.minv, xf)
        self.maxv = np.maximum(self.maxv, xf)

        self.frac_abs_lt_eps_cnt += (np.abs(xf) < self.eps).astype(np.int64)
        if self.bounded_01:
            self.frac_lt_eps_cnt += (xf < self.eps).astype(np.int64)
            self.frac_gt_1m_eps_cnt += (xf > (1.0 - self.eps)).astype(np.int64)

        # reservoir: keep first k, then replace uniformly
        if self.res_n < self.k:
            self.res[self.res_n, :] = xf
            self.res_n += 1
        else:
            j = np.random.randint(0, self.n)
            if j < self.k:
                self.res[j, :] = xf

    def finalize(self) -> dict:
        n = max(1, int(self.n))
        var = self.M2 / max(1, n - 1)
        std = np.sqrt(np.maximum(var, 0.0))

        out = {
            "n": n,
            "mean": self.mean.astype(np.float32),
            "std": std.astype(np.float32),
            "min": self.minv.astype(np.float32),
            "max": self.maxv.astype(np.float32),
            "nan": self.nan.astype(np.int64),
            "inf": self.inf.astype(np.int64),
            "frac_abs_lt_eps": (self.frac_abs_lt_eps_cnt / n).astype(np.float32),
        }
        if self.bounded_01:
            out["frac_lt_eps"] = (self.frac_lt_eps_cnt / n).astype(np.float32)
            out["frac_gt_1m_eps"] = (self.frac_gt_1m_eps_cnt / n).astype(np.float32)

        if self.res_n > 0:
            samp = self.res[: self.res_n]
            out["p1"] = np.percentile(samp, 1, axis=0).astype(np.float32)
            out["p5"] = np.percentile(samp, 5, axis=0).astype(np.float32)
            out["p50"] = np.percentile(samp, 50, axis=0).astype(np.float32)
            out["p95"] = np.percentile(samp, 95, axis=0).astype(np.float32)
            out["p99"] = np.percentile(samp, 99, axis=0).astype(np.float32)
        return out


@dataclass
class TLSControllerState:
    next_decision_time: float = 0.0
    pending_state: Optional[np.ndarray] = None
    pending_action: Optional[int] = None
    pending_epsilon: float = 0.0
    in_control_when_pending: bool = False
    next_target_update_time: float = 0.0
    pending_segments: Deque[Tuple[int, float]] = field(
        default_factory=deque
    )  # (phase_idx, duration_s)
    segment_end_time: float = 0.0  # when current segment ends
    # log_step: int = 0


"""
statistical structures and algorithms
"""

# ===========================
# Proposal 1 / Proposal 2 wheels (algorithm only, no TensorBoard I/O)
# ===========================


def _np_or_none(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x)
    except Exception:
        return None


def get_attr_np(obj: Any, names: Sequence[str]) -> Optional[np.ndarray]:
    for n in names:
        if hasattr(obj, n):
            arr = _np_or_none(getattr(obj, n))
            if arr is not None:
                return arr
    return None


def split_core_sem_from_states(
    states: np.ndarray, sem_dim: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split combined state [core, expert] -> (core, expert).
    Returns empty arrays if shape is invalid.
    """
    x = np.asarray(states, dtype=np.float64)
    if x.ndim != 2:
        return np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=np.float64)

    d = int(x.shape[1])
    sem_dim = int(sem_dim)
    if sem_dim <= 0 or sem_dim >= d:
        return np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=np.float64)

    core = x[:, : d - sem_dim]
    sem = x[:, d - sem_dim :]
    return core, sem


def _safe_pearson_from_sums(
    n: np.ndarray,
    sum_x: np.ndarray,
    sum_x2: np.ndarray,
    sum_y: np.ndarray,
    sum_y2: np.ndarray,
    sum_xy: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64)
    sum_x = np.asarray(sum_x, dtype=np.float64)
    sum_x2 = np.asarray(sum_x2, dtype=np.float64)
    sum_y = np.asarray(sum_y, dtype=np.float64)
    sum_y2 = np.asarray(sum_y2, dtype=np.float64)
    sum_xy = np.asarray(sum_xy, dtype=np.float64)

    n_safe = np.maximum(n, 1.0)
    num = sum_xy - (sum_x * sum_y) / n_safe
    den_x = sum_x2 - (sum_x * sum_x) / n_safe
    den_y = sum_y2 - (sum_y * sum_y) / n_safe
    den = np.sqrt(np.maximum(den_x, 0.0) * np.maximum(den_y, 0.0))

    r = np.full_like(num, np.nan, dtype=np.float64)
    valid = (n >= 2.0) & (den > float(eps))
    r[valid] = num[valid] / den[valid]
    return np.clip(r, -1.0, 1.0)


def _pearson_per_dim(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pearson corr for each column of x against scalar target y.
    Returns (corr[D], n_valid[D]).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if x.ndim != 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)

    n_rows = min(x.shape[0], y.shape[0])
    if n_rows <= 0:
        d = x.shape[1]
        return np.full((d,), np.nan, dtype=np.float64), np.zeros((d,), dtype=np.int64)

    x = x[:n_rows, :]
    y = y[:n_rows]
    y_fin = np.isfinite(y)

    d = x.shape[1]
    n = np.zeros(d, dtype=np.int64)
    sum_x = np.zeros(d, dtype=np.float64)
    sum_x2 = np.zeros(d, dtype=np.float64)
    sum_y = np.zeros(d, dtype=np.float64)
    sum_y2 = np.zeros(d, dtype=np.float64)
    sum_xy = np.zeros(d, dtype=np.float64)

    for j in range(d):
        xj = x[:, j]
        m = y_fin & np.isfinite(xj)
        if not np.any(m):
            continue
        xv = xj[m]
        yv = y[m]
        n[j] = int(m.sum())
        sum_x[j] = float(np.sum(xv))
        sum_x2[j] = float(np.sum(xv * xv))
        sum_y[j] = float(np.sum(yv))
        sum_y2[j] = float(np.sum(yv * yv))
        sum_xy[j] = float(np.sum(xv * yv))

    r = _safe_pearson_from_sums(n, sum_x, sum_x2, sum_y, sum_y2, sum_xy)
    return r, n


def _corr_matrix(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Cross-correlation matrix corr(x[:,i], y[:,j]) with row-wise finite mask.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim != 2 or y.ndim != 2:
        return np.empty((0, 0), dtype=np.float64)

    n_rows = min(x.shape[0], y.shape[0])
    x = x[:n_rows, :]
    y = y[:n_rows, :]

    if n_rows <= 1 or x.shape[1] == 0 or y.shape[1] == 0:
        return np.full((x.shape[1], y.shape[1]), np.nan, dtype=np.float64)

    m = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    if int(m.sum()) <= 1:
        return np.full((x.shape[1], y.shape[1]), np.nan, dtype=np.float64)

    xf = x[m, :]
    yf = y[m, :]
    n = float(xf.shape[0])

    xc = xf - np.mean(xf, axis=0, keepdims=True)
    yc = yf - np.mean(yf, axis=0, keepdims=True)

    cov = (xc.T @ yc) / max(1.0, n - 1.0)
    sx = np.sqrt(np.maximum(np.var(xf, axis=0, ddof=1), 0.0))
    sy = np.sqrt(np.maximum(np.var(yf, axis=0, ddof=1), 0.0))
    den = sx[:, None] * sy[None, :]

    out = np.full_like(cov, np.nan, dtype=np.float64)
    valid = den > float(eps)
    out[valid] = cov[valid] / den[valid]
    return np.clip(out, -1.0, 1.0)


class RunningExpertAdvPearson:
    """
    Proposal 1 tracker:
      Pearson corr of each expert feature dim vs PPO advantage.
    """

    def __init__(self, sem_dim: int) -> None:
        self.sem_dim = int(sem_dim)
        self.n = np.zeros(self.sem_dim, dtype=np.int64)
        self.sum_x = np.zeros(self.sem_dim, dtype=np.float64)
        self.sum_x2 = np.zeros(self.sem_dim, dtype=np.float64)
        self.sum_y = np.zeros(self.sem_dim, dtype=np.float64)
        self.sum_y2 = np.zeros(self.sem_dim, dtype=np.float64)
        self.sum_xy = np.zeros(self.sem_dim, dtype=np.float64)

    def update(self, sem: np.ndarray, adv: np.ndarray) -> None:
        sem = np.asarray(sem, dtype=np.float64)
        adv = np.asarray(adv, dtype=np.float64).reshape(-1)

        if sem.ndim != 2 or sem.shape[1] != self.sem_dim:
            return

        n_rows = min(sem.shape[0], adv.shape[0])
        if n_rows <= 0:
            return

        sem = sem[:n_rows, :]
        adv = adv[:n_rows]
        adv_fin = np.isfinite(adv)

        for j in range(self.sem_dim):
            xj = sem[:, j]
            m = adv_fin & np.isfinite(xj)
            if not np.any(m):
                continue
            xv = xj[m]
            yv = adv[m]
            self.n[j] += int(m.sum())
            self.sum_x[j] += float(np.sum(xv))
            self.sum_x2[j] += float(np.sum(xv * xv))
            self.sum_y[j] += float(np.sum(yv))
            self.sum_y2[j] += float(np.sum(yv * yv))
            self.sum_xy[j] += float(np.sum(xv * yv))

    def corr(self) -> np.ndarray:
        return _safe_pearson_from_sums(
            self.n, self.sum_x, self.sum_x2, self.sum_y, self.sum_y2, self.sum_xy
        )

    def finalize(self) -> Dict[str, Any]:
        r = self.corr()
        finite = np.isfinite(r)
        abs_r = np.abs(r[finite]) if np.any(finite) else np.array([], dtype=np.float64)
        return {
            "corr": r.astype(np.float32),
            "n": self.n.astype(np.int64),
            "mean_abs": float(np.mean(abs_r)) if abs_r.size else float("nan"),
            "max_abs": float(np.max(abs_r)) if abs_r.size else float("nan"),
            "frac_abs_gt_0p10": (
                float(np.mean(abs_r > 0.10)) if abs_r.size else float("nan")
            ),
        }


class RunningExpertCoreCrossCorr:
    """
    Proposal 2 tracker:
      Cross-correlation matrix between expert dims and core encoder dims.
    """

    def __init__(self, sem_dim: int, core_dim: int) -> None:
        self.sem_dim = int(sem_dim)
        self.core_dim = int(core_dim)

        self.n = 0
        self.sum_sem = np.zeros(self.sem_dim, dtype=np.float64)
        self.sum_sem2 = np.zeros(self.sem_dim, dtype=np.float64)
        self.sum_core = np.zeros(self.core_dim, dtype=np.float64)
        self.sum_core2 = np.zeros(self.core_dim, dtype=np.float64)
        self.sum_outer = np.zeros((self.sem_dim, self.core_dim), dtype=np.float64)

    def update(self, sem: np.ndarray, core: np.ndarray) -> None:
        sem = np.asarray(sem, dtype=np.float64)
        core = np.asarray(core, dtype=np.float64)

        if sem.ndim != 2 or core.ndim != 2:
            return
        if sem.shape[1] != self.sem_dim or core.shape[1] != self.core_dim:
            return

        n_rows = min(sem.shape[0], core.shape[0])
        if n_rows <= 0:
            return

        sem = sem[:n_rows, :]
        core = core[:n_rows, :]

        m = np.isfinite(sem).all(axis=1) & np.isfinite(core).all(axis=1)
        if not np.any(m):
            return

        s = sem[m, :]
        c = core[m, :]
        self.n += int(s.shape[0])

        self.sum_sem += np.sum(s, axis=0)
        self.sum_sem2 += np.sum(s * s, axis=0)
        self.sum_core += np.sum(c, axis=0)
        self.sum_core2 += np.sum(c * c, axis=0)
        self.sum_outer += s.T @ c

    def corr(self) -> np.ndarray:
        if self.n <= 1 or self.sem_dim <= 0 or self.core_dim <= 0:
            return np.full((self.sem_dim, self.core_dim), np.nan, dtype=np.float64)

        n = float(self.n)
        mu_s = self.sum_sem / n
        mu_c = self.sum_core / n

        cov = (self.sum_outer / n) - (mu_s[:, None] * mu_c[None, :])
        var_s = np.maximum((self.sum_sem2 / n) - (mu_s * mu_s), 0.0)
        var_c = np.maximum((self.sum_core2 / n) - (mu_c * mu_c), 0.0)
        den = np.sqrt(var_s)[:, None] * np.sqrt(var_c)[None, :]

        out = np.full_like(cov, np.nan, dtype=np.float64)
        valid = den > 1e-12
        out[valid] = cov[valid] / den[valid]
        return np.clip(out, -1.0, 1.0)

    def finalize(self) -> Dict[str, Any]:
        c = self.corr()
        finite = np.isfinite(c)
        abs_c = np.abs(c[finite]) if np.any(finite) else np.array([], dtype=np.float64)
        sem_max = (
            np.nanmax(np.abs(c), axis=1)
            if (c.ndim == 2 and c.shape[1] > 0)
            else np.array([], dtype=np.float64)
        )
        return {
            "corr": c.astype(np.float32),
            "n": int(self.n),
            "mean_abs": float(np.mean(abs_c)) if abs_c.size else float("nan"),
            "p95_abs": (
                float(np.percentile(abs_c, 95.0)) if abs_c.size else float("nan")
            ),
            "frac_abs_gt_0p30": (
                float(np.mean(abs_c > 0.30)) if abs_c.size else float("nan")
            ),
            "sem_max_abs_mean": (
                float(np.nanmean(sem_max)) if sem_max.size else float("nan")
            ),
        }


def proposal1_expert_adv_corr_from_rollout(
    buf: Any,
    sem_dim: int,
    tracker: Optional[RunningExpertAdvPearson] = None,
) -> Dict[str, Any]:
    """
    Proposal 1 one-rollout algorithm.
    Uses state suffix as expert slice and correlates with advantages.
    """
    states = get_attr_np(buf, ["states", "obs", "observations"])
    advs = get_attr_np(buf, ["advs", "advantages", "adv"])
    if states is None or advs is None:
        return {"ok": False, "reason": "missing_states_or_advs"}

    core, sem = split_core_sem_from_states(states, sem_dim)
    if sem.size == 0:
        return {"ok": False, "reason": "invalid_sem_dim_or_state_shape"}

    y = np.asarray(advs, dtype=np.float64).reshape(-1)
    n_rows = min(sem.shape[0], y.shape[0])
    if n_rows <= 1:
        return {"ok": False, "reason": "too_few_samples"}

    sem = sem[:n_rows, :]
    y = y[:n_rows]

    r, n = _pearson_per_dim(sem, y)
    if tracker is not None:
        tracker.update(sem, y)

    finite = np.isfinite(r)
    abs_r = np.abs(r[finite]) if np.any(finite) else np.array([], dtype=np.float64)
    return {
        "ok": True,
        "corr": r.astype(np.float32),
        "n": n.astype(np.int64),
        "mean_abs": float(np.mean(abs_r)) if abs_r.size else float("nan"),
        "max_abs": float(np.max(abs_r)) if abs_r.size else float("nan"),
        "frac_abs_gt_0p10": (
            float(np.mean(abs_r > 0.10)) if abs_r.size else float("nan")
        ),
    }


def proposal2_expert_core_xcorr_from_rollout(
    buf: Any,
    sem_dim: int,
    tracker: Optional[RunningExpertCoreCrossCorr] = None,
) -> Dict[str, Any]:
    """
    Proposal 2 one-rollout algorithm.
    Computes cross-correlation matrix between expert slice and core slice.
    """
    states = get_attr_np(buf, ["states", "obs", "observations"])
    if states is None:
        return {"ok": False, "reason": "missing_states"}

    core, sem = split_core_sem_from_states(states, sem_dim)
    if sem.size == 0:
        return {"ok": False, "reason": "invalid_sem_dim_or_state_shape"}
    if core.shape[1] == 0:
        return {"ok": False, "reason": "no_core_dims"}

    c = _corr_matrix(sem, core)
    if tracker is not None:
        tracker.update(sem, core)

    finite = np.isfinite(c)
    abs_c = np.abs(c[finite]) if np.any(finite) else np.array([], dtype=np.float64)
    sem_max = (
        np.nanmax(np.abs(c), axis=1)
        if c.shape[1] > 0
        else np.array([], dtype=np.float64)
    )

    return {
        "ok": True,
        "corr": c.astype(np.float32),
        "mean_abs": float(np.mean(abs_c)) if abs_c.size else float("nan"),
        "p95_abs": float(np.percentile(abs_c, 95.0)) if abs_c.size else float("nan"),
        "frac_abs_gt_0p30": (
            float(np.mean(abs_c > 0.30)) if abs_c.size else float("nan")
        ),
        "sem_max_abs_mean": (
            float(np.nanmean(sem_max)) if sem_max.size else float("nan")
        ),
    }


def proposal2_topk_abs_pairs(
    corr: np.ndarray, k: int = 20
) -> List[Tuple[int, int, float]]:
    c = np.asarray(corr, dtype=np.float64)
    if c.ndim != 2 or c.size == 0:
        return []

    abs_c = np.abs(c)
    abs_c[~np.isfinite(abs_c)] = -np.inf
    flat = abs_c.reshape(-1)

    k = max(0, min(int(k), flat.size))
    if k == 0:
        return []

    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]

    out: List[Tuple[int, int, float]] = []
    ncol = c.shape[1]
    for f in idx:
        i = int(f // ncol)
        j = int(f % ncol)
        out.append((i, j, float(c[i, j])))
    return out


"""
phase management (finding out major green phases and register auxilliary phases)
"""

# [NEW BLOCK] TLS phase-plan helpers (major greens + auxiliary phases)


@dataclass(frozen=True)
class TLSPhasePlan:
    program_id: str
    phases: List[Tuple[int, float, str]]  # (idx, duration_s, state_str)
    major_greens: List[
        int
    ]  # indices of "major green" phases (agent actions map to these)
    owner_major: List[
        int
    ]  # owner_major[phase_idx] -> major green phase idx that owns it
    aux_after_major: Dict[
        int, List[int]
    ]  # major green idx -> aux phase indices after it until next major
    phase_duration: Dict[int, float]  # phase_idx -> configured duration (seconds)


def _get_active_program_logic(tls_id: str):
    """Return the active ProgramLogic object for tls_id."""
    program_id = traci.trafficlight.getProgram(tls_id)
    logics = traci.trafficlight.getAllProgramLogics(tls_id)

    logic = None
    for lg in logics:
        try:
            if lg.getSubID() == program_id:
                logic = lg
                break
        except Exception:
            if getattr(lg, "programID", None) == program_id:
                logic = lg
                break
    if logic is None:
        logic = logics[0]
    return program_id, logic


def _default_is_major_green(
    state: str, duration_s: float, *, min_major_green_s: float
) -> bool:
    """
    Heuristic "major green":
      - contains any green signal (G/g)
      - contains NO yellow (y/Y)
      - duration >= min_major_green_s  (filters out short clearance phases)
    """
    has_green = ("G" in state) or ("g" in state)
    has_yellow = ("y" in state) or ("Y" in state)
    return (
        has_green
        and (not has_yellow)
        and (float(duration_s) >= float(min_major_green_s))
    )


def get_tls_phase_plan(
    tls_id: str,
    cache: Dict[str, Any],
    *,
    min_major_green_s: float = 5.0,
    is_major_green: Optional[Callable[[str, float], bool]] = None,
) -> TLSPhasePlan:
    """
    Build (and cache) a relaxed phase plan:
      - agent chooses only major green phases
      - any phases between two major greens are treated as auxiliary phases owned by the earlier major green
    """
    program_id = traci.trafficlight.getProgram(tls_id)
    key = "_tls_phase_plan"

    # reuse cached plan if program unchanged
    if key in cache:
        plan: TLSPhasePlan = cache[key]
        if plan.program_id == program_id:
            return plan

    program_id, logic = _get_active_program_logic(tls_id)
    try:
        phases_obj = logic.getPhases()
    except Exception:
        phases_obj = getattr(logic, "phases")

    phases: List[Tuple[int, float, str]] = []
    phase_duration: Dict[int, float] = {}
    for i, ph in enumerate(phases_obj):
        dur = float(getattr(ph, "duration"))
        st = str(getattr(ph, "state"))
        phases.append((int(i), dur, st))
        phase_duration[int(i)] = dur

    if is_major_green is None:

        def is_major_green_local(s: str, d: float) -> bool:
            return _default_is_major_green(
                s, d, min_major_green_s=float(min_major_green_s)
            )

        is_major_green = is_major_green_local

    major_greens = [idx for (idx, dur, st) in phases if is_major_green(st, dur)]
    if not major_greens:
        raise RuntimeError(
            f"[{tls_id}] No major green phases found. "
            f"Adjust min_major_green_s or provide is_major_green()."
        )

    n = len(phases)
    major_set = set(major_greens)

    # owner_major: each phase belongs to the most recent major green in cyclic order
    owner_major: List[int] = [-1] * n
    last_major = major_greens[-1]  # for phases before first major in list
    for i in range(n):
        if i in major_set:
            last_major = i
        owner_major[i] = int(last_major)

    # aux_after_major: phases strictly between major k and next major (cyclic)
    aux_after_major: Dict[int, List[int]] = {mg: [] for mg in major_greens}
    for j, mg in enumerate(major_greens):
        nxt = major_greens[(j + 1) % len(major_greens)]
        aux: List[int] = []
        k = (mg + 1) % n
        while k != nxt:
            aux.append(int(k))
            k = (k + 1) % n
        aux_after_major[int(mg)] = aux

    plan = TLSPhasePlan(
        program_id=str(program_id),
        phases=phases,
        major_greens=[int(x) for x in major_greens],
        owner_major=owner_major,
        aux_after_major=aux_after_major,
        phase_duration=phase_duration,
    )
    cache[key] = plan
    return plan


def tls_major_action_dim(
    tls_id: str, cache: Dict[str, Any], *, min_major_green_s: float = 5.0
) -> int:
    plan = get_tls_phase_plan(tls_id, cache, min_major_green_s=min_major_green_s)
    return int(len(plan.major_greens))


def tls_action_to_major_phase(
    tls_id: str, cache: Dict[str, Any], action: int, *, min_major_green_s: float = 5.0
) -> int:
    plan = get_tls_phase_plan(tls_id, cache, min_major_green_s=min_major_green_s)
    a = int(action)
    if a < 0 or a >= len(plan.major_greens):
        raise ValueError(
            f"action out of range: {a} (num_major={len(plan.major_greens)})"
        )
    return int(plan.major_greens[a])


def tls_current_major_phase(
    tls_id: str,
    cache: Dict[str, Any],
    current_phase: Optional[int] = None,
    *,
    min_major_green_s: float = 5.0,
) -> int:
    plan = get_tls_phase_plan(tls_id, cache, min_major_green_s=min_major_green_s)
    cur = int(
        traci.trafficlight.getPhase(tls_id) if current_phase is None else current_phase
    )
    if cur < 0 or cur >= len(plan.owner_major):
        # fall back to the first major green
        return int(plan.major_greens[0])
    return int(plan.owner_major[cur])


def tls_build_switch_segments(
    tls_id: str,
    cache: Dict[str, Any],
    *,
    target_major_phase: int,
    hold_s: float,
    current_phase: Optional[int] = None,
    min_major_green_s: float = 5.0,
    min_aux_dur_s: float = 0.1,
) -> List[Tuple[int, float]]:
    """
    Build segments for one macro-action:
      - if switching away from current major: play all auxiliary phases owned by current major (configured durations)
      - then play target major green for hold_s (hold_s excludes aux time)
    Returns list[(phase_idx, duration_s)].
    """
    plan = get_tls_phase_plan(tls_id, cache, min_major_green_s=min_major_green_s)
    cur_major = tls_current_major_phase(
        tls_id, cache, current_phase=current_phase, min_major_green_s=min_major_green_s
    )
    tgt = int(target_major_phase)

    segs: List[Tuple[int, float]] = []
    if tgt != int(cur_major):
        for aux_idx in plan.aux_after_major.get(int(cur_major), []):
            dur = float(plan.phase_duration.get(int(aux_idx), 0.0))
            if dur < float(min_aux_dur_s):
                dur = float(min_aux_dur_s)
            segs.append((int(aux_idx), dur))

    segs.append((tgt, float(hold_s)))
    return segs


def tls_set_phase_frozen(tls_id: str, phase_idx: int) -> None:
    """Set a TLS phase and prevent SUMO from auto-advancing."""
    traci.trafficlight.setPhase(tls_id, int(phase_idx))
    traci.trafficlight.setPhaseDuration(tls_id, 1e6)


def tls_advance_pending_segments(
    *,
    tls_id: str,
    pending_segments: Deque[Tuple[int, float]],
    segment_end_time: float,
    sim_t: float,
) -> float:
    """
    If the currently-playing segment has ended (sim_t >= segment_end_time),
    pop the next (phase, dur) from pending_segments, set it frozen, and return
    the new segment_end_time. If nothing remains, return 0.0.
    """
    if segment_end_time <= 0.0:
        return 0.0
    if sim_t < float(segment_end_time):
        return float(segment_end_time)

    if not pending_segments:
        return 0.0

    next_phase, next_dur = pending_segments.popleft()
    tls_set_phase_frozen(tls_id, int(next_phase))
    return float(sim_t) + float(next_dur)


"""
helper functions
"""


# =======================
# [NEW] shared queue extractor
# =======================
def _extract_queues_from_encoded_state(
    state_vec: Sequence[float],
    *,
    num_lanes: int,
    lane_block_size: int,
    queue_offset_in_block: int,
    scale: float = 1.0,
    clip_nonnegative: bool = True,
) -> List[float]:
    """
    Extract per-lane queue lengths from the encoded state vector.

    Returns a list of length num_lanes.
    If scale > 0, each queue is divided by scale (useful for normalization).
    """
    if num_lanes <= 0:
        raise ValueError("num_lanes must be > 0")

    expected_min_len = num_lanes * lane_block_size
    if len(state_vec) < expected_min_len:
        raise ValueError(
            f"state_vec too short: len={len(state_vec)} < {expected_min_len} "
            f"(num_lanes={num_lanes}, lane_block_size={lane_block_size})"
        )

    s = float(scale) if float(scale) > 0.0 else 1.0

    queues: List[float] = []
    for i in range(num_lanes):
        idx = i * lane_block_size + queue_offset_in_block
        q = float(state_vec[idx])
        if clip_nonnegative and q < 0.0:
            q = 0.0
        queues.append(q / s)
    return queues


def start_sumo(
    sumocfg: str, *, gui: bool, delay_ms: int, sumo_seed: int, traffic_scale: float
) -> None:
    binary = checkBinary("sumo-gui" if gui else "sumo")
    cmd: List[str] = [
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


"""
various reward functions
"""


def zone_exceedance_ratio_from_encoded_state(
    state_vec,
    num_lanes,
    lane_block_size=4,
    queue_offset_in_block=0,
    q0=0.25,
    tau=0.07,
):
    # q_i are already normalized queue features in [0,1] from your encoder
    qs = []
    base = 0
    for _ in range(num_lanes):
        q = float(state_vec[base + queue_offset_in_block])
        qs.append(np.clip(q, 0.0, 1.0))
        base += lane_block_size
    qv = np.asarray(qs, dtype=np.float64)

    # soft exceedance per lane
    ei = 1.0 / (1.0 + np.exp(-(qv - q0) / max(1e-6, tau)))
    z = float(np.mean(ei))  # in [0,1]
    return z


def zone_deadband_reward(
    z,
    z_lo=0.15,
    z_hi=0.45,
    r_good=0.10,
    m1=0.05,
    m2=0.05,
    lam=0.6,
    p=2.0,
):
    z = float(np.clip(z, 0.0, 1.0))
    if z <= z_lo:
        return float(r_good - m1 * (z / max(1e-6, z_lo)))
    elif z <= z_hi:
        return float(-m2 * ((z - z_lo) / max(1e-6, z_hi - z_lo)))
    else:
        x = (z - z_hi) / max(1e-6, 1.0 - z_hi)
        return float(-lam * (x**p))


def reward_avg_queue_from_encoded_state(
    state_vec: Sequence[float],
    *,
    num_lanes: int,
    lane_block_size: int = 4,
    queue_offset_in_block: int = 0,
    reduce: str = "mean",
) -> float:
    """
    Compute a reward from an encoded state vector based on (average) queue length.

    Assumptions about your encoding (matching your current encoder, excluding time_in_phase):
      - Per incoming lane block: [queue, veh_count, mean_speed, waiting_time]  (lane_block_size=4)
      - Then: is_green_now per lane  (num_lanes values)
      - Then: phase one-hot          (remaining values)
      - time_in_phase has been dropped before calling this function.

    Args:
      state_vec: encoded feature vector (time_in_phase already removed if you dropped it)
      num_lanes: number of incoming lanes encoded (must match your encoder's lane order)
      lane_block_size: number of features per lane in the lane block (default 4)
      queue_offset_in_block: index of queue feature within each lane block (default 0)
      reduce: "mean" or "sum" over lanes

    Returns:
      Reward as float. Typical choice is negative mean queue:  reward = -avg_queue
    """
    if num_lanes <= 0:
        raise ValueError("num_lanes must be > 0")

    expected_min_len = num_lanes * lane_block_size
    if len(state_vec) < expected_min_len:
        raise ValueError(
            f"state_vec too short: len={len(state_vec)} < {expected_min_len} "
            f"(num_lanes={num_lanes}, lane_block_size={lane_block_size})"
        )

    queues = []
    base = 0
    for i in range(num_lanes):
        idx = base + i * lane_block_size + queue_offset_in_block
        q = float(state_vec[idx])
        # queues should never be negative; clamp defensively
        if q < 0.0:
            q = 0.0
        queues.append(q)

    if reduce == "mean":
        val = sum(queues) / float(num_lanes)
    elif reduce == "sum":
        val = sum(queues)
    else:
        raise ValueError("reduce must be 'mean' or 'sum'")
    r = -(0.5 * val + 0.5 * max(queues))
    return r


def reward_top2_queue_from_encoded_state(
    state_vec: Sequence[float],
    *,
    num_lanes: int,
    lane_block_size: int = 4,
    queue_offset_in_block: int = 0,
    weights: Tuple[float, float] = (0.7, 0.3),
    power: float = 1.0,
    scale: float = 1.0,
    clip_nonnegative: bool = True,
) -> float:
    """
    Reward based on the TOP-2 longest queues across incoming lanes.

    Assumptions about encoding (matching your current encoder, time_in_phase dropped):
      - Per incoming lane block: [queue, veh_count, mean_speed, waiting_time] (lane_block_size=4)
      - Then: is_green_now per lane (num_lanes values)
      - Then: phase one-hot (remaining values)

    Args:
      state_vec: encoded feature vector.
      num_lanes: number of incoming lanes encoded.
      lane_block_size: number of features per lane in the lane block.
      queue_offset_in_block: index of queue feature within each lane block.
      weights: (w1, w2) weights for largest and 2nd-largest queues. Should sum to 1.0 (recommended).
      power: >=1.0. If >1, penalizes long queues more heavily (starvation/spillback).
      scale: optional scaling divisor applied to queues before computing penalty (useful when combining with throughput).
      clip_nonnegative: clamp negative queue values to 0 defensively.

    Returns:
      Reward (float): higher is better. Default is negative penalty:
        reward = - (w1*q1^p + w2*q2^p), with optional scaling.

    Notes:
      - If num_lanes == 1, q2 is taken equal to q1.
      - If scale <= 0, scale is treated as 1.0.
    """
    w1, w2 = float(weights[0]), float(weights[1])
    if w1 < 0.0 or w2 < 0.0:
        raise ValueError("weights must be nonnegative")
    if power < 1.0:
        raise ValueError("power must be >= 1.0")

    queues = _extract_queues_from_encoded_state(
        state_vec,
        num_lanes=num_lanes,
        lane_block_size=lane_block_size,
        queue_offset_in_block=queue_offset_in_block,
        scale=scale,
        clip_nonnegative=clip_nonnegative,
    )
    # Top-2 largest
    q_sorted = sorted(queues, reverse=True)
    q1 = q_sorted[0]
    q2 = q_sorted[1] if len(q_sorted) >= 2 else q_sorted[0]

    penalty = w1 * (q1**power) + w2 * (q2**power)
    return -penalty


# =======================
# [NEW] softmax-weighted queue penalty (smooth "top" approximation)
# =======================
def reward_softmax_queue_from_encoded_state(
    state_vec: Sequence[float],
    *,
    num_lanes: int,
    lane_block_size: int = 4,
    queue_offset_in_block: int = 0,
    power: float = 1.0,
    scale: float = 1.0,
    softmax_beta: float = 5.0,
    clip_nonnegative: bool = True,
) -> float:
    """
    Smooth alternative to "top-k" queue penalty: softmax-weighted penalty over ALL lanes.

      weights_i = softmax(beta * q_i)
      penalty   = sum_i weights_i * (q_i ** power)
      reward    = -penalty

    - weights sum to 1 (normalization)
    - as beta -> +inf, weights concentrate on the maximum queue (approaches max-like behavior)
    - power > 1 penalizes long queues super-linearly

    All queues are first normalized by `scale` (same as in reward_top2_queue_from_encoded_state).
    """
    if power < 1.0:
        raise ValueError("power must be >= 1.0")

    qs = _extract_queues_from_encoded_state(
        state_vec,
        num_lanes=num_lanes,
        lane_block_size=lane_block_size,
        queue_offset_in_block=queue_offset_in_block,
        scale=scale,
        clip_nonnegative=clip_nonnegative,
    )

    beta = float(softmax_beta)
    if beta <= 0.0:
        # beta<=0: raise error instead of uniform weights
        raise ValueError("softmax_beta must be > 0.0")
    else:
        logits = beta * np.asarray(qs, dtype=np.float64)
        logits = logits - float(np.max(logits))  # stabilize
        exps = np.exp(logits)
        weights = exps / float(np.sum(exps) + 1e-12)

    q_pow = np.asarray(qs, dtype=np.float64) ** float(power)
    penalty = float(np.sum(weights * q_pow))
    return -penalty


# =======================
# [NEW] softmax-weighted waiting-time barrier (smooth "max-wait" approximation)
# =======================
def reward_softmax_wait_barrier_from_encoded_state(
    state_vec: Sequence[float],
    *,
    num_lanes: int,
    lane_block_size: int = 4,
    wait_offset_in_block: int = 3,  # [queue, veh_count, speed, waiting_time]
    wait_ref_s: float = 60.0,
    softmax_beta: float = 10.0,
    barrier_start_s: float = 30.0,
    barrier_power: float = 1.0,
    clip_nonnegative: bool = True,
    # [NEW] If True, state_vec already contains the bounded/normalized wait feature
    # produced by encode_tsc_state_vector_bounded_v2 (soft-saturated to [0,1]).
    wait_is_encoded: bool = False,
) -> float:
    """
    Smooth waiting-time barrier reward (negative).
    Two input conventions:
      A) wait_is_encoded == False:
         - state_vec wait is in seconds (unbounded)
         - normalize by wait_ref_s
         - threshold is barrier_start_s / wait_ref_s

      B) wait_is_encoded == True (encode_tsc_state_vector_bounded_v2):
         - state_vec wait already is bounded/normalized to [0,1]:
             w_enc = soft_sat(mean_wait_stopped / wait_ref_s, sat=1.0)
         - DO NOT divide by wait_ref_s again
         - map barrier_start_s into same encoded space:
             start_enc = soft_sat(barrier_start_s / wait_ref_s, sat=1.0)

    Steps:
      1) Extract per-lane waiting times from encoded state
      2) Normalize by wait_ref_s
      3) Compute softmax-weighted mean wait (smooth max-like)
      4) Apply a threshold barrier: penalty = max(0, soft_wait - start)^power
      5) reward = -penalty

    Returns:
      wait_reward <= 0
    """
    if num_lanes <= 0:
        raise ValueError("num_lanes must be > 0")
    if wait_ref_s <= 0.0:
        raise ValueError("wait_ref_s must be > 0")
    if softmax_beta <= 0.0:
        raise ValueError("softmax_beta must be > 0")
    if barrier_power < 1.0:
        raise ValueError("barrier_power must be >= 1.0")

    expected_min_len = num_lanes * lane_block_size
    if len(state_vec) < expected_min_len:
        raise ValueError(
            f"state_vec too short: len={len(state_vec)} < {expected_min_len} "
            f"(num_lanes={num_lanes}, lane_block_size={lane_block_size})"
        )

    # ---- extract + normalize waiting times ----
    waits = []
    inv_ref = 1.0 / float(wait_ref_s)
    for i in range(num_lanes):
        idx = i * lane_block_size + wait_offset_in_block
        w = float(state_vec[idx])
        if clip_nonnegative and w < 0.0:
            w = 0.0
        # waits.append(w * inv_ref)  # normalized wait
        if wait_is_encoded:
            waits.append(w)
        else:
            waits.append(w * inv_ref)

    waits = np.asarray(waits, dtype=np.float64)

    # ---- softmax weights over waits (smooth max) ----
    beta = float(softmax_beta)
    logits = beta * waits
    logits = logits - float(np.max(logits))  # stabilize
    exps = np.exp(logits)
    weights = exps / float(np.sum(exps) + 1e-12)

    soft_wait = float(np.sum(weights * waits))

    # ---- barrier threshold in normalized units ----
    # start = float(barrier_start_s) * inv_ref
    if wait_is_encoded:
        start = _soft_sat(float(barrier_start_s) * inv_ref, sat=1.0)
    else:
        start = float(barrier_start_s) * inv_ref
    overflow = soft_wait - start
    if overflow <= 0.0:
        return 0.0

    penalty = overflow ** float(barrier_power)
    return -float(penalty)


# =======================
# [NEW] time-since-served starvation cost from encoded state (for potential shaping)
# =======================
def starvation_cost_from_encoded_state(
    *,
    tls_id: str,
    state_vec: Sequence[float],
    cache: Dict,
    num_lanes: int,
    lane_block_size: int = 4,
    softmax_beta: float = 10.0,
    power: float = 1.0,
    min_major_green_s: float = 5.0,
) -> float:
    """
    Extract the per-major "time-since-served" features appended by
    encode_tsc_state_vector_bounded_v2() and convert them into a smooth
    starvation cost in [0, 1] (approximately).
    """
    plan = get_tls_phase_plan(tls_id, cache, min_major_green_s=float(min_major_green_s))
    num_phases = int(len(plan.phases))
    num_major = int(len(plan.major_greens))
    offset = int(num_lanes) * int(lane_block_size) + int(num_lanes) + int(num_phases)
    end = offset + num_major
    if len(state_vec) < end:
        return 0.0
    since = np.asarray(state_vec[offset:end], dtype=np.float64)
    since = np.clip(since, 0.0, 1.0)
    logits = float(softmax_beta) * since
    logits = logits - float(np.max(logits))
    exps = np.exp(logits)
    weights = exps / float(np.sum(exps) + 1e-12)
    cost = float(np.sum(weights * (since ** float(power))))
    return float(np.clip(cost, 0.0, 1.0))


def _get_tls_out_lanes(tls_id: str) -> list[str]:
    """
    Outgoing lanes (downstream of the intersection) derived from controlled links.
    Each link tuple is (inLane, outLane, viaLane).
    """
    out_lanes: Set[str] = set()
    for conn_group in traci.trafficlight.getControlledLinks(tls_id):
        for link in conn_group:
            if not link:
                continue
            out_lane = link[1]
            if out_lane:
                out_lanes.add(out_lane)
    return sorted(out_lanes)


def throughput_tracker_step(
    tls_id: str,
    cache: Dict,
    *,
    out_lanes: Optional[Sequence[str]] = None,
) -> None:
    """
    Call this EVERY simulationStep() to accumulate which vehicles newly ENTERED
    the downstream lanes since the last decision reward was computed.

    It counts a vehicle exactly once at the moment it first appears on ANY downstream lane.

    cache keys used:
      - "_tp_out_lanes": list[str]
      - "_tp_seen_total": set[str]
      - "_tp_new_since_last": set[str]
    """
    if out_lanes is None:
        out_lanes = cache.get("_tp_out_lanes")
        if out_lanes is None:
            out_lanes = _get_tls_out_lanes(tls_id)
            cache["_tp_out_lanes"] = out_lanes

    seen_total: Set[str] = cache.setdefault("_tp_seen_total", set())

    # NEW: integer counter since last reward call
    count_since_last: int = cache.setdefault("_tp_count_since_last", 0)

    for ln in out_lanes:
        for vid in traci.lane.getLastStepVehicleIDs(ln):
            if vid not in seen_total:
                seen_total.add(vid)
                count_since_last += 1

    # write back updated counter
    cache["_tp_count_since_last"] = count_since_last


def reward_throughput_per_second_on_decision(
    sim_time: float,
    cache: Dict,
) -> float:
    """
    Call this at EACH DECISION POINT to get:
        reward = (# vehicles that newly entered downstream lanes since last decision) / dt

    cache keys used:
      - "_tp_last_decision_t": float
      - "_tp_new_since_last": set[str]  (cleared after reward is computed)

    Returns:
      throughput_veh_per_sec (float)

    Notes:
      - First call returns 0.0 (no previous decision interval).
      - This relies on throughput_tracker_step(...) being called every simulation step
        during the interval; otherwise fast vehicles may be missed.
    """
    last_t = cache.get("_tp_last_decision_t")
    cache["_tp_last_decision_t"] = float(sim_time)

    if last_t is None:
        cache["_tp_count_since_last"] = 0
        return 0.0

    dt = float(sim_time) - float(last_t)
    if dt <= 0.0:
        cache["_tp_count_since_last"] = 0
        return 0.0

    count = float(cache.get("_tp_count_since_last", 0))

    # reset interval counter (keep seen_total so vehicles are not double-counted)
    cache["_tp_count_since_last"] = 0

    return count / dt


# =======================
# [NEW] composite reward variant using softmax queue term
# =======================
def reward_throughput_plus_softmax_queue(
    *,
    tls_id: str,
    sim_time: float,
    state_vec: Sequence[float],
    cache: Dict,
    num_lanes: int,
    throughput_ref_veh_per_s: float,
    queue_ref_veh: float,
    w_throughput: float = 1.0,
    w_queue: float = 1.0,
    queue_power: float = 1.0,
    softmax_beta: float = 5.0,
    lane_block_size: int = 4,
    queue_offset_in_block: int = 0,
    reward_clip: Optional[Tuple[float, float]] = (-1.0, 1.0),
) -> float:
    def _clip(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x

    if throughput_ref_veh_per_s <= 0.0:
        raise ValueError("throughput_ref_veh_per_s must be > 0")
    if queue_ref_veh <= 0.0:
        raise ValueError("queue_ref_veh must be > 0")

    thr = reward_throughput_per_second_on_decision(sim_time=sim_time, cache=cache)
    thr_norm = _clip(float(thr) / float(throughput_ref_veh_per_s), 0.0, 1.0)

    q_reward = reward_softmax_queue_from_encoded_state(
        state_vec,
        num_lanes=num_lanes,
        lane_block_size=lane_block_size,
        queue_offset_in_block=queue_offset_in_block,
        power=queue_power,
        scale=queue_ref_veh,
        softmax_beta=softmax_beta,
        clip_nonnegative=True,
    )

    print(
        f">> reward: thr={thr:.3f} (norm {thr_norm:.3f}), softmax_queue_reward={q_reward:.5f}"
    )

    r = float(w_throughput) * thr_norm + float(w_queue) * float(q_reward)

    if reward_clip is not None:
        lo, hi = float(reward_clip[0]), float(reward_clip[1])
        if lo > hi:
            lo, hi = hi, lo
        r = _clip(r, lo, hi)

    return float(r)


# =======================
# [NEW] composite reward: throughput + softmax queue + delta softmax queue + softmax wait barrier
# =======================
rcnt = 0


def reward_throughput_plus_softmax_queue_deltaq_plus_softmax_wait_barrier_v2(
    *,
    tls_id: str,
    sim_time: float,
    state_vec: Sequence[float],
    cache: Dict,
    num_lanes: int,
    throughput_ref_veh_per_s: float,
    queue_ref_veh: float,
    wait_ref_s: float = 60.0,
    wait_barrier_start_s: float = 30.0,
    # ---- weights (recommended defaults below) ----
    w_throughput: float = 1.0,
    w_queue: float = 1.0,
    w_delta_queue: float = 0.5,
    w_wait_barrier: float = 0.5,
    w_starve_potential: float = 0.0,
    w_queue_zone: float = 0.3,
    # ---- queue term params ----
    queue_power: float = 1.0,
    softmax_queue_beta: float = 5.0,
    # ---- wait term params ----
    softmax_wait_beta: float = 10.0,
    wait_barrier_power: float = 1.0,
    # ---- starvation potential params ----
    starve_softmax_beta: float = 5.0,
    starve_power: float = 1.0,
    gamma_dt: float = 1.0,
    min_major_green_s: float = 5.0,
    # ---- zone shaping params (hardcoded defaults; no run_ppo changes needed) ----
    zone_q0: float = 0.29,  # lane queue exceedance center
    zone_tau: float = 0.07,  # softness of exceedance sigmoid
    zone_lo: float = 0.15,  # good coverage boundary
    zone_hi: float = 0.45,  # bad coverage starts
    zone_r_good: float = 0.15,  # reward when clearly in good region
    zone_m1: float = 0.05,  # mild slope in good region
    zone_m2: float = 0.1,  # mild slope in middle region
    zone_lambda: float = 0.6,  # penalty scale in bad region
    zone_p: float = 2.0,  # curvature in bad region
    # ---- encoding layout ----
    lane_block_size: int = 4,
    queue_offset_in_block: int = 0,
    wait_offset_in_block: int = 3,
) -> float:
    global rcnt
    """
    Reward terms:
      1) Throughput term (normalized by throughput_ref_veh_per_s, NO CLIP)
      2) Softmax queue penalty (negative)
      3) Delta softmax queue improvement: (prev_penalty - cur_penalty) (positive if improved)
      4) Softmax wait barrier penalty (negative): smooth "max-wait" above a threshold

    Notes:
      - No final reward clipping: uses full reward range.
      - Delta term uses cached previous softmax-queue penalty per TLS.
    """
    if throughput_ref_veh_per_s <= 0.0:
        raise ValueError("throughput_ref_veh_per_s must be > 0")
    if queue_ref_veh <= 0.0:
        raise ValueError("queue_ref_veh must be > 0")
    if wait_ref_s <= 0.0:
        raise ValueError("wait_ref_s must be > 0")

    # 1) Throughput (veh/s) over last decision interval, normalized (no clamp)
    thr = reward_throughput_per_second_on_decision(sim_time=sim_time, cache=cache)
    thr_norm = float(thr) / float(throughput_ref_veh_per_s)

    # 2) Absolute softmax queue reward (negative)
    q_reward = reward_softmax_queue_from_encoded_state(
        state_vec,
        num_lanes=num_lanes,
        lane_block_size=lane_block_size,
        queue_offset_in_block=queue_offset_in_block,
        power=queue_power,
        scale=queue_ref_veh,
        softmax_beta=softmax_queue_beta,
        clip_nonnegative=True,
    )
    # convert reward -> penalty scalar for delta
    q_penalty = -float(q_reward)  # >= 0

    # 3) Delta softmax queue improvement (positive if queues improved)
    prev_key = f"_rw_prev_softmax_q_penalty::{tls_id}"
    prev_penalty = cache.get(prev_key, None)
    cache[prev_key] = q_penalty

    if prev_penalty is None:
        delta_q = 0.0
    else:
        delta_q = float(prev_penalty) - float(q_penalty)  # >0 if improved

    # 4) Softmax wait barrier (negative)
    wait_reward = reward_softmax_wait_barrier_from_encoded_state(
        state_vec,
        num_lanes=num_lanes,
        lane_block_size=lane_block_size,
        wait_offset_in_block=wait_offset_in_block,
        wait_ref_s=wait_ref_s,
        softmax_beta=softmax_wait_beta,
        barrier_start_s=wait_barrier_start_s,
        barrier_power=wait_barrier_power,
        clip_nonnegative=True,
        wait_is_encoded=True,
    )
    # 5) Potential-based starvation shaping (optional)
    starve_shape = 0.0
    if float(w_starve_potential) != 0.0:
        cur_cost = starvation_cost_from_encoded_state(
            tls_id=tls_id,
            state_vec=state_vec,
            cache=cache,
            num_lanes=num_lanes,
            lane_block_size=lane_block_size,
            softmax_beta=float(starve_softmax_beta),
            power=float(starve_power),
            min_major_green_s=float(min_major_green_s),
        )
        prev_cost_key = f"_rw_prev_starve_cost::{tls_id}"
        prev_cost = cache.get(prev_cost_key, None)
        cache[prev_cost_key] = float(cur_cost)
        if prev_cost is not None:
            starve_shape = float(w_starve_potential) * (
                float(prev_cost) - float(gamma_dt) * float(cur_cost)
            )

    # 6) [NEW] queue coverage zone term (deadband/acceptable-region shaping)
    # Requires you already pasted:
    #   - zone_exceedance_ratio_from_encoded_state(...)
    #   - zone_deadband_reward(...)
    z_cov = zone_exceedance_ratio_from_encoded_state(
        state_vec,
        num_lanes=num_lanes,
        lane_block_size=lane_block_size,
        queue_offset_in_block=queue_offset_in_block,
        q0=zone_q0,
        tau=zone_tau,
    )
    r_zone = zone_deadband_reward(
        z_cov,
        z_lo=zone_lo,
        z_hi=zone_hi,
        r_good=zone_r_good,
        m1=zone_m1,
        m2=zone_m2,
        lam=zone_lambda,
        p=zone_p,
    )

    # Final combined reward (no clipping)
    r = (
        float(w_throughput) * float(thr_norm)
        + float(w_queue) * float(q_reward)
        + float(w_delta_queue) * float(delta_q)
        + float(w_wait_barrier) * float(wait_reward)
        + float(starve_shape)
        + float(w_queue_zone) * float(r_zone)
    )
    if rcnt % 100 == 0:
        print(
            f">> reward: {r} thr={thr:.3f} (norm {thr_norm:.3f}), q={q_reward:.3f}, delta_q={delta_q:.3f}, wait_barrier={wait_reward:.3f}, zone={r_zone:.3f}"
            f"{', starve_shape=' + format(starve_shape, '.3f') if starve_shape != 0.0 else ''}"
        )
    rcnt += 1
    return float(r)
