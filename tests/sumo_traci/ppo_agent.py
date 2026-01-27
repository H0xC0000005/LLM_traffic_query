# [NEW FILE] ppo_agent.py

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F


"""
main chunk of actor critic blocks
"""


class ActorCritic(nn.Module):
    """
    Minimal discrete-action Actor-Critic used by PPO:
      - shared MLP trunk
      - policy logits head (categorical)
      - value head
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            # nn.Linear(hidden_dim, hidden_dim),  # 3 layers
            # nn.Tanh(),
        )
        self.pi = nn.Linear(hidden_dim, action_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value


def _ortho_init_linear(layer: nn.Linear, gain: float = 1.0) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)


class _MLPBlock(nn.Module):
    """
    MLP trunk with:
      - Linear -> LayerNorm -> SiLU
      - optional residual connections (when shapes allow)
      - orthogonal init
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        *,
        use_skip: bool,
        ln_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.use_skip = bool(use_skip)
        self.hidden_dim = int(hidden_dim)

        layers: List[nn.Module] = []
        dims = [int(in_dim)] + [int(hidden_dim)] * int(num_layers)

        # Build pre-norm blocks: Linear -> LN -> SiLU
        self.linears = nn.ModuleList()
        self.lns = nn.ModuleList()

        for i in range(num_layers):
            lin = nn.Linear(dims[i], dims[i + 1])
            ln = nn.LayerNorm(dims[i + 1], eps=ln_eps)

            self.linears.append(lin)
            self.lns.append(ln)

            # used only for readability; forward does the actual sequencing
            layers.append(lin)
            layers.append(ln)
            layers.append(nn.SiLU())

        # keep for debugging/introspection
        self.net = nn.Sequential(*layers)

        # orthogonal init for trunk
        for lin in self.linears:
            _ortho_init_linear(lin, gain=math.sqrt(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, (lin, ln) in enumerate(zip(self.linears, self.lns)):
            z = lin(h)
            z = ln(z)
            z = torch.nn.functional.silu(z)

            # Residual when possible (same shape)
            # no shape check; dont let the inconsistency pass silently
            if self.use_skip and h.shape == z.shape:
                h = h + z
            else:
                h = z
        return h


class ActorCriticV2(nn.Module):
    """
    Discrete-action Actor-Critic for PPO:
      - separate actor/critic trunks (same architecture, separate params)
      - SiLU activation
      - LayerNorm before activation (pre-norm)
      - adjustable number of layers
      - optional skip connections
      - orthogonal init
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        *,
        num_layers: int = 2,
        use_skip: bool = False,
        ln_eps: float = 1e-5,
        pi_head_gain: float = 0.01,
        v_head_gain: float = 1.0,
    ) -> None:
        super().__init__()

        self.actor = _MLPBlock(
            in_dim=int(state_dim),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            use_skip=bool(use_skip),
            ln_eps=float(ln_eps),
        )
        self.critic = _MLPBlock(
            in_dim=int(state_dim),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            use_skip=bool(use_skip),
            ln_eps=float(ln_eps),
        )

        self.pi = nn.Linear(int(hidden_dim), int(action_dim))
        self.v = nn.Linear(int(hidden_dim), 1)

        # orthogonal init for heads
        _ortho_init_linear(self.pi, gain=float(pi_head_gain))
        _ortho_init_linear(self.v, gain=float(v_head_gain))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ha = self.actor(x)
        hc = self.critic(x)

        logits = self.pi(ha)
        value = self.v(hc).squeeze(-1)
        return logits, value


class _ScalarEmbed(nn.Module):
    """Embed a scalar feature (e.g., Δt) into a vector."""

    def __init__(self, out_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # match your trunk style: orthogonal init
        _ortho_init_linear(self.net[0], gain=math.sqrt(2.0))
        _ortho_init_linear(self.net[2], gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1)
        return self.net(x)


class _OneHotEmbed(nn.Module):
    """Embed a one-hot previous-action vector into a vector."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        _ortho_init_linear(self.net[0], gain=math.sqrt(2.0))
        _ortho_init_linear(self.net[2], gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,A)
        return self.net(x)


class _GatedResBlock(nn.Module):
    """
    Pre-LN gated residual MLP block:
      u = LN(h)
      f = Linear(u->4d)->SiLU->Linear(4d->d)
      g = sigmoid(Linear(u->d))
      h = h + g * f
    """

    def __init__(
        self,
        d: int,
        *,
        expansion: int = 4,
        ln_eps: float = 1e-5,
        gate_bias_init: float = -2.0,  # start near "mostly-skip" for stability
    ) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d, eps=ln_eps)

        self.fc1 = nn.Linear(d, expansion * d)
        self.fc2 = nn.Linear(expansion * d, d)

        self.gate = nn.Linear(d, d)

        # trunk init consistent with your code (orthogonal, sqrt(2) gain) :contentReference[oaicite:0]{index=0}
        _ortho_init_linear(self.fc1, gain=math.sqrt(2.0))
        # small/zero init for residual update helps PPO stability
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        # gate: keep it initially in a non-saturated region and mostly "closed"
        _ortho_init_linear(self.gate, gain=0.1)
        nn.init.constant_(self.gate.bias, gate_bias_init)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        u = self.ln(h)
        f = self.fc2(F.silu(self.fc1(u)))
        g = torch.sigmoid(self.gate(u))
        return h + g * f


class _GatedResTrunk(nn.Module):
    """Input projection + N gated residual blocks + output LN."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_blocks: int,
        *,
        expansion: int = 4,
        ln_eps: float = 1e-5,
        gate_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        _ortho_init_linear(self.in_proj, gain=math.sqrt(2.0))

        self.blocks = nn.ModuleList(
            [
                _GatedResBlock(
                    hidden_dim,
                    expansion=expansion,
                    ln_eps=ln_eps,
                    gate_bias_init=gate_bias_init,
                )
                for _ in range(int(n_blocks))
            ]
        )
        self.out_ln = nn.LayerNorm(hidden_dim, eps=ln_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)  # always residual (skip-like)
        return self.out_ln(h)


class ActorCriticV3(nn.Module):
    """
    Drop-in replacement for ActorCriticV2:
      - keeps forward(x)->(logits, value) expected by PPOAgent :contentReference[oaicite:1]{index=1}
      - exposes .actor, .critic, .pi, .v so your optimizer param groups still work :contentReference[oaicite:2]{index=2}
      - gated residual MLP trunks (good default capability: hidden_dim=256, n_blocks=4..6)

    Optional conditioning without changing PPO code: pack extra fields into your state vector:
      state = [stats..., dt, prev_a]                     (prev_a as index)
      state = [stats..., dt, prev_a_one_hot (len=A)]     (prev_a one-hot)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        *,
        n_blocks: int = 6,
        expansion: int = 4,
        ln_eps: float = 1e-5,
        # parsing / conditioning
        obs_dim: Optional[int] = None,  # number of "stats" features at the front
        use_dt: bool = False,  # if True, expects 1 scalar after obs
        prev_action_mode: Literal["none", "index", "one_hot"] = "none",
        dt_embed_dim: int = 16,
        prev_act_embed_dim: int = 32,
        head_hidden: int = 128,
        pi_head_gain: float = 0.01,
        v_head_gain: float = 1.0,
        gate_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.use_dt = bool(use_dt)
        self.prev_action_mode = prev_action_mode

        if obs_dim is None:
            # default: treat everything as plain stats (no extra conditioning)
            obs_dim = int(state_dim)
        self.obs_dim = int(obs_dim)

        # embedders (only created if enabled)
        self.dt_embed = _ScalarEmbed(dt_embed_dim) if self.use_dt else None

        if self.prev_action_mode == "index":
            self.prev_act_emb = nn.Embedding(self.action_dim, prev_act_embed_dim)
            nn.init.normal_(self.prev_act_emb.weight, mean=0.0, std=0.02)
            self.prev_act_oh_embed = None
        elif self.prev_action_mode == "one_hot":
            self.prev_act_emb = None
            self.prev_act_oh_embed = _OneHotEmbed(self.action_dim, prev_act_embed_dim)
        else:
            self.prev_act_emb = None
            self.prev_act_oh_embed = None

        # effective trunk input dim after embedding
        trunk_in_dim = self.obs_dim
        if self.use_dt:
            trunk_in_dim += int(dt_embed_dim)
        if self.prev_action_mode != "none":
            trunk_in_dim += int(prev_act_embed_dim)

        # separate actor/critic trunks like your V2 :contentReference[oaicite:3]{index=3}
        self.actor = _GatedResTrunk(
            trunk_in_dim,
            int(hidden_dim),
            int(n_blocks),
            expansion=int(expansion),
            ln_eps=float(ln_eps),
            gate_bias_init=float(gate_bias_init),
        )
        self.critic = _GatedResTrunk(
            trunk_in_dim,
            int(hidden_dim),
            int(n_blocks),
            expansion=int(expansion),
            ln_eps=float(ln_eps),
            gate_bias_init=float(gate_bias_init),
        )

        # heads (small MLP per the "capability" answer)
        self.pi = nn.Sequential(
            nn.Linear(int(hidden_dim), int(head_hidden)),
            nn.SiLU(),
            nn.Linear(int(head_hidden), self.action_dim),
        )
        self.v = nn.Sequential(
            nn.Linear(int(hidden_dim), int(head_hidden)),
            nn.SiLU(),
            nn.Linear(int(head_hidden), 1),
        )

        # init heads: hidden layers sqrt(2), final layers PPO-friendly
        _ortho_init_linear(self.pi[0], gain=math.sqrt(2.0))
        _ortho_init_linear(self.pi[2], gain=float(pi_head_gain))
        _ortho_init_linear(self.v[0], gain=math.sqrt(2.0))
        _ortho_init_linear(self.v[2], gain=float(v_head_gain))

    def _pack_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parse x into (stats, dt, prev_action) according to flags, then concatenate
        as [stats, embed(dt), embed(prev_a)] for the gated trunks.
        """
        # x is float tensor from PPOAgent :contentReference[oaicite:4]{index=4}
        stats = x[..., : self.obs_dim]
        off = self.obs_dim

        feats = [stats]

        if self.use_dt:
            dt = x[..., off : off + 1]
            off += 1
            feats.append(self.dt_embed(dt))
        if self.prev_action_mode == "index":
            prev_a = x[..., off : off + 1]
            # state is float -> map to int index (user must store exact ints)
            prev_idx = prev_a.round().to(torch.long).squeeze(-1)
            feats.append(self.prev_act_emb(prev_idx))
            off += 1
        elif self.prev_action_mode == "one_hot":
            prev_oh = x[..., off : off + self.action_dim]
            feats.append(self.prev_act_oh_embed(prev_oh))
            off += self.action_dim

        return torch.cat(feats, dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self._pack_features(x)
        ha = self.actor(z)
        hc = self.critic(z)

        logits = self.pi(ha)
        value = self.v(hc).squeeze(
            -1
        )  # keep (B,) like V2 :contentReference[oaicite:5]{index=5}
        return logits, value


"""
utilities of PPO agent
"""


class RolloutBuffer:
    """
    Stores on-policy rollouts at decision points:
      (s_t, a_t, logp_t, v_t, r_t, done_t)

    After collection, compute:
      returns_t and advantages_t (GAE-Lambda)
    """

    def __init__(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.logps: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.durations_s: List[float] = (
            []
        )  # [NEW] elapsed time for each transition (seconds)

        self.returns: Optional[np.ndarray] = None
        self.advs: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.states)

    def clear(self) -> None:
        self.__init__()

    def add(
        self,
        *,
        state: np.ndarray,
        action: int,
        logp: float,
        value: float,
        reward: float,
        done: bool,
        duration_s: float,
    ) -> None:
        self.states.append(state.astype(np.float32, copy=False))
        self.actions.append(int(action))
        self.logps.append(float(logp))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.durations_s.append(float(duration_s))

    def compute_gae(
        self,
        *,
        last_value: float,
        gamma: float,
        gae_lambda: float,
        base_dt_s: float,  # [NEW] reference duration corresponding to "one step" gamma (e.g., hold_s)
    ) -> None:
        """
        Compute GAE advantages and returns.

        If the rollout ends mid-episode, last_value should be V(s_T) for bootstrapping.
        If the rollout ends at terminal, last_value should be 0.
        """
        n = len(self.rewards)
        advs = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        next_adv = 0.0
        next_value = float(last_value)

        for t in reversed(range(n)):
            done = float(self.dones[t])
            mask = 1.0 - done

            # delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            # next_adv = delta + gamma * gae_lambda * mask * next_adv

            # NEW: time-aware (SMDP) discounting
            dt = self.durations_s[t]
            gamma_t = float(gamma) ** (
                float(dt) / float(base_dt_s)
            )  # [NEW] effective discount for this transition

            delta = self.rewards[t] + gamma_t * next_value * mask - self.values[t]
            next_adv = delta + gamma_t * gae_lambda * mask * next_adv

            advs[t] = next_adv
            returns[t] = advs[t] + self.values[t]

            next_value = self.values[t]

        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        self.advs = advs
        self.returns = returns

    def minibatches(
        self, *, batch_size: int, shuffle: bool = True
    ) -> Iterator[np.ndarray]:
        idx = np.arange(len(self.states))
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            yield idx[start : start + batch_size]


@dataclass
class PPOAgent:
    state_dim: int
    action_dim: int
    seed: int = 0
    hidden_dim: int = 128
    n_layer: int = 2
    use_skip: bool = False
    # lr: float = 3e-4
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    device: Optional[str] = None

    # PPO hyperparameters (defaults)
    clip_eps: float = 0.2
    # [NEW] value-function clipping; set to None to disable
    vf_clip_eps: Optional[float] = None
    epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    # -------------------------------
    # [NEW] Entropy schedule (linear)
    # If ent_coef_end is None -> fixed ent_coef.
    # -------------------------------
    ent_coef_end: Optional[float] = None
    ent_coef_decay_updates: int = 0

    # -------------------------------
    # [NEW] Uniform ridge exploration (mixture policy μ)
    # μ = (1-α)π + αU, with optional linear decay of α.
    # -------------------------------
    explore_alpha_start: float = 0.0
    explore_alpha_end: float = 0.0
    explore_alpha_decay_updates: int = 0

    # -------------------------------
    # [NEW] PPO safety knobs
    # -------------------------------
    target_kl: Optional[float] = None  # e.g. 0.02 to early-stop PPO epochs
    adv_clip: Optional[float] = None  # e.g. 5.0 to clip advantages

    max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_t = torch.device(self.device)

        torch.manual_seed(int(self.seed))
        self.model = ActorCriticV3(
            self.state_dim,
            self.action_dim,
            hidden_dim=self.hidden_dim,
            n_blocks=self.n_layer,
        ).to(self.device_t)
        # self.opt = optim.Adam(self.model.parameters(), lr=float(self.lr))
        self.opt = optim.Adam(
            [
                {"params": self.model.actor.parameters(), "lr": self.actor_lr},
                {"params": self.model.pi.parameters(), "lr": self.actor_lr},
                {"params": self.model.critic.parameters(), "lr": self.critic_lr},
                {"params": self.model.v.parameters(), "lr": self.critic_lr},
            ],
        )
        self._update_idx: int = 0
        pass

    # =======================
    # [NEW] schedule helpers
    # =======================
    @staticmethod
    def _linear_schedule(start: float, end: float, t: int, t_end: int) -> float:
        if t_end <= 0:
            return float(end)
        frac = min(max(float(t) / float(t_end), 0.0), 1.0)
        return float(start + frac * (end - start))

    def _current_ent_coef(self) -> float:
        if self.ent_coef_end is None:
            return float(self.ent_coef)
        return self._linear_schedule(
            float(self.ent_coef),
            float(self.ent_coef_end),
            int(self._update_idx),
            int(self.ent_coef_decay_updates),
        )

    def _current_explore_alpha(self) -> float:
        # α for mixture policy μ = (1-α)π + αU
        return self._linear_schedule(
            float(self.explore_alpha_start),
            float(self.explore_alpha_end),
            int(self._update_idx),
            int(self.explore_alpha_decay_updates),
        )

    # =======================

    @torch.no_grad()
    def act(self, state_vec: np.ndarray) -> tuple[int, float, float]:
        """
        Sample action from the *behavior policy* (training).

        Behavior policy is a mixture:
          μ(a|s) = (1-α)π(a|s) + α * Uniform(a)

        IMPORTANT:
          - We store logp under μ so PPO ratios stay consistent.

        Returns: (action, logp_mu, value)
        """
        x = (
            torch.from_numpy(state_vec.astype(np.float32))
            .unsqueeze(0)
            .to(self.device_t)
        )
        logits, value = self.model(x)

        # π distribution
        dist_pi = Categorical(logits=logits)
        probs_pi = dist_pi.probs  # [1, A]

        alpha = float(self._current_explore_alpha())
        if alpha > 0.0:
            a_dim = int(self.action_dim)
            probs_mu = (1.0 - alpha) * probs_pi + alpha * (1.0 / float(a_dim))
            dist_mu = Categorical(probs=probs_mu)
            a = dist_mu.sample()
            logp = dist_mu.log_prob(a)  # log μ(a|s)
        else:
            a = dist_pi.sample()
            logp = dist_pi.log_prob(a)  # log π(a|s) == log μ(a|s) when α=0

        return int(a.item()), float(logp.item()), float(value.item())

    @torch.no_grad()
    def act_greedy(self, state_vec: np.ndarray) -> int:
        x = (
            torch.from_numpy(state_vec.astype(np.float32))
            .unsqueeze(0)
            .to(self.device_t)
        )
        logits, _ = self.model(x)
        return int(torch.argmax(logits, dim=1).item())

    @torch.no_grad()
    def forward_logits_value(
        self,
        state: Union[np.ndarray, torch.Tensor],
        *,
        return_probs: bool = False,
        to_cpu: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass for evaluation / analysis.

        Returns:
        logits: (B, action_dim)
        probs:  (B, action_dim) if return_probs=True else None
        value:  (B,)
        """
        if isinstance(state, np.ndarray):
            x = torch.from_numpy(state.astype(np.float32, copy=False))
        else:
            x = state

        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, state_dim)

        x = x.to(self.device_t, dtype=torch.float32)

        logits, value = self.model(x)  # logits: (B,A), value: (B,)
        probs = torch.softmax(logits, dim=-1) if return_probs else None

        if to_cpu:
            logits = logits.detach().cpu()
            value = value.detach().cpu()
            if probs is not None:
                probs = probs.detach().cpu()

        return logits, probs, value

    def _evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        *,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log-prob under behavior policy μ and entropy under π.
        alpha is the ridge exploration coefficient as regularization.

        Returns:
          logps_mu: [B]
          entropy_pi: [B]
          values: [B]
        """
        logits, values = self.model(states)
        dist_pi = Categorical(logits=logits)
        probs_pi = dist_pi.probs  # [B, A]
        entropy_pi = dist_pi.entropy()

        if alpha > 0.0:
            a_dim = int(self.action_dim)
            probs_mu = (1.0 - float(alpha)) * probs_pi + float(alpha) * (
                1.0 / float(a_dim)
            )
            sel = probs_mu.gather(1, actions.view(-1, 1)).squeeze(1).clamp_min(1e-12)
            logps_mu = torch.log(sel)
        else:
            logps_mu = dist_pi.log_prob(actions)

        return logps_mu, entropy_pi, values

    def update(self, buf: RolloutBuffer) -> Dict[str, float]:
        assert (
            buf.returns is not None and buf.advs is not None
        ), "Call compute_gae() before update()."

        states = torch.from_numpy(np.stack(buf.states)).to(self.device_t)
        actions = torch.tensor(buf.actions, dtype=torch.long).to(self.device_t)
        old_logps = torch.tensor(buf.logps, dtype=torch.float32).to(self.device_t)
        # [NEW] old value predictions saved during rollout collection
        old_values = torch.tensor(buf.values, dtype=torch.float32).to(self.device_t)
        returns = torch.from_numpy(buf.returns).to(self.device_t)
        advs = torch.from_numpy(buf.advs).to(self.device_t)

        # [NEW] per-update scheduled coefficients (held constant during this PPO update)
        ent_coef_cur = float(self._current_ent_coef())
        alpha_cur = float(self._current_explore_alpha())

        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "vf_clip_frac": 0.0,
            "ent_coef": 0.0,
            "explore_alpha": 0.0,
            "early_stop": 0.0,
        }
        n_updates = 0
        early_stop = False

        for _ in range(int(self.epochs)):
            for mb_idx in buf.minibatches(
                batch_size=int(self.minibatch_size), shuffle=True
            ):
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logps = old_logps[mb_idx]
                mb_old_values = old_values[mb_idx]  # [NEW]
                mb_returns = returns[mb_idx]
                mb_advs = advs[mb_idx]

                # [NEW] advantage clipping
                if self.adv_clip is not None:
                    mb_advs = torch.clamp(
                        mb_advs, -float(self.adv_clip), float(self.adv_clip)
                    )

                # [NEW] logp under μ, entropy under π
                logps, entropy, values = self._evaluate(
                    mb_states, mb_actions, alpha=alpha_cur
                )

                ratio = torch.exp(logps - mb_old_logps)
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                pg_loss = -torch.min(ratio * mb_advs, clipped * mb_advs).mean()

                # --- [NEW] PPO value clipping ------------------------------------
                if self.vf_clip_eps is None:
                    v_loss = 0.5 * (mb_returns - values).pow(2).mean()
                    vf_clip_frac = torch.tensor(0.0, device=values.device)
                else:
                    eps_v = float(self.vf_clip_eps)
                    v_pred_clipped = mb_old_values + torch.clamp(
                        values - mb_old_values, -eps_v, eps_v
                    )

                    v_loss_unclipped = (values - mb_returns).pow(2)
                    v_loss_clipped = (v_pred_clipped - mb_returns).pow(2)

                    # max matches common PPO implementations (SB3/baselines)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    vf_clip_frac = (
                        (torch.abs(values - mb_old_values) > eps_v).float().mean()
                    )
                # ----------------------------------------------------------------
                ent = entropy.mean()

                # [NEW] scheduled entropy coefficient
                loss = pg_loss + self.vf_coef * v_loss - ent_coef_cur * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    approx_kl = 0.5 * ((logps - mb_old_logps).pow(2)).mean()

                    # [NEW] target-KL early stop
                    if self.target_kl is not None and float(approx_kl.item()) > float(
                        self.target_kl
                    ):
                        early_stop = True

                    clip_frac = (torch.abs(ratio - 1.0) > self.clip_eps).float().mean()

                stats["policy_loss"] += float(pg_loss.item())
                stats["value_loss"] += float(v_loss.item())
                stats["entropy"] += float(ent.item())
                stats["approx_kl"] += float(approx_kl.item())
                stats["clip_frac"] += float(clip_frac.item())
                stats["vf_clip_frac"] += float(vf_clip_frac.item())  # [NEW]
                n_updates += 1

                if early_stop:
                    break
            if early_stop:
                break
        if n_updates > 0:
            for k in list(stats.keys()):
                stats[k] /= n_updates

        # [NEW] record current schedule coefficients and early-stop flag
        stats["ent_coef"] = float(ent_coef_cur)
        stats["explore_alpha"] = float(alpha_cur)
        stats["early_stop"] = float(1.0 if early_stop else 0.0)
        stats["update_idx"] = int(self._update_idx)
        stats["updates_done"] = float(n_updates)
        # advance schedules AFTER finishing this PPO update
        self._update_idx += 1

        return stats
