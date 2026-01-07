# [NEW FILE] ppo_agent.py

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


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
    epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_t = torch.device(self.device)

        torch.manual_seed(int(self.seed))
        self.model = ActorCriticV2(
            self.state_dim,
            self.action_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.n_layer,
            use_skip=self.use_skip,
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
        pass

    @torch.no_grad()
    def act(self, state_vec: np.ndarray) -> tuple[int, float, float]:
        """
        Sample action from the policy (training).
        Returns: (action, logp, value)
        """
        x = (
            torch.from_numpy(state_vec.astype(np.float32))
            .unsqueeze(0)
            .to(self.device_t)
        )
        logits, value = self.model(x)
        dist = Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
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

    def _evaluate(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.model(states)
        dist = Categorical(logits=logits)
        logps = dist.log_prob(actions)
        entropy = dist.entropy()
        return logps, entropy, values

    def update(self, buf: RolloutBuffer) -> Dict[str, float]:
        assert (
            buf.returns is not None and buf.advs is not None
        ), "Call compute_gae() before update()."

        states = torch.from_numpy(np.stack(buf.states)).to(self.device_t)
        actions = torch.tensor(buf.actions, dtype=torch.long).to(self.device_t)
        old_logps = torch.tensor(buf.logps, dtype=torch.float32).to(self.device_t)
        returns = torch.from_numpy(buf.returns).to(self.device_t)
        advs = torch.from_numpy(buf.advs).to(self.device_t)

        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
        }
        n_updates = 0

        for _ in range(int(self.epochs)):
            for mb_idx in buf.minibatches(
                batch_size=int(self.minibatch_size), shuffle=True
            ):
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logps = old_logps[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advs = advs[mb_idx]

                logps, entropy, values = self._evaluate(mb_states, mb_actions)

                ratio = torch.exp(logps - mb_old_logps)
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                pg_loss = -torch.min(ratio * mb_advs, clipped * mb_advs).mean()

                v_loss = 0.5 * (mb_returns - values).pow(2).mean()
                ent = entropy.mean()

                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    approx_kl = 0.5 * ((logps - mb_old_logps).pow(2)).mean()
                    clip_frac = (torch.abs(ratio - 1.0) > self.clip_eps).float().mean()

                stats["policy_loss"] += float(pg_loss.item())
                stats["value_loss"] += float(v_loss.item())
                stats["entropy"] += float(ent.item())
                stats["approx_kl"] += float(approx_kl.item())
                stats["clip_frac"] += float(clip_frac.item())
                n_updates += 1

        if n_updates > 0:
            for k in list(stats.keys()):
                stats[k] /= n_updates

        return stats
