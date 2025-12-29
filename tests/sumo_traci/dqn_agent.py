from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn




@dataclass
class RunningQStats:
    # running mean over q elements seen so far
    q_sum: float = 0.0
    q_count: int = 0

    # running maxima
    q_abs_max: float = 0.0
    y_abs_max: float = 0.0

    def update(self, q_all: torch.Tensor, y: torch.Tensor) -> None:
        # q_all: [B, A], y: [B]
        q_mean_batch = q_all.mean().item()
        # For running mean over *all q elements* (not mean-of-means):
        self.q_sum += q_all.sum().item()
        self.q_count += q_all.numel()

        self.q_abs_max = max(self.q_abs_max, q_all.abs().max().item())
        self.y_abs_max = max(self.y_abs_max, y.abs().max().item())

    @property
    def q_mean_running(self) -> float:
        return self.q_sum / max(1, self.q_count)


class DQN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self._he_init()

    def _he_init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNAgent:
    state_dim: int
    action_dim: int
    seed: int = 0
    hidden_dim: int = 128
    lr: float = 1e-3
    device: Optional[str] = None
    log_path: Optional[str] = "out.txt"
    _log_header_written: bool = False

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_t = torch.device(self.device)

        torch.manual_seed(self.seed)

        # Online Q-network
        self.model = DQN(self.state_dim, self.action_dim, hidden_dim=self.hidden_dim).to(self.device_t)

        # Target Q-network
        self.target = DQN(self.state_dim, self.action_dim, hidden_dim=self.hidden_dim).to(self.device_t)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber

        self.train_steps: int = 0

        # running aggregation across training
        self.running_stats = RunningQStats()

    @torch.no_grad()
    def act(self, state_vec: np.ndarray, epsilon: float = 0.0) -> int:
        """
        epsilon=0.0 -> greedy argmax
        epsilon>0.0 -> epsilon-greedy random action
        """
        if state_vec.ndim != 1 or state_vec.shape[0] != self.state_dim:
            raise ValueError(f"state_vec shape {state_vec.shape} does not match state_dim={self.state_dim}")

        if epsilon > 0.0 and float(np.random.rand()) < epsilon:
            return int(np.random.randint(self.action_dim))

        x = torch.from_numpy(state_vec.astype(np.float32)).unsqueeze(0).to(self.device_t)  # [1, state_dim]
        q = self.model(x)  # [1, action_dim]
        return int(torch.argmax(q, dim=1).item())

    def update_target(self) -> None:
        self.target.load_state_dict(self.model.state_dict())

    def update(
        self,
        batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        *,
        gamma: float = 0.99,
        grad_clip: float = 10.0,
    ) -> float:
        """
        One DQN gradient step.

        batch:
          states      [B, state_dim] float32
          actions     [B] int64
          rewards     [B] float32
          next_states [B, state_dim] float32
          dones       [B] float32 (0.0 or 1.0)

        Returns:
          loss (float)
        """
        states, actions, rewards, next_states, dones = batch

        self.model.train()

        s = torch.from_numpy(states).to(self.device_t)              # [B, D]
        a = torch.from_numpy(actions).to(self.device_t)            # [B]
        r = torch.from_numpy(rewards).to(self.device_t)            # [B]
        s2 = torch.from_numpy(next_states).to(self.device_t)       # [B, D]
        d = torch.from_numpy(dones).to(self.device_t)              # [B]

        q_all: torch.Tensor = self.model(s)                                      # [B, A]
        q_sa = q_all.gather(1, a.unsqueeze(1)).squeeze(1)          # [B]

        # vanilla dqn
        # with torch.no_grad():
        #     q2_all = self.target(s2)                               # [B, A]
        #     q2_max = q2_all.max(dim=1).values                      # [B]
        #     y = r + gamma * (1.0 - d) * q2_max                     # [B]

        with torch.no_grad():
            # Double DQN:
            # 1) action selection using ONLINE net
            a2 = self.model(s2).argmax(dim=1)                      # [B]

            # 2) action evaluation using TARGET net
            q2_sa = self.target(s2).gather(1, a2.unsqueeze(1)).squeeze(1)  # [B]

            y = r + gamma * (1.0 - d) * q2_sa                      # [B]

        loss = self.loss_fn(q_sa, y)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        self.optim.step()

        self.train_steps += 1
        self.model.eval()

        # ---- logging ----
        # per-step stats
        log = {
            "step": self.train_steps,
            "loss": float(loss.detach().cpu().item()),
            "q_mean": float(q_all.detach().mean().cpu().item()),
            "q_abs_max": float(q_all.detach().abs().max().cpu().item()),
            "y_abs_max": float(y.detach().abs().max().cpu().item()),
        }

        # running aggregation across *entire training so far*
        self.running_stats.update(q_all.detach(), y.detach())
        self._append_log_line(log)

        return float(loss.detach().cpu().item())

    def _append_log_line(self, log: Dict[str, Any]) -> None:
        if self.log_path is None:
            return

        p = Path(self.log_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        fields = [
            "step",
            "loss",
            "q_mean",
            "q_abs_max",
            "y_abs_max",
        ]

        with p.open("a", encoding="utf-8") as f:
            if not self._log_header_written and (p.stat().st_size == 0):
                f.write("\t".join(fields) + "\n")
                self._log_header_written = True

            f.write("\t".join(str(log[k]) for k in fields) + "\n")
            f.flush()