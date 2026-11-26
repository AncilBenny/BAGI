"""Small recurrent Q baseline for transfer diagnostics, not a new RL algorithm.

Uses PyTorch GRU and standard target-network TD learning. Whole public episodes
are replayed from zero recurrent state, so no stale hidden states enter targets.
"""
from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Episode:
    observations: np.ndarray  # T+1 public observations, including history features
    actions: np.ndarray
    rewards: np.ndarray
    terminal: np.ndarray
    domain: str
    legal_actions: int


class RecurrentQ(nn.Module):
    def __init__(self, width: int, actions: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(width, hidden, batch_first=True)
        self.head = nn.Linear(hidden, actions)

    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        return self.head(out), hidden


class RecurrentReplay:
    """Persistent policy, target, optimizer, episode replay and private RNG."""

    def __init__(self, width: int, actions: int, *, seed: int):
        self.width, self.actions = width, actions
        self.seed = seed
        self.rng = random.Random(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.online = RecurrentQ(width, actions)
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=0.0003)
        self.replay: dict[str, deque[Episode]] = {}
        self.hidden = None
        self.updates = 0
        self.presentations = 0
        self.training_actions = 0

    def begin_episode(self):
        self.hidden = None

    def act(self, observation: np.ndarray, legal_actions: int, *, epsilon: float) -> int:
        if observation.shape != (self.width,) or not np.isfinite(observation).all():
            raise ValueError("Invalid public observation")
        if not 1 <= legal_actions <= self.actions or not 0 <= epsilon <= 1:
            raise ValueError("Invalid action interface or exploration")
        with torch.no_grad():
            q, self.hidden = self.online(
                torch.as_tensor(observation, dtype=torch.float32).reshape(1, 1, -1),
                self.hidden,
            )
        if epsilon and self.rng.random() < epsilon:
            return self.rng.randrange(legal_actions)
        return int(q[0, 0, :legal_actions].argmax().item())

    def remember(self, episode: Episode):
        n = len(episode.actions)
        if (
            n == 0 or episode.observations.shape != (n + 1, self.width)
            or episode.rewards.shape != (n,) or episode.terminal.shape != (n,)
            or not np.isfinite(episode.observations).all()
            or not np.isfinite(episode.rewards).all()
            or not 1 <= episode.legal_actions <= self.actions
            or not np.issubdtype(episode.actions.dtype, np.integer)
            or episode.actions.min() < 0 or episode.actions.max() >= episode.legal_actions
            or episode.terminal.dtype != np.bool_ or episode.terminal[:-1].any()
        ):
            raise ValueError("Invalid actual-action replay episode")
        self.replay.setdefault(episode.domain, deque(maxlen=2048)).append(copy.deepcopy(episode))
        self.training_actions += n

    def update(self, batch_size: int = 4) -> float:
        """Equal domain sampling, then uniform whole episodes within domain."""
        domains = sorted(key for key, rows in self.replay.items() if rows)
        if not domains or batch_size < 1:
            raise ValueError("Replay is empty or batch size invalid")
        # Randomize the first domain, then cycle for balanced sampling.
        offset = self.rng.randrange(len(domains))
        sampled = [self.rng.choice(self.replay[domains[(offset + i) % len(domains)]])
                   for i in range(batch_size)]
        losses = []
        for ep in sampled:
            x = torch.as_tensor(ep.observations, dtype=torch.float32).unsqueeze(0)
            a = torch.as_tensor(ep.actions, dtype=torch.long)
            r = torch.as_tensor(ep.rewards, dtype=torch.float32)
            terminal = torch.as_tensor(ep.terminal, dtype=torch.float32)
            q, _ = self.online(x)
            predicted = q[0, :-1].gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                target_q, _ = self.target(x)
                next_action = q[0, 1:, :ep.legal_actions].argmax(dim=-1)
                next_value = target_q[0, 1:].gather(1, next_action.unsqueeze(1)).squeeze(1)
                target = r + 0.99 * (1 - terminal) * next_value
            losses.append(F.smooth_l1_loss(predicted, target))
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite recurrent TD loss")
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.updates += 1
        self.presentations += sum(len(ep.actions) for ep in sampled)
        if self.updates % 20 == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def reset_head(self, seed: int):
        """Single boundary-reset intervention inspired by primacy-bias work."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.online.head.reset_parameters()
        self.target.load_state_dict(self.online.state_dict())
        for parameter in self.online.head.parameters():
            self.optimizer.state.pop(parameter, None)
        self.begin_episode()
