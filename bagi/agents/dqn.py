"""Small DQN baseline used as a sample-efficiency reference."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class QNetwork(nn.Module):
    def __init__(self, observation_shape: tuple[int, int], actions: int, hidden: int = 128):
        super().__init__()
        inputs = int(np.prod(observation_shape))
        self.net = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, actions),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float().flatten(start_dim=1) / 6.0
        return self.net(x)


@dataclass
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self._data: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._data)

    def add(self, transition: Transition) -> None:
        self._data.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self._data, batch_size)


class DQNAgent:
    def __init__(
        self,
        observation_shape: tuple[int, int],
        actions: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.actions = actions
        self.gamma = gamma
        self.online = QNetwork(observation_shape, actions).to(self.device)
        self.target = QNetwork(observation_shape, actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer()

    def select_action(self, observation: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.actions)
        obs = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.online(obs).argmax(dim=1).item())

    def learn(self, batch_size: int = 64) -> float | None:
        if len(self.buffer) < batch_size:
            return None
        batch = self.buffer.sample(batch_size)
        observations = torch.as_tensor(
            np.stack([t.observation for t in batch]), device=self.device
        )
        actions = torch.tensor([t.action for t in batch], device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(
            np.stack([t.next_observation for t in batch]), device=self.device
        )
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        q_values = self.online(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_values = self.target(next_observations).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - dones) * next_values

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
