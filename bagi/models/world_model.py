"""A small predictive model for Workshop dynamics."""

from __future__ import annotations

import math

import torch
from torch import nn


class WorldModel(nn.Module):
    """Predict next local observation and reward from observation + action."""

    def __init__(
        self,
        observation_shape: tuple[int, int],
        actions: int,
        cell_values: int = 7,
        hidden: int = 192,
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.actions = actions
        self.cell_values = cell_values
        cells = math.prod(observation_shape)
        inputs = cells + actions
        self.backbone = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.next_observation_head = nn.Linear(hidden, cells * cell_values)
        self.reward_head = nn.Linear(hidden, 1)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = observations.shape[0]
        obs = observations.float().flatten(start_dim=1) / max(1, self.cell_values - 1)
        action_one_hot = torch.nn.functional.one_hot(
            actions.long(), num_classes=self.actions
        ).float()
        features = self.backbone(torch.cat([obs, action_one_hot], dim=1))
        logits = self.next_observation_head(features)
        logits = logits.view(batch, *self.observation_shape, self.cell_values)
        reward = self.reward_head(features).squeeze(1)
        return logits, reward

    @torch.no_grad()
    def predict(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, reward = self(observations, actions)
        return logits.argmax(dim=-1), reward
