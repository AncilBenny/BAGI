"""Model-predictive planning through a learned Workshop world model."""

from __future__ import annotations

import torch

from bagi.models import WorldModel


class ModelPredictivePlanner:
    """Choose actions by scoring short imagined action sequences.

    The planner never calls the real environment while searching. Every candidate
    trajectory is rolled forward entirely inside the learned world model.
    """

    def __init__(
        self,
        model: WorldModel,
        actions: int,
        horizon: int = 5,
        candidates: int = 128,
        gamma: float = 0.97,
        seed: int = 42,
    ):
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if candidates < actions:
            raise ValueError("candidates must be >= number of actions")
        self.model = model.eval()
        self.actions = actions
        self.horizon = horizon
        self.candidates = candidates
        self.gamma = gamma
        self.generator = torch.Generator().manual_seed(seed)
        self.imagined_steps = 0

    @torch.no_grad()
    def select_action(self, observation) -> int:
        observation = torch.as_tensor(observation, dtype=torch.long)
        states = observation.unsqueeze(0).repeat(self.candidates, 1, 1)

        sequences = torch.randint(
            0,
            self.actions,
            (self.candidates, self.horizon),
            generator=self.generator,
        )
        # Guarantee that every real action is considered at least once at t=0.
        sequences[: self.actions, 0] = torch.arange(self.actions)

        scores = torch.zeros(self.candidates, dtype=torch.float32)
        discount = 1.0
        for step in range(self.horizon):
            actions = sequences[:, step]
            states, predicted_rewards = self.model.predict(states, actions)
            scores += discount * predicted_rewards.cpu()
            discount *= self.gamma
            self.imagined_steps += self.candidates

        best = int(torch.argmax(scores).item())
        return int(sequences[best, 0].item())
