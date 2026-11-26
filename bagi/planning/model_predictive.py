"""Model-predictive planning through a learned Workshop world model."""

from __future__ import annotations

import torch

from bagi.models import WorldModel


class ModelPredictivePlanner:
    """Choose actions by scoring short imagined action sequences."""

    def __init__(
        self,
        model: WorldModel,
        actions: int,
        horizon: int = 5,
        candidates: int = 128,
        gamma: float = 0.97,
        seed: int = 42,
        event_weights: list[float] | None = None,
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
        self.event_weights = (
            torch.tensor(event_weights, dtype=torch.float32) if event_weights is not None else None
        )

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
        sequences[: self.actions, 0] = torch.arange(self.actions)

        scores = torch.zeros(self.candidates, dtype=torch.float32)
        discount = 1.0
        for step in range(self.horizon):
            actions = sequences[:, step]
            if self.event_weights is not None and hasattr(self.model, "predict_events"):
                states, predicted_rewards, event_probs = self.model.predict_events(states, actions)
                event_score = event_probs.cpu() @ self.event_weights
                step_score = predicted_rewards.cpu() + event_score
            else:
                states, predicted_rewards = self.model.predict(states, actions)
                step_score = predicted_rewards.cpu()
            scores += discount * step_score
            discount *= self.gamma
            self.imagined_steps += self.candidates

        best = int(torch.argmax(scores).item())
        return int(sequences[best, 0].item())
