"""An optional uncertainty penalty for model-predictive causal planning.

The baseline remains unchanged: a zero penalty delegates to its implementation.
Predictive confidence is a heuristic, not a calibrated uncertainty estimate.
"""

from __future__ import annotations

import torch

from bagi.planning.model_predictive import ModelPredictivePlanner


class UncertaintyAwarePlanner(ModelPredictivePlanner):
    """Penalize low-confidence imagined observations when ranking action plans."""

    def __init__(self, *args, uncertainty_weight: float = 0.0, **kwargs):
        if uncertainty_weight < 0:
            raise ValueError("uncertainty_weight must be non-negative")
        super().__init__(*args, **kwargs)
        self.uncertainty_weight = float(uncertainty_weight)
        self.last_selected_uncertainty: float | None = None

    @torch.no_grad()
    def select_action(self, observation) -> int:
        if self.uncertainty_weight == 0:
            self.last_selected_uncertainty = None
            return super().select_action(observation)

        states = torch.as_tensor(observation, dtype=torch.long).unsqueeze(0).repeat(
            self.candidates, 1, 1
        )
        sequences = torch.randint(
            0, self.actions, (self.candidates, self.horizon), generator=self.generator
        )
        sequences[: self.actions, 0] = torch.arange(self.actions)

        scores = torch.zeros(self.candidates, dtype=torch.float32)
        selected_risk = torch.zeros(self.candidates, dtype=torch.float32)
        discount = 1.0
        for step in range(self.horizon):
            actions = sequences[:, step]
            logits, predicted_rewards, event_logits = self.model.forward_with_events(
                states, actions
            )
            states = logits.argmax(dim=-1)
            # Confidence is calculated from every cell's predicted distribution,
            # then averaged over the observation grid per candidate trajectory.
            confidence = torch.softmax(logits, dim=-1).amax(dim=-1)
            risk = 1.0 - confidence.flatten(start_dim=1).mean(dim=1)
            event_score = (
                torch.sigmoid(event_logits).cpu() @ self.event_weights
                if self.event_weights is not None
                else 0.0
            )
            step_score = predicted_rewards.cpu() + event_score
            scores += discount * (step_score - self.uncertainty_weight * risk.cpu())
            selected_risk += discount * risk.cpu()
            discount *= self.gamma
            self.imagined_steps += self.candidates

        best = int(torch.argmax(scores).item())
        self.last_selected_uncertainty = float(selected_risk[best])
        return int(sequences[best, 0].item())
