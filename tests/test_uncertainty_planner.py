"""Unit tests for uncertainty-aware action selection with matched search budgets."""

import pytest
import torch

from bagi.planning.model_predictive import ModelPredictivePlanner
from bagi.planning.uncertainty_aware import UncertaintyAwarePlanner


class ToyModel:
    def eval(self):
        return self

    def predict(self, observations, actions):
        return observations, torch.where(actions == 0, 0.30, 0.10).float()

    def forward_with_events(self, observations, actions):
        logits = torch.full((len(actions), 1, 1, 3), -20.0)
        logits[:, :, :, 0] = 20.0
        logits[actions == 0] = 0.0  # action 0 is more rewarding but uncertain
        rewards = torch.where(actions == 0, 0.30, 0.10).float()
        return logits, rewards, torch.zeros((len(actions), 4))


def test_zero_penalty_is_identical_to_baseline():
    model = ToyModel()
    baseline = ModelPredictivePlanner(model, actions=2, horizon=1, candidates=2, seed=17)
    no_penalty = UncertaintyAwarePlanner(
        model, actions=2, horizon=1, candidates=2, seed=17,
        uncertainty_weight=0.0,
    )
    observation = [[0]]
    assert baseline.select_action(observation) == 0
    assert no_penalty.select_action(observation) == 0
    assert baseline.imagined_steps == no_penalty.imagined_steps == 2
    assert no_penalty.last_selected_uncertainty is None


def test_confidence_penalty_can_change_preferred_action():
    model = ToyModel()
    planner = UncertaintyAwarePlanner(
        model, actions=2, horizon=1, candidates=2, seed=17,
        uncertainty_weight=1.0,
    )
    assert planner.select_action([[0]]) == 1
    assert planner.imagined_steps == 2
    assert planner.last_selected_uncertainty == pytest.approx(0.0, abs=1e-6)


def test_negative_uncertainty_weight_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        UncertaintyAwarePlanner(
            ToyModel(), actions=2, horizon=1, candidates=2,
            uncertainty_weight=-0.1,
        )
