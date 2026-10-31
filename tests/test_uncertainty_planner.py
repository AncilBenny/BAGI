"""Unit tests for uncertainty-aware action selection with matched search budgets."""

import pytest
import torch

from bagi.experiments.planner_uncertainty_ablation import paired_stage_statistics
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


def test_paired_statistics_exact_sign_flip_and_no_goal_positives():
    rows = []
    for seed in (401, 402, 403, 404):
        base = {"planner_seed": seed, "uncertainty_weight": 0.0}
        treated = {"planner_seed": seed, "uncertainty_weight": 1.0}
        for stage in ("switch", "door", "key", "lock", "goal"):
            base[f"{stage}_rate"] = 0.0
            treated[f"{stage}_rate"] = 1.0 if stage == "switch" else 0.0
        rows.extend((base, treated))
    result = paired_stage_statistics(rows, 1.0)
    assert result["switch"]["mean_paired_difference"] == 1.0
    assert result["switch"]["improved_seeds"] == 4
    assert result["switch"]["exact_two_sided_sign_flip_p"] == 0.125
    assert result["goal"]["tied_seeds"] == 4
    assert result["goal"]["exact_two_sided_sign_flip_p"] == 1.0


def test_paired_statistics_requires_identical_planner_seeds():
    base = {"uncertainty_weight": 0.0, "planner_seed": 401}
    treated = {"uncertainty_weight": 1.0, "planner_seed": 402}
    with pytest.raises(ValueError, match="matched"):
        paired_stage_statistics([base, treated], 1.0)
