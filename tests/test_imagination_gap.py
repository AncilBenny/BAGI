"""Pure score and rollout-accounting checks for the E6 imagination diagnostic."""
import torch

from bagi.experiments.imagination_gap import (
    evaluate_trajectory,
    outcome_errors,
    summarize_event_errors,
)


class IdentityModel:
    @torch.no_grad()
    def predict_events(self, observations, actions):
        return observations, torch.zeros(len(actions)), torch.zeros((len(actions), 4))


class IncrementModel:
    @torch.no_grad()
    def predict_events(self, observations, actions):
        return observations + 1, torch.zeros(len(actions)), torch.zeros((len(actions), 4))


def test_outcome_errors_perfect_prediction():
    obs = torch.tensor([[0, 1], [2, 3]])
    errors = outcome_errors(obs, obs, 0.0, 0.0, [0, 0, 0, 0], set())
    assert errors["cell_accuracy"] == 1.0
    assert errors["whole_observation_accuracy"] == 1.0
    assert errors["reward_mae"] == 0.0
    assert errors["event_brier"] == 0.0


def test_event_false_positives_are_measured():
    obs = torch.zeros((2, 2), dtype=torch.long)
    errors = outcome_errors(obs, obs, 0, 0, [1, 0, 0, 0], set())
    assert errors["event_brier"] == 0.25
    assert errors["false_event_probability"] == 0.25


def test_rollout_metrics_count_horizons_and_observation_drift():
    zero = [[0, 0], [0, 0]]
    one = [[1, 0], [0, 0]]
    trajectory = [
        {"observation": zero, "action": 0, "next_observation": zero,
         "reward": 0.0, "events": set()},
        {"observation": zero, "action": 1, "next_observation": one,
         "reward": 0.0, "events": set()},
    ]
    result = evaluate_trajectory(IdentityModel(), trajectory, [1, 2], 1, 5)
    assert result[1]["comparisons"] == 2
    assert result[2]["comparisons"] == 1
    assert result[2]["whole_observation_accuracy"] == 0.0
    assert result[2]["cell_accuracy"] == 0.75


def test_event_positive_brier_is_not_hidden_by_rare_events():
    obs = torch.zeros((2, 2), dtype=torch.long)
    missed_event = outcome_errors(
        obs, obs, 0, 0, [0, 0, 0, 0], {"switch_activated"}
    )
    no_event = outcome_errors(obs, obs, 0, 0, [0, 0, 0, 0], set())
    summary = summarize_event_errors([missed_event, no_event])
    assert missed_event["event_brier"] == 0.25
    assert summary["positive_event_count"] == 1
    assert summary["negative_event_count"] == 7
    assert summary["positive_event_brier"] == 1.0
    assert summary["negative_event_brier"] == 0.0


def test_event_positive_brier_is_null_without_positive_labels():
    obs = torch.zeros((2, 2), dtype=torch.long)
    no_event = outcome_errors(obs, obs, 0, 0, [0, 0, 0, 0], set())
    summary = summarize_event_errors([no_event])
    assert summary["positive_event_count"] == 0
    assert summary["positive_event_brier"] is None
    assert summary["negative_event_count"] == 4
    assert summary["negative_event_brier"] == 0.0


def test_teacher_forced_vs_recursive_use_identical_actions_and_targets():
    zero = [[0, 0], [0, 0]]
    one = [[1, 1], [1, 1]]
    trajectory = [
        {"observation": zero, "action": 0, "next_observation": zero,
         "reward": 0.0, "events": set()},
        {"observation": zero, "action": 0, "next_observation": one,
         "reward": 0.0, "events": set()},
    ]
    scores = evaluate_trajectory(IncrementModel(), trajectory, [1, 2], 1, 1)
    assert scores[1]["comparisons"] == scores[2]["comparisons"] == 1
    assert scores[1]["whole_observation_accuracy"] == 0.0
    assert scores[1]["teacher_forced_whole_observation_accuracy"] == 0.0
    assert scores[2]["whole_observation_accuracy"] == 0.0
    assert scores[2]["teacher_forced_whole_observation_accuracy"] == 1.0
    assert scores[2]["teacher_forced_cell_accuracy"] == 1.0
    assert scores[2]["cell_accuracy"] == 0.0
