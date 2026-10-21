"""Information-matched reachable-graph holdout tests."""

import torch

from bagi.experiments.dataset_foundations import model_input
from bagi.experiments.reachable_state_holdout import (
    enumerate_reachable,
    make_groups,
    score_transitions,
    split_training_keys,
)
from bagi.worlds import WorkshopEnv


def test_reachable_graph_determinism_and_full_causal_event_support():
    env = WorkshopEnv(size=9, max_steps=100, view_size=5)
    first, meta = enumerate_reachable(env)
    second, same = enumerate_reachable(env)
    assert first == second
    assert meta == same
    assert meta["state_action_transitions"] == len(first)
    assert len(first) == meta["reachable_nonterminal_states"] * 4
    positives = {event for row in first for event in row["info"]["events"]}
    assert {"switch_activated", "key_picked_up", "lock_opened", "goal_reached"} <= positives


def test_train_holdout_keys_use_the_training_seed_and_fraction():
    rows = [
        {"observation": [[i]], "action": 0} for i in range(10)
    ]
    seen = split_training_keys(rows, seed=42, fraction=0.2)
    assert len(seen) == 8
    assert seen == split_training_keys(rows, seed=42, fraction=0.2)
    groups = make_groups(rows, seen)
    assert len(groups["seen_in_training"]) == 8
    assert len(groups["unseen_observation_action"]) == 2


class RecordingWorldModel:
    def __init__(self):
        self.calls = []

    def forward_with_events(self, observations, actions):
        assert observations.ndim == 3
        assert actions.ndim == 1
        self.calls.append((tuple(observations.shape), tuple(actions.shape)))
        batch, height, width = observations.shape
        logits = torch.zeros(batch, height, width, 9)
        return logits, torch.zeros(batch), torch.zeros(batch, 4)


def test_evaluator_does_not_pass_privileged_state_to_world_model():
    model = RecordingWorldModel()
    row = {
        "observation": [[0]], "action": 1, "next_observation": [[0]],
        "reward": 0.0, "info": {"events": ["switch_activated"],
                              "position": [1, 2], "has_key": True},
    }
    metrics = score_transitions(model, [row])
    assert metrics["whole_observation_accuracy"] == 1.0
    assert metrics["positive_event_labels"] == 1
    assert metrics["event_positive_counts"]["switch_activated"] == 1
    assert metrics["event_true_positives_at_threshold_0_5"]["switch_activated"] == 1
    assert model.calls == [((1, 1, 1), (1,))]


def test_absent_positive_event_labels_are_null_not_zero():
    model = RecordingWorldModel()
    row = {
        "observation": [[0]], "action": 0, "next_observation": [[0]],
        "reward": 0.0, "info": {"events": []},
    }
    metrics = score_transitions(model, [row])
    assert metrics["positive_event_labels"] == 0
    assert metrics["positive_event_brier"] is None
    assert metrics["negative_event_labels"] == 4
