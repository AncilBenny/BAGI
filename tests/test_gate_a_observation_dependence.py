import numpy as np
import torch

from bagi.experiments.gate_a_observation_dependence import StaleImageActor, tensor_tree_equal


class RecordingModel:
    def __init__(self):
        self.seen = []

    def predict(self, obs, *, state, episode_start, deterministic):
        self.seen.append(obs.copy())
        return np.array(2), np.zeros(1)


def test_stale_view_preserves_actual_action_history_and_resets():
    model = RecordingModel()
    actor = StaleImageActor(model, deterministic=False)
    actor.begin_episode()
    first = np.zeros(161, dtype=np.float32)
    second = np.ones(161, dtype=np.float32)
    actor.act(first, 7, epsilon=0)
    actor.act(second, 7, epsilon=0)
    assert np.array_equal(model.seen[1][:147], first[:147])
    assert np.array_equal(model.seen[1][147:], second[147:])
    assert np.all(second == 1)
    actor.begin_episode()
    actor.act(second, 7, epsilon=0)
    assert np.array_equal(model.seen[2], second)


def test_tensor_tree_equality_detects_adam_moment_change():
    a = {"state": {0: {"exp_avg": torch.zeros(2)}}, "groups": [1, 2]}
    b = {"state": {0: {"exp_avg": torch.ones(2)}}, "groups": [1, 2]}
    assert tensor_tree_equal(a, a)
    assert not tensor_tree_equal(a, b)
