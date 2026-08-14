import numpy as np
import torch

from bagi.planning import ModelPredictivePlanner


class DummyWorldModel:
    def eval(self):
        return self

    def predict(self, observations, actions):
        # Reward action 2 most highly; preserve observation exactly.
        rewards = (actions == 2).float()
        return observations.clone(), rewards


def test_planner_uses_imagined_rewards():
    planner = ModelPredictivePlanner(
        model=DummyWorldModel(),
        actions=4,
        horizon=3,
        candidates=16,
        seed=1,
    )
    observation = np.zeros((5, 5), dtype=np.int8)
    action = planner.select_action(observation)
    assert action in range(4)
    assert planner.imagined_steps == 48


def test_planner_rejects_too_few_candidates():
    try:
        ModelPredictivePlanner(DummyWorldModel(), actions=4, candidates=3)
    except ValueError as exc:
        assert "candidates" in str(exc)
    else:
        raise AssertionError("expected ValueError")
