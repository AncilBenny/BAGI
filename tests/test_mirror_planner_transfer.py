"""Tests for model-predictive mirror adapter on exposed observations and actions."""
import torch

from bagi.experiments.mirror_planner_transfer import MirrorPlannerModel
from bagi.planning.model_predictive import ModelPredictivePlanner


class Dummy:
    def eval(self):
        return self

    def forward_with_events(self, observations, actions):
        batch, height, width = observations.shape
        logits = torch.full((batch, height, width, 9), -20.0)
        logits[..., 0] = 20.0
        return logits, torch.ones(batch), torch.zeros(batch, 4)


def test_planner_adapter_reuses_model_without_privileged_state():
    model = MirrorPlannerModel(Dummy())
    obs = torch.zeros(1, 5, 5, dtype=torch.long)
    predicted, reward, events = model.predict_events(obs, torch.tensor([1]))
    assert predicted.shape == obs.shape
    assert reward.tolist() == [1.0]
    assert events.shape == (1, 4)
    planner = ModelPredictivePlanner(
        model=model, actions=4, horizon=2, candidates=4,
        seed=7, event_weights=[0.05, 0.1, 0.2, 1.0],
    )
    assert planner.select_action(obs[0]) in range(4)
    assert planner.imagined_steps == 8
