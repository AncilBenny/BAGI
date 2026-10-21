"""Unit tests for the exact horizontal-mirror canonicalization wrapper."""
import torch

from bagi.experiments.symmetry_canonicalization import (
    CanonicalMirrorModel,
    mirror_action,
)


def test_mirror_action_is_involution():
    actions = torch.tensor([0, 1, 2, 3])
    assert mirror_action(actions).tolist() == [0, 3, 2, 1]
    assert torch.equal(mirror_action(mirror_action(actions)), actions)


class Recorder:
    def forward_with_events(self, obs, actions):
        self.inputs = (obs.clone(), actions.clone())
        batch, h, w = obs.shape
        logits = torch.arange(w, dtype=torch.float32).view(1, 1, w, 1).expand(
            batch, h, w, 9
        ).clone()
        return logits, torch.ones(batch), torch.zeros(batch, 4)


def test_wrapper_mirrors_observation_action_and_next_view_only():
    model = Recorder()
    wrapper = CanonicalMirrorModel(model)
    visible = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    logits, reward, event = wrapper.forward_with_events(visible, torch.tensor([1]))
    assert torch.equal(model.inputs[0], visible.flip(dims=(-1,)))
    assert model.inputs[1].tolist() == [3]
    assert logits[0, 0, :, 0].tolist() == [2, 1, 0]
    assert reward.tolist() == [1.0]
    assert event.shape == (1, 4)
