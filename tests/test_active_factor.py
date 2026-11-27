import pytest
import torch
from torch.distributions import Categorical

from bagi.agents.active_factor import active_statistics


@pytest.mark.parametrize('active', [0, 1])
def test_inactive_logits_have_no_direct_gradient(active):
    torch.manual_seed(18)
    logits = torch.randn(12, 9, requires_grad=True)
    actions = torch.stack([torch.randint(7, (12,)), torch.randint(2, (12,))], dim=1)
    lp, h = active_statistics(logits, actions, active)
    (lp.mean() + 0.1 * h.mean()).backward()
    start, end = (0, 7) if active == 0 else (7, 9)
    inactive = torch.ones(9, dtype=torch.bool)
    inactive[start:end] = False
    assert torch.count_nonzero(logits.grad[:, inactive]) == 0
    assert torch.count_nonzero(logits.grad[:, ~inactive]) > 0
    expected = Categorical(logits=logits[:, start:end].detach())
    torch.testing.assert_close(lp, expected.log_prob(actions[:, active]))
    torch.testing.assert_close(h, expected.entropy())


@pytest.mark.parametrize('active', [0, 1])
def test_ratio_ignores_inactive_changes(active):
    torch.manual_seed(19)
    old = torch.randn(6, 9)
    new = old.clone()
    inactive = slice(7, 9) if active == 0 else slice(0, 7)
    new[:, inactive] = torch.randn_like(new[:, inactive]) * 4
    actions = torch.tensor([[0, 1], [1, 0], [2, 1], [3, 0], [4, 1], [6, 0]])
    before, entropy = active_statistics(old, actions, active)
    after, new_entropy = active_statistics(new, actions, active)
    assert torch.equal(before, after)
    assert torch.equal(entropy, new_entropy)
    assert torch.equal((after - before).exp(), torch.ones(6))
    def joint(x):
        return sum(Categorical(logits=v).log_prob(actions[:, i])
                   for i, v in enumerate(x.split((7, 2), dim=1)))
    assert not torch.equal(joint(old), joint(new))


@pytest.mark.parametrize('active', [-1, 2, True])
def test_invalid_domain_rejected(active):
    with pytest.raises(ValueError, match='actuator'):
        active_statistics(torch.zeros(2, 9), torch.zeros(2, 2), active)


@pytest.mark.parametrize('bad', [float('nan'), 0.5, 7.0, -1.0])
def test_invalid_actions_rejected(bad):
    actions = torch.zeros(2, 2)
    actions[0, 0] = bad
    with pytest.raises(ValueError):
        active_statistics(torch.zeros(2, 9), actions, 0)
