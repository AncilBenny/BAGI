import copy

import pytest
import torch

from bagi.agents.critic_reset import PREFIXES, reset_critic


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_actor = torch.nn.LSTM(3, 4)
        self.lstm_critic = torch.nn.LSTM(3, 4)
        self.mlp_extractor = torch.nn.Module()
        self.mlp_extractor.policy_net = torch.nn.Linear(4, 2)
        self.mlp_extractor.value_net = torch.nn.Linear(4, 2)
        self.action_net = torch.nn.Linear(2, 9)
        self.value_net = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.Adam(self.parameters())


def test_critic_only_reset_preserves_actor_and_identity():
    torch.manual_seed(19)
    p, fresh = Policy(), Policy()
    before = copy.deepcopy(p.state_dict())
    ids = {k: id(v) for k, v in p.named_parameters()}
    opt = p.optimizer
    keys = reset_critic(p, fresh)
    assert set(keys) == {k for k in before if k.startswith(PREFIXES)}
    for k, v in p.state_dict().items():
        assert torch.equal(v, fresh.state_dict()[k] if k in keys else before[k])
    assert p.optimizer is opt and ids == {k: id(v) for k, v in p.named_parameters()}
    assert not opt.state


def test_reset_rejects_stale_adam_and_shared_critic():
    p, fresh = Policy(), Policy()
    p.value_net(torch.ones(1, 2)).sum().backward()
    p.optimizer.step()
    with pytest.raises(ValueError, match="Adam"):
        reset_critic(p, fresh)
    p.optimizer.state.clear()
    p.lstm_critic = p.lstm_actor
    with pytest.raises(ValueError, match="independent"):
        reset_critic(p, fresh)


def test_reset_rejects_bad_source_without_partial_change():
    p, fresh = Policy(), Policy()
    before = copy.deepcopy(p.state_dict())
    with torch.no_grad():
        fresh.value_net.bias.fill_(float("nan"))
    with pytest.raises(ValueError, match="Incompatible"):
        reset_critic(p, fresh)
    assert all(torch.equal(v, p.state_dict()[k]) for k, v in before.items())
