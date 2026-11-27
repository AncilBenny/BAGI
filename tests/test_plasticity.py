import copy

import pytest
import torch

from bagi.agents.plasticity import reset_adam_history, reset_control_head


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(3, 4)
        self.action_net = torch.nn.Linear(4, 9)
        self.optimizer = torch.optim.Adam(self.parameters())

    def step(self):
        self.optimizer.zero_grad()
        self.action_net(self.shared(torch.ones(1, 3))).square().sum().backward()
        self.optimizer.step()


def test_adam_reset_keeps_weights_and_optimizer_identity():
    p = Policy()
    p.step()
    optimizer = p.optimizer
    before = copy.deepcopy(p.state_dict())
    groups = copy.deepcopy(optimizer.state_dict()["param_groups"])
    assert reset_adam_history(p) == 4
    assert p.optimizer is optimizer and not optimizer.state
    assert optimizer.state_dict()["param_groups"] == groups
    assert all(torch.equal(v, p.state_dict()[k]) for k, v in before.items())
    p.step()
    assert all(s["step"].item() == 1 for s in optimizer.state.values())


def test_only_control_rows_change():
    p, fresh = Policy(), Policy()
    p.step()
    before = copy.deepcopy(p.state_dict())
    with pytest.raises(ValueError, match="Adam"):
        reset_control_head(p, fresh)
    reset_adam_history(p)
    reset_control_head(p, fresh)
    for k, v in p.state_dict().items():
        if k.startswith("action_net."):
            assert torch.equal(v[:7], before[k][:7])
            assert torch.equal(v[7:], fresh.state_dict()[k][7:])
        else:
            assert torch.equal(v, before[k])
