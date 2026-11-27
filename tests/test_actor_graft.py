import pytest
import torch

from bagi.agents.actor_graft import MODES, actor_graft, temporary_actor


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_actor = torch.nn.LSTM(4, 3)
        self.lstm_critic = torch.nn.LSTM(4, 3)
        self.mlp_extractor = torch.nn.Module()
        self.mlp_extractor.policy_net = torch.nn.Linear(3, 2)
        self.mlp_extractor.value_net = torch.nn.Linear(3, 2)
        self.action_net = torch.nn.Linear(2, 9)
        self.value_net = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.Adam(self.parameters())


def snapshot(policy):
    return {k: v.detach().clone() for k, v in policy.state_dict().items()}


@pytest.mark.parametrize("mode", MODES)
def test_graft_exact_component_scope(mode):
    p, donor = Policy(), Policy()
    before, source = snapshot(p), snapshot(donor)
    result = actor_graft(before, source, mode)
    for k, v in result.items():
        if (k.startswith("lstm_actor.") and mode in ("A_lstm", "A_path", "A_actor")
                or k.startswith("mlp_extractor.policy_net.")
                and mode in ("A_mlp", "A_path", "A_actor")):
            assert torch.equal(v, source[k])
        elif k.startswith("action_net.") and mode in ("A_head", "A_path", "A_actor"):
            n = 9 if mode == "A_actor" else 7
            assert torch.equal(v[:n], source[k][:n])
            assert torch.equal(v[n:], before[k][n:])
        else:
            assert torch.equal(v, before[k])
        assert v.data_ptr() != before[k].data_ptr()
        assert torch.equal(before[k], p.state_dict()[k])
        assert torch.equal(source[k], donor.state_dict()[k])


def test_restore_on_exception_and_preserve_optimizer():
    p, donor = Policy(), Policy()
    before = snapshot(p)
    opt = p.optimizer
    ids = [id(v) for v in p.parameters()]
    with pytest.raises(RuntimeError, match="probe failed"), temporary_actor(
        p, snapshot(donor), "A_actor"
    ):
        assert torch.equal(p.action_net.weight, donor.action_net.weight)
        raise RuntimeError("probe failed")
    assert all(torch.equal(v, p.state_dict()[k]) for k, v in before.items())
    assert p.optimizer is opt and ids == [id(v) for v in p.parameters()]


def test_invalid_graft_does_not_mutate_recipient():
    p, donor = Policy(), Policy()
    before, source = snapshot(p), snapshot(donor)
    source["action_net.bias"][0] = float("nan")
    with pytest.raises(ValueError, match="Incompatible"), temporary_actor(p, source, "A_head"):
        pytest.fail("Invalid graft entered probe")
    assert all(torch.equal(v, p.state_dict()[k]) for k, v in before.items())
