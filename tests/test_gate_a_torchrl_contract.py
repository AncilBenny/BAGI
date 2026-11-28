"""TorchRL 0.15 compatibility contracts, not native learning evidence."""
import importlib.util

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torchrl") is None,
    reason="Requires the optional pinned recurrent replay extra",
)


class MaskValues(nn.Module):
    def forward(self, values, legal):
        return values.masked_fill(~legal, torch.finfo(values.dtype).min)


@pytest.mark.parametrize("terminated", [False, True])
def test_legal_bootstrap_and_actual_optimizer_step(terminated):
    from tensordict import TensorDict
    from tensordict.nn import TensorDictModule, TensorDictSequential
    from torchrl.modules import QValueModule
    from torchrl.objectives import DQNLoss, SoftUpdate

    layer = nn.Linear(994, 7)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.copy_(torch.tensor([1., 2., 100., 101., 102., 103., 104.]))
    policy = TensorDictSequential(
        TensorDictModule(layer, ["observation"], ["raw_values"]),
        TensorDictModule(MaskValues(), ["raw_values", "legal"], ["action_value"]),
        QValueModule(action_space="categorical"),
    )
    legal = torch.tensor([[True, True, False, False, False, False, False]])
    root = TensorDict(
        {"observation": torch.zeros(1, 994), "legal": legal}, batch_size=[1],
    )
    assert policy(root.clone())["action"].item() == 1
    loss_module = DQNLoss(
        policy, action_space="categorical", delay_value=True, double_dqn=True,
    )
    loss_module.make_value_estimator(gamma=0.99)
    updater = SoftUpdate(loss_module, eps=0.95)
    transition = root.clone()
    transition["action"] = torch.zeros(1, dtype=torch.long)
    transition["next"] = TensorDict(
        {
            "observation": torch.zeros(1, 994), "legal": legal.clone(),
            "reward": torch.tensor([[0.5]]), "done": torch.ones(1, 1, dtype=torch.bool),
            "terminated": torch.tensor([[terminated]]),
            "truncated": torch.tensor([[not terminated]]),
        },
        batch_size=[1],
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    before = layer.bias.detach().clone()
    loss = loss_module(transition)["loss"]
    target = 0.5 if terminated else 0.5 + 0.99 * 2.
    assert loss.item() == pytest.approx((1. - target) ** 2, abs=1e-5)
    optimizer.zero_grad()
    loss.backward()
    assert torch.count_nonzero(layer.bias.grad[2:]).item() == 0
    optimizer.step()
    updater.step()
    assert not torch.equal(layer.bias[0], before[0])
    torch.testing.assert_close(layer.bias[2:], before[2:], rtol=0, atol=0)


def test_recurrent_sequence_and_reset():
    from tensordict import TensorDict
    from torchrl.modules import LSTMModule, set_recurrent_mode

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        lstm = LSTMModule(
            input_size=994, hidden_size=8, in_key="observation", out_key="embedding",
        )
        observations = torch.randn(1, 4, 994)
    init = torch.tensor([[[True], [False], [True], [False]]])
    sequence = TensorDict(
        {
            "observation": observations,
            "is_init": init,
            "recurrent_state_h": torch.zeros(1, 4, 1, 8),
            "recurrent_state_c": torch.zeros(1, 4, 1, 8),
        },
        batch_size=[1, 4],
    )
    with torch.no_grad(), set_recurrent_mode("recurrent"):
        expected = lstm(sequence)["embedding"]
    h = c = torch.zeros(1, 1, 8)
    results = []
    with torch.no_grad(), set_recurrent_mode(False):
        for i in range(4):
            row = TensorDict(
                {"observation": observations[:, i], "is_init": init[:, i],
                 "recurrent_state_h": h, "recurrent_state_c": c},
                batch_size=[1],
            )
            out = lstm(row)
            results.append(out["embedding"])
            h = out["next", "recurrent_state_h"]
            c = out["next", "recurrent_state_c"]
        fresh = lstm(TensorDict(
            {"observation": observations[:, 2], "is_init": init[:, 2],
             "recurrent_state_h": torch.zeros_like(h),
             "recurrent_state_c": torch.zeros_like(c)},
            batch_size=[1],
        ))["embedding"]
    torch.testing.assert_close(torch.stack(results, dim=1), expected)
    torch.testing.assert_close(results[2], fresh)


def test_replay_contiguous_slices_and_rng_restart(tmp_path):
    from tensordict import TensorDict
    from torchrl.data import LazyTensorStorage, SliceSampler, TensorDictReplayBuffer

    episodes = torch.arange(3).repeat_interleave(4)
    positions = torch.arange(4).repeat(3)
    data = TensorDict(
        {"episode": episodes, "position": positions,
         "is_init": (positions == 0).unsqueeze(-1),
         "next": TensorDict({"done": (positions == 3).unsqueeze(-1)}, batch_size=[12])},
        batch_size=[12],
    )

    def make_buffer(generator):
        buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(32),
            sampler=SliceSampler(traj_key="episode", slice_len=4, strict_length=True),
            batch_size=8, generator=generator,
        )
        buffer.extend(data.clone())
        return buffer

    rng = torch.Generator().manual_seed(23)
    original = make_buffer(rng)
    original.sample()
    path = tmp_path / "rng.pt"
    torch.save(rng.get_state(), path)
    expected = original.sample()
    restored_rng = torch.Generator()
    restored_rng.set_state(torch.load(path, weights_only=True))
    restored = make_buffer(restored_rng).sample()
    for key in ("episode", "position", "is_init"):
        torch.testing.assert_close(expected[key], restored[key], rtol=0, atol=0)
    assert expected["episode"].reshape(2, 4).unique(dim=1).shape[1] == 1
    assert torch.equal(expected["position"].reshape(2, 4), torch.arange(4).repeat(2, 1))


def test_native_public_interface_and_executed_action():
    from bagi.experiments.cumulative_transfer import TASKS
    from bagi.experiments.gate_a_competence import NativePPOEnv
    from bagi.experiments.gate_a_shared_interface import WIDTH, SharedActuators

    for index, domain in enumerate(TASKS):
        native = NativePPOEnv(domain, 83_000_000 + index)
        env = SharedActuators(native)
        try:
            observation, _ = env.reset()
            assert observation.shape == (WIDTH,)
            assert env.observation_space.contains(observation)
            for i in range(4):
                action = i % (7 if index == 0 else 2)
                proposal = [action, 0] if index == 0 else [0, action]
                observation, _, terminated, truncated, _ = env.step(proposal)
                assert native.episode_actions[-1] == action
                assert env.observation_space.contains(observation)
                if terminated or truncated:
                    break
            assert native.steps == i + 1
        finally:
            env.close()
