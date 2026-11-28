"""Persistent recurrent replay integration contracts; no scientific claim."""
import importlib.util

import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torchrl") is None,
    reason="Requires pinned recurrent replay extra",
)


def sample(learner, domain, index, *, terminated=False):
    from bagi.agents.torchrl_recurrent import WIDTH

    generator = torch.Generator().manual_seed(index + 111)
    n = 2 + index % 3
    observations = torch.randn(n + 1, WIDTH, generator=generator)
    actions = torch.arange(n) % (7 if domain == 0 else 2)
    terminal = torch.zeros(n, dtype=torch.bool)
    truncated = terminal.clone()
    terminal[-1] = terminated
    truncated[-1] = not terminated
    learner.remember(domain, observations, actions, torch.arange(n).float(),
                     terminal, truncated)


def equal_nested(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_nested(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal_nested(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("terminated", [False, True])
def test_recurrent_target_matches_current_history(terminated):
    from bagi.agents.torchrl_recurrent import (
        PersistentRecurrentDQN,
        transition_batch,
    )

    learner = PersistentRecurrentDQN(41, capacity=3, hidden=8)
    sample(learner, 1, 1, terminated=terminated)
    learner.update(1)
    row = learner.replay[1].sample()[0]
    batch = transition_batch(row, 1)
    n = int(row["length"])
    obs = row["observations"][:n + 1].unsqueeze(0)
    # Independent oracle: unroll all T+1 observations from zero, for BOTH networks.
    with torch.no_grad():
        online = learner.net.head(learner.net.lstm(obs)[0])[0, :, :2]
        with learner.loss.target_value_network_params.to_module(learner.policy):
            target = learner.net.head(learner.net.lstm(obs)[0])[0, :, :2].clone()
        predicted = online[:-1].gather(-1, row["actions"][:n, None]).squeeze(-1)
        selected = online[1:].argmax(-1, keepdim=True)
        bootstrap = target[1:].gather(-1, selected).squeeze(-1)
        expected = row["reward"][:n] + 0.99 * (~row["terminated"][:n]) * bootstrap
        assert learner.loss(batch)["loss"].item() == pytest.approx(
            (predicted - expected).square().mean().item(), abs=1e-5,
        )


def test_wrapped_replay_full_disk_resume(tmp_path):
    from bagi.agents.torchrl_recurrent import WIDTH, PersistentRecurrentDQN

    torch.set_num_threads(1)
    original = PersistentRecurrentDQN(43, capacity=3, hidden=8)
    for index in range(8):
        sample(original, index % 2, index)
    assert [len(buffer) for buffer in original.replay] == [3, 3]
    original.update()
    original.update()
    path = tmp_path / "checkpoint"
    original.save(path)
    restored = PersistentRecurrentDQN.load(path)
    equal_nested(original.loss.state_dict(), restored.loss.state_dict())
    equal_nested(original.optimizer.state_dict(), restored.optimizer.state_dict())
    for learner in (original, restored):
        sample(learner, 0, 8)
    for domain in (0, 1):
        for i in range(3):
            x = torch.full((WIDTH,), i / 10)
            assert original.act(x, domain, epsilon=0.5) == restored.act(
                x, domain, epsilon=0.5,
            )
        original.begin_episode()
        restored.begin_episode()
    for _ in range(3):
        assert original.update() == restored.update()
        assert original.last_samples == restored.last_samples
    equal_nested(original.loss.state_dict(), restored.loss.state_dict())
    equal_nested(original.optimizer.state_dict(), restored.optimizer.state_dict())
    for key in ("episodes", "training_actions", "updates", "presentations"):
        assert getattr(original, key) == getattr(restored, key)
    with pytest.raises(FileExistsError):
        original.save(path)
    original.act(torch.zeros(WIDTH), 0)
    with pytest.raises(ValueError):
        original.act(torch.zeros(WIDTH), 1)
    with pytest.raises(ValueError):
        original.save(tmp_path / "dirty")


def test_native_short_prefix_replay():
    from bagi.agents.torchrl_recurrent import PersistentRecurrentDQN
    from bagi.experiments.cumulative_transfer import TASKS
    from bagi.experiments.gate_a_competence import NativePPOEnv
    from bagi.experiments.gate_a_shared_interface import SharedActuators

    learner = PersistentRecurrentDQN(47, capacity=3, hidden=8)
    for domain, task in enumerate(TASKS):
        env = SharedActuators(NativePPOEnv(task, 84_000_000 + domain))
        try:
            obs, _ = env.reset()
            observations, actions, rewards, terms, truncs = [obs], [], [], [], []
            for i in range(4):
                action = learner.act(obs, domain, epsilon=0.2)
                proposal = [action, 0] if domain == 0 else [0, action]
                obs, reward, term, trunc, _ = env.step(proposal)
                assert env.env.episode_actions[-1] == action
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                terms.append(term)
                truncs.append(trunc or (i == 3 and not term))
                if term or trunc:
                    break
            learner.begin_episode()
            # Stack before converting to tensor (avoid list-of-arrays warning).
            import numpy as np

            learner.remember(domain, np.stack(observations), actions, rewards, terms, truncs)
        finally:
            env.close()
    assert learner.training_actions == sum(
        int(buffer.sample()[0]["length"]) for buffer in learner.replay
    )
    assert learner.updates == 0
    assert learner.update(2) >= 0
    assert learner.updates == 1
    assert learner.presentations == learner.training_actions
