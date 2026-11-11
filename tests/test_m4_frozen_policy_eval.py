"""Regression contracts for frozen M3 restoration and independent M4 episodes."""
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from bagi.experiments import m4_frozen_policy_eval as m4


def test_m4_config_is_exact_frozen_study():
    config = yaml.safe_load(Path("configs/m4_frozen_policy.yaml").read_text())
    assert m4.validate_config(config) == config
    for field, replacement in (
        ("sampled_repeats", 3),
        ("model_seeds", [17, 29]),
        ("evaluation_seeds", list(range(96001, 96013))),
        ("output_dir", "runs/m3_goal_learning"),
    ):
        changed = {**config, field: replacement}
        with pytest.raises(ValueError, match="Registered M4"):
            m4.validate_config(changed)


def test_m4_rng_seeds_are_unique_and_outcome_independent():
    seeds = []
    for arm in m4.ARMS:
        for model_seed in m4.MODEL_SEEDS:
            for task, layouts in (
                (m4.TRAIN_TASK, m4.EVAL_SEEDS),
                (m4.TRANSFER_TASK, m4.TRANSFER_SEEDS),
            ):
                for layout in layouts:
                    for repeat in range(1, m4.REPEATS + 1):
                        seeds.append(m4.action_seed(
                            arm, model_seed, task, layout, repeat
                        ))
    assert len(seeds) == len(set(seeds)) == 8 * 2 * 12 * 5
    with pytest.raises(ValueError):
        m4.action_seed("recurrent", 17, m4.TRAIN_TASK, 98001, 1)
    with pytest.raises(ValueError):
        m4.action_seed("recurrent", 37, m4.TRAIN_TASK, 96001, 1)


def test_m4_numeric_load_rejects_wrong_digest_shape_dtype_and_pickle(tmp_path):
    model = torch.nn.Linear(2, 1, bias=False)

    class Agent:
        policy = model

    arr = model.state_dict()["weight"].detach().cpu().numpy()
    path = tmp_path / "policy.npz"
    np.savez_compressed(path, tensor_000=arr)
    manifest = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "allow_pickle": False,
        "members": {"tensor_000": {
            "name": "weight", "shape": list(arr.shape), "dtype": str(arr.dtype)
        }},
    }
    assert m4.restore_numeric_policy(Agent(), path, manifest) == manifest["sha256"]
    with pytest.raises(ValueError, match="digest"):
        m4.restore_numeric_policy(Agent(), path, {**manifest, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="digest/pickle"):
        m4.restore_numeric_policy(Agent(), path, {**manifest, "allow_pickle": True})
    changed = {**manifest, "members": {
        "tensor_000": {"name": "weight", "shape": [1, 3], "dtype": "float32"}
    }}
    with pytest.raises(ValueError, match="shape"):
        m4.restore_numeric_policy(Agent(), path, changed)


def test_m4_native_goal_scoring_counts_actions_and_resets(monkeypatch):
    class ActionSpace:
        n = 7

        def seed(self, seed):
            self.seed_value = seed

    class Env:
        def __init__(self, reward):
            self.action_space = ActionSpace()
            self.reward = reward
            self.native_steps = 0
            self.native_resets = 0

        def reset(self, seed):
            self.native_resets += 1
            return np.zeros(2), {}

        def step(self, action):
            assert action == 2
            self.native_steps += 1
            return np.zeros(2), self.reward, True, False, {}

        def close(self):
            pass

    class Agent:
        def predict(self, observation, state, episode_start, deterministic):
            assert episode_start.tolist() == [True]
            return np.array([2]), None

    for reward, success in ((1.0, 1), (0.0, 0)):
        monkeypatch.setattr(m4, "make_task", lambda *args, reward=reward, **kwargs: Env(reward))
        meter = {"independent_actions": 0, "independent_resets": 0}
        episode = m4.evaluate_episode(
            Agent(), "reactive", m4.TRAIN_TASK, 98001,
            sampled_seed=None, meter=meter, reference=False,
        )
        assert episode["goal_success"] == success
        assert episode["executed_action_histogram"][2] == 1
        assert meter == {"independent_actions": 1, "independent_resets": 1}


def test_m4_recurrent_episode_state_resets_then_carries(monkeypatch):
    class ActionSpace:
        n = 7

        def seed(self, seed):
            pass

    class Env:
        action_space = ActionSpace()

        def __init__(self):
            self.native_steps = 0
            self.native_resets = 0

        def reset(self, seed):
            self.native_resets += 1
            return np.zeros(2), {}

        def step(self, action):
            self.native_steps += 1
            return np.zeros(2), 1.0 if self.native_steps == 2 else 0.0, (
                self.native_steps == 2
            ), False, {}

        def close(self):
            pass

    class Agent:
        def __init__(self):
            self.seen = []

        def predict(self, observation, state, episode_start, deterministic):
            self.seen.append((bool(episode_start[0]), state))
            return np.array([2]), (1 if state is None else state + 1)

    monkeypatch.setattr(m4, "make_task", lambda *args, **kwargs: Env())
    agent = Agent()
    meter = {"reference_actions": 0, "reference_resets": 0}
    result = m4.evaluate_episode(
        agent, "recurrent", m4.TRAIN_TASK, 96001,
        sampled_seed=None, meter=meter, reference=True,
    )
    assert result["goal_success"] == 1
    assert agent.seen == [(True, None), (False, 1)]
    assert meter == {"reference_actions": 2, "reference_resets": 1}
