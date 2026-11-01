"""G2-E4-P2 synthetic action, RNG, provenance and paired-goal guards; no MiniGrid."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch

from bagi.experiments import e4_p2_frozen_contract as p2
from bagi.experiments import e4_p2_frozen_runner as runner


class Space:
    n: ClassVar[int] = 7

    def contains(self, action):
        return isinstance(action, (int, np.integer)) and 0 <= action < self.n

    def seed(self, seed):
        self.seed_value = seed


class Grid:
    def __init__(self, seed):
        self.seed = seed

    def encode(self):
        return np.full((3, 3, 3), self.seed % 251, dtype=np.uint8)


class World:
    def __init__(self):
        self.action_space = Space()
        self.unwrapped = self
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        self.grid = Grid(0)
        self.steps = 0

    def reset(self, *, seed):
        self.steps = 0
        self.grid = Grid(seed)
        return np.array([seed % 10, 1, 2], dtype=np.uint8), {"private": 999}

    def step(self, action):
        self.steps += 1
        done = self.steps == 3
        return (np.array([self.steps, action, 2], dtype=np.uint8),
                float(done and action == 2), done, False, {"private": 999})


class Agent:
    def __init__(self):
        self.flags = []

    def predict(self, public, deterministic):
        assert public.ndim == 1 and public.shape == (3,)
        self.flags.append(deterministic)
        action = 2 if deterministic else int(torch.randint(0, 7, ()).item())
        return np.asarray(action), None


def test_frozen_public_only_greedy_native_accounting():
    agent, env = Agent(), World()
    ledger = {"test_actions": 0}
    row, trace = runner.evaluate_episode(
        env, p2.TASKS[0], 27000, "greedy_trained", 17, 0, 0, agent, ledger,
    )
    assert ledger["test_actions"] == env.steps == row["native_actions"] == 3
    assert row["goal_success"] == 1
    assert row["action_histogram"][2] == 3
    assert trace["public_pre_action"].shape == (3, 3)
    assert agent.flags == [True, True, True]
    assert row["action_sampling_seed"] == p2.episode_rng_seed(0, 0, 0, 27000)


def test_stochastic_torch_episode_stream_is_reproducible_and_independent():
    a, b = Agent(), Agent()
    first, t1 = runner.evaluate_episode(
        World(), p2.TASKS[0], 27001, "stochastic_trained", 17, 0, 0,
        a, {"test_actions": 0},
    )
    second, t2 = runner.evaluate_episode(
        World(), p2.TASKS[0], 27001, "stochastic_trained", 17, 0, 0,
        b, {"test_actions": 0},
    )
    assert a.flags == b.flags == [False] * 3
    assert first == second
    np.testing.assert_array_equal(t1["actions"], t2["actions"])
    random, _ = runner.evaluate_episode(
        World(), p2.TASKS[0], 27001, "uniform_random", 17, 0, 0,
        None, {"test_actions": 0},
    )
    assert random["action_sampling_seed"] != first["action_sampling_seed"]


def test_reject_invalid_policy_assignment_and_action_budget():
    with pytest.raises(ValueError, match="assignment"):
        runner.evaluate_episode(
            World(), p2.TASKS[0], 27000, "stochastic_trained", 17, 0, 0,
            None, {"test_actions": 0},
        )
    with pytest.raises(AssertionError, match="ceiling"):
        runner.evaluate_episode(
            World(), p2.TASKS[0], 27000, "greedy_trained", 17, 0, 0,
            Agent(), {"test_actions": p2.MAX_TEST_ACTIONS},
        )


def _paired_rows(task, model_index=0):
    rows = {}
    for arm_index, arm in enumerate(p2.ARMS):
        episodes = []
        for seed in p2.TEST_SEEDS:
            success = (arm == "stochastic_trained" and seed < 27005)
            episodes.append({
                "reset_seed": seed, "model_index": model_index,
                "evaluator_layout_sha256": str(seed),
                "initial_public_sha256": str(seed),
                "native_actions": 3, "action_histogram": [3, 0, 0, 0, 0, 0, 0],
                "native_reward": 1.0 if success else 0.0,
                "native_terminated": True, "goal_success": int(success),
                "action_sampling_seed": p2.episode_rng_seed(
                    p2.TASKS.index(task), model_index, arm_index, seed,
                ),
            })
        rows[arm] = episodes
    return rows


def test_paired_gate_requires_native_goals_and_each_control():
    rows = _paired_rows(p2.TASKS[0])
    scored = runner.score(p2.TASKS[0], rows)
    assert scored["per_seed_gate"]
    assert scored["native_successes"]["stochastic_trained"] == 5
    assert scored["trained_stochastic_min_gain_over_all_controls"] == 5
    rows["uniform_random"][0]["goal_success"] = 1
    rows["uniform_random"][0]["native_reward"] = 1.0
    assert runner.score(p2.TASKS[0], rows)["per_seed_gate"]
    rows["uniform_random"][1]["goal_success"] = 1
    rows["uniform_random"][1]["native_reward"] = 1.0
    rows["uniform_random"][2]["goal_success"] = 1
    rows["uniform_random"][2]["native_reward"] = 1.0
    assert not runner.score(p2.TASKS[0], rows)["per_seed_gate"]


def test_paired_score_rejects_divergent_resets_and_unregistered_stream():
    rows = _paired_rows(p2.TASKS[1])
    rows["uniform_random"][2]["evaluator_layout_sha256"] = "not paired"
    with pytest.raises(ValueError, match="physical reset"):
        runner.score(p2.TASKS[1], rows)
    rows = _paired_rows(p2.TASKS[0])
    rows["stochastic_trained"][3]["action_sampling_seed"] = 0
    with pytest.raises(ValueError, match="accounting"):
        runner.score(p2.TASKS[0], rows)


def test_strict_numeric_restore_without_checkpoint_or_training(monkeypatch):
    class FakePolicy:
        def __init__(self):
            self.state = {"actor.weight": torch.zeros((2, 3), dtype=torch.float32)}
            self.eval_mode = False

        def state_dict(self):
            return self.state

        def load_state_dict(self, loaded, strict):
            assert strict and set(loaded) == set(self.state)
            self.state = loaded

        def set_training_mode(self, mode):
            self.eval_mode = not mode

    class FakePPO:
        def __init__(self, *_args, **_kwargs):
            self.policy = FakePolicy()

        def learn(self, *_args, **_kwargs):
            raise AssertionError("P2 must NEVER train")

    monkeypatch.setitem(sys.modules, "stable_baselines3", SimpleNamespace(PPO=FakePPO))
    weights = {"policy__actor__weight": np.ones((2, 3), dtype=np.float32)}
    restored = runner._restore_policy(p2.TASKS[0], 17, weights, World())
    assert restored.policy.eval_mode
    assert torch.all(restored.policy.state["actor.weight"] == 1)
    with pytest.raises(ValueError, match="tensor"):
        runner._restore_policy(
            p2.TASKS[0], 17,
            {"policy__actor__weight": np.ones((2, 4), dtype=np.float32)}, World(),
        )


def test_corrupt_source_fails_before_any_native_environment(monkeypatch, tmp_path):
    from bagi.experiments import learned_competence_e4 as p1

    def fail_archive(_path):
        raise ValueError("wrong pinned archive")

    def must_not_make_env(_task):
        raise AssertionError("P2 must not create an env before provenance")

    monkeypatch.setattr(p2, "archive_weights", fail_archive)
    monkeypatch.setattr(p1, "_make_env", must_not_make_env)
    with pytest.raises(ValueError, match="pinned archive"):
        runner.run(tmp_path / "missing.zip", tmp_path / "out")
