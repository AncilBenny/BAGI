"""G2-E4-P1 synthetic contracts; never executes MiniGrid or PPO."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from bagi.experiments import learned_competence_e4 as e4


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
    action_space: ClassVar[Space] = Space()

    def __init__(self):
        self.step_count = 0
        self.reset_count = 0
        self.unwrapped = self
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        self.grid = Grid(0)

    def reset(self, *, seed):
        self.step_count = 0
        self.reset_count += 1
        self.grid = Grid(seed)
        return np.array([seed % 10, 1, 2], dtype=np.uint8), {"secret": 999}

    def step(self, action):
        self.step_count += 1
        done = self.step_count == 3
        return (np.array([self.step_count, action, 2], dtype=np.uint8),
                float(done and action == 2), done, False, {"secret": 999})


class Agent:
    def predict(self, obs, deterministic):
        assert deterministic and obs.ndim == 1
        return np.asarray(2), None


def test_frozen_budgets():
    e4.verify_registration()
    assert (e4.MAX_TRAIN_ACTIONS, e4.MAX_TEST_ACTIONS, e4.MAX_ALL_ACTIONS) == (
        65536, 23640, 89176
    )


def test_guard_rejects_modified_budget(monkeypatch):
    monkeypatch.setattr(e4, "TRAIN_STEPS_PER_MODEL", 8192)
    with pytest.raises(ValueError, match="preregistration"):
        e4.verify_registration()


def test_public_shape_and_nonfinite_fail_closed():
    for obs in (np.zeros((2, 2)), np.array([float("nan")]),
                np.array(["secret"], dtype=object)):
        with pytest.raises(ValueError, match="PUBLIC"):
            e4._public_array(obs)


def test_world_hash_evaluator_only():
    env = World()
    env.reset(seed=26000)
    first = e4._layout_digest(env)
    public = e4._public_digest(np.array([1, 2, 3], dtype=np.uint8))
    env.grid = Grid(26001)
    assert first != e4._layout_digest(env)
    assert public == e4._public_digest(np.array([1, 2, 3], dtype=np.uint8))


def test_episode_native_step_count_and_no_secret():
    env = World()
    ledger = {"train_actions": 0, "test_actions": 0}
    row, trace = e4.evaluate_episode(env, e4.TASKS[0], 26000, "trained", 17,
                                     Agent(), ledger)
    assert row["goal_success"] == 1
    assert row["native_actions"] == ledger["test_actions"] == 3
    assert row["native_terminated"] and not row["native_truncated"]
    assert trace["public_pre_action"].shape == (3, 3)
    assert trace["actions"].tolist() == [2, 2, 2]
    assert "secret" not in row


def test_random_never_calls_agent():
    class FailAgent:
        def predict(self, *_args, **_kwargs):
            raise AssertionError("random called agent")

    row, trace = e4.evaluate_episode(World(), e4.TASKS[0], 26001, "random",
                                     29, FailAgent(), {"train_actions": 0,
                                                       "test_actions": 0})
    assert row["native_actions"] == len(trace["actions"]) == 3


def test_unregistered_seed_rejected_before_reset():
    env = World()
    with pytest.raises(ValueError, match="Unregistered"):
        e4.evaluate_episode(env, e4.TASKS[0], 21000, "trained", 17, Agent(),
                            {"train_actions": 0, "test_actions": 0})
    assert env.reset_count == 0


def test_safe_numeric_npz_roundtrip(tmp_path):
    _, trace = e4.evaluate_episode(World(), e4.TASKS[0], 26000, "trained", 17,
                                   Agent(), {"train_actions": 0, "test_actions": 0})
    path = tmp_path / "safe.npz"
    digest = e4._save_traces(path, [trace, trace])
    assert len(digest) == 64
    with np.load(path, allow_pickle=False) as loaded:
        assert loaded["episode_offsets"].tolist() == [0, 3, 6]
        assert loaded["actions"].tolist() == [2] * 6
        assert all(loaded[key].dtype.kind in "biuf" for key in loaded.files)


def test_score_rejects_mismatched_reset_layout():
    rows = {arm: [{"reset_seed": seed, "evaluator_layout_sha256": str(seed),
                   "goal_success": 0, "native_actions": 3} for seed in e4.TEST_SEEDS]
            for arm in e4.ARMS}
    assert e4.score(rows)["valid_test_diversity"]
    rows["untrained"][2]["evaluator_layout_sha256"] = "different"
    with pytest.raises(ValueError, match="identical physical"):
        e4.score(rows)


def test_score_rejects_false_diversity_despite_success():
    rows = {arm: [{"reset_seed": seed, "evaluator_layout_sha256": "same",
                   "goal_success": int(arm == "trained"), "native_actions": 3}
                  for seed in e4.TEST_SEEDS] for arm in e4.ARMS}
    result = e4.score(rows)
    assert result["native_successes"]["trained"] == 10
    assert not result["valid_test_diversity"]


def test_action_ceiling_checks_before_native_step():
    env = World()
    with pytest.raises(AssertionError, match="TEST native-action ceiling"):
        e4.evaluate_episode(env, e4.TASKS[0], 26000, "trained", 17, Agent(),
                            {"train_actions": 0, "test_actions": e4.MAX_TEST_ACTIONS})
    assert env.step_count == 0
