"""Simulator-free G2-E4-P4 contracts: scale is the ONLY actor input change."""

from __future__ import annotations

import hashlib
import re
import sys
import types
from typing import ClassVar

import numpy as np
import pytest
import torch

from bagi.experiments import e4_p4_raw_vs_scaled_ppo as p4


class FakeBox:
    def __init__(self, *, low, high, shape, dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype


class FakeGymWrapper:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.unwrapped = env.unwrapped

    def close(self):
        self.env.close()


class FakeSpace:
    n: ClassVar[int] = 7

    def contains(self, action):
        return isinstance(action, (int, np.integer)) and 0 <= action < self.n

    def seed(self, seed):
        self.seed_value = seed


class FakeGrid:
    def __init__(self, seed):
        self.seed = seed

    def encode(self):
        return np.full((3, 3, 3), self.seed % 251, dtype=np.uint8)


class FakeWorld:
    action_space: ClassVar[FakeSpace] = FakeSpace()

    def __init__(self):
        self.unwrapped = self
        self.observation_space = FakeBox(
            low=0, high=10, shape=(147,), dtype=np.uint8
        )
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        self.grid = FakeGrid(0)
        self.steps = 0
        self.resets = []

    def reset(self, *, seed):
        self.steps = 0
        self.resets.append(seed)
        self.grid = FakeGrid(seed)
        return np.full(147, 8, dtype=np.uint8), {"hidden": 999}

    def step(self, action):
        self.steps += 1
        done = self.steps == 3
        return np.full(147, 10 - self.steps, dtype=np.uint8), float(done), done, False, {
            "hidden": 999,
        }

    def close(self):
        return None


class FakeStochasticPolicy:
    def __init__(self):
        self.inputs = []

    def predict(self, observation, *, deterministic):
        assert not deterministic
        self.inputs.append(np.asarray(observation).copy())
        return np.array(torch.multinomial(torch.full((7,), 1 / 7), 1).item()), None


def test_registration_and_correct_ceiling():
    p4.verify_registration()
    assert p4.TRAIN_CEILING == 4 * 16384 == 65536
    assert p4.TEST_CEILING == 2 * 2 * 2 * 16 * 144 == 18432
    assert p4.ALL_CEILING == 83968
    assert set(p4.TEST_SEEDS).isdisjoint(set(range(21000, 21016)))
    assert set(p4.TEST_SEEDS).isdisjoint(set(range(26000, 28016)))


def test_registration_fails_when_budget_changes(monkeypatch):
    monkeypatch.setattr(p4, "NORMALIZER", 255.0)
    with pytest.raises(ValueError, match="preregistration"):
        p4.verify_registration()


def test_scale_preserves_public_information_and_dimensionality():
    raw = np.tile(np.asarray([0, 5, 10], dtype=np.uint8), 49)
    original = p4.actor_public(raw, "raw")
    scaled = p4.actor_public(raw, "scaled")
    assert original.dtype == scaled.dtype == np.float32
    assert original.shape == scaled.shape == (147,)
    np.testing.assert_array_equal(original, scaled * 10)
    assert raw[2] == 10


def test_public_contract_rejects_privileged_or_malformed_input():
    for invalid in (
        np.ones((7, 7, 3), dtype=np.uint8),
        np.ones(148, dtype=np.uint8),
        np.array([float("nan")] * 147),
        np.full(147, 11, dtype=np.int16),
        np.full(147, -1, dtype=np.int16),
        np.full(147, "hidden", dtype=object),
        np.zeros(147, dtype=np.float32),
    ):
        with pytest.raises(ValueError, match="PUBLIC"):
            p4.native_public(invalid)
    with pytest.raises(ValueError, match="scale"):
        p4.actor_public(np.zeros(147, dtype=np.uint8), "other")


def test_scale_wrapper_matches_reset_stream_and_sanitizes_info(monkeypatch):
    monkeypatch.setitem(sys.modules, "gymnasium", types.SimpleNamespace(
        Wrapper=FakeGymWrapper, spaces=types.SimpleNamespace(Box=FakeBox)
    ))
    worlds = []

    def make_world():
        w = FakeWorld()
        worlds.append(w)
        return w

    monkeypatch.setattr(p4, "make_world", make_world)
    ledger = {"train": 0, "test": 0}
    raw = p4.make_train_env("raw", 17, ledger)
    scaled = p4.make_train_env("scaled", 17, ledger)
    for env, exp in ((raw, 8.0), (scaled, 0.8)):
        obs, info = env.reset(seed=1234)
        assert info == {}
        np.testing.assert_allclose(obs, np.full(147, exp))
        assert env.train_reset_seeds == [p4.TRAIN_BASE]
        for _ in range(3):
            _next, reward, term, trunc, info = env.step(2)
            assert info == {}
        assert term and not trunc and reward == 1.0
        assert env.native_goals == 1
        assert env.real_actions == 3
        assert env.rows["actions"] == [2, 2, 2]
    assert ledger["train"] == 6
    assert [w.resets for w in worlds] == [[p4.TRAIN_BASE], [p4.TRAIN_BASE]]


def test_eval_live_vs_stale_only_actor_input_differs(monkeypatch):
    # Does NOT run the real MiniGrid simulator; mock world has a 3-step goal.
    live, stale = FakeWorld(), FakeWorld()
    a, b = FakeStochasticPolicy(), FakeStochasticPolicy()
    ledger = {"train": 0, "test": 0}
    left, lt = p4.test_episode(live, a, "scaled", "live", 17, 29000, ledger)
    right, rt = p4.test_episode(stale, b, "scaled", "stale", 17, 29000, ledger)
    assert left["rng_seed"] == right["rng_seed"]
    assert left["layout_sha256"] == right["layout_sha256"]
    assert left["initial_public_sha256"] == right["initial_public_sha256"]
    assert left["native_goal"] == right["native_goal"] == 1
    assert left["actions"] == right["actions"] == 3
    np.testing.assert_array_equal(lt["actions"], rt["actions"])
    np.testing.assert_array_equal(lt["actor_input"][0], rt["actor_input"][0])
    assert not np.array_equal(lt["actor_input"][1], rt["actor_input"][1])
    assert np.all(rt["actor_input"] == rt["actor_input"][0])
    assert ledger["test"] == 6


def test_eval_rejects_wrong_scale_feedback_or_seed():
    with pytest.raises(ValueError, match="arm"):
        p4.test_episode(FakeWorld(), FakeStochasticPolicy(), "bad", "live", 17, 29000, {})
    with pytest.raises(ValueError, match="seed"):
        p4.sampling_seed(17, 28000)


def test_evaluator_hash_never_goes_into_actor_input():
    world = FakeWorld()
    world.reset(seed=29000)
    a = p4._hash_layout(world)
    assert re.fullmatch("[0-9a-f]{64}", a)
    assert p4.actor_public(np.full(147, 8, dtype=np.uint8), "scaled").shape == (147,)


def test_numeric_save_rejects_unsafe_and_has_true_lengths(tmp_path):
    evidence = {
        "public_codes": np.ones((2, 147), dtype=np.float32),
        "actor_input": np.ones((2, 147), dtype=np.float32),
        "actions": np.array([0, 1], dtype=np.int16),
        "native_rewards": np.zeros(2, dtype=np.float64),
        "terminated": np.array([0, 1], dtype=np.int8),
        "truncated": np.array([0, 0], dtype=np.int8),
    }
    path = tmp_path / "safe.npz"
    digest = p4._pack(path, [evidence])
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["episode_offsets"], [0, 2])
        assert arrays["actions"].dtype == np.int16
    with pytest.raises(ValueError, match="Unsafe"):
        p4._pack(tmp_path / "bad.npz", [{**evidence, "actions": np.array(["secret"])}])


def test_score_requires_identical_physical_reset_and_gates_per_seed():
    def case(seed, goal):
        return {
            "reset_seed": seed, "rng_seed": p4.sampling_seed(17, seed),
            "layout_sha256": str(seed), "initial_public_sha256": str(seed),
            "native_goal": goal, "native_reward": float(goal),
            "terminated": bool(goal), "actions": 3,
            "action_counts": [0, 0, 3, 0, 0, 0, 0],
        }

    def row(model_seed, scale):
        def matching(seed, goal):
            x = case(seed, goal)
            x["rng_seed"] = p4.sampling_seed(model_seed, seed)
            return x

        return {
            "model_seed": model_seed, "scale": scale,
            "train": {"real_actions": p4.TRAIN_PER_MODEL, "native_goals": 1},
            "eval": {
                "live": [matching(s, 1) for s in p4.TEST_SEEDS],
                "stale": [matching(s, int(i > 5) if scale == "scaled" else 1)
                          for i, s in enumerate(p4.TEST_SEEDS)],
            },
        }

    rows = [row(seed, scale) for seed in p4.MODEL_SEEDS for scale in p4.SCALES]
    output = p4.score(rows)
    assert output["strong_feedback_improvement_both_seeds"]
    rows[0]["eval"]["live"][0]["layout_sha256"] = "privileged_wrong"
    with pytest.raises(ValueError, match="layout"):
        p4.score(rows)


def test_training_reset_pair_check_rejects_mismatched_initialization():
    details = [
        {
            "model_seed": seed, "scale": scale,
            "initial_state_sha256": "SAME" if scale == "raw" else "DIFFERENT",
            "train": {
                "reset_seeds": [
                    p4.TRAIN_BASE + p4.MODEL_SEEDS.index(seed) * p4.TRAIN_STRIDE
                ],
                "first_public_sha256": "identical",
            },
        }
        for seed in p4.MODEL_SEEDS for scale in p4.SCALES
    ]
    with pytest.raises(ValueError, match="initial"):
        p4._train_seed_pair_check(details)
