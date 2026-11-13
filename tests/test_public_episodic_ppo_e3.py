"""G2-E3-P1 synthetic-only source and budget contracts; no native worlds."""

from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from bagi.experiments import public_episodic_ppo_e3 as pilot


class FakePublicWorld(gym.Env):
    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self, terminate_at: int = 3):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 8, shape=(2,), dtype=np.int8)
        self.terminate_at = terminate_at
        self.native_calls = 0
        self.episode_steps = 0
        self.seeds = []
        self.secret_state = {"position": 12345}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.seeds.append(seed)
        self.episode_steps = 0
        return np.array([0, 0], dtype=np.int8), {"has_key": True}

    def step(self, action):
        self.native_calls += 1
        self.episode_steps += 1
        # Distinct public view after action 1, then repeated view at action 2.
        view = 1 if self.episode_steps < 3 else 2
        observation = np.array([view, 0], dtype=np.int8)
        completed = self.episode_steps >= self.terminate_at
        return observation, float(completed), completed, False, {
            "events": ["switch_activated"],
            "private_position": self.secret_state,
        }


def test_preregistered_exact_native_action_ceiling(monkeypatch):
    pilot.verify_registration()
    assert pilot.MAX_TRAIN == 32768
    assert pilot.MAX_EVAL == 7360
    assert pilot.MAX_ACTIONS == 40128
    assert pilot.SEEDS == (17, 29)
    assert pilot.ARMS == ("extrinsic", "first_visit")
    assert pilot.TASKS == ("workshop", "minigrid_empty_6x6")
    monkeypatch.setattr(pilot, "BETA", 0.1)
    with pytest.raises(ValueError, match="preregistration"):
        pilot.verify_registration()


def test_public_bonus_reset_and_evaluator_only_info():
    native = FakePublicWorld()
    env = pilot.NativePublicRecorder(native, "workshop", "first_visit", 0, True)
    initial, private_info = env.reset(seed=123)
    np.testing.assert_array_equal(initial, [0, 0])
    assert private_info == {}
    assert native.seeds == [18000]
    obs1, shaped1, terminal1, truncated1, info1 = env.step(0)
    obs2, shaped2, terminal2, truncated2, info2 = env.step(1)
    obs3, shaped3, terminal3, truncated3, info3 = env.step(0)
    assert [shaped1, shaped2, shaped3] == pytest.approx([0.05, 0, 1.05])
    assert not any((terminal1, truncated1, terminal2, truncated2, truncated3))
    assert terminal3
    assert info1 == info2 == info3 == {}
    np.testing.assert_array_equal(obs1, obs2)
    assert not np.array_equal(obs2, obs3)
    assert env.episodes[0]["native_goal_success"]
    assert env.episodes[0]["events"]["switch_activated"] == 3
    assert env.report()["native_reward_sum"] == 1.0
    assert env.report()["intrinsic_reward_sum"] == 0.1
    assert env.report()["first_native_reward_at_action"] == 3
    env.reset()
    assert native.seeds == [18000, 18001]
    assert env.step(0)[1] == pytest.approx(0.05)
    assert native.native_calls == 4
    env.close()


def test_extrinsic_only_and_testing_are_true_native_rewards():
    for training, arm in ((True, "extrinsic"), (False, "first_visit")):
        native = FakePublicWorld(terminate_at=2)
        env = pilot.NativePublicRecorder(native, "workshop", arm, 1, training)
        env.reset(seed=None if training else 16000)
        assert env.step(0)[1] == 0.0
        assert env.step(0)[1] == 1.0
        assert env.report()["intrinsic_reward_sum"] == 0
        assert env.report()["native_goals"] == 1
        assert native.seeds == ([19000] if training else [16000])
        env.close()


def test_budget_and_validation_reject_without_extra_native_actions():
    native = FakePublicWorld(terminate_at=10)
    env = pilot.NativePublicRecorder(native, "workshop", "extrinsic", 0, True)
    with pytest.raises(ValueError, match="No live"):
        env.step(0)
    env.reset()
    for invalid in (-1, 2):
        with pytest.raises(ValueError):
            env.step(invalid)
    for invalid in ("1", True):
        with pytest.raises(TypeError):
            env.step(invalid)
    assert native.native_calls == 0
    with pytest.raises(ValueError, match="Unexpected reset"):
        env.reset()
    env.limit = 1
    env.step(0)
    with pytest.raises(ValueError, match="budget"):
        env.step(0)
    assert native.native_calls == 1
    env.close()


def test_heldout_seed_pairing_and_numeric_pickle_safety(tmp_path):
    native = FakePublicWorld(terminate_at=1)
    env = pilot.NativePublicRecorder(native, "workshop", "first_visit", 0, False)
    with pytest.raises(ValueError, match="explicit"):
        env.reset()
    with pytest.raises(ValueError, match="heldout"):
        env.reset(seed=15000)
    env.reset(seed=16000)
    assert env.step(0)[1] == 1.0
    env.reset(seed=16001)
    assert env.step(0)[1] == 1.0
    numeric = env.numeric_arrays()
    assert len(numeric["action"]) == env.native_actions == 2
    assert all(arr.dtype.kind != "O" for arr in numeric.values())
    path = tmp_path / "numeric.npz"
    np.savez_compressed(path, **numeric)
    with np.load(path, allow_pickle=False) as saved:
        assert saved["observation"].shape == (2, 2)
        assert saved["native_reward"].tolist() == [1.0, 1.0]
        assert saved["intrinsic_reward"].tolist() == [0.0, 0.0]
    env.close()


def test_no_reward_or_goal_promotion_from_intrinsic_bonus():
    native = FakePublicWorld(terminate_at=100)
    env = pilot.NativePublicRecorder(native, "workshop", "first_visit", 0, True)
    env.reset()
    # Only native reward can establish true goal completion.
    assert env.step(0)[1] == pytest.approx(0.05)
    assert not env._current["native_goal_success"]
    assert env.rows["native_reward"] == [0.0]
    assert env.rows["intrinsic_reward"] == [0.05]
    env.close()


def test_positive_reward_requires_actual_goal_termination():
    class FalseRewardWorld(FakePublicWorld):
        def step(self, action):
            self.native_calls += 1
            return np.array([1, 0], dtype=np.int8), 1.0, False, False, {}

    world = FalseRewardWorld()
    recorder = pilot.NativePublicRecorder(world, "workshop", "first_visit", 0, True)
    recorder.reset()
    with pytest.raises(ValueError, match="without native goal"):
        recorder.step(0)
    recorder.close()


def test_paired_scoring_uses_only_registered_native_success():
    def rows(outcomes):
        return {"episodes": [
            {"native_reset_seed": 16000 + i, "native_goal_success": b}
            for i, b in enumerate(outcomes)
        ]}

    a = rows([False] * 10)
    b = rows([True, False, True] + [False] * 7)
    scored = pilot._paired_from_rows(a, b)
    assert scored == {
        "count_wins": 2, "count_losses": 0,
        "ties": 8, "count_minus_extrinsic": 2,
    }
    b["episodes"][0]["native_reset_seed"] = 9999
    with pytest.raises(ValueError, match="seeds differ"):
        pilot._paired_from_rows(a, b)
