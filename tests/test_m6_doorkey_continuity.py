"""Tests for M6 native episode-reset ablation, registered seeds and goal gates."""
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import yaml
from gymnasium.spaces import Box, Discrete

from bagi.experiments import m6_doorkey_continuity as m6


class TinyPublic(gym.Env):
    observation_space = Box(0, 1, shape=(2,), dtype=np.float32)
    action_space = Discrete(7)

    def __init__(self):
        super().__init__()
        self.step_in_episode = 0
        self.last_seed = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_in_episode = 0
        self.last_seed = seed
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        assert 0 <= action < 7
        self.step_in_episode += 1
        end = self.step_in_episode == 256
        return np.zeros(2, dtype=np.float32), 0.0, False, end, {}


def test_registered_m6_config_and_fresh_splits():
    config = yaml.safe_load(Path("configs/m6_doorkey_continuity.yaml").read_text())
    assert m6.validate_config(config) == config
    for field, changed in (
        ("training_seeds", [71, 83]),
        ("heldout_seeds", list(range(103001, 103013))),
        ("rollout_steps", 256),
        ("output_dir", "runs/m5_continual_retention"),
    ):
        with pytest.raises(ValueError, match="Registered M6"):
            m6.validate_config({**config, field: changed})
    assert set(m6.HELDOUT).isdisjoint(m6.MONITOR)
    assert set(m6.SEEDS).isdisjoint(m6.HELDOUT + m6.MONITOR)


def test_native_reset_interrupts_only_incomplete_episodes():
    interrupted = m6.NativeDoorTraining(TinyPublic(), model_seed=97)
    try:
        interrupted.reset(seed=106001)
        assert interrupted.first_reset_seed != 106001
        for _ in range(m6.ROLLOUT):
            interrupted.step(0)
        interrupted.reset()
        stats = interrupted.measured()
        assert stats["native_actions"] == 128
        assert stats["native_resets"] == 2
        assert stats["incomplete_episode_resets"] == 1
        assert stats["max_episode_native_actions"] == 128
        assert sum(stats["executed_action_histogram"]) == 128
    finally:
        interrupted.close()

    continuing = m6.NativeDoorTraining(TinyPublic(), model_seed=97)
    try:
        continuing.reset()
        for _ in range(256):
            continuing.step(0)
        continuing.reset()
        stats = continuing.measured()
        assert stats["native_actions"] == 256
        assert stats["native_resets"] == 2
        assert stats["incomplete_episode_resets"] == 0
        assert stats["native_time_limit_episodes"] == 1
        assert stats["max_episode_native_actions"] == 256
        assert sum(stats["executed_action_histogram"]) == 256
    finally:
        continuing.close()


def test_identical_seed_stream_initial_conditions_before_reset_intervention():
    first = m6.NativeDoorTraining(TinyPublic(), model_seed=109)
    second = m6.NativeDoorTraining(TinyPublic(), model_seed=109)
    try:
        first.reset()
        second.reset()
        assert first.first_reset_seed == second.first_reset_seed
        assert first.seed_hash.hexdigest() == second.seed_hash.hexdigest()
        first.step(1)
        second.step(1)
        first.reset()
        second.reset()
        assert first.seed_hash.hexdigest() == second.seed_hash.hexdigest()
    finally:
        first.close()
        second.close()


def _models(*, good=True):
    records = []
    for arm in m6.ARMS:
        for seed in m6.SEEDS:
            if arm == "reset128":
                physical, goals = 0, 0
                resets, maxlen = 510, 128
            else:
                physical = (3 if seed == 97 else 2) if good else (
                    3 if seed == 97 else 0
                )
                goals = physical
                resets, maxlen = 0, 256
            records.append({
                "arm": arm, "model_seed": seed,
                "training": {
                    "incomplete_episode_resets": resets,
                    "max_episode_native_actions": maxlen,
                },
                "independent_heldout": {
                    "rewarded_goals": goals,
                    "verified_physical_goals": physical,
                },
            })
    return records


def test_continuity_preheldout_integrity_checks_and_primary_gate():
    records = _models()
    checks = m6.compare_reset_integrity(records)
    assert all(value["valid_continuity_intervention"] for value in checks.values())
    result = m6.score(records, checks)
    assert result["physical_goal_difference_of_24"] == 5
    assert result["both_continuation_model_seeds_have_physical_goal"]
    assert result["primary_gate_pass"]
    weaker = _models(good=False)
    assert not m6.score(weaker, m6.compare_reset_integrity(weaker))["primary_gate_pass"]
    altered = _models()
    altered[0]["training"]["max_episode_native_actions"] = 256
    assert not all(row["valid_continuity_intervention"] for row in
                   m6.compare_reset_integrity(altered).values())
    with pytest.raises(ValueError, match="models or unique seeds"):
        m6.compare_reset_integrity(records[:-1])


def test_m6_reuses_exact_public_evaluator_and_no_training_private_state():
    assert m6.DOORKEY == "MiniGrid-DoorKey-8x8-v0"
    from bagi.experiments.m5_continual_retention import evaluate

    assert m6.evaluate_door is evaluate
    # No independent final evaluation can happen until M6's four-model
    # native-action/reset continuity validation has succeeded.
    source = Path(m6.__file__).read_text(encoding="utf-8")
    assert source.index("continuity = compare_reset_integrity(records)") < (
        source.index('phase = "independent_heldout"')
    )
    assert "agent.set_env(agent.get_env(), force_reset=(arm == \"reset128\"))" in source


def test_m6_pinned_sb3_vecenv_rebinding_preserves_continuous_rollout():
    # Optional in lightweight default CI, REQUIRED when the admitted research
    # workflow installs [dev,rl] and runs its platform tests before fitting.
    sb3 = pytest.importorskip("stable_baselines3")
    for arm in m6.ARMS:
        env = m6.NativeDoorTraining(TinyPublic(), model_seed=97)
        try:
            agent = sb3.PPO(
                "MlpPolicy", env, seed=97, device="cpu",
                n_steps=8, batch_size=4, n_epochs=1,
                policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}}, verbose=0,
            )
            for _ in range(2):
                agent.set_env(agent.get_env(), force_reset=(arm == "reset128"))
                agent.learn(total_timesteps=8, reset_num_timesteps=False)
            assert env.native_steps == agent.num_timesteps == 16
            if arm == "reset128":
                assert env.incomplete_episode_resets >= 1
                assert env.max_episode_length <= 8
            else:
                assert env.incomplete_episode_resets == 0
                assert env.max_episode_length == 16
        finally:
            env.close()
