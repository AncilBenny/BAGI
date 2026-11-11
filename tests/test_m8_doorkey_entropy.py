"""M8 entropy ablation protocol, equal native work and evidence-gate tests."""
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import yaml
from gymnasium.spaces import Box, Discrete

from bagi.experiments import m8_doorkey_entropy as m8


class TinyPublic(gym.Env):
    observation_space = Box(0.0, 1.0, shape=(243,), dtype=np.float32)
    action_space = Discrete(7)

    def __init__(self):
        super().__init__()
        self.episode_step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_step = 0
        return np.zeros(243, dtype=np.float32), {}

    def step(self, action):
        assert 0 <= action < 7
        self.episode_step += 1
        done = self.episode_step == 256
        return np.zeros(243, dtype=np.float32), 0.0, False, done, {}


def test_m8_exact_config_and_independent_seed_splits():
    config = yaml.safe_load(Path("configs/m8_doorkey_entropy.yaml").read_text())
    assert m8.validate_config(config) == config
    for key, wrong in (
        ("heldout_seeds", list(range(109001, 109013))),
        ("train_steps_per_fit", 65536),
        ("entropy_coefficients", {"standard_entropy": 0, "bonus_entropy": 0}),
        ("output_dir", "runs/m7_frozen_doorkey_sampling"),
    ):
        with pytest.raises(ValueError, match="Registered M8"):
            m8.validate_config({**config, key: wrong})
    assert set(m8.HELDOUT).isdisjoint(m8.MONITOR)
    assert set(m8.HELDOUT).isdisjoint(range(109001, 109013))
    assert set(m8.HELDOUT).isdisjoint(range(106001, 106013))
    assert m8.UPDATES_PER_MODEL == 4096
    seeds = {
        m8.action_seed(arm, seed, layout, rep)
        for arm in m8.ARMS for seed in m8.SEEDS
        for layout in m8.HELDOUT for rep in range(1, m8.REPEATS + 1)
    }
    assert len(seeds) == 2 * 2 * 12 * 3


def test_m8_reuses_m6_counter_with_new_seeds_without_changing_m6_defaults():
    from bagi.experiments.m6_doorkey_continuity import NativeDoorTraining

    with pytest.raises(ValueError, match="preregistered"):
        NativeDoorTraining(TinyPublic(), model_seed=127)
    a = NativeDoorTraining(
        TinyPublic(), model_seed=127, allowed_seeds=m8.SEEDS,
        max_native_steps=m8.TRAIN_STEPS,
    )
    b = NativeDoorTraining(
        TinyPublic(), model_seed=127, allowed_seeds=m8.SEEDS,
        max_native_steps=m8.TRAIN_STEPS,
    )
    try:
        a.reset()
        b.reset()
        for _ in range(256):
            a.step(0)
            b.step(0)
        a.reset()
        b.reset()
        assert a.measured()["max_episode_native_actions"] == 256
        assert a.measured()["incomplete_episode_resets"] == 0
        assert a.measured()["reset_seed_stream_sha256"] == (
            b.measured()["reset_seed_stream_sha256"]
        )
    finally:
        a.close()
        b.close()


def _row(arm, seed, successes=0):
    return {
        "arm": arm, "model_seed": seed, "entropy_coefficient": m8.COEFFICIENTS[arm],
        "training": {
            "native_actions": m8.TRAIN_STEPS,
            "incomplete_episode_resets": 0,
            "max_episode_native_actions": 256,
            "executed_action_histogram": [m8.TRAIN_STEPS, 0, 0, 0, 0, 0, 0],
            "first_reset_seed": 1_000_000 + seed * 500_000,
        },
        "training_real_actions": m8.TRAIN_STEPS,
        "optimizer_step_calls": m8.UPDATES_PER_MODEL,
        "policy_parameters": 15880,
        "numeric_policy": {"sha256": "valid", "allow_pickle": False},
        "independent": {
            "sampled_totals": {"verified_physical_goals": successes},
            "deterministic_totals": {"verified_physical_goals": 0},
        },
    }


def test_preheldout_gate_and_both_seed_success_rule():
    rows = [
        _row("standard_entropy", 127),
        _row("standard_entropy", 139),
        _row("bonus_entropy", 127, successes=3),
        _row("bonus_entropy", 139, successes=2),
    ]
    integrity = m8.preheldout_integrity(rows)
    assert all(integrity.values())
    assert m8.score(rows, integrity)["primary_gate_pass"]
    rows[-1]["independent"]["sampled_totals"]["verified_physical_goals"] = 0
    assert not m8.score(rows, integrity)["primary_gate_pass"]
    rows[-1]["independent"]["sampled_totals"]["verified_physical_goals"] = 2
    rows[0]["training"]["incomplete_episode_resets"] = 1
    with pytest.raises(ValueError, match="control interrupted native episodes"):
        m8.preheldout_integrity(rows)
    rows[0]["training"]["incomplete_episode_resets"] = 0
    rows[2]["training"]["incomplete_episode_resets"] = 1
    integrity = m8.preheldout_integrity(rows)
    assert not integrity["bonus_entropy:127"]
    with pytest.raises(ValueError, match="missing or duplicated fresh fits"):
        m8.preheldout_integrity(rows[:-1])


def test_m8_no_privileged_training_state_and_holdout_order():
    from bagi.experiments.m7_frozen_doorkey_sampling import evaluate_episode

    assert m8.evaluate_episode is evaluate_episode
    src = Path(m8.__file__).read_text(encoding="utf-8")
    assert "agent.learn(total_timesteps=TRAIN_STEPS, reset_num_timesteps=False)" in src
    assert "ent_coef=COEFFICIENTS[arm]" in src
    assert src.index("checks = preheldout_integrity(records)") < (
        src.index('phase = "independent_heldout"')
    )
    assert "numeric_state(agent, output.parent" in src
    assert "PublicGoalActions(timed)" in src
    assert "private" not in m8.make_training_env.__doc__ if (
        m8.make_training_env.__doc__
    ) else True


def test_upstream_sb3_ent_coef_is_only_arm_parameter_when_installed():
    sb3 = pytest.importorskip("stable_baselines3")
    from bagi.experiments.m6_doorkey_continuity import NativeDoorTraining

    rows = []
    for arm in m8.ARMS:
        env = NativeDoorTraining(
            TinyPublic(), model_seed=127, allowed_seeds=m8.SEEDS,
            max_native_steps=m8.TRAIN_STEPS,
        )
        try:
            agent = sb3.PPO(
                "MlpPolicy", env, seed=127, device="cpu", n_steps=8,
                batch_size=4, n_epochs=1, ent_coef=m8.COEFFICIENTS[arm],
                policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
                verbose=0,
            )
            agent.learn(total_timesteps=16, reset_num_timesteps=False)
            assert env.native_steps == agent.num_timesteps == 16
            assert env.incomplete_episode_resets == 0
            rows.append(agent.ent_coef)
        finally:
            env.close()
    assert rows == [0.0, 0.02]
