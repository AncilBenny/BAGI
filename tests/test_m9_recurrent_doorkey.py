"""M9 public-only recurrent control, episode state and source-bound evaluation."""
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import yaml
from gymnasium.spaces import Box, Discrete

from bagi.experiments import m9_recurrent_doorkey as m9


class TinyPublic(gym.Env):
    observation_space = Box(0, 1, shape=(243,), dtype=np.float32)
    action_space = Discrete(7)

    def __init__(self):
        super().__init__()
        self.step_in_episode = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_in_episode = 0
        return np.zeros(243, dtype=np.float32), {}

    def step(self, action):
        assert 0 <= action < 7
        self.step_in_episode += 1
        return np.zeros(243, dtype=np.float32), 0.0, False, self.step_in_episode == 256, {}


def test_m9_exact_preregistered_config_rng_and_independent_splits():
    config = yaml.safe_load(Path("configs/m9_recurrent_doorkey.yaml").read_text())
    assert m9.validate_config(config) == config
    for key, changed in (
        ("heldout_seeds", list(range(112001, 112013))),
        ("model_seeds", [127, 139]),
        ("lstm_hidden_size", 64),
        ("output_dir", "runs/m8_doorkey_entropy"),
    ):
        with pytest.raises(ValueError, match="Registered M9"):
            m9.validate_config({**config, key: changed})
    assert set(m9.HELDOUT).isdisjoint(m9.MONITOR)
    assert set(m9.HELDOUT).isdisjoint(range(112001, 112013))
    assert set(m9.HELDOUT).isdisjoint(range(109001, 109013))
    assert m9.UPDATES_PER_MODEL == 4096
    seeds = {
        m9.action_seed(arm, model_seed, layout, repeat)
        for arm in m9.ARMS for model_seed in m9.SEEDS
        for layout in m9.HELDOUT for repeat in range(1, m9.REPEATS + 1)
    }
    assert len(seeds) == 2 * 2 * 12 * 3
    with pytest.raises(ValueError):
        m9.action_seed("recurrent_history", 151, 112001, 1)


def test_carried_recurrent_state_resets_exactly_per_native_episode():
    class DummyRecurrent:
        def __init__(self):
            self.seen = []

        def predict(self, obs, *, state, episode_start, deterministic):
            assert obs.shape == (243,)
            assert deterministic
            self.seen.append((None if state is None else state, episode_start.copy()))
            return np.asarray(2), (0 if state is None else state) + 1

    agent = DummyRecurrent()
    for _ in range(2):
        policy = m9.StatefulPredict(agent)
        for _ in range(3):
            action, _ = policy.predict(np.zeros(243), deterministic=True)
            assert int(action) == 2
    assert [state for state, _ in agent.seen] == [None, 1, 2, None, 1, 2]
    assert [int(start[0]) for _, start in agent.seen] == [1, 0, 0, 1, 0, 0]


def _row(arm, seed, goals=0):
    return {
        "arm": arm, "policy_family": arm, "model_seed": seed,
        "training": {
            "native_actions": m9.TRAIN_STEPS,
            "incomplete_episode_resets": 0,
            "max_episode_native_actions": 256,
            "executed_action_histogram": [m9.TRAIN_STEPS, 0, 0, 0, 0, 0, 0],
            "first_reset_seed": 1_000_000 + seed * 500_000,
        },
        "training_real_actions": m9.TRAIN_STEPS,
        "optimizer_step_calls": m9.UPDATES_PER_MODEL,
        "policy_parameters": 36000 if arm == "recurrent_history" else 15880,
        "numeric_policy": {"sha256": "valid", "allow_pickle": False},
        "independent": {
            "sampled_totals": {"verified_physical_goals": goals},
            "deterministic_totals": {"verified_physical_goals": 0},
        },
    }


def test_m9_heldout_barrier_and_both_seed_physical_gate():
    rows = [
        _row("reactive_history", 151),
        _row("reactive_history", 167),
        _row("recurrent_history", 151, goals=3),
        _row("recurrent_history", 167, goals=2),
    ]
    checks = m9.preheldout_integrity(rows)
    assert all(checks.values())
    assert m9.score(rows, checks)["primary_gate_pass"]
    rows[-1]["independent"]["sampled_totals"]["verified_physical_goals"] = 0
    assert not m9.score(rows, checks)["primary_gate_pass"]
    rows[-1]["independent"]["sampled_totals"]["verified_physical_goals"] = 2
    rows[2]["training"]["incomplete_episode_resets"] = 1
    assert not m9.preheldout_integrity(rows)["recurrent_history:151"]
    rows[2]["training"]["incomplete_episode_resets"] = 0
    rows[1]["training"]["first_reset_seed"] += 1
    with pytest.raises(ValueError, match="first training reset"):
        m9.preheldout_integrity(rows)
    with pytest.raises(ValueError, match="missing or duplicated"):
        m9.preheldout_integrity(rows[:-1])


def test_m9_native_counter_reuse_and_heldout_only_after_all_training():
    with pytest.raises(ValueError, match="preregistered"):
        from bagi.experiments.m6_doorkey_continuity import NativeDoorTraining

        NativeDoorTraining(TinyPublic(), model_seed=151)
    from bagi.experiments.m6_doorkey_continuity import NativeDoorTraining

    env = NativeDoorTraining(
        TinyPublic(), model_seed=151, allowed_seeds=m9.SEEDS,
        max_native_steps=m9.TRAIN_STEPS,
    )
    try:
        env.reset()
        for _ in range(256):
            env.step(0)
        env.reset()
        assert env.native_steps == 256
        assert env.incomplete_episode_resets == 0
        assert env.max_episode_length == 256
    finally:
        env.close()
    src = Path(m9.__file__).read_text()
    assert src.index("checks = preheldout_integrity(records)") < (
        src.index('stage = "independent_heldout"')
    )
    assert "StatefulPredict(agent)" in src
    assert "PublicGoalActions(timed)" in src
    assert "numeric_state(agent, output.parent" in src


def test_m9_real_upstream_recurrent_predict_and_train_when_installed():
    sb3 = pytest.importorskip("stable_baselines3")
    contrib = pytest.importorskip("sb3_contrib")
    from bagi.experiments.m6_doorkey_continuity import NativeDoorTraining

    rows = []
    for arm in m9.ARMS:
        env = NativeDoorTraining(
            TinyPublic(), model_seed=151, allowed_seeds=m9.SEEDS,
            max_native_steps=m9.TRAIN_STEPS,
        )
        try:
            recurrent = arm == "recurrent_history"
            kwargs = {"net_arch": {"pi": [8], "vf": [8]}}
            if recurrent:
                kwargs.update({"lstm_hidden_size": 8, "n_lstm_layers": 1})
            agent = (contrib.RecurrentPPO if recurrent else sb3.PPO)(
                "MlpLstmPolicy" if recurrent else "MlpPolicy", env,
                seed=151, device="cpu", n_steps=8, batch_size=4, n_epochs=1,
                policy_kwargs=kwargs, verbose=0,
            )
            agent.learn(total_timesteps=16, reset_num_timesteps=False)
            assert env.native_steps == agent.num_timesteps == 16
            assert env.incomplete_episode_resets == 0
            adapted = m9.StatefulPredict(agent) if recurrent else agent
            action, _ = adapted.predict(np.zeros(243, dtype=np.float32),
                                        deterministic=True)
            assert 0 <= int(np.asarray(action).item()) < 7
            rows.append(sum(p.numel() for p in agent.policy.parameters()))
        finally:
            env.close()
    assert rows[0] > 0 and rows[1] > 0 and rows[0] != rows[1]
