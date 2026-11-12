"""Source-contract and real-MiniGrid checks for M10-P1 public representations."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import yaml
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX

from bagi.experiments.m5_continual_retention import DOORKEY
from bagi.experiments.m10_public_representation import (
    ARMS,
    EXPECTED_SHAPES,
    HELDOUT,
    MONITOR,
    PublicDirectionalActions,
    SEEDS,
    TRAIN_STEPS,
    UPDATES_PER_MODEL,
    action_seed,
    public_env,
    score,
    training_integrity,
    validate_config,
)


def test_m10p1_config_seed_isolation_and_fixed_episode_denominators():
    config = yaml.safe_load(
        Path("configs/m10_public_representation.yaml").read_text(encoding="utf-8")
    )
    assert validate_config(config) == config
    assert len(set(MONITOR)) == 6
    assert len(set(HELDOUT)) == 12
    assert not set(MONITOR).intersection(HELDOUT)
    assert not set(HELDOUT).intersection(range(118001, 118013))
    assert not set(HELDOUT).intersection(range(119001, 119013))
    assert UPDATES_PER_MODEL == 4096
    assert TRAIN_STEPS == 131072
    assert len(ARMS) * len(SEEDS) * UPDATES_PER_MODEL == 24576
    assert action_seed(ARMS[0], SEEDS[0], MONITOR[0], 1) != action_seed(
        ARMS[0], SEEDS[0], HELDOUT[0], 1
    )
    for name, other in (
        ("heldout_seeds", list(MONITOR)),
        ("arms", [ARMS[0]]),
        ("sampled_repeats", 1),
    ):
        changed = deepcopy(config)
        changed[name] = other
        with pytest.raises(ValueError):
            validate_config(changed)
    with pytest.raises(ValueError):
        action_seed(ARMS[0], SEEDS[0], 118001, 1)


@pytest.mark.parametrize("arm", ARMS)
def test_actual_minigrid_upstream_public_observation_native_actions(arm):
    env = public_env(gym.make(DOORKEY), arm)
    try:
        observation, info = env.reset(seed=31005)
        assert isinstance(info, dict)
        assert observation.shape == EXPECTED_SHAPES[arm]
        assert env.observation_space.contains(observation)
        if arm == "onehot_direction":
            image = observation[:7 * 7 * 20].reshape(7, 7, 20)
            assert set(np.unique(image)).issubset({0.0, 1.0})
            assert np.all(image.sum(axis=2) == 3)
            assert (len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)) == 20
        if arm != "raw_history":
            assert observation[-4:].sum() == 1
            assert set(np.unique(observation[-4:])).issubset({0.0, 1.0})
            assert isinstance(env, PublicDirectionalActions)
        for action in (0, 1, 2, 3, 4, 5, 6):
            observation, reward, terminated, truncated, info = env.step(action)
            assert observation.shape == EXPECTED_SHAPES[arm]
            assert env.observation_space.contains(observation)
            assert isinstance(info, dict)
            assert np.isfinite(float(reward))
            if terminated or truncated:
                break
    finally:
        env.close()


def test_raw_plus_direction_preserves_original_243_public_features():
    raw = public_env(gym.make(DOORKEY), "raw_history")
    direct = public_env(gym.make(DOORKEY), "raw_direction")
    try:
        left, _ = raw.reset(seed=31007)
        right, _ = direct.reset(seed=31007)
        np.testing.assert_array_equal(left, right[:243])
        for action in (0, 2, 1, 3):
            left, _, _, _, _ = raw.step(action)
            right, _, _, _, _ = direct.step(action)
            np.testing.assert_array_equal(left, right[:243])
    finally:
        raw.close()
        direct.close()


def test_training_integrity_must_hold_before_final_access():
    with pytest.raises(ValueError):
        training_integrity([])
    with pytest.raises(ValueError):
        score([], {})
