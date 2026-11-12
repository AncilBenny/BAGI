"""M12 test public action-conditioned predictions on actual MiniGrid observations."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml

from bagi.experiments.m5_continual_retention import DOORKEY, MEMORY
from bagi.experiments.m12_public_front_effect import (
    FEATURES,
    FINAL,
    FRONT,
    SEEDS,
    TASKS,
    TRAIN,
    FrontLogit,
    evaluate,
    fitted_tables,
    frozen_config,
    public_feature,
    split_metrics,
    visible_change,
)


def test_registered_protocol_does_not_reuse_old_final_episodes():
    cfg = yaml.safe_load(Path("configs/m12_public_front_effect.yaml").read_text())
    assert frozen_config(cfg) == cfg
    all_seeds = sum(TRAIN + FINAL, ())
    assert len(all_seeds) == len(set(all_seeds)) == 96
    assert len(SEEDS) == 2
    assert not set(all_seeds).intersection(range(120001, 120013))
    assert not set(all_seeds).intersection(range(142101, 142113))


@pytest.mark.parametrize("task", TASKS)
def test_real_public_front_cell_action_and_next_image_contract(task):
    env = gym.make(task)
    try:
        obs, _ = env.reset(seed=711117)
        for action in range(7):
            before = np.asarray(obs["image"])
            features, obj = public_feature(before, action)
            assert features.shape == (FEATURES,)
            assert features.sum() == 3
            assert obj == int(before[FRONT][0])
            after, reward, ended, truncated, info = env.step(action)
            assert visible_change(before, np.asarray(after["image"])) in (0, 1)
            assert np.isfinite(float(reward))
            assert isinstance(info, dict)
            obs = after
            if ended or truncated:
                break
    finally:
        env.close()


def test_invalid_private_shapes_and_actions_rejected():
    with pytest.raises(ValueError):
        public_feature(np.zeros((8, 8, 3)), 2)
    with pytest.raises(ValueError):
        public_feature(np.zeros((7, 7, 3)), 7)
    with pytest.raises(ValueError):
        visible_change(np.zeros((7, 7, 3)), np.zeros((8, 8, 3)))
    with pytest.raises(ValueError):
        FrontLogit()(torch.zeros(4, 243))


def test_action_frequency_control_has_zero_optimizer_requirements():
    actions = np.array([0, 2, 2, 5], dtype=np.int64)
    x = np.zeros((4, FEATURES), dtype=np.float32)
    for index, action in enumerate(actions):
        for offset in (0, 11, 17):
            x[index, action * 20 + offset] = 1
    episodes = [{
        "x": x, "y": np.array([1, 1, 0, 0], dtype=np.float32),
        "actions": actions,
        "objects": np.zeros(4, dtype=np.int64),
    }]
    control = fitted_tables(episodes)
    assert len(control["action_rates"]) == 7
    assert len(control["front_object_rates"]) == 7
    assert evaluate(episodes, control)["all"]["native_actions"] == 4
    scores = split_metrics(
        np.array([1.0, 0.0]), np.array([2, 3]), np.array([0.8, 0.1])
    )
    assert scores["forward"]["positives"] == 1
    assert scores["interaction"]["negatives"] == 1
    with pytest.raises(ValueError):
        split_metrics(
            np.array([1.0, 0.0]), np.array([2, 3]), np.array([0.8, np.nan])
        )
