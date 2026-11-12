"""Contracts for new public-effect continual learning; no hidden state input."""
from __future__ import annotations

import copy
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml

from bagi.experiments.m11_public_effect_retention import (
    FINAL,
    SEEDS,
    TASKS,
    TRAIN,
    EffectGRU,
    Episode,
    collect,
    evaluate,
    learn,
    public_input,
    public_label,
    validate_config,
)
from bagi.experiments.m5_continual_retention import PublicGoalActions


def test_fixed_untouched_episode_split_and_protocol():
    c = yaml.safe_load(Path("configs/m11_public_effect_retention.yaml").read_text())
    assert validate_config(c) == c
    seeds = sum(TRAIN + FINAL, ())
    assert len(seeds) == len(set(seeds))
    assert len(SEEDS) == 2
    assert len(TRAIN[0]) == len(TRAIN[1]) == 24
    assert len(FINAL[0]) == len(FINAL[1]) == 12
    assert not set(seeds).intersection(range(120001, 120013))
    changed = copy.deepcopy(c)
    changed["target"] = "private_door_opening"
    with pytest.raises(ValueError):
        validate_config(changed)


@pytest.mark.parametrize("task", TASKS)
def test_real_native_public_only_observations(task):
    env = PublicGoalActions(gym.make(task))
    try:
        obs, _ = env.reset(seed=31009)
        assert obs.shape == (243,)
        action = 0
        x = public_input(obs, action)
        after, _, _, _, _ = env.step(action)
        assert x.shape == (250,)
        assert x[-7:].tolist() == [1, 0, 0, 0, 0, 0, 0]
        assert public_label(obs, after) in (0.0, 1.0)
        assert np.array_equal(x[:243], obs)
    finally:
        env.close()


def fake_episode(seed, positive):
    x = torch.zeros((4, 250), dtype=torch.float32)
    y = torch.tensor([positive, 1 - positive, positive, 0], dtype=torch.float32)
    return Episode(seed=seed, x=x, y=y, actions=[0, 1, 2, 3], rewarded_goals=0)


def test_rehearsal_and_naive_have_equal_new_task_presentation_and_updates():
    old = [fake_episode(1, 0), fake_episode(2, 1)]
    new = [fake_episode(3, 1), fake_episode(4, 0)]
    torch.manual_seed(7)
    initial = EffectGRU()
    naive = copy.deepcopy(initial)
    replay = copy.deepcopy(initial)
    one = learn(naive, new, seed=11)
    two = learn(replay, new, seed=11, replay=old)
    assert one["optimizer_step_calls"] == two["optimizer_step_calls"] == 8
    assert one["new_task_sample_presentations"] == two[
        "new_task_sample_presentations"
    ]
    assert one["old_cached_replay_sample_presentations"] == 0
    assert two["old_cached_replay_sample_presentations"] == 32
    assert 0 <= evaluate(replay, old)["brier"] <= 1


def test_unknown_split_rejected_and_wrong_public_shape_rejected():
    with pytest.raises(ValueError):
        collect(TASKS[0], FINAL[1])
    with pytest.raises(ValueError):
        public_input(np.zeros(242, dtype=np.float32), 1)
    with pytest.raises(ValueError):
        public_input(np.zeros(243, dtype=np.float32), 7)
    with pytest.raises(ValueError):
        public_label(np.zeros(243), np.zeros(242))
    with pytest.raises(ValueError):
        EffectGRU()(torch.zeros((4, 243)))
