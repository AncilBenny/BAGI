"""M14 preserves M13 baseline and changes only public open-door traversal."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import yaml
from minigrid.core.constants import OBJECT_TO_IDX

from bagi.experiments.m13_public_interaction_coverage import Controller
from bagi.experiments.m14_public_open_door import (
    FINAL,
    MONITOR,
    TASKS,
    TraversableOpenDoorController,
    judge,
    validate_config,
)


def test_new_frozen_seeds_not_reused():
    c = yaml.safe_load(Path("configs/m14_public_open_door.yaml").read_text())
    assert validate_config(c) == c
    all_seeds = sum(MONITOR + FINAL, ())
    assert len(all_seeds) == len(set(all_seeds))
    assert len(FINAL[0]) == len(FINAL[1]) == 36
    assert not set(all_seeds).intersection(range(172001, 172037))
    assert not set(all_seeds).intersection(range(182001, 182037))
    wrong = deepcopy(c)
    wrong["minimum_pooled_goal_gain"] = 0
    with pytest.raises(ValueError):
        validate_config(wrong)


def fake(obj, state=0):
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    image[3, 5] = (obj, 0, state)
    return {"image": image, "direction": 0, "mission": "use the key"}


def test_public_open_door_corrects_blocked_navigation_only():
    door = fake(OBJECT_TO_IDX["door"], 0)
    key = fake(OBJECT_TO_IDX["key"])
    locked = fake(OBJECT_TO_IDX["door"], 2)
    empty = fake(OBJECT_TO_IDX["empty"])
    original = Controller("public_memory")
    corrected = TraversableOpenDoorController()
    rng_old = np.random.default_rng(44)
    rng_new = np.random.default_rng(44)
    assert original.choose(door, rng_old) in (0, 1)
    assert corrected.choose(door, rng_new) == 2
    assert original.choose(key, rng_old) == corrected.choose(key, rng_new) == 3
    original.update(key, 3, empty)
    corrected.update(key, 3, empty)
    assert original.choose(locked, rng_old) == corrected.choose(locked, rng_new) == 5


@pytest.mark.parametrize("task", TASKS)
def test_real_native_public_obs_and_open_state_contract(task):
    env = gym.make(task)
    try:
        obs, _ = env.reset(seed=31955)
        actor = TraversableOpenDoorController()
        action = actor.choose(obs, np.random.default_rng(55))
        assert type(action) is int and 0 <= action <= 6
        successor, _, _, _, _ = env.step(action)
        actor.update(obs, action, successor)
    finally:
        env.close()


def test_incomplete_paired_final_scoring_refused():
    with pytest.raises(ValueError):
        judge({})
