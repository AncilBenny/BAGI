"""M13 public actor, real DoorKey contract and physical count invariants."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import yaml
from minigrid.core.constants import OBJECT_TO_IDX

from bagi.experiments.m13_public_interaction_coverage import (
    FINAL,
    MODES,
    MONITOR,
    TASKS,
    Controller,
    evaluate_gate,
    front,
    registered,
    summarize,
)


def test_m13_frozen_disjoint_seed_protocol():
    config = yaml.safe_load(Path("configs/m13_public_interaction_coverage.yaml").read_text())
    assert registered(config) == config
    all_seeds = sum(MONITOR + FINAL, ())
    assert len(all_seeds) == len(set(all_seeds))
    assert len(MONITOR[0]) == len(MONITOR[1]) == 24
    assert len(FINAL[0]) == len(FINAL[1]) == 36
    assert not set(all_seeds).intersection(range(120001, 120013))
    incorrect = deepcopy(config)
    incorrect["opening_gain_gate"] = 0
    with pytest.raises(ValueError):
        registered(incorrect)


@pytest.mark.parametrize("task", TASKS)
def test_real_native_public_door_key_observation(task):
    env = gym.make(task)
    try:
        obs, _ = env.reset(seed=31911)
        assert isinstance(front(obs), tuple)
        for mode in MODES:
            actor = Controller(mode)
            rng = np.random.default_rng(19)
            choice = actor.choose(obs, rng)
            assert type(choice) is int and 0 <= choice <= 6
            after, _, _, _, _ = env.step(choice)
            actor.update(obs, choice, after)
            assert isinstance(actor.key_observed_after_pickup, bool)
            obs, _ = env.reset(seed=31911)
    finally:
        env.close()


def _observation(obj, state=0):
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    image[3, 5] = (obj, 0, state)
    return {"image": image, "mission": "use the key", "direction": 0}


def test_public_executed_pickup_evidence_controls_future_toggle():
    key = _observation(OBJECT_TO_IDX["key"])
    empty = _observation(OBJECT_TO_IDX["empty"])
    door = _observation(OBJECT_TO_IDX["door"], 2)
    actor = Controller("public_memory")
    rng = np.random.default_rng(333)
    assert actor.choose(key, rng) == 3
    assert actor.choose(door, rng) in (0, 1)
    actor.update(key, 3, empty)
    assert actor.key_observed_after_pickup
    assert actor.public_pickup_evidence == 1
    assert actor.choose(door, rng) == 5
    with pytest.raises(ValueError):
        Controller("private_carrying_state")


def test_funnel_does_not_accept_empty_coverage():
    with pytest.raises(ValueError):
        summarize([])
    with pytest.raises(ValueError):
        evaluate_gate({})
    with pytest.raises(TypeError):
        front(np.zeros((7, 7, 3)))
