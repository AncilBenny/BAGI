"""M15 learns only from actual public observations and executed actions."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml

from bagi.experiments.m15_public_action_bc import (
    ARMS,
    FINAL,
    MONITOR,
    SEEDS,
    TASKS,
    TRAIN,
    WIDTH,
    BCPolicy,
    class_weights,
    features,
    fit,
    preregistered_gate,
    validate_config,
)


def test_frozen_training_monitor_final_layouts_and_arms():
    cfg = yaml.safe_load(Path("configs/m15_public_bc.yaml").read_text())
    assert validate_config(cfg) == cfg
    layouts = sum(TRAIN + MONITOR + FINAL, ())
    assert len(layouts) == len(set(layouts))
    assert not set(layouts).intersection(range(202001, 202037))
    assert len(TRAIN[0]) == len(TRAIN[1]) == 32
    assert len(MONITOR[0]) == len(MONITOR[1]) == 8
    assert len(FINAL[0]) == len(FINAL[1]) == 24
    assert len(SEEDS) == len(ARMS) == 2
    altered = deepcopy(cfg)
    altered["balanced_weight_cap"] = 1.0
    with pytest.raises(ValueError):
        validate_config(altered)


@pytest.mark.parametrize("task", TASKS)
def test_real_native_public_features_and_history_bit(task):
    env = gym.make(task)
    try:
        obs, _ = env.reset(seed=33017)
        before = features(obs, False)
        after = features(obs, True)
        assert before.shape == after.shape == (WIDTH,)
        assert before.dtype == np.float32
        assert before[-1] == 0 and after[-1] == 1
        np.testing.assert_array_equal(before[:-1], after[:-1])
        assert np.isfinite(before).all()
        logits = BCPolicy()(torch.from_numpy(before).unsqueeze(0))
        assert logits.shape == (1, 7)
    finally:
        env.close()


def test_fresh_cross_entropy_optimizer_counts_and_label_weights():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((256, WIDTH)).astype(np.float32)
    y = np.concatenate((
        np.zeros(240, dtype=np.int64),
        np.ones(16, dtype=np.int64),
    ))
    weights = class_weights(y)
    assert weights[1] > weights[0]
    assert weights.max() <= 8
    torch.manual_seed(SEEDS[0])
    ordinary = BCPolicy()
    metrics = fit(ordinary, x, y, seed=SEEDS[0], balanced=False)
    assert metrics["optimizer_step_calls"] == 20
    assert metrics["cached_sample_presentations"] == 2560
    assert metrics["class_weights"] is None
    assert metrics["parameter_count"] > 0


def test_missing_native_final_model_refuses_positive_claim():
    with pytest.raises(ValueError):
        preregistered_gate({})
    with pytest.raises(ValueError):
        features({}, False)
    with pytest.raises(ValueError):
        BCPolicy()(torch.zeros((5, 243)))
