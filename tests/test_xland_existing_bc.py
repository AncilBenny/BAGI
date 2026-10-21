"""Information-boundary and reproducibility checks for the reused BC baseline."""
import numpy as np
import pytest

from bagi.experiments.xland_existing_bc import (
    ALL_TASKS,
    TRAIN_TASKS,
    collect,
    get_rules,
    validate_config,
    visible_features,
)


def good():
    return {
        "environment": "XLand-MiniGrid-R1-9x9",
        "demonstration_labels_per_curriculum": 256,
        "batch_size": 32, "epochs": 12,
        "episode_cap": 40, "max_collection_attempts": 200,
        "demonstration_seed_base": 5100,
        "evaluation_seeds": [9100, 9101, 9102],
        "policy_training_seeds": [17, 29],
    }


def test_exact_positive_budget_and_nonoverlapping_seeds():
    assert validate_config(good()) == (256, 40, [9100, 9101, 9102], [17, 29])
    with pytest.raises(ValueError, match="overlap"):
        validate_config({**good(), "evaluation_seeds": [5150]})


def test_only_training_skills_permitted_for_demonstrations():
    assert ALL_TASKS[-1] == "hold_then_near"
    with pytest.raises(ValueError, match="TRAIN"):
        collect(None, None, ("hold_then_near",), budget=8,
                seed_base=0, cap=40, max_attempts=5)
    with pytest.raises(ValueError, match="Unapproved"):
        get_rules("hidden_extra_skill")
    assert len(TRAIN_TASKS) == 3


def test_learning_observation_rejects_privileged_components():
    visible = {
        "img": np.arange(98, dtype=np.uint8).reshape(7, 7, 2),
        "direction": np.array([0, 1, 0, 0]),
        "grid": np.full((9, 9, 2), 99), "rule_encoding": np.arange(8),
    }
    result = visible_features(visible)
    assert result.shape == (102,)
    assert result[-4:].tolist() == [0, 1, 0, 0]
    assert 99 not in result
    assert result.dtype == np.float32


def test_reject_malformed_direction():
    with pytest.raises(ValueError, match="4-way"):
        visible_features({"img": np.zeros((7, 7, 2)), "direction": [1, 0]})
