"""Guardrails for an exact-budget, no-pickle upstream PPO integration."""
import pytest

from bagi.experiments.xland_upstream_recurrent_pilot import validate_settings


def settings():
    return {
        "environment": "XLand-MiniGrid-R1-9x9",
        "training": {"num_envs": 4, "steps_per_env": 64, "steps_per_update": 16,
                     "num_minibatches": 2, "real_steps": 8192},
        "evaluation": {"seeds": [6800, 6801, 6802],
                       "episode_cap": 40},
    }


def test_fixed_budget_and_separate_evaluation_seeds():
    assert validate_settings(settings()) == (8192, [6800, 6801, 6802])


@pytest.mark.parametrize("overrides", [
    {"real_steps": 8191},
    {"steps_per_env": 63},
    {"num_envs": 0},
    {"num_minibatches": 3},
])
def test_bad_training_budgets_fail_closed(overrides):
    cfg = settings()
    cfg["training"].update(overrides)
    with pytest.raises(ValueError, match="divisible"):
        validate_settings(cfg)


def test_duplicate_evaluation_seeds_rejected():
    cfg = settings()
    cfg["evaluation"]["seeds"] = [6800, 6800]
    with pytest.raises(ValueError, match="unique"):
        validate_settings(cfg)


def test_only_approved_environment():
    cfg = settings()
    cfg["environment"] = "XLand-MiniGrid-R4-13x13"
    with pytest.raises(ValueError, match="9x9"):
        validate_settings(cfg)
