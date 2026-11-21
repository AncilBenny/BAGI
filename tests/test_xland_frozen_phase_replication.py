"""REUSE-24 new-layout disjointness and exact frozen reward anchoring."""
import pytest

from bagi.experiments.xland_frozen_phase_replication import (
    FRESH_SEEDS,
    REFERENCE_SEEDS,
    check_anchor_rewards,
    validate_seed_partition,
)


def test_new_preregistered_layouts_exclude_training_and_prior_evaluation():
    validate_seed_partition(REFERENCE_SEEDS, FRESH_SEEDS)
    assert len(REFERENCE_SEEDS) == 5
    assert len(FRESH_SEEDS) == 10
    with pytest.raises(ValueError, match="changed"):
        validate_seed_partition(REFERENCE_SEEDS, (9604, *FRESH_SEEDS[1:]))


def test_reward_seed_exact_replication_guard():
    source = {
        "models": [{
            "model_seed": 17,
            "conditions": [{
                "condition": "sham_action_history",
                "independent_evaluation": {
                    "hold": {"cases": [
                        {"seed": 9600, "goal_success": 1},
                        {"seed": 9601, "goal_success": 0},
                    ]},
                    "near": {"cases": [
                        {"seed": 9600, "goal_success": 0},
                    ]},
                    "putdown": {"cases": [
                        {"seed": 9600, "goal_success": 0},
                    ]},
                },
            }],
        }]
    }
    measured = {
        "hold": {"rewarded_layout_seeds": [9600]},
        "near": {"rewarded_layout_seeds": []},
        "putdown": {"rewarded_layout_seeds": []},
    }
    assert check_anchor_rewards(
        measured, source, 17, "sham_action_history"
    )
    measured["putdown"]["rewarded_layout_seeds"] = [9600]
    with pytest.raises(AssertionError, match="mismatch"):
        check_anchor_rewards(
            measured, source, 17, "sham_action_history"
        )


def test_not_allowed_to_claim_wrong_policy_seed():
    with pytest.raises(ValueError, match="policy seed"):
        check_anchor_rewards({}, {"models": []}, 19, "sham_action_history")
