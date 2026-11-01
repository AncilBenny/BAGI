"""REUSE-31 synthetic schema/accounting guards; no simulator or training."""
from copy import deepcopy

import pytest

from bagi.experiments.xland_gate_b_preflight import summarize


def archived_shaped_inputs():
    cases = []
    for seed, checkpoint, actions, stages in (
        (9701, 4, [1, 0, 0, 1, 0, 0, 0, 0, 3, 2, 4],
         {"seek_square": 9, "carrying_square": 2}),
        (9702, 10, [2, 3, 1, 0, 1, 0, 0, 0, 4],
         {"seek_square": 2, "carrying_square": 7}),
    ):
        prefix = [0] * checkpoint
        cases.append({
            "seed": seed, "checkpoint": checkpoint,
            "excluded_initial_adjacency": False, "initial_distance": 3,
            "teacher_assisted_physical_placement": True,
            "learned_success": False,
            "effective_blue_pickups_on_full_trace": 1,
            "effective_blue_drops_on_full_trace": 1,
            "ordered_blue_pickup_then_drop": True,
            "final_blue_adjacent_to_red": True,
            "teacher_errors": [],
            "learner_only": {
                "reward_success": 0, "real_steps": 40,
                "executed_actions": [*prefix, *([1] * (40 - checkpoint))],
            },
            "learner_prefix_actions": prefix,
            "teacher_actions": actions,
            "teacher_label_queries": len(actions),
            "teacher_correction_stage_counts": stages,
            "native_simulator_step_calls": 40 + checkpoint + len(actions),
        })
    reuse30 = {
        "registered_probes": [[9701, 4], [9702, 10]],
        "policy_model_seed": 29,
        "fixed_model_training_labels_source": 39637067506,
        "valid_distinct_checkpoints": True,
        "new_policy_training_steps": 0,
        "heldout_composition_teacher_queries": 0,
        "total_train_only_teacher_label_queries": 20,
        "total_native_simulator_step_calls": 114,
        "cases": cases,
    }
    control = {
        "policy_seed": 29,
        "recovery_probe": {
            "native_actions_sha256_le_int64":
                "3b620a8c6090f756a0dcf4e9af276a1526c59f040c5708dcad77f8ba8a630d36",
            "teacher_wrong_red_release_queries": 1,
            "subsequent_train_only_correction_queries": 19,
        },
        "actual_train_only_expert_label_queries": 20,
        "heldout_composition_teacher_queries": 0,
    }
    return reuse30, control


def test_preflight_counts_only_real_learner_checkpoint_labels():
    result = summarize(*archived_shaped_inputs())
    assert result["reuse30_learner_visited_checkpoint_labels"] == 2
    assert result["reuse30_expert_controlled_continuation_labels"] == 18
    assert result["reuse30_learner_checkpoint_fraction"] == 0.1
    assert result["combined_reuse29_30_learner_checkpoint_labels"] == 3
    assert result["archived_public_observation_rows"] == 0
    assert result["new_real_simulator_steps"] == 0


def test_preflight_rejects_prefix_not_observed_from_learner():
    source, control = archived_shaped_inputs()
    source = deepcopy(source)
    source["cases"][0]["learner_prefix_actions"][0] = 5
    with pytest.raises(ValueError, match="real learner checkpoint"):
        summarize(source, control)


def test_preflight_rejects_missing_teacher_query_and_vacuous_layout():
    source, control = archived_shaped_inputs()
    source["cases"][1]["teacher_label_queries"] -= 1
    with pytest.raises(ValueError, match="actions/queries"):
        summarize(source, control)
    source, control = archived_shaped_inputs()
    source["cases"][0]["initial_distance"] = 1
    with pytest.raises(ValueError, match="vacuous"):
        summarize(source, control)


def test_preflight_rejects_claim_of_new_policy_training():
    source, control = archived_shaped_inputs()
    source["new_policy_training_steps"] = 1
    with pytest.raises(ValueError, match="retraining"):
        summarize(source, control)
