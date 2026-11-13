"""Train-skill score requires physical manipulation, not just goal reward."""
import numpy as np
import pytest

from bagi.experiments.xland_putdown_reward_validity import (
    blue_square_transition,
    parse_saved_reference,
    physically_nonvacuous,
    tiles_adjacent,
)


def test_reward_without_verified_blue_pickup_then_drop_not_skill():
    assert not physically_nonvacuous(1, True, 1, 1, True, True)
    assert not physically_nonvacuous(1, False, 0, 1, True, True)
    assert not physically_nonvacuous(1, False, 1, 0, True, True)
    assert not physically_nonvacuous(1, False, 1, 1, False, True)
    assert not physically_nonvacuous(1, False, 1, 1, True, False)
    assert not physically_nonvacuous(0, False, 1, 1, True, True)
    assert physically_nonvacuous(1, False, 1, 1, True, True)


def test_only_actual_blue_square_native_moves_count():
    empty = np.array([0, 0])
    blue = np.array([3, 2])
    red = np.array([4, 1])
    grid_with_blue = np.zeros((3, 3, 2), dtype=int)
    grid_with_blue[1, 1] = blue
    grid_without_blue = grid_with_blue.copy()
    grid_without_blue[1, 1] = empty
    assert blue_square_transition(
        3, empty, blue, grid_with_blue, grid_without_blue, blue, 0
    ) == (True, False)
    assert blue_square_transition(
        4, blue, empty, grid_without_blue, grid_with_blue, blue, 0
    ) == (False, True)
    # The wrong RED ball can be picked up and put down, but is not BLUE skill.
    assert blue_square_transition(
        3, empty, red, grid_with_blue, grid_without_blue, blue, 0
    ) == (False, False)
    assert blue_square_transition(
        4, red, empty, grid_without_blue, grid_with_blue, blue, 0
    ) == (False, False)
    # Selecting the right action without any real grid change is not enough.
    assert blue_square_transition(
        4, blue, empty, grid_without_blue, grid_without_blue, blue, 0
    ) == (False, False)
    final_grid = grid_with_blue.copy()
    final_grid[1, 2] = red
    assert tiles_adjacent(final_grid, blue, red)
    final_grid[1, 2] = empty
    final_grid[0, 0] = red
    assert not tiles_adjacent(final_grid, blue, red)


def test_saved_source_native_trace_requires_grounded_learned_reward():
    document = {
        "policy_seed": 29,
        "independent_reference_layout": {
            "layout_seed": 9604,
            "reward_success": 1,
            "real_steps": 2,
            "executed_actions": [0, 4],
        },
        "new_online_policy_training_steps": 0,
        "heldout_composition_teacher_queries": 0,
    }
    assert parse_saved_reference(document) == [0, 4]
    document["independent_reference_layout"]["reward_success"] = 0
    with pytest.raises(ValueError, match="source"):
        parse_saved_reference(document)

def test_archived_recovery_trace_frozen_and_train_only():
    from bagi.experiments.xland_putdown_reward_validity import parse_saved_recovery

    actions = [0, 0, 1, 3, 4, 2, 0, 0, 0, 0, 1, 0,
               0, 0, 0, 3, 1, 0, 0, 0, 0, 1, 0, 4]
    document = {
        "policy_seed": 29,
        "new_online_policy_training_steps": 0,
        "heldout_composition_teacher_queries": 0,
        "original_failed_layout": {"layout_seed": 9708, "reward_success": 0},
        "recovery_probe": {
            "actual_learner_failure_seed": 9708,
            "checkpoint_wrong_red_held": True,
            "learner_executed_prefix": actions[:4],
            "manual_wrong_red_release_steps": 1,
            "wrong_red_release_restored_empty_pocket": True,
            "wrong_red_release_restored_red_tile": True,
            "subsequent_train_only_correction_queries": 19,
            "successful_goal_recovery": 1,
            "recovered_in_40_step_cap": True,
            "total_sequence_steps": 24,
            "correction_actions": actions[5:],
        },
    }
    assert parse_saved_recovery(document) == actions
    document["recovery_probe"]["correction_actions"][-1] = 3
    with pytest.raises(ValueError, match="changed"):
        parse_saved_recovery(document)
    document["recovery_probe"]["correction_actions"][-1] = 4
    document["heldout_composition_teacher_queries"] = 1
    with pytest.raises(ValueError, match="provenance"):
        parse_saved_recovery(document)


