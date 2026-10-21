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
