"""Coverage of action put-down via already-implemented XLand goal primitives."""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("xminigrid")

from bagi.experiments.xland_action_coverage import (
    ACTIONS,
    locate_exact,
    plan_putdown_after_pickup,
    putdown_ruleset,
)


def test_upstream_goal_rewards_action_four_and_no_new_rule():
    ruleset = putdown_ruleset()
    assert int(ruleset.goal[0]) == 4  # XLand TileNearGoal
    assert list(np.asarray(ruleset.rules)[:, 0]) == [0]  # XLand EmptyRule
    assert len(ruleset.init_tiles) == 2
    assert ACTIONS["put_down"] == 4


def test_navigate_to_free_drop_cell_near_object():
    grid = np.ones((7, 7, 2), dtype=np.uint8)
    grid[0, :, 0] = grid[-1, :, 0] = 2
    grid[:, 0, 0] = grid[:, -1, 0] = 2
    grid[3, 3, 0] = 3  # Red ball
    route, target = plan_putdown_after_pickup(grid, (1, 1), 0, (3, 3))
    assert route
    assert abs(target[0] - 3) + abs(target[1] - 3) == 1
    assert int(grid[target[0], target[1], 0]) == 1


def test_exact_object_lookup_rejects_missing_or_duplicate_objects():
    grid = np.ones((5, 5, 2), dtype=np.uint8)
    tile = np.array([4, 3], dtype=np.uint8)
    with pytest.raises(ValueError, match="exactly one"):
        locate_exact(grid, tile)
    grid[1, 1] = tile
    assert locate_exact(grid, tile) == (1, 1)
    grid[2, 2] = tile
    with pytest.raises(ValueError, match="exactly one"):
        locate_exact(grid, tile)
