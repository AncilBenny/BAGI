"""Guard phase probes and REUSE-17 reward anchors."""
from types import SimpleNamespace

import numpy as np

from bagi.experiments.xland_phase_diagnostics import (
    FROZEN_FRESH_SUCCESSES,
    inventory_and_grid,
)


def test_phase_flags_detect_carried_tile_and_adjacent_objects():
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles

    grid = np.zeros((5, 5, 2), dtype=np.uint8)
    square = np.asarray(TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE])
    red = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.RED])
    grid[2, 2] = square
    grid[2, 3] = red
    state = SimpleNamespace(
        grid=grid, agent=SimpleNamespace(pocket=square)
    )
    phases = inventory_and_grid(state)
    assert phases["carrying_square"] and phases["square_next_to_red"]
    assert not phases["pocket_empty"] and not phases["green_on_grid"]


def test_empty_pocket_not_confused_with_correct_drop():
    from xminigrid.core.constants import Tiles

    state = SimpleNamespace(
        grid=np.zeros((5, 5, 2), dtype=np.uint8),
        agent=SimpleNamespace(pocket=np.array([Tiles.EMPTY, 0])),
    )
    phases = inventory_and_grid(state)
    assert phases["pocket_empty"]
    assert not phases["square_next_to_red"]
    assert not phases["carrying_red"]


def test_frozen_reward_seed_sets_not_tuned():
    assert FROZEN_FRESH_SUCCESSES[17]["hold"] == (9202, 9203, 9204)
    assert FROZEN_FRESH_SUCCESSES[17]["near"] == (9203, 9204)
    assert FROZEN_FRESH_SUCCESSES[17]["putdown"] == ()
    assert all(not v for v in FROZEN_FRESH_SUCCESSES[29].values())
