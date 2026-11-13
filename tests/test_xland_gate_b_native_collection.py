"""Native runner boundary tests; no simulator or learned-success claims."""

from dataclasses import dataclass

import numpy as np
import pytest

from bagi.experiments.xland_gate_b_native_collection import (
    screen_source_layout,
    source_record_checks,
)


@dataclass
class FakeAgent:
    grid: np.ndarray


@dataclass
class FakeTimestep:
    state: FakeAgent


def grid_at(a=(1, 1), b=(4, 4)):
    board = np.zeros((7, 7, 2), dtype=np.int64)
    blue = np.asarray((5, 2))
    red = np.asarray((4, 0))
    board[a] = blue
    board[b] = red
    return FakeTimestep(FakeAgent(board)), blue, red


def test_source_screen_only_true_distinct_nonadjacent_objects():
    ts, blue, red = grid_at()
    assert screen_source_layout(ts, blue, red)
    ts, blue, red = grid_at((1, 1), (1, 2))
    assert not screen_source_layout(ts, blue, red)
    ts, blue, red = grid_at((1, 1), (1, 1))
    assert not screen_source_layout(ts, blue, red)
    ts, blue, red = grid_at()
    ts.state.grid[0, 1] = blue
    assert not screen_source_layout(ts, blue, red)


def test_rejects_untrusted_source_metadata():
    valid = {"total_expert_queries": 582, "heldout_teacher_labels": 0,
             "model_results": [{"model_seed": 17}, {"model_seed": 29}]}
    source_record_checks(valid)
    with pytest.raises(ValueError, match="Unrecognized"):
        source_record_checks({**valid, "total_expert_queries": 583})
    with pytest.raises(ValueError, match="Unrecognized"):
        source_record_checks({**valid, "heldout_teacher_labels": 1})
    with pytest.raises(ValueError, match="Unrecognized"):
        source_record_checks({**valid, "model_results": [{"model_seed": 17}]})
