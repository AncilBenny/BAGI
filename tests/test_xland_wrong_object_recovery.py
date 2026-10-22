"""Guard the legal learner input and source control on recovery probe."""
from typing import ClassVar

import numpy as np
import pytest

from bagi.experiments.xland_wrong_object_recovery import (
    CAP,
    CONDITION,
    FAILED_LAYOUT,
    MODEL_SEED,
    REFERENCE_LAYOUT,
    action_from_policy,
    native_trace_sha256,
)


def test_recovery_was_precommitted_to_actual_failure():
    assert MODEL_SEED == 29
    assert FAILED_LAYOUT == 9708
    assert REFERENCE_LAYOUT == 9604
    assert CAP == 40
    assert CONDITION == "real_executed_action_history"


def test_policy_input_uses_only_public_obs_and_action_history():
    class Dummy:
        observation: ClassVar[dict] = {
            "img": np.zeros((5, 5, 2), dtype=np.uint8),
            "direction": np.array([1, 0, 0, 0], dtype=np.float32),
            "goal_encoding": np.zeros(5, dtype=np.int32),
            "rule_encoding": np.zeros((1, 7), dtype=np.int32),
        }

    class Policy:
        def predict(self, values, deterministic):
            assert values.shape == (115,)
            assert deterministic
            return np.array(4), None

    assert action_from_policy(Policy(), Dummy(), [6, 6, 6, 6]) == 4


def test_out_of_range_policy_action_is_forbidden():
    class Policy:
        def predict(self, values, deterministic):
            return np.array(7), None

    class Dummy:
        observation: ClassVar[dict] = {
            "img": np.zeros((5, 5, 2), dtype=np.uint8),
            "direction": np.array([1, 0, 0, 0], dtype=np.float32),
            "goal_encoding": np.zeros(5, dtype=np.int32),
            "rule_encoding": np.zeros((1, 7), dtype=np.int32),
        }

    with pytest.raises(AssertionError, match="Non-native"):
        action_from_policy(Policy(), Dummy(), [6, 6, 6, 6])


def test_train_only_teacher_releases_wrong_red_only_into_free_floor():
    """RED is missing from the grid precisely because it is held."""
    from types import SimpleNamespace

    pytest.importorskip("xminigrid")
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles

    from bagi.experiments.xland_dagger_teacher_feasibility import correction_plan

    grid = np.zeros((9, 9, 2), dtype=np.int32)
    grid[:, :, 0] = int(Tiles.FLOOR)
    red = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.RED])
    agent = SimpleNamespace(
        position=np.asarray([4, 4]), direction=np.asarray(1),
        pocket=red.copy(),
    )
    ts = SimpleNamespace(state=SimpleNamespace(grid=grid, agent=agent))
    assert correction_plan(ts, "putdown") == ([4], "release_wrong_red")
    grid[4, 5, 0] = int(Tiles.WALL)
    with pytest.raises(ValueError, match="blocked cell"):
        correction_plan(ts, "putdown")


def test_native_recovery_trace_hash_uses_executed_release_once():
    import hashlib

    prefix = [0, 0, 1, 3]
    release = 4
    corrective_actions = [0, 1, 3]
    expected_actions = [0, 0, 1, 3, 4, 0, 1, 3]
    expected_digest = hashlib.sha256(
        np.asarray(expected_actions, dtype="<i8").tobytes()
    ).hexdigest()
    assert native_trace_sha256(prefix, release, corrective_actions) == expected_digest
