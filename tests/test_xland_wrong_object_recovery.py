"""Guard the legal learner input and source control on recovery probe."""
import numpy as np
import pytest

from bagi.experiments.xland_wrong_object_recovery import (
    CAP,
    CONDITION,
    FAILED_LAYOUT,
    MODEL_SEED,
    REFERENCE_LAYOUT,
    action_from_policy,
)


def test_recovery_was_precommitted_to_actual_failure():
    assert MODEL_SEED == 29
    assert FAILED_LAYOUT == 9708
    assert REFERENCE_LAYOUT == 9604
    assert CAP == 40
    assert CONDITION == "real_executed_action_history"


def test_policy_input_uses_only_public_obs_and_action_history():
    class Dummy:
        observation = {
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
        observation = {
            "img": np.zeros((5, 5, 2), dtype=np.uint8),
            "direction": np.array([1, 0, 0, 0], dtype=np.float32),
            "goal_encoding": np.zeros(5, dtype=np.int32),
            "rule_encoding": np.zeros((1, 7), dtype=np.int32),
        }

    with pytest.raises(AssertionError, match="Non-native"):
        action_from_policy(Policy(), Dummy(), [6, 6, 6, 6])
