"""REUSE-23: exact prior-action sourcing and equal-width instruction history."""
import numpy as np
import pytest

from bagi.experiments.xland_instruction_action_history import (
    CONDITIONS,
    WIDTH,
    assemble,
    check_true_history,
    encode_context,
)


def audit_rows():
    return {
        "round": np.array([0, 0, 1], dtype=np.int64),
        "episode": np.array([0, 0, 1], dtype=np.int64),
        "step_in_episode": np.array([0, 1, 0], dtype=np.int64),
        "done": np.array([False, True, True], dtype=bool),
        "executed_action": np.array([1, 2, 0], dtype=np.int64),
        "expert_label": np.array([3, 4, 3], dtype=np.int64),
        "prior_executed_actions": np.array([
            [6, 6, 6, 6], [6, 6, 6, 1], [6, 6, 6, 6],
        ], dtype=np.int64),
    }


def test_previous_real_actions_are_not_current_expert_labels():
    arr = audit_rows()
    check_true_history(arr)
    arr["prior_executed_actions"][1, -1] = 3
    with pytest.raises(ValueError, match="actual execution"):
        check_true_history(arr)


def test_matched_inputs_and_identical_public_instructions():
    data = audit_rows()
    local = np.zeros((3, 54), dtype=np.float32)
    public = np.ones((3, 33), dtype=np.float32)
    a = assemble(
        local, public, data["prior_executed_actions"],
        CONDITIONS[0],
    )
    b = assemble(
        local, public, data["prior_executed_actions"],
        CONDITIONS[1],
    )
    assert a.shape == b.shape == (3, WIDTH)
    assert WIDTH == 115
    np.testing.assert_array_equal(a[:, :87], b[:, :87])
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.array_equal(a[1, 87:], b[1, 87:])


def test_invalid_native_history_actions_rejected():
    with pytest.raises(ValueError, match="Invalid"):
        encode_context(np.array([[6, 6, 6, 7]]), condition=CONDITIONS[1])
    with pytest.raises(ValueError, match="Unapproved"):
        encode_context(np.array([[6, 6, 6, 6]]), condition="expert_history")
