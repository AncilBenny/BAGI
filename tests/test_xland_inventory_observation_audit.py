"""REUSE-22: observation equality and no hidden fields in learner view."""
import numpy as np
import pytest

from bagi.experiments.xland_inventory_observation_audit import (
    assert_identical_learner_views,
    hash_array,
)


def dummy():
    return {
        "img": np.zeros((5, 5, 2), dtype=np.uint8),
        "direction": np.array([1, 0, 0, 0], dtype=np.float32),
        "goal_encoding": np.array([1, 2, 3, 4, 5], dtype=np.int32),
        "rule_encoding": np.ones((1, 7), dtype=np.int32),
    }


def test_upstream_fields_are_identical_under_opaque_hidden_state():
    a, b = dummy(), dummy()
    assert all(assert_identical_learner_views(a, b).values())
    assert hash_array(a["img"]) == hash_array(b["img"])


def test_visible_goal_or_image_change_is_caught():
    a, b = dummy(), dummy()
    b["rule_encoding"][0, 0] += 1
    with pytest.raises(AssertionError, match="visible"):
        assert_identical_learner_views(a, b)
    b = dummy()
    b["img"][0, 0, 0] = 9
    with pytest.raises(AssertionError, match="visible"):
        assert_identical_learner_views(a, b)


def test_missing_task_instruction_field_rejected():
    a, b = dummy(), dummy()
    del a["goal_encoding"]
    with pytest.raises(ValueError, match="upstream"):
        assert_identical_learner_views(a, b)
