"""Upstream task-instruction encoding guard, no XLand simulator required."""
import numpy as np
import pytest

from bagi.experiments.xland_upstream_instruction_bc import (
    INPUT_WIDTH, MAX_RULES, observation_with_instruction, task_instructions,
)


def test_instructed_and_sham_have_same_input_width():
    goal = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    rule = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
    instructed = task_instructions(goal, rule, enabled=True)
    sham = task_instructions(goal, rule, enabled=False)
    assert MAX_RULES == 4
    assert instructed.shape == sham.shape == (33,)
    assert not np.array_equal(instructed, sham)
    assert np.count_nonzero(sham) == 0
    a = observation_with_instruction(np.zeros(54, dtype=np.float32), instructed)
    b = observation_with_instruction(np.zeros(54, dtype=np.float32), sham)
    assert a.shape == b.shape == (INPUT_WIDTH,)
    np.testing.assert_array_equal(a[:54], b[:54])


def test_multiple_rules_pad_but_never_truncate():
    x = task_instructions(np.ones(5), np.ones((2, 7)), enabled=True)
    assert x.shape == (33,)
    with pytest.raises(ValueError, match="count"):
        task_instructions(np.ones(5), np.ones((5, 7)), enabled=True)


def test_wrong_encoder_shapes_rejected():
    with pytest.raises(ValueError, match="shape"):
        task_instructions(np.ones(4), np.ones((1, 7)), enabled=True)
    with pytest.raises(ValueError, match="size"):
        observation_with_instruction(np.ones(53), np.ones(33))
