"""Synthetic contract tests: no environment calls and no reward leakage."""

import numpy as np
import pytest

from bagi.exploration.public_count import EpisodicFirstVisit


def test_first_visit_and_reset():
    count = EpisodicFirstVisit(bonus=0.25)
    start = np.array([0, 1], dtype=np.int8)
    new = np.array([1, 1], dtype=np.int8)
    count.reset(start)
    assert count.unique_public_views == 1
    assert count.observe(start) == 0
    assert count.observe(new) == 0.25
    assert count.observe(new) == 0
    assert count.unique_public_views == 2
    count.reset(start)
    assert count.observe(new) == 0.25


def test_equal_public_views_with_different_hidden_states_same_bonus():
    # No private state is accepted or even available as an argument.
    view = np.array([[4, 0], [1, 2]], dtype=np.int8)
    first = EpisodicFirstVisit()
    second = EpisodicFirstVisit()
    first.reset(view.copy())
    second.reset(view.copy())
    hidden_state_a, hidden_state_b = {"has_key": False}, {"has_key": True}
    assert hidden_state_a != hidden_state_b
    assert first.observe(view.copy()) == second.observe(view.copy()) == 0.0


def test_invalid_observation_and_uninitialized_rejected():
    counter = EpisodicFirstVisit()
    with pytest.raises(RuntimeError):
        counter.observe(np.array([1], dtype=np.int8))
    with pytest.raises(ValueError):
        counter.reset(np.array([float("nan")]))
    with pytest.raises(ValueError):
        EpisodicFirstVisit(float("inf"))
    with pytest.raises(ValueError):
        EpisodicFirstVisit(-1)
    counter.reset(np.array([0, 0], dtype=np.int8))
    with pytest.raises(ValueError):
        counter.observe(np.array([1], dtype=np.int64))
    with pytest.raises(ValueError):
        counter.observe(np.array([1, 1, 1], dtype=np.int8))


def test_view_storage_does_not_mutate_public_inputs():
    view = np.array([[0, 1], [2, 3]], dtype=np.int8)
    original = view.copy()
    counter = EpisodicFirstVisit(0.3)
    counter.reset(view)
    assert counter.observe(view[::-1]) == 0.3
    np.testing.assert_array_equal(view, original)


def test_reward_is_intrinsic_only_and_input_is_observation_only():
    import inspect

    counter = EpisodicFirstVisit(bonus=0.2)
    assert list(inspect.signature(counter.observe).parameters) == ["next_public_observation"]
    counter.reset(np.array([0], dtype=np.int8))
    native_extrinsic = 1.0
    intrinsic = counter.observe(np.array([1], dtype=np.int8))
    assert native_extrinsic == 1.0
    assert intrinsic == 0.2
    assert counter.observe(np.array([1], dtype=np.int8)) == 0.0
