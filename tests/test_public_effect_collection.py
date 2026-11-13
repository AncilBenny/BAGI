"""Synthetic action/provenance tests only; no native G2-E1 result."""
from __future__ import annotations

import numpy as np
import pytest

from bagi.agents.public_effect_collection import (
    collect_public_episode,
    verify_fixed_collection,
)


def rollout(layout="train", ident="train-1", *, steps=4, behavior="synthetic"):
    state = np.zeros(5, dtype=np.float32)
    calls = []

    def actor(view):
        assert not view.flags.writeable
        assert np.array_equal(view, state)
        return 1

    def step(action):
        assert action == 1
        calls.append(action)
        state[4] = len(calls)  # synthetic actual-history-like feature
        if len(calls) == 2:
            state[0] = 1  # genuine public change
        return state.copy(), 0.0, len(calls) == steps, len(calls) == 1

    audit = collect_public_episode(
        layout=layout, episode_id=ident, behavior=behavior,
        initial_public=state.copy(), choose_action=actor,
        native_step=step, width=5, n_actions=3, cap=10, history_bits=1,
    )
    return audit, calls


def test_actual_steps_and_history_exclusion_and_hidden_disagreement():
    audit, calls = rollout()
    assert calls == [1] * 4
    assert audit.native_actions == 4 and audit.terminal
    assert audit.episode.visible_changes.tolist() == [0, 1, 0, 0]
    assert audit.episode.actions.tolist() == calls
    assert audit.hidden_without_public_change == 1
    assert audit.evaluator_hidden_changes == 1
    assert audit.episode.observations[0, 4] == 0
    assert audit.episode.observations[-1, 4] == 3


def test_episode_cap_never_calls_more_native_steps():
    state = np.zeros(3)
    called = []

    def step(action):
        called.append(action)
        return state.copy(), 0.0, False, False

    result = collect_public_episode(
        layout="x", episode_id="x", behavior="random",
        initial_public=state, choose_action=lambda _: 0,
        native_step=step, width=3, n_actions=2, cap=2,
    )
    assert result.native_actions == len(called) == 2
    assert not result.terminal


def test_reject_invalid_native_actions_without_step():
    calls = []
    with pytest.raises(ValueError, match="out-of-range"):
        collect_public_episode(
            layout="x", episode_id="x", behavior="bad",
            initial_public=np.zeros(3), choose_action=lambda _: 4,
            native_step=lambda a: calls.append(a), width=3, n_actions=2, cap=2,
        )
    assert calls == []


def test_reject_malformed_native_result():
    with pytest.raises(ValueError, match="evaluator flags"):
        collect_public_episode(
            layout="x", episode_id="x", behavior="bad",
            initial_public=np.zeros(3), choose_action=lambda _: 0,
            native_step=lambda _: (np.zeros(3), 0.0, False, 1),
            width=3, n_actions=2, cap=2,
        )


def test_exact_layout_split_and_budget_gate():
    train = [rollout("train", "train-1")[0], rollout("train", "train-2")[0]]
    test = [rollout("test", "test-1")[0], rollout("test", "test-2")[0]]
    out = verify_fixed_collection(
        train, test, train_layouts=("train",), test_layouts=("test",),
        episodes_per_layout=2, width=5, n_actions=3, cap=10,
        max_native_actions=16,
    )
    assert out["native_actions"] == 16
    assert out["hidden_without_public_change"] == 4
    with pytest.raises(ValueError, match="budget"):
        verify_fixed_collection(
            train, test, train_layouts=("train",), test_layouts=("test",),
            episodes_per_layout=2, width=5, n_actions=3, cap=10,
            max_native_actions=15,
        )
    with pytest.raises(ValueError, match="registered layout"):
        verify_fixed_collection(
            train, test[:1], train_layouts=("train",), test_layouts=("test",),
            episodes_per_layout=2, width=5, n_actions=3, cap=10,
            max_native_actions=16,
        )
    with pytest.raises(ValueError, match="layout registration"):
        verify_fixed_collection(
            train, test, train_layouts=("train",), test_layouts=("train",),
            episodes_per_layout=2, width=5, n_actions=3, cap=10,
            max_native_actions=16,
        )
