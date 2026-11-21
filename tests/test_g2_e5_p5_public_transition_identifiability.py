"""Synthetic-only P5 aliasing and no-future/public-history contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from bagi.experiments.g2_e5_p5_public_transition_identifiability import (
    _shift_within_episodes,
    audit_transitions,
)


def test_repeated_public_action_can_have_different_public_outcomes():
    obs = np.zeros((4, 3), dtype=np.int8)
    act = np.array([1, 0, 1, 0])
    nxt = np.array([[1, 0, 0], [0, 0, 0], [2, 0, 0], [0, 0, 0]], dtype=np.int8)
    groups = np.array([0, 0, 1, 1])
    out = audit_transitions(obs, act, nxt, groups, n_actions=3)
    direct = out["variants"]["executed_history_0"]
    history = out["variants"]["executed_history_4"]
    assert direct["ambiguous_keys"] == 1
    assert direct["ambiguous_key_rows"] == 2
    assert direct["cross_episode_ambiguous_keys"] == 1
    assert history["ambiguous_keys"] == 1  # identical episode-start context


def test_only_earlier_executed_actions_can_disambiguate():
    obs = np.zeros((4, 2), dtype=np.int8)
    act = np.array([0, 1, 2, 1])
    nxt = np.array([[0, 0], [1, 0], [0, 0], [2, 0]], dtype=np.int8)
    groups = np.array([0, 0, 1, 1])
    out = audit_transitions(obs, act, nxt, groups, n_actions=3)
    assert out["variants"]["executed_history_0"]["ambiguous_keys"] == 1
    assert out["variants"]["executed_history_4"]["ambiguous_keys"] == 0
    assert out["variants"]["executed_history_4"]["repeated_key_rows"] == 0


def test_target_history_suffix_excluded_from_effect_label():
    obs = np.zeros((2, 5), dtype=np.int8)
    act = np.array([1, 1])
    nxt = np.zeros((2, 5), dtype=np.int8)
    nxt[1, -1] = 1  # action-history encoding, not an environment effect
    groups = np.array([0, 1])
    assert audit_transitions(obs, act, nxt, groups, n_actions=3)["variants"][
        "executed_history_0"]["ambiguous_keys"] == 1
    assert audit_transitions(obs, act, nxt, groups, n_actions=3,
                             target_history_bits=1)["variants"][
        "executed_history_0"]["ambiguous_keys"] == 0


def test_episode_tails_are_never_paired_with_next_episode_start():
    obs = np.arange(8).reshape(4, 2)
    act = np.array([0, 1, 2, 0])
    groups = np.array([0, 0, 1, 1])
    x, a, nxt, e = _shift_within_episodes(obs, act, groups)
    assert x[:, 0].tolist() == [0, 4]
    assert nxt[:, 0].tolist() == [2, 6]
    assert a.tolist() == [0, 2]
    assert e.tolist() == [0, 1]


def test_private_or_malformed_arrays_rejected():
    obs = np.zeros((2, 2))
    action = np.array([0, 1])
    episode = np.array([0, 1])
    with pytest.raises(ValueError, match="Invalid public"):
        audit_transitions(obs.astype(object), action, obs, episode, n_actions=3)
    with pytest.raises(ValueError, match="Invalid public"):
        audit_transitions(obs, action, np.full((2, 2), np.nan), episode, n_actions=3)
    with pytest.raises(ValueError, match="Invalid excluded"):
        audit_transitions(obs, action, obs, episode, n_actions=3,
                          target_history_bits=2)
