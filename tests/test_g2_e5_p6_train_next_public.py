"""Synthetic-only tests; original archive TRAIN/test labels are not accessed."""

from __future__ import annotations

import numpy as np
import pytest

from bagi.experiments.g2_e5_p6_train_next_public import (
    ARMS,
    _dataset,
    _executed_history,
    _features,
    _fit_predict,
    _metrics,
)


def _toy():
    obs = np.array([[0.0, 1], [1.0, 0], [2.0, 0], [0.0, 2]], dtype=np.float32)
    act = np.array([1, 2, 1, 2], dtype=np.int8)
    nxt = np.array([[1.0, 1], [1.0, 0], [2.0, 1], [0.0, 2]], dtype=np.float32)
    ep = np.array([0, 0, 1, 1], dtype=np.int32)
    history = _executed_history(act, ep, 3)
    return _dataset("synthetic", obs, act, nxt, np.array([0, 0, 1, 1]),
                    history, n_actions=3, target_width=2)


def test_executed_history_resets_at_episode_boundary():
    d = _toy()
    assert d.history.shape == (4, 12)
    assert d.history[0].sum() == d.history[2].sum() == 0
    assert d.history[1, 1] == d.history[3, 1] == 1
    assert d.history[1].sum() == d.history[3].sum() == 1


def test_arms_match_dimensions_and_frozen_actions():
    d = _toy()
    indices = np.array([1, 2, 3])
    x = {arm: _features(d, indices, arm, np.random.default_rng(101)) for arm in ARMS}
    assert {v.shape for v in x.values()} == {(3, 17)}
    assert all(np.array_equal(v[:, -3:], x["real_history"][:, -3:])
               for v in x.values())
    assert np.array_equal(x["reactive"][:, :2], x["real_history"][:, :2])
    assert np.all(x["reactive"][:, 2:14] == 0)
    assert np.all(x["action_only"][:, :14] == 0)
    assert sorted(x["shuffled_history"][:, 2:14].tobytes()) == sorted(
        x["real_history"][:, 2:14].tobytes())
    assert np.array_equal(np.sort(x["shuffled_history"][:, 2:14], axis=0),
                          np.sort(x["real_history"][:, 2:14], axis=0))


def test_group_disjoint_train_folds_and_no_hidden_sources():
    d = _toy()
    left = np.flatnonzero(d.fold == 0)
    right = np.flatnonzero(d.fold == 1)
    assert left.tolist() == [0, 1]
    assert right.tolist() == [2, 3]
    assert not set(left).intersection(right)
    with pytest.raises(ValueError, match="Invalid"):
        _dataset("bad", d.observation, d.action, d.next_observation,
                 np.zeros(4, dtype=np.int8), d.history, n_actions=3, target_width=2)


def test_training_deterministic_fixed_recipe_and_bounds():
    d = _toy()
    x = _features(d, np.arange(3), "real_history", np.random.default_rng(10))
    y = d.next_observation[:3] / 8
    a = _fit_predict(x, y, x[:1], seed=17)
    b = _fit_predict(x, y, x[:1], seed=17)
    assert a.shape == (1, 2)
    assert np.array_equal(a, b)
    assert np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all()


def test_metrics_separate_changed_cell_support_from_global_mean():
    y = np.array([[0, 1]], dtype=np.float32)
    x = np.zeros_like(y)
    perfect = _metrics(y, y, x)
    assert perfect["public_cell_mse"] == 0
    assert perfect["changed_public_cells"] == 1
    persistence = _metrics(x, y, x)
    assert persistence["public_cell_mse"] == 0.5
    assert persistence["changed_cell_mse"] == 1.0
    assert _metrics(x, x, x)["changed_cell_mse"] is None
