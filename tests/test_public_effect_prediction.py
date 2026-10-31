"""Synthetic G2-E1 public-only effect prediction contracts, not native results."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from bagi.agents.public_effect_prediction import (
    PublicEffectEpisode,
    binary_metrics,
    checked_episodes,
    disjoint_layouts,
    duplicate_trajectory_groups,
    effect_predictions,
    episode_bootstrap_brier_difference,
    fit_public_effects,
    public_change,
    shuffled_training_actions,
)


def episode(layout: str, ident: str, *, n: int = 12) -> PublicEffectEpisode:
    x = np.arange(n * 3, dtype=np.float32).reshape(n, 3) % 2
    a = np.arange(n, dtype=np.int64) % 3
    y = (a == 1).astype(np.int8)
    return PublicEffectEpisode(layout, ident, "synthetic", x, a, y)


def test_xland_history_bits_excluded_from_visible_effect():
    before = np.zeros(115, dtype=np.float32)
    after = before.copy()
    after[-28] = 1
    assert public_change(before, after, history_bits=28) == 0
    assert public_change(before, after) == 1
    after[7] = 1
    assert public_change(before, after, history_bits=28) == 1


def test_workshop_5x5_observation_and_hidden_change_not_claimed():
    x = np.zeros((5, 5), dtype=np.int8)
    assert public_change(x, x.copy()) == 0
    y = x.copy()
    y[1, 1] = 2
    assert public_change(x, y) == 1


@pytest.mark.parametrize("before,after,bits", [
    (np.zeros(115), np.zeros(114), 28),
    (np.array([np.nan, 1]), np.array([0, 1]), 0),
    (np.zeros(3), np.zeros(3), 3),
    (np.zeros(3), np.zeros(3), -1),
])
def test_invalid_public_signatures_fail_closed(before, after, bits):
    with pytest.raises(ValueError):
        public_change(before, after, history_bits=bits)


def test_episode_action_histogram_and_train_only_shuffle():
    e = [episode("train1", "e1"), episode("train2", "e2")]
    _, a, _, g = checked_episodes(e, width=3, n_actions=3)
    shuffled = shuffled_training_actions(a, g, seed=17)
    for group in np.unique(g):
        assert sorted(a[g == group]) == sorted(shuffled[g == group])
    assert np.any(shuffled != a)
    assert np.array_equal(shuffled, shuffled_training_actions(a, g, seed=17))


def test_no_heldout_layout_or_episode_leakage():
    disjoint_layouts([episode("train", "e1")], [episode("test", "e2")])
    with pytest.raises(ValueError, match="overlap"):
        disjoint_layouts([episode("train", "e1")], [episode("train", "e2")])
    with pytest.raises(ValueError, match="overlap"):
        disjoint_layouts([episode("train", "e1")], [episode("test", "e1")])


def test_invalid_actions_labels_or_duplicate_episode_ids_rejected():
    e = episode("train", "e1")
    bad = PublicEffectEpisode(e.layout, e.episode_id, e.behavior, e.observations,
                              np.full(12, 7), e.visible_changes)
    with pytest.raises(ValueError):
        checked_episodes([bad], width=3, n_actions=3)
    with pytest.raises(ValueError):
        checked_episodes([e, e], width=3, n_actions=3)
    bad_label = PublicEffectEpisode(e.layout, e.episode_id, e.behavior, e.observations,
                                    e.actions, np.full(12, 2))
    with pytest.raises(ValueError):
        checked_episodes([bad_label], width=3, n_actions=3)


def test_both_train_label_classes_required():
    e = episode("train", "e1")
    bad = PublicEffectEpisode(e.layout, e.episode_id, e.behavior, e.observations,
                              e.actions, np.zeros(12, dtype=np.int8))
    with pytest.raises(ValueError, match="Both public-change classes"):
        fit_public_effects([bad], width=3, n_actions=3, seed=17)


def test_matched_capacity_and_deterministic_training_no_test_fit():
    train = [episode("train1", "e1"), episode("train2", "e2")]
    test = [episode("test", "e3")]
    disjoint_layouts(train, test)
    f1 = fit_public_effects(train, width=3, n_actions=3, seed=17, epochs=2)
    f2 = fit_public_effects(train, width=3, n_actions=3, seed=17, epochs=2)
    assert f1.train_rows == 24 and f1.train_episodes == 2
    assert f1.train_action_counts == (8, 8, 8)
    assert sum(p.numel() for p in f1.actual_action.parameters()) == sum(
        p.numel() for p in f1.shuffled_action.parameters()
    )
    for key, value in f1.actual_action.state_dict().items():
        assert torch.equal(value, f2.actual_action.state_dict()[key])
    pred = effect_predictions(f1, test, width=3, n_actions=3)
    assert set(pred) == {"actual_action", "shuffled_action", "action_frequency", "constant"}
    assert all(v.shape == (12,) and np.isfinite(v).all() for v in pred.values())
    assert np.allclose(pred["constant"], 1 / 3)


def test_binary_metrics_calibration_and_episode_bootstrap():
    y = np.array([1, 0, 1, 0])
    a = np.array([0.8, 0.2, 0.8, 0.2])
    b = np.array([0.5, 0.5, 0.5, 0.5])
    m = binary_metrics(y, a)
    assert m["tp"] == 2 and m["fp"] == 0 and m["fn"] == 0
    assert m["brier"] < binary_metrics(y, b)["brier"]
    out = episode_bootstrap_brier_difference(y, a, b, np.array([0, 0, 1, 1]),
                                              seed=29, resamples=100)
    assert out["difference_first_minus_second"] < 0
    assert out["ci_95_low"] <= out["ci_95_high"]
    with pytest.raises(ValueError, match="two heldout episodes"):
        episode_bootstrap_brier_difference(y, a, b, np.zeros(4), seed=17)


def test_scripted_duplicate_trajectories_not_independent():
    first = episode("default", "first")
    second = episode("default", "second")
    third = episode("mirror", "third", n=9)
    out = duplicate_trajectory_groups([first, second, third])
    assert out == {"episodes": 3, "unique_trajectories": 2, "duplicate_episodes": 1}
