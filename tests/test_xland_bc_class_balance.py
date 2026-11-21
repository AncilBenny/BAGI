"""Verify established inverse-frequency sampling and rare-action diagnostics."""
import numpy as np
import pytest

from bagi.experiments.xland_bc_class_balance import action_recall, balanced_indices


def test_balanced_sampler_reproducible_and_prioritizes_minor_action():
    labels = np.array([0] * 80 + [1] * 8 + [4] * 2, dtype=np.int64)
    first = balanced_indices(labels, seed=412)
    again = balanced_indices(labels, seed=412)
    assert np.array_equal(first, again)
    assert first.shape == labels.shape
    assert first.max() < len(labels) and first.min() >= 0
    assert int(np.sum(labels[first] == 4)) > 10


def test_class_diagnostic_does_not_hide_missing_causal_actions():
    y = np.array([0, 0, 1, 3, 4], dtype=np.int64)
    prediction = np.array([0, 0, 1, 0, 0], dtype=np.int64)
    scores = action_recall(prediction, y)
    assert scores["0"] == 1.0
    assert scores["3"] == 0.0
    assert scores["4"] == 0.0
    assert scores["5"] is None


def test_bad_recall_lengths_are_rejected():
    with pytest.raises(ValueError, match="mismatch"):
        action_recall(np.array([0]), np.array([0, 1]))
