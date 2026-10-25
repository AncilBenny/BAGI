"""Pure NumPy synthetic tests, no simulator or checkpoint loads."""

from __future__ import annotations

import numpy as np
import pytest

from bagi.experiments.e4_p1_distribution_audit import (
    ACTION_NAMES,
    policy_probabilities,
    summarize_probabilities,
)


def fake_arrays() -> dict[str, np.ndarray]:
    # All-zero weights yield a *uniform* categorical policy; greedy chooses 0.
    return {
        "public_pre_action": np.array([[1.0], [2.0], [3.0]]),
        "actions": np.array([0, 0, 0], dtype=np.int16),
        "policy__mlp_extractor__policy_net__0__weight": np.zeros((3, 1)),
        "policy__mlp_extractor__policy_net__0__bias": np.zeros(3),
        "policy__mlp_extractor__policy_net__2__weight": np.zeros((3, 3)),
        "policy__mlp_extractor__policy_net__2__bias": np.zeros(3),
        "policy__action_net__weight": np.zeros((7, 3)),
        "policy__action_net__bias": np.zeros(7),
    }


def test_uniform_policy_entropy_and_top_probability():
    arrays = fake_arrays()
    probs = policy_probabilities(arrays)
    np.testing.assert_allclose(probs, 1 / 7, atol=1e-12)
    stats = summarize_probabilities(arrays)
    assert stats["policy_action_reproduction_matches"] == 3
    assert stats["mean_normalized_entropy"] == 1.0
    assert stats["mean_top_action_probability"] == round(1 / 7, 6)
    assert len(ACTION_NAMES) == 7


def test_nonuniform_action_logit_probabilities():
    arrays = fake_arrays()
    arrays["policy__action_net__bias"][2] = 4
    arrays["actions"] = np.array([2, 2, 2])
    stats = summarize_probabilities(arrays)
    assert stats["mean_action_probability"][2] > 0.9
    assert stats["mean_normalized_entropy"] < 0.5
    assert stats["action_histogram"] == [0, 0, 3, 0, 0, 0, 0]


def test_reject_record_action_disagreement():
    arrays = fake_arrays()
    arrays["actions"] = np.array([2, 0, 0])
    with pytest.raises(ValueError, match="did not match"):
        summarize_probabilities(arrays)


def test_reject_unsafe_weights_and_observations():
    arrays = fake_arrays()
    arrays["policy__action_net__bias"] = np.array(["unsafe"] * 7, dtype=object)
    with pytest.raises(ValueError, match="Unsafe"):
        policy_probabilities(arrays)
    arrays = fake_arrays()
    arrays["public_pre_action"][0, 0] = np.nan
    with pytest.raises(ValueError, match="PUBLIC"):
        policy_probabilities(arrays)


def test_reject_wrong_logits_shape():
    arrays = fake_arrays()
    arrays["policy__action_net__bias"] = np.zeros(6)
    arrays["policy__action_net__weight"] = np.zeros((6, 3))
    with pytest.raises(ValueError, match="actor output"):
        policy_probabilities(arrays)
