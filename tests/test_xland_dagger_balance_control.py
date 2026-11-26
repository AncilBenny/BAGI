"""REUSE-16: prove both conditions share source pool and reject invalid input."""
import numpy as np
import pytest

from bagi.experiments.xland_dagger_balance_control import (
    checked_pool,
    fixed_sampler_indices,
)


def mock_pool():
    obs = np.zeros((16, 54), dtype=np.float32)
    obs[:, 0] = np.arange(16)
    actions = np.array([0] * 12 + [3] * 3 + [4], dtype=np.int64)
    return obs, actions, obs.copy(), np.zeros(16, dtype=bool)


def test_dataset_hash_is_deterministic_and_sensitive_to_labels():
    obs, acts, nxt, dones = mock_pool()
    digest = checked_pool(obs, acts, nxt, dones, expected_rows=16)
    assert len(digest) == 64
    assert digest == checked_pool(obs, acts, nxt, dones, expected_rows=16)
    changed = acts.copy()
    changed[0] = 1
    assert checked_pool(obs, changed, nxt, dones, expected_rows=16) != digest


def test_sampler_cannot_change_source_pool_or_number_of_exposures():
    _, actions, _, _ = mock_pool()
    ordinary = fixed_sampler_indices(actions, 17, "ordinary")
    balanced = fixed_sampler_indices(actions, 17, "class_balanced")
    assert len(ordinary) == len(balanced) == len(actions)
    assert np.array_equal(ordinary, np.arange(16))
    assert np.array_equal(balanced, fixed_sampler_indices(
        actions, 17, "class_balanced"
    ))
    assert np.sum(actions[balanced] == 4) > np.sum(actions == 4)


def test_pool_rejects_wrong_feature_shape_missing_rare_labels_and_bad_counts():
    obs, actions, nxt, dones = mock_pool()
    with pytest.raises(ValueError, match="pool"):
        checked_pool(obs[:, :-1], actions, nxt, dones, expected_rows=16)
    with pytest.raises(ValueError, match="pool"):
        checked_pool(obs, actions, nxt, dones, expected_rows=17)
    with pytest.raises(ValueError, match="pickup"):
        checked_pool(obs, np.zeros(16, dtype=np.int64), nxt, dones, expected_rows=16)
    with pytest.raises(ValueError, match="pool"):
        checked_pool(obs, np.full(16, 7), nxt, dones, expected_rows=16)


def test_unknown_sampling_condition_rejected():
    _, actions, _, _ = mock_pool()
    with pytest.raises(ValueError, match="preregistered"):
        fixed_sampler_indices(actions, 17, "new_tuned_method")
