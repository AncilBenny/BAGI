"""REUSE-13: strict causal history features and episode reset boundaries."""
import numpy as np
import pytest

from bagi.experiments.xland_bc_action_history import (
    augment_training_observations,
    encode_history,
)


def sample():
    return {
        "obs": np.array([[1, 0], [2, 0], [3, 0]], dtype=np.float32),
        "next_obs": np.array([[2, 0], [3, 0], [4, 0]], dtype=np.float32),
        "acts": np.array([3, 4, 0], dtype=np.int64),
        "dones": np.array([False, True, False]),
    }


def test_history_is_previous_actions_only_and_resets_on_terminal():
    augmented, next_views = augment_training_observations(sample())
    assert augmented.shape == next_views.shape == (3, 26)
    assert np.all(augmented[0, 2:] == 0)
    assert augmented[1, 2:].reshape(4, 6)[-1, 3] == 1
    assert augmented[1, 2:].sum() == 1
    assert np.all(augmented[2, 2:] == 0)
    assert next_views[0, 2:].reshape(4, 6)[-1, 3] == 1
    assert next_views[1, 2:].reshape(4, 6)[-1, 4] == 1
    assert next_views[1, 2:].sum() == 2


def test_zero_padding_distinct_from_forward_action_zero():
    assert encode_history([]).sum() == 0
    assert encode_history([0]).sum() == 1
    assert encode_history([0]).reshape(4, 6)[-1, 0] == 1


def test_reject_privileged_or_invalid_action_history():
    with pytest.raises(ValueError, match="four"):
        encode_history([0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="action"):
        encode_history([6])


def test_reject_misaligned_transitions():
    corrupted = sample()
    corrupted["next_obs"] = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="aligned"):
        augment_training_observations(corrupted)
