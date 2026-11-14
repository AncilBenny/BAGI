"""M23 stage-3 split isolation and TRAIN-only loss weights."""

import numpy as np
import pytest

from bagi.experiments.m23_class_balance import training_object_weights
from bagi.experiments.m23_public_world_model import N_OBJECT
from bagi.experiments.m23_source_feasibility import run_source_episode
from bagi.experiments.m23_study_contract import (
    BALANCE_MONITOR_LAYOUTS,
    FINAL_LAYOUTS,
    MODEL_SEEDS,
    MONITOR_LAYOUTS,
    TRAIN_LAYOUTS,
    validate_split_contract,
)


def test_new_source_balance_monitor_disjoint_from_registered_other_splits():
    validate_split_contract()
    new = {x for rows in BALANCE_MONITOR_LAYOUTS.values() for x in rows}
    old = {
        x
        for split in (TRAIN_LAYOUTS, MONITOR_LAYOUTS, FINAL_LAYOUTS)
        for rows in split.values()
        for x in rows
    }
    assert len(new) == 48
    assert len(MODEL_SEEDS) == 8
    assert new.isdisjoint(old)
    assert "unlockpickup" not in BALANCE_MONITOR_LAYOUTS
    assert all(len(rows) == 16 for rows in BALANCE_MONITOR_LAYOUTS.values())


def test_final_layout_access_rejected_for_source_balance_monitor():
    with pytest.raises(ValueError, match="registered BALANCE_MONITOR"):
        run_source_episode("fetch", 443001, split="balance_monitor")
    with pytest.raises(ValueError, match="registered BALANCE_MONITOR"):
        run_source_episode("putnear", 463001, split="balance_monitor")
    with pytest.raises(ValueError):
        run_source_episode("unlockpickup", 453001, split="balance_monitor")


def test_training_only_rare_object_weighting_bounded_and_unobserved_zero():
    labels = np.zeros((2, 49), dtype=np.int64)
    labels[0, 0] = 5
    rows = {"next_obj": labels}
    before = labels.copy()
    weights = training_object_weights(rows)
    assert weights.shape == (N_OBJECT,)
    assert weights[5] > weights[0]
    assert 0.25 <= weights[0] <= 8.0
    assert weights[5] <= 8.0
    assert np.all(weights[2:5] == 0)
    np.testing.assert_array_equal(before, rows["next_obj"])


def test_invalid_weight_labels_rejected():
    with pytest.raises(ValueError):
        training_object_weights({"next_obj": np.array([[N_OBJECT] * 49])})
    with pytest.raises(ValueError):
        training_object_weights({"next_obj": np.zeros((0, 49), dtype=np.int64)})
    with pytest.raises(ValueError):
        training_object_weights({"next_obj": np.zeros((1, 48), dtype=np.int64)})
