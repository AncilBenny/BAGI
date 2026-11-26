"""M23 intervention evaluation contracts: independent layouts and matched exposure."""

import numpy as np
import pytest

from bagi.experiments.m23_probe_effects import (
    equal_exposure_success_only,
    source_probe_split,
)
from bagi.experiments.m23_source_feasibility import run_source_episode
from bagi.experiments.m23_study_contract import (
    BALANCE_MONITOR_LAYOUTS,
    EFFECT_MONITOR_LAYOUTS,
    FINAL_LAYOUTS,
    MEMORY_MONITOR_LAYOUTS,
    MONITOR_LAYOUTS,
    PROBE_MONITOR_LAYOUTS,
    PROBE_TRAIN_LAYOUTS,
    TRAIN_LAYOUTS,
    validate_split_contract,
)


def test_heldout_source_probe_and_train_splits_are_disjoint():
    validate_split_contract()
    train = {x for values in PROBE_TRAIN_LAYOUTS.values() for x in values}
    monitor = {x for values in PROBE_MONITOR_LAYOUTS.values() for x in values}
    other = {
        x for split in (
            TRAIN_LAYOUTS, MONITOR_LAYOUTS, BALANCE_MONITOR_LAYOUTS,
            MEMORY_MONITOR_LAYOUTS, EFFECT_MONITOR_LAYOUTS, FINAL_LAYOUTS,
        ) for values in split.values() for x in values
    }
    assert len(train) == 384
    assert len(monitor) == 96
    assert train.isdisjoint(monitor | other)
    assert monitor.isdisjoint(other)
    assert "unlockpickup" not in PROBE_TRAIN_LAYOUTS
    assert "unlockpickup" not in PROBE_MONITOR_LAYOUTS


def test_source_probe_rejects_protected_target_and_final_before_native_step():
    with pytest.raises(ValueError):
        run_source_episode("unlockpickup", 453001, split="probe_train")
    with pytest.raises(ValueError):
        run_source_episode("unlockpickup", 453001, split="probe_monitor")
    with pytest.raises(ValueError, match="PROBE_MONITOR"):
        run_source_episode("fetch", 443001, split="probe_monitor")
    with pytest.raises(ValueError, match="PROBE_TRAIN"):
        run_source_episode("doorkey", 433001, split="probe_train")


def test_matched_control_repeats_existing_not_future_data():
    old = {
        "proxy_labels": np.array([1, 2, 3, 1], dtype=np.int64),
        "features": np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
    }
    frozen = old["features"].copy()
    repeated = equal_exposure_success_only(old, 17, 941)
    assert len(repeated["proxy_labels"]) == 17
    assert set(repeated["proxy_labels"].tolist()) <= {1, 2, 3}
    np.testing.assert_array_equal(frozen, old["features"])
    again = equal_exposure_success_only(old, 17, 941)
    np.testing.assert_array_equal(repeated["features"], again["features"])
    with pytest.raises(ValueError):
        equal_exposure_success_only(old, 2, 941)


def test_only_registered_probe_split_can_be_collected(tmp_path):
    with pytest.raises(ValueError):
        source_probe_split("target_final", tmp_path)
    assert list(tmp_path.iterdir()) == []
