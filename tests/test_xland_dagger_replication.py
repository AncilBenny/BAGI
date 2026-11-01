"""REUSE-17: frozen numeric data and source-provenance checks."""
import numpy as np
import pytest

from bagi.experiments.xland_dagger_replication import (
    load_frozen_numeric_pool,
    successful_seeds,
)


def test_rewarded_seeds_exclude_failed_episodes():
    rows = {"hold": {"details": [
        {"seed": 9100, "goal_success": 0},
        {"seed": 9101, "goal_success": 1},
        {"seed": 9102, "goal_success": 0},
    ]}}
    assert successful_seeds(rows) == {"hold": (9101,)}


def test_seed_outside_verified_source_fails_before_read(tmp_path):
    with pytest.raises(ValueError, match="reviewed"):
        load_frozen_numeric_pool(tmp_path, 101)


def test_reject_object_dtype_without_pickle_loading(tmp_path):
    arr = np.array([{"rule": "hidden"}], dtype=object)
    np.savez_compressed(
        tmp_path / "expert_pool_seed17.npz",
        obs=arr, acts=np.array([0]), next_obs=arr,
        dones=np.array([False]),
    )
    with pytest.raises(ValueError):
        load_frozen_numeric_pool(tmp_path, 17)


def test_reject_wrong_numeric_sha(tmp_path):
    np.savez_compressed(
        tmp_path / "expert_pool_seed17.npz",
        obs=np.zeros((291, 54), dtype=np.float32),
        acts=np.array([0] * 280 + [3] * 5 + [4] * 6, dtype=np.int64),
        next_obs=np.zeros((291, 54), dtype=np.float32),
        dones=np.zeros(291, dtype=bool),
    )
    with pytest.raises(ValueError, match="SHA256"):
        load_frozen_numeric_pool(tmp_path, 17)
