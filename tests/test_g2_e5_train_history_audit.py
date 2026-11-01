"""Synthetic, simulator-free contracts for read-only G2-E5-P0 TRAIN audit."""
from __future__ import annotations

import hashlib
import io
import zipfile

import numpy as np
import pytest

from bagi.experiments import g2_e5_train_history_audit as study


def test_true_executed_action_history_resets_at_episode_boundary():
    action = np.array([0, 2, 1, 3, 2], dtype=np.int8)
    episode = np.array([0, 0, 0, 1, 1])
    got = study._history(action, episode, 4)
    assert got.tolist() == [
        [-1, -1, -1, -1], [0, -1, -1, -1], [2, 0, -1, -1],
        [-1, -1, -1, -1], [3, -1, -1, -1],
    ]


def test_shuffled_control_is_deterministic_and_within_episode():
    action = np.array([0, 1, 2, 3, 1, 1, 1], dtype=np.int8)
    episode = np.array([0, 0, 0, 0, 1, 1, 1])
    a = study._history(action, episode, 4, shuffle=True)
    b = study._history(action, episode, 4, shuffle=True)
    assert np.array_equal(a, b)
    assert np.array_equal(a[0], np.full(4, -1))
    assert np.array_equal(a[4], np.full(4, -1))
    assert a[5, 0] == a[6, 0] == 1


def test_history_changes_lookup_key_without_future_observation():
    obs = np.zeros((2, 3), dtype=np.float32)
    action = np.array([1, 1], dtype=np.int8)
    assert len(set(study._keys(obs, action, np.empty((2, 0), dtype=np.int16)))) == 1
    hist = np.array([[0, -1], [2, -1]], dtype=np.int16)
    assert len(set(study._keys(obs, action, hist))) == 2


def test_crossfit_excludes_heldout_fold_even_when_keys_repeat():
    obs = np.zeros((4, 2), dtype=np.float32)
    action = np.zeros(4, dtype=np.int8)
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    episode = np.array([0, 0, 1, 1])
    result = study._crossfit(obs, action, y, episode, episode, 2, 0)
    assert result["lookup_covered_rows"] == 4
    # Each heldout episode sees ONLY the opposite label: smoothed p=3/4 or 1/4.
    assert result["brier"] == pytest.approx(0.5625)
    assert result["action_only_brier"] == pytest.approx(0.5625)


def test_reject_invalid_actions_and_interleaved_episodes():
    obs = np.zeros((4, 2), dtype=np.float32)
    action = np.array([0, 1, 0, 1], dtype=np.int8)
    y = np.array([0, 1, 0, 1], dtype=np.int8)
    with pytest.raises(ValueError, match="Malformed"):
        study._crossfit(obs, np.array([0, 9, 0, 1]), y, np.array([0, 0, 1, 1]),
                        np.array([0, 0, 1, 1]), 2, 0)
    with pytest.raises(ValueError, match="not contiguous"):
        study._crossfit(obs, action, y, np.array([0, 1, 0, 1]),
                        np.array([0, 1, 0, 1]), 2, 0)


def test_e1_zip_reader_never_touches_test_arrays(tmp_path, monkeypatch):
    arrays = {}
    for domain in ("xland", "workshop"):
        arrays[f"{domain}_train_observations"] = np.zeros((2, 2), dtype=np.float32)
        arrays[f"{domain}_train_actions"] = np.zeros(2, dtype=np.int64)
        arrays[f"{domain}_train_labels"] = np.zeros(2, dtype=np.int8)
        arrays[f"{domain}_train_groups"] = np.array([0, 1], dtype=np.int64)
        arrays[f"{domain}_train_episode_ids"] = np.array(["a", "b"])
        arrays[f"{domain}_train_behaviors"] = np.array(["uniform_random"] * 2)
        # Object TEST content would require unsafe pickle if accessed.
        arrays[f"{domain}_test_observations"] = np.array([{"private": 1}], dtype=object)
    numeric = io.BytesIO()
    np.savez_compressed(numeric, **arrays)
    path = tmp_path / "safe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("summary.json", "{}")
        archive.writestr("public_effect_numeric.npz", numeric.getvalue())
    monkeypatch.setitem(study.SOURCE_SHA256, "e1", hashlib.sha256(path.read_bytes()).hexdigest())
    out = study._safe_archive(path, "e1")
    assert all("_train_" in key for key in out)
    assert len(out) == 12


def test_sha256_gate_fails_closed_before_archive_read(tmp_path):
    path = tmp_path / "invalid.zip"
    path.write_bytes(b"not a trusted source")
    with pytest.raises(ValueError, match="digest"):
        study._safe_archive(path, "e1")
