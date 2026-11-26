"""Synthetic G2-E4-P2 archive and preregistration contracts; no native steps."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from bagi.experiments import e4_p2_frozen_contract as p2


def test_frozen_p2_budgets_and_disjoint_starts():
    p2.verify_registration()
    assert p2.MAX_TRAIN_ACTIONS == 0
    assert p2.MAX_TEST_ACTIONS == 2 * 2 * 4 * 10 * (144 + 250) // 2
    assert not set(p2.TEST_SEEDS) & set(range(26000, 26010))
    assert not set(p2.TEST_SEEDS) & set(range(21000, 21016))


def test_all_160_action_sampling_streams_are_distinct():
    seeds = [
        p2.episode_rng_seed(t, m, a, s)
        for t in range(2) for m in range(2)
        for a in range(4) for s in p2.TEST_SEEDS
    ]
    assert len(seeds) == len(set(seeds)) == 160
    with pytest.raises(ValueError, match="Unregistered"):
        p2.episode_rng_seed(0, 0, 0, 26000)


def test_reject_changed_scientific_budget(monkeypatch):
    monkeypatch.setattr(p2, "MAX_TEST_ACTIONS", 31521)
    with pytest.raises(ValueError, match="preregistration"):
        p2.verify_registration()


def test_numeric_weight_identity_and_fail_closed():
    reference = {
        "actor.weight": np.ones((2, 3), dtype=np.float32),
        "actor.bias": np.zeros(2, dtype=np.float32),
    }
    saved = {
        "policy__actor__weight": np.ones((2, 3), dtype=np.float32),
        "policy__actor__bias": np.zeros(2, dtype=np.float32),
    }
    checked = p2.validated_state_dict(saved, reference)
    assert set(checked) == set(reference)
    np.testing.assert_array_equal(checked["actor.weight"], reference["actor.weight"])
    checked["actor.weight"][0, 0] = 7
    assert saved["policy__actor__weight"][0, 0] == 1
    for changed in (
        {key: value for key, value in saved.items() if key != "policy__actor__bias"},
        {**saved, "policy__actor__weight": np.zeros((2, 4), dtype=np.float32)},
        {**saved, "policy__actor__weight": np.zeros((2, 3), dtype=np.float64)},
        {**saved, "policy__actor__weight": np.full((2, 3), np.nan)},
        {**saved, "policy__actor__weight": np.array([["unsafe"]], dtype=object)},
    ):
        with pytest.raises(ValueError, match="tensor"):
            p2.validated_state_dict(changed, reference)


def _synthetic_zip(path, *, bad_inner_sha=False):
    members = {}
    entries = []
    for task in p2.TASKS:
        for seed in p2.MODEL_SEEDS:
            for arm in p2.SOURCE_ARMS:
                arrays = {
                    "public_pre_action": np.array([[1, 2]], dtype=np.uint8),
                    "actions": np.array([0], dtype=np.int16),
                }
                if arm != "random":
                    for key in p2.ACTOR_KEYS:
                        arrays["policy__" + key.replace(".", "__")] = np.ones(
                            (2, 2), dtype=np.float32,
                        )
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **arrays)
                name = f"{task.replace('-', '_')}_seed{seed}_{arm}.npz"
                blob = buffer.getvalue()
                members[name] = blob
                entries.append({
                    "task": task, "model_seed": seed, "arm": arm,
                    "file": name, "sha256": hashlib.sha256(blob).hexdigest(),
                })
    if bad_inner_sha:
        entries[-1]["sha256"] = "0" * 64
    summary = {
        "study": "G2-E4-P1", "status": "complete",
        "tasks": list(p2.TASKS), "model_seeds": list(p2.MODEL_SEEDS),
        "test_seeds": list(range(26000, 26010)),
        "arms": list(p2.SOURCE_ARMS),
        "native_action_ledger": {"train_actions": 65536, "test_actions": 23110},
        "results": [
            {"task": task, "model_seed": seed}
            for task in p2.TASKS for seed in p2.MODEL_SEEDS
        ],
        "trace_artifacts": entries,
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, blob in members.items():
            z.writestr(name, blob)
        z.writestr("summary.json", json.dumps(summary))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_archive_requires_exact_source_zip_hash(tmp_path):
    path = tmp_path / "not-source.zip"
    path.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="SHA256"):
        p2.archive_weights(path)


def test_all_inner_archives_checked_without_pickle_or_simulator(tmp_path, monkeypatch):
    path = tmp_path / "fake.zip"
    fake_sha = _synthetic_zip(path)
    monkeypatch.setattr(p2, "ARCHIVE_SHA256", fake_sha)
    policies = p2.archive_weights(path)
    assert len(policies) == 8
    assert all(len(weights) == len(p2.ACTOR_KEYS) for weights in policies.values())
    corrupted_sha = _synthetic_zip(path, bad_inner_sha=True)
    monkeypatch.setattr(p2, "ARCHIVE_SHA256", corrupted_sha)
    with pytest.raises(ValueError, match="checksum"):
        p2.archive_weights(path)
