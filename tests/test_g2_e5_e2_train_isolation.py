"""Synthetic-only E2 actor0 provenance checks; no original EVAL or simulators."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from bagi.experiments import g2_e5_e2_train_isolation as study


def _synthetic_source(path: Path, *, corrupt_episode: bool = False,
                      corrupt_actor: bool = False) -> str:
    actor = np.ones(study.SOURCE_ROWS, dtype=np.int8)
    actor[0:2048] = 0
    actor[2848:3648] = 2
    actor[3648:5696] = 0
    actor[6496:7296] = 2
    if corrupt_actor:
        actor[0] = 1
    seed = np.zeros(study.SOURCE_ROWS, dtype=np.int8)
    seed[3648:] = 1
    episode = np.zeros(study.SOURCE_ROWS, dtype=np.int32)
    template = np.r_[np.repeat(np.arange(51), 40), np.repeat(51, 8)]
    episode[:2048] = template
    episode[3648:5696] = template + 92
    if corrupt_episode:
        episode[100] = 101
    obs = np.full((study.SOURCE_ROWS, 5, 5), 99, dtype=np.int8)
    obs[actor == 0] = 1
    # Evaluation sentinel (99) must never appear in the derived TRAIN archive.
    numeric = io.BytesIO()
    np.savez_compressed(
        numeric, observation=obs, next_observation=obs.copy(),
        action=np.zeros(study.SOURCE_ROWS, dtype=np.int8),
        episode_index=episode, seed_index=seed, actor_index=actor,
        private_eval_object=np.array([{"do_not_load": True}], dtype=object),
    )
    with zipfile.ZipFile(path, "w") as source:
        source.writestr("summary.json", "{}")
        source.writestr("public_policy_numeric.npz", numeric.getvalue())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_derived(tmp_path: Path, monkeypatch) -> Path:
    source = tmp_path / "mixed.zip"
    checksum = _synthetic_source(source)
    monkeypatch.setattr(study, "SOURCE_ZIP_SHA256", checksum)
    dest = tmp_path / "isolated.zip"
    audit = study.derive_actor0(source, dest)
    assert audit["source_eval_rows_materialized_during_derivation"] == 3200
    assert audit["output_eval_rows"] == 0
    return dest


def _modified_npz_archive(path: Path, *, inject_actor: bool = False,
                          inject_extra: bool = False) -> tuple[Path, str]:
    with zipfile.ZipFile(path) as archive:
        with np.load(io.BytesIO(archive.read("train_numeric.npz")), allow_pickle=False) as data:
            fields = {name: data[name].copy() for name in study.FIELDS}
    if inject_actor:
        fields["actor_index"][0] = 2
    if inject_extra:
        fields["private_eval_metadata"] = np.zeros(1, dtype=np.int8)
    payload = io.BytesIO()
    np.savez_compressed(payload, **fields)
    malicious = path.parent / "spoofed.zip"
    npz = payload.getvalue()
    with zipfile.ZipFile(malicious, "w") as archive:
        archive.writestr("manifest.json", json.dumps(study._manifest(npz)))
        archive.writestr("train_numeric.npz", npz)
    return malicious, hashlib.sha256(malicious.read_bytes()).hexdigest()


def test_derived_actor0_only_output_has_no_eval_or_private_field(tmp_path, monkeypatch):
    derived = _synthetic_derived(tmp_path, monkeypatch)
    digest = hashlib.sha256(derived.read_bytes()).hexdigest()
    result = study.load_isolated_train(derived, expected_sha256=digest)
    assert len(result["action"]) == 4096
    assert np.all(result["actor_index"] == 0)
    assert np.all(result["observation"] == 1)
    assert set(result) == set(study.FIELDS)
    with zipfile.ZipFile(derived) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["source_actor_counts"] == {"0": 4096, "1": 1600, "2": 1600}
        assert manifest["derivation_accessed_mixed_eval_arrays"] is True
        assert manifest["consumer_archive_contains_eval_rows"] is False


def test_original_mixed_source_rejected_even_when_digest_is_known(tmp_path):
    mixed = tmp_path / "mixed.zip"
    digest = _synthetic_source(mixed)
    with pytest.raises(ValueError, match="Not an isolated"):
        study.load_isolated_train(mixed, expected_sha256=digest)


def test_source_hash_fails_before_any_original_archive_access(tmp_path):
    source = tmp_path / "source.zip"
    _synthetic_source(source)
    with pytest.raises(ValueError, match="original E2 source SHA256"):
        study.derive_actor0(source, tmp_path / "never_created.zip")
    assert not (tmp_path / "never_created.zip").exists()


def test_modified_archive_rejected_by_pinned_sha(tmp_path, monkeypatch):
    derived = _synthetic_derived(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Isolated TRAIN archive SHA256 mismatch"):
        study.load_isolated_train(derived)


def test_eval_sentinel_rejected_even_if_metadata_and_hash_are_updated(tmp_path, monkeypatch):
    derived = _synthetic_derived(tmp_path, monkeypatch)
    malicious, digest = _modified_npz_archive(derived, inject_actor=True)
    with pytest.raises(ValueError, match="EVAL rows are forbidden"):
        study.load_isolated_train(malicious, expected_sha256=digest)


def test_extra_private_array_rejected_even_if_hash_is_updated(tmp_path, monkeypatch):
    derived = _synthetic_derived(tmp_path, monkeypatch)
    malicious, digest = _modified_npz_archive(derived, inject_extra=True)
    with pytest.raises(ValueError, match="unexpected fields"):
        study.load_isolated_train(malicious, expected_sha256=digest)


def test_corrupt_episode_rejected_at_derivation(tmp_path, monkeypatch):
    mixed = tmp_path / "mixed.zip"
    digest = _synthetic_source(mixed, corrupt_episode=True)
    monkeypatch.setattr(study, "SOURCE_ZIP_SHA256", digest)
    with pytest.raises(ValueError, match="episode"):
        study.derive_actor0(mixed, tmp_path / "never_created.zip")
    assert not (tmp_path / "never_created.zip").exists()


def test_corrupt_actor_partition_rejected_at_derivation(tmp_path, monkeypatch):
    mixed = tmp_path / "mixed.zip"
    digest = _synthetic_source(mixed, corrupt_actor=True)
    monkeypatch.setattr(study, "SOURCE_ZIP_SHA256", digest)
    with pytest.raises(ValueError, match="partition changed"):
        study.derive_actor0(mixed, tmp_path / "never_created.zip")
