"""Synthetic-only P0B bundle and launch-workflow contract tests.

No real Workshop reset, step, optimizer, or GH workflow dispatch occurs.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from bagi.experiments import g2_e6_p0b_bundle as bundle


def _source_fixture(tmp_path: Path, *, status: str = "complete") -> Path:
    root = tmp_path / "numeric"
    root.mkdir()
    digests = {}
    for name in bundle.NUMERIC_MEMBERS:
        file = root / name
        np.savez_compressed(file, executed_action=np.asarray([1, 2], dtype=np.int8))
        digests[name] = hashlib.sha256(file.read_bytes()).hexdigest()
    summary = {
        "study": "G2-E6-P0B", "status": status,
        "workshop_git_blob_sha1": bundle.WORKSHOP_SHA,
        "dqn_git_blob_sha1": bundle.DQN_SHA,
        "promotion": False, "size13_native_calls": 0,
        "native_actions": 8192, "native_train_actions": 8192,
        "native_diagnostic_actions": 0, "optimizer_updates": 8066,
        "native_action_count_exact": True, "native_step_exceptions": 0,
        "recorded_actions": 8192, "numeric_file_sha256": digests,
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return root


def test_bundle_one_allowlisted_numeric_artifact_without_materializing_arrays(tmp_path):
    source = _source_fixture(tmp_path)
    out = tmp_path / "bundle"
    result = bundle.package(source, out)
    assert result["status"] == "packaged"
    assert result["run_status"] == "complete"
    assert result["original_test_eval_arrays_materialized"] is False
    assert result["members"] == list(bundle.MEMBERS)
    zipped = out / bundle.ZIP_NAME
    assert zipped.is_file()
    assert hashlib.sha256(zipped.read_bytes()).hexdigest() == result["zip_sha256"]
    assert result["zip_sha256"] in (out / (bundle.ZIP_NAME + ".sha256")).read_text()
    with zipfile.ZipFile(zipped) as archive:
        assert archive.namelist() == list(bundle.MEMBERS)
        assert archive.testzip() is None
    assert json.loads((out / "packaging_status.json").read_text())["status"] == "packaged"


def test_bundle_failed_run_keeps_invalid_scientific_status(tmp_path):
    source = _source_fixture(tmp_path, status="failed")
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["native_train_actions"] = 2
    summary["native_actions"] = 2
    summary["recorded_actions"] = 2
    summary["optimizer_updates"] = 0
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    result = bundle.package(source, tmp_path / "bundle")
    assert result["status"] == "packaged"
    assert result["run_status"] == "failed"
    with zipfile.ZipFile(tmp_path / "bundle" / bundle.ZIP_NAME) as archive:
        saved = json.loads(archive.read("summary.json"))
    assert saved["status"] == "failed"
    assert saved["optimizer_updates"] == 0


def test_bundle_rejects_hash_mismatch_and_records_reason(tmp_path):
    source = _source_fixture(tmp_path)
    (source / "train_public_numeric.npz").write_bytes(b"corrupted")
    out = tmp_path / "bundle"
    with pytest.raises(ValueError, match="Corrupted numeric member"):
        bundle.package(source, out)
    assert not (out / bundle.ZIP_NAME).exists()
    record = json.loads((out / "packaging_status.json").read_text())
    assert record["status"] == "invalid"
    assert "Corrupted" in record["error"]


def test_bundle_rejects_extra_members_and_symlinks(tmp_path):
    source = _source_fixture(tmp_path)
    (source / "eval_private.pkl").write_bytes(b"unsafe")
    with pytest.raises(ValueError, match="unknown"):
        bundle.package(source, tmp_path / "reject_extra")
    (source / "eval_private.pkl").unlink()
    saved = source / "frozen_dqn_weights.npz"
    saved.unlink()
    saved.symlink_to(source / "train_public_numeric.npz")
    with pytest.raises(ValueError, match="regular files"):
        bundle.package(source, tmp_path / "reject_symlink")


def test_bundle_rejects_incorrect_complete_run_and_missing_output(tmp_path):
    source = _source_fixture(tmp_path)
    summary_file = source / "summary.json"
    summary = json.loads(summary_file.read_text())
    summary["optimizer_updates"] = 8065
    summary_file.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="exact native accounting"):
        bundle.package(source, tmp_path / "bad_updates")
    missing = tmp_path / "does_not_exist"
    with pytest.raises(ValueError, match="Missing or unsafe"):
        bundle.package(missing, tmp_path / "missing")
    assert (tmp_path / "missing" / "packaging_status.json").is_file()


def test_bundle_never_overwrites_existing_output(tmp_path):
    source = _source_fixture(tmp_path)
    out = tmp_path / "bundle"
    out.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        bundle.package(source, out)


def test_one_shot_workflow_statically_does_not_launch_on_source_commit():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github/workflows/bagi-g2-e6-p0b-one-shot.yml"
    )
    cfg = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    triggers = cfg.get("on", cfg.get(True))
    assert set(triggers) == {"push"}
    assert triggers["push"]["branches"] == ["setup/initial-platform"]
    assert triggers["push"]["paths"] == [
        "docs/research-launches/G2_E6_P0B_ONE_SHOT_AUTHORIZATION.txt"
    ]
    assert cfg["concurrency"]["cancel-in-progress"] is False
    assert len(cfg["jobs"]) == 2
    preflight = cfg["jobs"]["preflight"]
    native = cfg["jobs"]["native-acquisition"]
    assert preflight["timeout-minutes"] <= 15
    assert native["timeout-minutes"] <= 55
    assert native["needs"] == "preflight"
    assert cfg["permissions"] == {"contents": "read", "actions": "read"}
    steps = preflight["steps"] + native["steps"]
    text = "\n".join(step.get("run", "") for step in steps)
    assert "A\\t" in text
    assert "sha = os.environ['GITHUB_SHA']" in text
    assert "if ref['object']['sha'] != sha:" in text
    assert "pytest" in text and "ruff check" in text
    assert "g2_e6_p0b_workshop_dual_size_acquisition" in text
    assert "g2_e6_p0b_bundle" in text
    uploads = [step for step in native["steps"]
               if step.get("uses", "").startswith("actions/upload-artifact@")]
    assert len(uploads) == 1
    assert uploads[0].get("if") == "always()"
