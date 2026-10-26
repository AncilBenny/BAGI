"""No-network, no-import, no-environment-action tests for P4 metadata gate."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from bagi.experiments.g2_e5_p4_wheel_preflight import audit_wheel


def _synthetic_wheel(
    tmp_path: Path, upstream_matplotlib: str, *, package: str = "rllte-core"
) -> tuple[Path, str]:
    path = tmp_path / "rllte_core-1.0.1-py3-none-any.whl"
    metadata = (
        "Metadata-Version: 2.1\n"
        f"Name: {package}\nVersion: 1.0.1\n"
        f"Requires-Dist: matplotlib{upstream_matplotlib}\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("rllte_core-1.0.1.dist-info/METADATA", metadata)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _project(tmp_path: Path) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text('[project]\nname="bagi"\ndependencies=["matplotlib>=3.9"]\n')
    return path


def test_pinned_upstream_conflict_blocks_training(tmp_path):
    wheel, digest = _synthetic_wheel(tmp_path, "==3.6.0")
    result = audit_wheel(wheel, _project(tmp_path), expected_sha256=digest)
    assert result["matplotlib_verdict"] == "proven_conflict"
    assert result["status"] == "STOP_NO_NATIVE_TRAINING"
    assert not result["authorized_to_train"]
    assert result["native_actions"] == result["native_resets"] == 0


def test_non_conflicting_metadata_does_not_auto_authorize_training(tmp_path):
    wheel, digest = _synthetic_wheel(tmp_path, ">=3.9")
    result = audit_wheel(wheel, _project(tmp_path), expected_sha256=digest)
    assert result["matplotlib_verdict"] == "needs_full_dependency_resolver"
    assert not result["runtime_import_tested"]
    assert not result["all_dependencies_resolved"]
    assert not result["authorized_to_train"]


def test_tampered_or_unpinned_wheel_rejected_before_read(tmp_path):
    wheel, _digest = _synthetic_wheel(tmp_path, "==3.6.0")
    with pytest.raises(ValueError, match="SHA256"):
        audit_wheel(wheel, _project(tmp_path))


def test_wrong_identity_fails_after_verified_hash(tmp_path):
    wheel, digest = _synthetic_wheel(tmp_path, "==3.6.0", package="not-rllte")
    with pytest.raises(ValueError, match="package name"):
        audit_wheel(wheel, _project(tmp_path), expected_sha256=digest)


def test_conditional_requirement_is_not_a_proven_conflict(tmp_path):
    wheel, digest = _synthetic_wheel(tmp_path, "==3.6.0; extra == 'plots'")
    result = audit_wheel(wheel, _project(tmp_path), expected_sha256=digest)
    assert result["matplotlib_verdict"] == "needs_full_dependency_resolver"
    assert not result["authorized_to_train"]
