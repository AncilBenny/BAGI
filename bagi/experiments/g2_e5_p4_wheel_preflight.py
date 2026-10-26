"""Fail-closed, offline wheel-metadata preflight for the registered RND study.

This is NOT an RND import, native environment run, or permission to train.
Only inspect a hash-pinned public wheel and BAGI's declared requirements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

PACKAGE = "rllte-core"
VERSION = "1.0.1"
# SHA256 published by PyPI; verify before inspecting *any* untrusted archive.
WHEEL_SHA256 = "3a0ec99c27eae05def49fee79cf868bca298fbf2f7db9c3b90f9be3fd467d6b3"
MATPLOTLIB = re.compile(r"^matplotlib\s*(?:\(([^)]+)\)|([<>=!~].*))?$", re.I)
SIMPLE_VERSION = re.compile(r"^\s*(==|>=)\s*(\d+(?:\.\d+)*)\s*$")


def _matplotlib_specs(requirements: list[str]) -> list[str]:
    """Extract only the precise simple dependency format we can safely compare."""
    found = []
    for requirement in requirements:
        # A conditional marker, even for a currently false extra, needs a resolver.
        base, *markers = requirement.split(";", 1)
        match = MATPLOTLIB.fullmatch(base.strip())
        if match:
            # Do not assert a mandatory conflict from an optional/conditional extra.
            spec = (match.group(1) or match.group(2) or "").strip()
            found.append("conditional" if markers else spec)
    return found


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(piece) for piece in value.split("."))


def _compare_matplotlib(upstream: list[str], project: list[str]) -> str:
    """Prove only the known ==x vs >=y disjoint case; never infer global feasibility."""
    if len(upstream) != 1 or len(project) != 1:
        return "needs_full_dependency_resolver"
    a = SIMPLE_VERSION.fullmatch(upstream[0])
    b = SIMPLE_VERSION.fullmatch(project[0])
    if a is None or b is None or a.group(1) != "==" or b.group(1) != ">=":
        return "needs_full_dependency_resolver"
    exact = _version_tuple(a.group(2))
    floor = _version_tuple(b.group(2))
    n = max(len(exact), len(floor))
    exact += (0,) * (n - len(exact))
    floor += (0,) * (n - len(floor))
    return "proven_conflict" if exact < floor else "needs_full_dependency_resolver"


def audit_wheel(
    wheel: Path,
    project_file: Path,
    *,
    expected_sha256: str = WHEEL_SHA256,
) -> dict[str, object]:
    """Read wheel metadata without importing, installing, extracting, or executing it.

    An incorrect hash is a hard stop, not an invitation to inspect untrusted ZIP.
    expected_sha256 is injectable only for tests; CLI uses the pinned PyPI hash.
    """
    with wheel.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_sha256:
        raise ValueError("Wheel SHA256 does not match the published pinned artifact")
    with zipfile.ZipFile(wheel) as archive:
        entries = [
            name
            for name in archive.namelist()
            if name == "rllte_core-1.0.1.dist-info/METADATA"
        ]
        if len(entries) != 1:
            raise ValueError("Pinned wheel must have exactly one expected METADATA file")
        meta = BytesParser().parsebytes(archive.read(entries[0]))
    if meta.get("Name", "").lower().replace("_", "-") != PACKAGE:
        raise ValueError("Unexpected wheel package name")
    if meta.get("Version") != VERSION:
        raise ValueError("Unexpected wheel package version")
    project = tomllib.loads(project_file.read_text(encoding="utf-8"))
    local = _matplotlib_specs(project["project"]["dependencies"])
    remote = _matplotlib_specs(meta.get_all("Requires-Dist", []))
    result = _compare_matplotlib(remote, local)
    return {
        "status": "STOP_NO_NATIVE_TRAINING",
        "package": PACKAGE,
        "version": VERSION,
        "wheel_sha256": actual,
        "matplotlib_verdict": result,
        "upstream_matplotlib": remote,
        "bagi_matplotlib": local,
        "runtime_import_tested": False,
        "all_dependencies_resolved": False,
        "native_actions": 0,
        "native_resets": 0,
        "authorized_to_train": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "pyproject.toml",
    )
    args = parser.parse_args()
    print(json.dumps(audit_wheel(args.wheel, args.pyproject), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
