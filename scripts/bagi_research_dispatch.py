"""Select the requested study before execution; reject accidental wrong-branch runs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

RESEARCH_REF = "refs/heads/setup/initial-platform"
REQUEST_PATHS = {
    ".github/bagi-m1-once.json": ("m1", "BAGI-M1-2026-11-11"),
    ".github/bagi-m2-once.json": ("m2", "BAGI-M2-2026-11-11"),
}


def selected_study(event: str, study: str, ref: str) -> str:
    """Require an explicit, research-branch-only manual study; keep schedules legacy."""
    if event == "schedule":
        if study:
            raise ValueError("Scheduled research cannot select a manual study")
        return "legacy"
    if event != "workflow_dispatch":
        raise ValueError("Unregistered research trigger")
    if study not in ("legacy", "m1", "m2"):
        raise ValueError("Manual research dispatch requires an explicit registered study")
    if ref != RESEARCH_REF:
        raise ValueError("Manual research dispatch must target the protected research branch")
    return study


def push_request_study() -> str:
    """Accept only one new, source-bound M1 request file from a branch push."""
    if os.environ.get("GITHUB_REF") != RESEARCH_REF:
        raise ValueError("M1 push must target the research branch")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        raise ValueError("M1 push event provenance missing")
    event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, timeout=5
    ).strip()
    parent = subprocess.check_output(
        ["git", "rev-parse", "HEAD^"], text=True, timeout=5
    ).strip()
    changed = subprocess.check_output(
        ["git", "diff-tree", "--no-commit-id", "--name-status", "-r", "HEAD"],
        text=True, timeout=5,
    ).strip()
    if (event.get("ref") != RESEARCH_REF
            or event.get("after") != actual
            or os.environ.get("GITHUB_SHA") != actual
            or event.get("before") != parent):
        raise ValueError("M1 push commit/event/parent mismatch")
    matches = [
        (path, study, request_id)
        for path, (study, request_id) in REQUEST_PATHS.items()
        if changed == f"A{chr(9)}{path}"
    ]
    if len(matches) != 1:
        raise ValueError("Research request commit must add one registered file")
    path, study, request_id = matches[0]
    request = json.loads(Path(path).read_text(encoding="utf-8"))
    if request != {
        "study": study,
        "force_retrain": True,
        "expected_parent_sha": parent,
        "request_id": request_id,
    }:
        raise ValueError("Research request metadata invalid or not fresh")
    return study


def main() -> int:
    try:
        if os.environ["GITHUB_EVENT_NAME"] == "push":
            mode = push_request_study()
        else:
            mode = selected_study(
                os.environ["GITHUB_EVENT_NAME"],
                os.environ.get("BAGI_DISPATCH_STUDY", ""),
                os.environ.get("GITHUB_REF", ""),
            )
    except (KeyError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"BAGI research dispatch HOLD: {exc}", file=sys.stderr)
        return 2
    print(f"BAGI research dispatch admitted mode: {mode}")
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            output.write("study=" + mode + chr(10))
    return 0


if __name__ == "__main__":
    sys.exit(main())
