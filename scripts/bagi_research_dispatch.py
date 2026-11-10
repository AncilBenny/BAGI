"""Select the requested study before execution; reject accidental wrong-branch runs."""
from __future__ import annotations

import os
import sys

RESEARCH_REF = "refs/heads/setup/initial-platform"


def selected_study(event: str, study: str, ref: str) -> str:
    """Require an explicit, research-branch-only manual study; keep schedules legacy."""
    if event == "schedule":
        if study:
            raise ValueError("Scheduled research cannot select a manual study")
        return "legacy"
    if event != "workflow_dispatch":
        raise ValueError("Unregistered research trigger")
    if study not in ("legacy", "m1"):
        raise ValueError("Manual research dispatch requires an explicit registered study")
    if ref != RESEARCH_REF:
        raise ValueError("Manual research dispatch must target the protected research branch")
    return study


def main() -> int:
    try:
        mode = selected_study(
            os.environ["GITHUB_EVENT_NAME"],
            os.environ.get("BAGI_DISPATCH_STUDY", ""),
            os.environ.get("GITHUB_REF", ""),
        )
    except (KeyError, ValueError) as exc:
        print(f"BAGI research dispatch HOLD: {exc}", file=sys.stderr)
        return 2
    print(f"BAGI research dispatch admitted mode: {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
