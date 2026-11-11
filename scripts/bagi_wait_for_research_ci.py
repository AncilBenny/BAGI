"""Wait for independent branch and PR CI before one-shot M1 compute."""
from __future__ import annotations

import os
import sys
import time

from bagi_evidence_gate import BRANCH, GateError, Github, ci_snapshot


def main() -> int:
    api = Github(os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"])
    source = os.environ["REQUEST_SHA"]
    endpoint = "branches/" + BRANCH.replace("/", "%2F")
    # These are short, read-only CI admission polls, not training/proposal limits.
    for attempt in range(90):
        actual = api.request(endpoint)["commit"]["sha"]
        if actual != source:
            print("BAGI M1 HOLD: research HEAD changed during CI", file=sys.stderr)
            return 2
        try:
            ci_snapshot(api, source)
        except GateError as error:
            if attempt == 89:
                print(f"BAGI M1 HOLD: exact CI still unavailable: {error}", file=sys.stderr)
                return 2
            time.sleep(10)
            continue
        print(f"BAGI M1 exact push/PR CI admitted for {source}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
