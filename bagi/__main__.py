"""Canonical entry point: current capability is a synthetic runtime smoke test."""
import json

from bagi.experiments.general_brain_smoke import run


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
