"""Registered feed-forward sanity control, using the shared native runtime."""
import argparse
from pathlib import Path

from bagi.experiments.gate_a_replay_competence import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.output, arms=("feedforward",), domains=(1,), updates_per_block=64)
