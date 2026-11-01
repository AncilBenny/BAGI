"""Compare model-free and model-based agents using real interaction counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def success_rate(rows: list[dict], reward_key: str, window: int = 25) -> float:
    tail = rows[-window:]
    if not tail:
        return 0.0
    return sum(float(row[reward_key]) > 0 for row in tail) / len(tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn", default="runs/dqn/metrics.jsonl")
    parser.add_argument("--planner", default="runs/planner/metrics.jsonl")
    args = parser.parse_args()

    dqn = load_jsonl(Path(args.dqn))
    planner = load_jsonl(Path(args.planner))
    dqn_steps = int(dqn[-1]["environment_steps"])
    planner_steps = int(planner[-1]["total_real_steps"])
    dqn_success = success_rate(dqn, "reward")
    planner_success = success_rate(planner, "episode_reward")

    print("BAGI sample-efficiency snapshot")
    print("--------------------------------")
    print(f"DQN:     real_steps={dqn_steps:6d} recent_success={dqn_success:.1%}")
    print(f"Planner: real_steps={planner_steps:6d} recent_success={planner_success:.1%}")
    print(f"Planner imagined steps: {int(planner[-1]['cumulative_imagined_steps'])}")
    print("\nInterpretation: imagined steps are compute, not real environment experience.")


if __name__ == "__main__":
    main()
