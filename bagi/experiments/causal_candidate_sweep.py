"""Test whether random-shooting candidate coverage limits causal planning."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

from bagi.experiments.causal_stage_progression import evaluate


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    root = Path(config["run_dir"])
    budget = int(config["guided_budget"])
    checkpoint = root / f"guided_{budget:03d}" / "world_model" / "world_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    candidate_counts = [int(value) for value in config["candidate_counts"]]
    planner_seeds = [int(value) for value in config["planner_seeds"]]
    episodes = int(config["episodes"])
    evaluation_seed_base = int(config["evaluation_seed_base"])
    env_cfg = config["environment"]
    model_cfg = config["model"]
    base_plan_cfg = config["planning"]

    rows: list[dict] = []
    for candidates in candidate_counts:
        per_seed = []
        for planner_seed in planner_seeds:
            result = evaluate(
                checkpoint,
                env_cfg,
                model_cfg,
                {**base_plan_cfg, "candidates": candidates},
                planner_seed,
                episodes,
                evaluation_seed_base,
            )
            per_seed.append(result)
            rows.append(
                {
                    "candidates": candidates,
                    "planner_seed": planner_seed,
                    **result,
                }
            )

        means = {
            key: mean([float(row[key]) for row in per_seed])
            for key in per_seed[0]
        }
        print(
            f"candidates={candidates:>4} "
            f"switch={means['switch_rate']:.3f} "
            f"door={means['door_rate']:.3f} "
            f"key={means['key_rate']:.3f} "
            f"lock={means['lock_rate']:.3f} "
            f"goal={means['goal_rate']:.3f}"
        )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "hypothesis": (
            "Workshop v0.2 failure is dominated by sparse random-shooting search coverage; "
            "increasing candidate count at fixed model and horizon should improve causal-stage progression."
        ),
        "guided_budget": budget,
        "candidate_counts": candidate_counts,
        "planner_seeds": planner_seeds,
        "episodes": episodes,
        "horizon": int(base_plan_cfg["horizon"]),
        "results": rows,
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved candidate sweep to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/causal_candidate_sweep.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
