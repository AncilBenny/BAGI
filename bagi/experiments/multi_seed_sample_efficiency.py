"""Run fixed-background sample-efficiency experiments across training seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from copy import deepcopy
from pathlib import Path

import yaml

from bagi.experiments.sample_efficiency import run as run_sample_efficiency
from bagi.experiments.sample_efficiency import write_yaml


METRICS = (
    "planner_success_rate",
    "planner_evaluation_real_steps",
    "planner_imagined_steps",
    "validation_cell_accuracy",
    "validation_reward_mae",
    "validation_event_f1",
)


def summarize_seed_results(seed_results: list[dict]) -> list[dict]:
    by_budget: dict[int, list[dict]] = {}
    for seed_result in seed_results:
        for condition in seed_result["conditions"]:
            budget = int(condition["guided_episodes"])
            by_budget.setdefault(budget, []).append(condition)

    summaries: list[dict] = []
    for budget in sorted(by_budget):
        conditions = by_budget[budget]
        training_steps = {int(row["training_real_steps"]) for row in conditions}
        if len(training_steps) != 1:
            raise ValueError(
                f"Training real-step count changed across seeds for guided budget {budget}: "
                f"{sorted(training_steps)}"
            )

        summary = {
            "guided_episodes": budget,
            "training_real_steps": training_steps.pop(),
            "seed_count": len(conditions),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in conditions]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    base_config_path = Path(config["base_config"])
    with base_config_path.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)

    training_seeds = [int(value) for value in config["training_seeds"]]
    output_root = Path(config["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    seed_results: list[dict] = []
    for training_seed in training_seeds:
        print(f"\n=== training seed: {training_seed} ===")
        seed_dir = output_root / f"seed_{training_seed}"
        seed_config = deepcopy(base_config)
        seed_config["training_seed"] = training_seed
        seed_config["sweep"]["output_dir"] = str(seed_dir)

        generated_config = output_root / "configs" / f"seed_{training_seed}.yaml"
        write_yaml(generated_config, seed_config)
        run_sample_efficiency(str(generated_config))

        summary_path = seed_dir / "summary.json"
        seed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        seed_results.append(seed_summary)

    aggregate = summarize_seed_results(seed_results)
    summary = {
        "base_config": str(base_config_path),
        "training_seeds": training_seeds,
        "seed_policy": seed_results[0]["seed_policy"],
        "random_control": seed_results[0]["random_control"],
        "conditions": aggregate,
        "seed_results": seed_results,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = output_root / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)

    raw_rows = [
        condition
        for seed_result in seed_results
        for condition in seed_result["conditions"]
    ]
    raw_csv_path = output_root / "seed_results.csv"
    with raw_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    print("\nFixed-background multi-seed sample-efficiency study")
    for row in aggregate:
        print(
            f"guided={row['guided_episodes']:>2} "
            f"real_steps={row['training_real_steps']:>5} "
            f"planner_success="
            f"{row['planner_success_rate_mean']:.3f}±"
            f"{row['planner_success_rate_std']:.3f}"
        )
    print(f"Saved aggregate summary to {summary_path}")
    print(f"Saved aggregate CSV to {csv_path}")
    print(f"Saved per-seed CSV to {raw_csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sample_efficiency_multiseed.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
