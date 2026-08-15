"""Test whether short MPC horizon causes failure in mandatory causal Workshop v0.2."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import yaml

from bagi.experiments.crossed_planner_seed_eval import evaluate


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    root = Path(config["run_dir"])
    budget = int(config["guided_budget"])
    checkpoint = root / f"guided_{budget:03d}" / "world_model" / "world_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    horizons = [int(value) for value in config["horizons"]]
    planner_seeds = [int(value) for value in config["planner_seeds"]]
    episodes = int(config["episodes"])
    evaluation_seed_base = int(config["evaluation_seed_base"])
    env_cfg = config["environment"]
    model_cfg = config["model"]
    base_plan_cfg = config["planning"]

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    summary_rows: list[dict] = []
    for horizon in horizons:
        values = []
        plan_cfg = {**base_plan_cfg, "horizon": horizon}
        for planner_seed in planner_seeds:
            rate = evaluate(
                checkpoint,
                env_cfg,
                model_cfg,
                plan_cfg,
                planner_seed,
                evaluation_seed_base,
                episodes,
            )
            values.append(rate)
            raw_rows.append(
                {
                    "guided_episodes": budget,
                    "horizon": horizon,
                    "planner_seed": planner_seed,
                    "success_rate": rate,
                }
            )
        mean, std = mean_std(values)
        summary_rows.append(
            {
                "guided_episodes": budget,
                "horizon": horizon,
                "success_mean": mean,
                "success_std": std,
            }
        )
        print(f"horizon={horizon} success={mean:.3f}±{std:.3f}")

    summary = {
        "hypothesis": (
            "Workshop v0.2 fails with horizon 5 because the next causal event after "
            "switch activation is six actions away; horizons >= 6 should restore planning."
        ),
        "guided_budget": budget,
        "planner_seeds": planner_seeds,
        "episodes": episodes,
        "results": summary_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (output_dir / "raw_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    print(f"Saved horizon sweep to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/causal_v2_horizon_sweep.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
