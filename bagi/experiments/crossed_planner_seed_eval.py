"""Cross model-training seeds with independent planner-search seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
import yaml

from bagi.models import WorldModel
from bagi.planning import ModelPredictivePlanner
from bagi.worlds import WorkshopEnv


def evaluate(
    checkpoint: Path,
    env_cfg: dict,
    model_cfg: dict,
    plan_cfg: dict,
    planner_seed: int,
    evaluation_seed_base: int,
    episodes: int,
) -> float:
    env = WorkshopEnv(
        size=int(env_cfg["size"]),
        max_steps=int(env_cfg["max_steps"]),
        view_size=int(env_cfg["view_size"]),
    )
    model = WorldModel(
        observation_shape=tuple(env.observation_space.shape),
        actions=int(model_cfg["actions"]),
        cell_values=int(model_cfg["cell_values"]),
        hidden=int(model_cfg["hidden"]),
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    planner = ModelPredictivePlanner(
        model=model,
        actions=int(model_cfg["actions"]),
        horizon=int(plan_cfg["horizon"]),
        candidates=int(plan_cfg["candidates"]),
        gamma=float(plan_cfg["gamma"]),
        seed=planner_seed,
        event_weights=[float(value) for value in plan_cfg.get("event_weights", [])] or None,
    )

    successes = 0
    for episode in range(episodes):
        observation, _ = env.reset(seed=evaluation_seed_base + episode)
        done = False
        reward_sum = 0.0
        while not done:
            action = planner.select_action(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            reward_sum += float(reward)
            done = terminated or truncated
        successes += int(reward_sum > 0.0)
    return successes / max(1, episodes)


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    root = Path(config["multiseed_run_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    training_seeds = [int(value) for value in config["training_seeds"]]
    planner_seeds = [int(value) for value in config["planner_seeds"]]
    budgets = [int(value) for value in config["guided_budgets"]]
    episodes = int(config["episodes"])
    evaluation_seed_base = int(config["evaluation_seed_base"])
    env_cfg = config["environment"]
    model_cfg = config["model"]
    plan_cfg = config["planning"]

    rows: list[dict] = []
    for training_seed in training_seeds:
        for budget in budgets:
            checkpoint = (
                root
                / f"seed_{training_seed}"
                / f"guided_{budget:03d}"
                / "world_model"
                / "world_model.pt"
            )
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            rates = []
            for planner_seed in planner_seeds:
                success_rate = evaluate(
                    checkpoint,
                    env_cfg,
                    model_cfg,
                    plan_cfg,
                    planner_seed,
                    evaluation_seed_base,
                    episodes,
                )
                rates.append(success_rate)
                rows.append(
                    {
                        "training_seed": training_seed,
                        "planner_seed": planner_seed,
                        "guided_episodes": budget,
                        "success_rate": success_rate,
                    }
                )
            mean, std = mean_std(rates)
            print(
                f"model_seed={training_seed} guided={budget:>2} "
                f"planner_seed_mean={mean:.3f}±{std:.3f}"
            )

    raw_csv = output_dir / "crossed_seed_results.csv"
    with raw_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    per_model: list[dict] = []
    aggregate: list[dict] = []
    for budget in budgets:
        model_means = []
        planner_stds = []
        for training_seed in training_seeds:
            values = [
                float(row["success_rate"])
                for row in rows
                if int(row["training_seed"]) == training_seed
                and int(row["guided_episodes"]) == budget
            ]
            mean, std = mean_std(values)
            model_means.append(mean)
            planner_stds.append(std)
            per_model.append(
                {
                    "guided_episodes": budget,
                    "training_seed": training_seed,
                    "planner_success_mean": mean,
                    "planner_success_std": std,
                }
            )
        model_mean, between_model_std = mean_std(model_means)
        aggregate.append(
            {
                "guided_episodes": budget,
                "success_mean": model_mean,
                "between_model_std": between_model_std,
                "mean_within_model_planner_std": statistics.mean(planner_stds),
            }
        )

    print("\nCrossed seed variance summary")
    for row in aggregate:
        print(
            f"guided={int(row['guided_episodes']):>2} "
            f"mean={row['success_mean']:.3f} "
            f"between_model_std={row['between_model_std']:.3f} "
            f"within_planner_std={row['mean_within_model_planner_std']:.3f}"
        )

    summary = {
        "question": (
            "How much observed planner-success variability comes from model-training "
            "randomness versus stochastic planner candidate sampling?"
        ),
        "training_seeds": training_seeds,
        "planner_seeds": planner_seeds,
        "evaluation_seed_base": evaluation_seed_base,
        "episodes": episodes,
        "aggregate": aggregate,
        "per_model": per_model,
    }
    json_path = output_dir / "crossed_seed_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_csv = output_dir / "crossed_seed_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)

    print(f"Saved summary to {json_path}")
    print(f"Saved CSV to {summary_csv}")
    print(f"Saved raw results to {raw_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crossed_planner_seed_eval.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
