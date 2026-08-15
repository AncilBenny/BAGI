"""Measure how causal-planning success changes with guided experience budget."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from bagi.experiments.collect_guided import run as collect_guided
from bagi.experiments.diagnose_planning import evaluate_random
from bagi.experiments.run_planner import run as run_planner
from bagi.experiments.train_world_model import run as train_world_model


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def final_jsonl_row(path: Path) -> dict:
    last: dict | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    if last is None:
        raise ValueError(f"No metrics found in {path}")
    return last


def planner_summary(path: Path) -> dict:
    episodes = 0
    successes = 0
    real_steps = 0
    imagined_steps = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            episodes += 1
            successes += int(float(row["episode_reward"]) > 0.0)
            real_steps += int(row["episode_real_steps"])
            imagined_steps = int(row["cumulative_imagined_steps"])
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / max(1, episodes),
        "evaluation_real_steps": real_steps,
        "imagined_steps": imagined_steps,
    }


def macro_event_f1(model_result: dict) -> float:
    scores = model_result["validation_event_f1"]
    return sum(float(value) for value in scores.values()) / max(1, len(scores))


def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config["seed"])
    training_seed = int(config.get("training_seed", seed))
    env_cfg = config["environment"]
    sweep_cfg = config["sweep"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    plan_cfg = config["planning"]
    eval_cfg = config["evaluation"]

    root = Path(sweep_cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    budgets = [int(value) for value in sweep_cfg["guided_budgets"]]
    random_episodes = int(sweep_cfg["random_episodes"])
    guided_seed_base = int(sweep_cfg.get("guided_seed_base", seed))
    random_seed_base = int(sweep_cfg.get("random_seed_base", seed + 100_000))
    evaluation_episodes = int(eval_cfg["episodes"])
    evaluation_seed_base = int(eval_cfg.get("seed_base", seed))

    random_control = evaluate_random(env_cfg, evaluation_episodes, evaluation_seed_base)
    rows: list[dict] = []

    print("BAGI causal sample-efficiency sweep")
    print(
        f"fixed collection seeds: guided_base={guided_seed_base} "
        f"random_base={random_seed_base}"
    )
    print(f"training_seed={training_seed} evaluation_seed_base={evaluation_seed_base}")
    print(
        f"random control: success_rate={random_control['success_rate']:.3f} "
        f"real_steps={random_control['evaluation_real_steps']}"
    )

    for budget in budgets:
        print(f"\n=== guided budget: {budget} episodes ===")
        budget_dir = root / f"guided_{budget:03d}"
        data_dir = budget_dir / "data"
        model_dir = budget_dir / "world_model"
        planner_dir = budget_dir / "planner"
        config_dir = budget_dir / "configs"

        collection_config = {
            "seed": seed,
            "environment": env_cfg,
            "collection": {
                "guided_episodes": budget,
                "random_episodes": random_episodes,
                "guided_seed_base": guided_seed_base,
                "random_seed_base": random_seed_base,
                "output_dir": str(data_dir),
            },
        }
        collection_path = config_dir / "collection.yaml"
        write_yaml(collection_path, collection_config)
        collect_guided(str(collection_path))

        transitions = data_dir / "transitions.jsonl"
        training_real_steps = count_rows(transitions)

        world_model_config = {
            "seed": training_seed,
            "data": {"transitions": str(transitions)},
            "model": model_cfg,
            "training": {**train_cfg, "output_dir": str(model_dir)},
        }
        world_model_path = config_dir / "world_model.yaml"
        write_yaml(world_model_path, world_model_config)
        train_world_model(str(world_model_path))

        planner_config = {
            "seed": training_seed,
            "environment": env_cfg,
            "model": {
                **model_cfg,
                "checkpoint": str(model_dir / "world_model.pt"),
            },
            "planning": plan_cfg,
            "evaluation": {
                "episodes": evaluation_episodes,
                "seed_base": evaluation_seed_base,
                "training_transitions": str(transitions),
                "output_dir": str(planner_dir),
            },
        }
        planner_path = config_dir / "planner.yaml"
        write_yaml(planner_path, planner_config)
        run_planner(str(planner_path))

        planner_result = planner_summary(planner_dir / "metrics.jsonl")
        model_result = final_jsonl_row(model_dir / "metrics.jsonl")
        event_f1 = macro_event_f1(model_result)
        row = {
            "training_seed": training_seed,
            "guided_episodes": budget,
            "random_episodes": random_episodes,
            "training_real_steps": training_real_steps,
            "planner_success_rate": planner_result["success_rate"],
            "planner_successes": planner_result["successes"],
            "planner_evaluation_real_steps": planner_result["evaluation_real_steps"],
            "planner_imagined_steps": planner_result["imagined_steps"],
            "validation_cell_accuracy": model_result["validation_cell_accuracy"],
            "validation_reward_mae": model_result["validation_reward_mae"],
            "validation_event_f1": event_f1,
        }
        rows.append(row)
        print(
            f"budget={budget} training_real_steps={training_real_steps} "
            f"success_rate={row['planner_success_rate']:.3f} "
            f"cell_acc={row['validation_cell_accuracy']:.3f} "
            f"event_f1={row['validation_event_f1']:.3f}"
        )

    summary = {
        "seed_policy": {
            "experiment_seed": seed,
            "training_seed": training_seed,
            "guided_seed_base": guided_seed_base,
            "random_seed_base": random_seed_base,
            "evaluation_seed_base": evaluation_seed_base,
        },
        "random_control": random_control,
        "conditions": rows,
    }
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = root / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSample-efficiency curve")
    print(f"random success rate: {random_control['success_rate']:.3f}")
    for row in rows:
        print(
            f"guided={row['guided_episodes']:>2} "
            f"real_steps={row['training_real_steps']:>5} "
            f"planner_success={row['planner_success_rate']:.3f}"
        )
    print(f"Saved summary to {summary_path}")
    print(f"Saved CSV to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sample_efficiency.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
