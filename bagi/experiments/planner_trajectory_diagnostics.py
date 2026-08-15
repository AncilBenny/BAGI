"""Diagnose world-model errors on actions selected by the planner."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from bagi.models import WorldModel
from bagi.planning import ModelPredictivePlanner
from bagi.worlds import WorkshopEnv

EVENT_NAMES = ("switch_activated", "key_picked_up", "lock_opened", "goal_reached")


def new_metrics() -> dict[str, float]:
    return {
        "steps": 0.0,
        "correct_cells": 0.0,
        "cells": 0.0,
        "reward_abs_error": 0.0,
        "event_brier_sum": 0.0,
        "event_labels": 0.0,
        "event_step_brier_sum": 0.0,
        "event_step_labels": 0.0,
        "non_event_step_brier_sum": 0.0,
        "non_event_step_labels": 0.0,
    }


def merge_metrics(target: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        target[key] += value


def summarize_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "steps": int(metrics["steps"]),
        "cell_accuracy": metrics["correct_cells"] / max(1.0, metrics["cells"]),
        "reward_mae": metrics["reward_abs_error"] / max(1.0, metrics["steps"]),
        "event_brier": metrics["event_brier_sum"] / max(1.0, metrics["event_labels"]),
        "event_step_brier": metrics["event_step_brier_sum"]
        / max(1.0, metrics["event_step_labels"]),
        "non_event_step_brier": metrics["non_event_step_brier_sum"]
        / max(1.0, metrics["non_event_step_labels"]),
    }


def update_metrics(
    metrics: dict[str, float],
    predicted_next: np.ndarray,
    actual_next: np.ndarray,
    predicted_reward: float,
    actual_reward: float,
    event_probs: np.ndarray,
    event_targets: np.ndarray,
) -> None:
    metrics["steps"] += 1
    metrics["correct_cells"] += float((predicted_next == actual_next).sum())
    metrics["cells"] += float(actual_next.size)
    metrics["reward_abs_error"] += abs(predicted_reward - actual_reward)

    brier = float(((event_probs - event_targets) ** 2).sum())
    labels = float(len(EVENT_NAMES))
    metrics["event_brier_sum"] += brier
    metrics["event_labels"] += labels
    if bool(event_targets.any()):
        metrics["event_step_brier_sum"] += brier
        metrics["event_step_labels"] += labels
    else:
        metrics["non_event_step_brier_sum"] += brier
        metrics["non_event_step_labels"] += labels


def evaluate_condition(
    checkpoint: Path,
    env_cfg: dict,
    model_cfg: dict,
    plan_cfg: dict,
    training_seed: int,
    evaluation_seed_base: int,
    episodes: int,
) -> dict:
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
        seed=training_seed,
        event_weights=[float(value) for value in plan_cfg.get("event_weights", [])] or None,
    )

    grouped = {name: new_metrics() for name in ("all", "success", "failure")}
    successes = 0
    for episode in range(episodes):
        observation, _ = env.reset(seed=evaluation_seed_base + episode)
        done = False
        episode_reward = 0.0
        episode_metrics = new_metrics()

        while not done:
            action = planner.select_action(observation)
            obs_tensor = torch.as_tensor(observation, dtype=torch.long).unsqueeze(0)
            action_tensor = torch.tensor([action], dtype=torch.long)
            with torch.no_grad():
                logits, predicted_rewards, event_logits = model.forward_with_events(
                    obs_tensor, action_tensor
                )
            predicted_next = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
            predicted_reward = float(predicted_rewards.item())
            event_probs = torch.sigmoid(event_logits).squeeze(0).cpu().numpy()

            next_observation, reward, terminated, truncated, info = env.step(action)
            observed_events = set(info.get("events", []))
            event_targets = np.asarray(
                [float(name in observed_events) for name in EVENT_NAMES], dtype=np.float32
            )
            update_metrics(
                episode_metrics,
                predicted_next,
                np.asarray(next_observation),
                predicted_reward,
                float(reward),
                event_probs,
                event_targets,
            )
            observation = next_observation
            episode_reward += float(reward)
            done = terminated or truncated

        success = episode_reward > 0.0
        successes += int(success)
        merge_metrics(grouped["all"], episode_metrics)
        merge_metrics(grouped["success" if success else "failure"], episode_metrics)

    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / max(1, episodes),
        **{
            f"{group}_{metric}": value
            for group, metrics in grouped.items()
            for metric, value in summarize_metrics(metrics).items()
        },
    }


def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    root = Path(config["multiseed_run_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    training_seeds = [int(value) for value in config["training_seeds"]]
    budgets = [int(value) for value in config["guided_budgets"]]
    evaluation_seed_base = int(config["evaluation_seed_base"])
    episodes = int(config["episodes"])
    env_cfg = config["environment"]
    model_cfg = config["model"]
    plan_cfg = config["planning"]

    rows: list[dict] = []
    for training_seed in training_seeds:
        seed_summary_path = root / f"seed_{training_seed}" / "summary.json"
        saved_summary = json.loads(seed_summary_path.read_text(encoding="utf-8"))
        expected = {
            int(row["guided_episodes"]): float(row["planner_success_rate"])
            for row in saved_summary["conditions"]
        }
        for budget in budgets:
            checkpoint = (
                root
                / f"seed_{training_seed}"
                / f"guided_{budget:03d}"
                / "world_model"
                / "world_model.pt"
            )
            result = evaluate_condition(
                checkpoint,
                env_cfg,
                model_cfg,
                plan_cfg,
                training_seed,
                evaluation_seed_base,
                episodes,
            )
            expected_rate = expected[budget]
            if abs(result["success_rate"] - expected_rate) > 1e-12:
                raise RuntimeError(
                    "Diagnostic replay did not reproduce saved planner success: "
                    f"seed={training_seed} budget={budget} "
                    f"expected={expected_rate:.3f} observed={result['success_rate']:.3f}"
                )
            row = {
                "training_seed": training_seed,
                "guided_episodes": budget,
                **result,
            }
            rows.append(row)
            print(
                f"seed={training_seed} guided={budget:>2} "
                f"success={result['success_rate']:.3f} "
                f"cell={result['all_cell_accuracy']:.3f} "
                f"event_step_brier={result['all_event_step_brier']:.4f}"
            )

    csv_path = output_dir / "planner_trajectory_diagnostics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "hypothesis": (
            "Guided causal experience improves planning by reducing model error on "
            "planner-selected trajectories, especially causal event transitions."
        ),
        "evaluation_seed_base": evaluation_seed_base,
        "episodes": episodes,
        "conditions": rows,
    }
    json_path = output_dir / "planner_trajectory_diagnostics.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved diagnostics to {json_path}")
    print(f"Saved CSV to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/planner_trajectory_diagnostics.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
