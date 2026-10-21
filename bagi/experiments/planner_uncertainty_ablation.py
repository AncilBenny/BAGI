"""E9: paired evaluation of baseline and uncertainty-penalized imagined planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from bagi.models import WorldModel
from bagi.planning.model_predictive import ModelPredictivePlanner
from bagi.planning.uncertainty_aware import UncertaintyAwarePlanner
from bagi.worlds import WorkshopEnv

STAGES = ("switch", "door", "key", "lock", "goal")


def evaluate(model, env_config: dict, plan_config: dict, weight: float,
             planner_seed: int, episodes: int, eval_seed_base: int) -> dict:
    env = WorkshopEnv(**env_config)
    args = {
        "model": model,
        "actions": env.action_space.n,
        "horizon": int(plan_config["horizon"]),
        "candidates": int(plan_config["candidates"]),
        "gamma": float(plan_config["gamma"]),
        "seed": planner_seed,
        "event_weights": [float(x) for x in plan_config["event_weights"]],
    }
    planner = (
        ModelPredictivePlanner(**args)
        if weight == 0.0
        else UncertaintyAwarePlanner(**args, uncertainty_weight=weight)
    )
    reached = {stage: 0 for stage in STAGES}
    total_real_steps = 0
    selected_risks = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=eval_seed_base + episode)
        episode_stages = {stage: False for stage in STAGES}
        done = False
        while not done:
            action = planner.select_action(observation)
            if isinstance(planner, UncertaintyAwarePlanner):
                selected_risks.append(planner.last_selected_uncertainty)
            observation, reward, terminated, truncated, info = env.step(action)
            events = set(info.get("events", []))
            position = np.asarray(info["position"])
            episode_stages["switch"] |= "switch_activated" in events
            episode_stages["door"] |= bool(np.array_equal(position, env._door))
            episode_stages["key"] |= "key_picked_up" in events
            episode_stages["lock"] |= "lock_opened" in events
            episode_stages["goal"] |= "goal_reached" in events or float(reward) > 0
            total_real_steps += 1
            done = terminated or truncated
        for stage in STAGES:
            reached[stage] += int(episode_stages[stage])
    return {
        **{f"{stage}_rate": reached[stage] / episodes for stage in STAGES},
        "eval_real_steps": total_real_steps,
        "mean_episode_steps": total_real_steps / episodes,
        "imagined_steps": planner.imagined_steps,
        "mean_selected_discounted_risk": (
            sum(selected_risks) / len(selected_risks) if selected_risks else None
        ),
    }


def run(config_path: str) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    env_cfg = cfg["environment"]
    model_cfg = cfg["model"]
    experiment = cfg["evaluation"]
    env = WorkshopEnv(**env_cfg)
    model = WorldModel(
        observation_shape=tuple(env.observation_space.shape),
        actions=int(model_cfg["actions"]),
        cell_values=int(model_cfg["cell_values"]),
        hidden=int(model_cfg["hidden"]),
    )
    checkpoint = Path(model_cfg["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing world-model checkpoint: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    seeds = [int(x) for x in experiment["planner_seeds"]]
    weights = [float(x) for x in experiment["uncertainty_weights"]]
    episodes = int(experiment["episodes_per_seed"])
    if not seeds or not weights or episodes < 1 or 0.0 not in weights:
        raise ValueError("Provide positive episodes, seeds, and a zero-weight baseline")
    rows = []
    for weight in weights:
        for seed in seeds:
            scores = evaluate(
                model, env_cfg, cfg["planning"], weight, seed, episodes,
                int(experiment["evaluation_seed_base"]),
            )
            row = {"uncertainty_weight": weight, "planner_seed": seed, **scores}
            rows.append(row)
            print(f"weight={weight:.2f} seed={seed} switch={scores['switch_rate']:.3f} "
                  f"door={scores['door_rate']:.3f} goal={scores['goal_rate']:.3f}")
    summaries = {}
    baselines = {row["planner_seed"]: row for row in rows
                 if row["uncertainty_weight"] == 0.0}
    for weight in weights:
        group = [row for row in rows if row["uncertainty_weight"] == weight]
        summaries[str(weight)] = {
            "mean_goal_rate": sum(x["goal_rate"] for x in group) / len(group),
            "mean_switch_rate": sum(x["switch_rate"] for x in group) / len(group),
            "mean_door_rate": sum(x["door_rate"] for x in group) / len(group),
            "paired_goal_gain": sum(
                x["goal_rate"] - baselines[x["planner_seed"]]["goal_rate"]
                for x in group
            ) / len(group),
        }
    output = {
        "hypothesis": "Penalizing uncertain imagined observations reduces model exploitation and improves causal-stage progression at matched real-interaction budgets.",
        "limitations": [
            "Predictive softmax confidence is not calibrated epistemic uncertainty.",
            "The environment has one fixed layout; seeds vary planner randomness, not layouts.",
            "The comparison is small and exploratory; a nonzero success gain needs replication.",
            "All settings share a pre-trained checkpoint; training data are not recollected.",
        ],
        "checkpoint": str(checkpoint),
        "config": config_path,
        "episodes_per_seed": episodes,
        "planner_seeds": seeds,
        "results": rows,
        "summary": summaries,
    }
    output_path = Path(experiment["output_dir"]) / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved E9 paired ablation: {output_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/planner_uncertainty_ablation.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
