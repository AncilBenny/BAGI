"""E6: Measure disagreement between recursively imagined and real causal trajectories.

This is an offline diagnostic: it never trains or updates a model.
The simulator's hidden state is never supplied to the learned model.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from bagi.experiments.collect_guided import GUIDED_ACTIONS
from bagi.models import WorldModel
from bagi.planning import ModelPredictivePlanner
from bagi.worlds import WorkshopEnv

EVENT_NAMES = ("switch_activated", "key_picked_up", "lock_opened", "goal_reached")


def outcome_errors(predicted_obs, actual_obs, predicted_reward, actual_reward,
                   predicted_events, actual_events) -> dict[str, float]:
    """Compare a predicted outcome against a real transition at one horizon."""
    expected = torch.as_tensor(actual_obs, dtype=torch.long)
    predicted = torch.as_tensor(predicted_obs, dtype=torch.long)
    probs = torch.as_tensor(predicted_events, dtype=torch.float32).flatten()
    targets = torch.tensor([float(name in actual_events) for name in EVENT_NAMES])
    if probs.numel() != len(EVENT_NAMES):
        raise ValueError("World model event count does not match the Workshop.")
    negatives = targets == 0
    return {
        "cell_accuracy": float((predicted == expected).float().mean()),
        "whole_observation_accuracy": float(torch.equal(predicted, expected)),
        "reward_mae": abs(float(predicted_reward) - float(actual_reward)),
        "event_brier": float(((probs - targets) ** 2).mean()),
        "false_event_probability": float(probs[negatives].mean()) if negatives.any() else 0.0,
        "true_event_count": float(targets.sum()),
    }


@torch.no_grad()
def evaluate_trajectory(model, trajectory: list[dict], horizons: list[int],
                        anchor_stride: int, max_anchors: int) -> dict[int, dict]:
    """Compare recursively rolled-out predictions with the same real action sequence."""
    if anchor_stride < 1 or max_anchors < 1 or not horizons or min(horizons) < 1:
        raise ValueError("Positive horizons, anchor_stride and max_anchors are required.")
    groups: dict[int, list[dict]] = defaultdict(list)
    starts = list(range(0, len(trajectory), anchor_stride))[:max_anchors]
    target_horizons = set(horizons)
    for start in starts:
        imagined_obs = torch.as_tensor(trajectory[start]["observation"],
                                       dtype=torch.long).unsqueeze(0)
        max_depth = min(max(horizons), len(trajectory) - start)
        for depth in range(1, max_depth + 1):
            real = trajectory[start + depth - 1]
            action = torch.tensor([real["action"]], dtype=torch.long)
            imagined_obs, predicted_reward, event_probs = model.predict_events(
                imagined_obs, action
            )
            if depth in target_horizons:
                groups[depth].append(outcome_errors(
                    imagined_obs.squeeze(0),
                    real["next_observation"],
                    float(predicted_reward.item()),
                    real["reward"],
                    event_probs.squeeze(0),
                    real["events"],
                ))
    metrics: dict[int, dict] = {}
    for horizon in horizons:
        values = groups[horizon]
        metrics[horizon] = {
            "comparisons": len(values),
            **({key: sum(row[key] for row in values) / len(values)
                for key in values[0]} if values else {}),
        }
    return metrics


def collect_trajectory(env, policy: str, seed: int, planner=None) -> list[dict]:
    observation, _ = env.reset(seed=seed)
    rng = random.Random(seed + 100_000)
    trajectory = []
    for step in range(env.max_steps):
        if policy == "guided":
            if step >= len(GUIDED_ACTIONS):
                break
            action = GUIDED_ACTIONS[step]
        elif policy == "random":
            action = rng.randrange(env.action_space.n)
        elif policy == "planner":
            if planner is None:
                raise ValueError("planner policy needs a planner")
            action = planner.select_action(observation)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        next_observation, reward, terminated, truncated, info = env.step(action)
        trajectory.append({
            "observation": observation,
            "action": int(action),
            "next_observation": next_observation,
            "reward": float(reward),
            "events": set(info.get("events", [])),
        })
        observation = next_observation
        if terminated or truncated:
            break
    return trajectory


def run(config_path: str) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    env_cfg = config["environment"]
    model_cfg = config["model"]
    eval_cfg = config["evaluation"]
    env = WorkshopEnv(**env_cfg)
    model = WorldModel(
        observation_shape=tuple(env.observation_space.shape),
        actions=int(model_cfg["actions"]),
        cell_values=int(model_cfg["cell_values"]),
        hidden=int(model_cfg["hidden"]),
    )
    checkpoint = Path(model_cfg["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training checkpoint not found: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    horizons = [int(h) for h in eval_cfg["horizons"]]
    episodes = int(eval_cfg["episodes_per_policy"])
    stride = int(eval_cfg["anchor_stride"])
    max_anchors = int(eval_cfg["max_anchors_per_episode"])
    seed_base = int(eval_cfg["evaluation_seed_base"])
    results = {}
    for policy in ("guided", "random", "planner"):
        grouped: dict[int, list[dict]] = defaultdict(list)
        stage_counts = {name: 0 for name in EVENT_NAMES}
        for episode in range(episodes):
            planner = None
            if policy == "planner":
                planner = ModelPredictivePlanner(
                    model=model, actions=int(model_cfg["actions"]),
                    horizon=int(eval_cfg["planner_horizon"]),
                    candidates=int(eval_cfg["planner_candidates"]),
                    seed=seed_base + episode,
                    event_weights=[float(v) for v in eval_cfg["event_weights"]],
                )
            trajectory = collect_trajectory(env, policy, seed_base + episode, planner)
            for step in trajectory:
                for name in EVENT_NAMES:
                    stage_counts[name] += int(name in step["events"])
            per_episode = evaluate_trajectory(model, trajectory, horizons, stride, max_anchors)
            for h, values in per_episode.items():
                if values["comparisons"]:
                    grouped[h].append(values)
        by_horizon = {}
        for h in horizons:
            rows = grouped[h]
            count = sum(row["comparisons"] for row in rows)
            by_horizon[str(h)] = {"comparisons": count}
            if count:
                for metric in rows[0]:
                    if metric != "comparisons":
                        by_horizon[str(h)][metric] = (
                            sum(row[metric] * row["comparisons"] for row in rows) / count
                        )
        results[policy] = {"episodes": episodes, "causal_events": stage_counts,
                           "by_horizon": by_horizon}
        print(f"{policy}: events={stage_counts}")
        for h, scores in by_horizon.items():
            print(f"  horizon={h} n={scores['comparisons']} "
                  f"cells={scores.get('cell_accuracy', float('nan')):.3f} "
                  f"whole_obs={scores.get('whole_observation_accuracy', float('nan')):.3f} "
                  f"event_brier={scores.get('event_brier', float('nan')):.4f}")

    summary = {
        "question": "Do recursive imagined rollouts disagree with reality, particularly on planner-selected actions?",
        "limits": [
            "Workshop layout is fixed; evaluation seeds do not create unseen layouts.",
            "Same actual action sequence is replayed; this is not a counterfactual outcome estimate.",
            "No recurrent memory or uncertainty head is tested in this diagnostic.",
        ],
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "results": results,
    }
    output = Path(eval_cfg["output_dir"]) / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved imagination-gap diagnostic to {output}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/imagination_gap.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
