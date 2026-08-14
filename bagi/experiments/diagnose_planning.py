"""Diagnose why learned-model planning succeeds or fails."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from bagi.models import WorldModel
from bagi.planning import ModelPredictivePlanner
from bagi.utils import seed_everything
from bagi.worlds import WorkshopEnv


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No transitions found in {path}")
    return rows


def build_model(env: WorkshopEnv, config: dict) -> WorldModel:
    model = WorldModel(
        observation_shape=tuple(env.observation_space.shape),
        actions=int(config["actions"]),
        cell_values=int(config["cell_values"]),
        hidden=int(config["hidden"]),
    )
    checkpoint = Path(config["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing {checkpoint}. Run `make world-model` first.")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return model


def one_step_accuracy(model: WorldModel, rows: list[dict], limit: int) -> float:
    chosen = rows[: min(limit, len(rows))]
    observations = torch.tensor(
        np.asarray([row["observation"] for row in chosen]), dtype=torch.long
    )
    actions = torch.tensor([int(row["action"]) for row in chosen], dtype=torch.long)
    targets = torch.tensor(
        np.asarray([row["next_observation"] for row in chosen]), dtype=torch.long
    )
    predicted, _ = model.predict(observations, actions)
    return float((predicted == targets).float().mean().item())


def reward_and_event_stats(rows: list[dict]) -> dict:
    positive_rewards = sum(float(row["reward"]) > 0 for row in rows)
    event_counts: Counter[str] = Counter()
    for row in rows:
        info = row.get("info", {})
        for event in info.get("events", []):
            event_counts[str(event)] += 1

    causal_events = {
        name: event_counts.get(name, 0)
        for name in [
            "switch_activated",
            "key_picked_up",
            "lock_opened",
            "goal_reached",
            "blocked",
        ]
    }
    return {
        "transitions": len(rows),
        "positive_reward_transitions": positive_rewards,
        "positive_reward_fraction": positive_rewards / max(1, len(rows)),
        "causal_event_counts": causal_events,
    }


def predicted_reward_by_action(
    model: WorldModel, rows: list[dict], actions: int, limit: int
) -> dict[int, dict[str, float]]:
    chosen = rows[: min(limit, len(rows))]
    observations = torch.tensor(
        np.asarray([row["observation"] for row in chosen]), dtype=torch.long
    )
    stats: dict[int, dict[str, float]] = {}
    with torch.no_grad():
        for action in range(actions):
            action_batch = torch.full(
                (observations.shape[0],), action, dtype=torch.long
            )
            _, rewards = model.predict(observations, action_batch)
            stats[action] = {
                "mean": float(rewards.mean().item()),
                "std": float(rewards.std(unbiased=False).item()),
                "min": float(rewards.min().item()),
                "max": float(rewards.max().item()),
            }
    return stats


def rollout_accuracy(
    model: WorldModel,
    env_cfg: dict,
    seed: int,
    horizons: list[int],
    samples: int,
) -> dict[int, float]:
    max_horizon = max(horizons)
    correct = {h: 0 for h in horizons}
    cells = {h: 0 for h in horizons}
    rng = random.Random(seed)

    for sample in range(samples):
        env = WorkshopEnv(
            size=int(env_cfg["size"]),
            max_steps=max(max_horizon + 1, int(env_cfg["max_steps"])),
            view_size=int(env_cfg["view_size"]),
        )
        real_obs, _ = env.reset(seed=seed + sample)
        imagined_obs = torch.tensor(real_obs, dtype=torch.long).unsqueeze(0)

        for step in range(1, max_horizon + 1):
            action = rng.randrange(env.action_space.n)
            real_obs, _, terminated, truncated, _ = env.step(action)
            action_tensor = torch.tensor([action], dtype=torch.long)
            imagined_obs, _ = model.predict(imagined_obs, action_tensor)

            if step in correct:
                target = torch.tensor(real_obs, dtype=torch.long).unsqueeze(0)
                correct[step] += int((imagined_obs == target).sum().item())
                cells[step] += int(target.numel())

            if terminated or truncated:
                break

    return {h: correct[h] / max(1, cells[h]) for h in horizons}


def evaluate_random(env_cfg: dict, episodes: int, seed: int) -> dict:
    env = WorkshopEnv(
        size=int(env_cfg["size"]),
        max_steps=int(env_cfg["max_steps"]),
        view_size=int(env_cfg["view_size"]),
    )
    rng = random.Random(seed)
    total_reward = 0.0
    real_steps = 0
    successes = 0
    action_counts: Counter[int] = Counter()

    for episode in range(episodes):
        env.reset(seed=seed + episode)
        done = False
        while not done:
            action = rng.randrange(env.action_space.n)
            action_counts[action] += 1
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            real_steps += 1
            if terminated:
                successes += 1
            done = terminated or truncated

    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / max(1, episodes),
        "reward": total_reward,
        "evaluation_real_steps": real_steps,
        "action_counts": dict(sorted(action_counts.items())),
    }


def evaluate_planner(
    model: WorldModel,
    env_cfg: dict,
    model_cfg: dict,
    horizon: int,
    candidates: int,
    gamma: float,
    episodes: int,
    seed: int,
) -> dict:
    env = WorkshopEnv(
        size=int(env_cfg["size"]),
        max_steps=int(env_cfg["max_steps"]),
        view_size=int(env_cfg["view_size"]),
    )
    planner = ModelPredictivePlanner(
        model=model,
        actions=int(model_cfg["actions"]),
        horizon=horizon,
        candidates=candidates,
        gamma=gamma,
        seed=seed,
    )
    total_reward = 0.0
    real_steps = 0
    successes = 0
    action_counts: Counter[int] = Counter()

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        done = False
        while not done:
            action = planner.select_action(observation)
            action_counts[action] += 1
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            real_steps += 1
            if terminated:
                successes += 1
            done = terminated or truncated

    return {
        "horizon": horizon,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / max(1, episodes),
        "reward": total_reward,
        "evaluation_real_steps": real_steps,
        "imagined_steps": planner.imagined_steps,
        "action_counts": dict(sorted(action_counts.items())),
    }


def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config["seed"])
    seed_everything(seed)
    env_cfg = config["environment"]
    model_cfg = config["model"]
    diag_cfg = config["diagnostics"]

    env = WorkshopEnv(
        size=int(env_cfg["size"]),
        max_steps=int(env_cfg["max_steps"]),
        view_size=int(env_cfg["view_size"]),
    )
    model = build_model(env, model_cfg)
    rows = load_rows(Path(diag_cfg["training_transitions"]))
    horizons = [int(value) for value in diag_cfg["horizons"]]
    reward_stats = reward_and_event_stats(rows)
    predicted_rewards = predicted_reward_by_action(
        model,
        rows,
        int(model_cfg["actions"]),
        int(diag_cfg["rollout_samples"]),
    )

    results = {
        "world_model_training_real_steps": len(rows),
        "training_signal": reward_stats,
        "predicted_reward_by_action": predicted_rewards,
        "one_step_cell_accuracy": one_step_accuracy(
            model, rows, int(diag_cfg["rollout_samples"])
        ),
        "closed_loop_cell_accuracy": rollout_accuracy(
            model,
            env_cfg,
            seed,
            horizons,
            int(diag_cfg["rollout_samples"]),
        ),
        "random_policy": evaluate_random(
            env_cfg, int(diag_cfg["episodes"]), seed
        ),
        "planner": [],
    }

    for horizon in horizons:
        results["planner"].append(
            evaluate_planner(
                model=model,
                env_cfg=env_cfg,
                model_cfg=model_cfg,
                horizon=horizon,
                candidates=int(diag_cfg["candidates"]),
                gamma=float(diag_cfg["gamma"]),
                episodes=int(diag_cfg["episodes"]),
                seed=seed,
            )
        )

    output_dir = Path(diag_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("BAGI planning diagnostics")
    print(f"world-model training real steps: {len(rows)}")
    print(
        "positive reward transitions: "
        f"{reward_stats['positive_reward_transitions']}/{reward_stats['transitions']} "
        f"({reward_stats['positive_reward_fraction']:.4f})"
    )
    print("causal events:")
    for event, count in reward_stats["causal_event_counts"].items():
        print(f"  {event}: {count}")
    print("predicted reward by action (mean ± std):")
    for action, stats in predicted_rewards.items():
        print(f"  action={action}: {stats['mean']:.5f} ± {stats['std']:.5f}")
    print(f"one-step cell accuracy: {results['one_step_cell_accuracy']:.3f}")
    for horizon in horizons:
        accuracy = results["closed_loop_cell_accuracy"][horizon]
        print(f"closed-loop accuracy @ {horizon}: {accuracy:.3f}")
    random_result = results["random_policy"]
    print(
        f"random: success_rate={random_result['success_rate']:.3f} "
        f"real_steps={random_result['evaluation_real_steps']} "
        f"actions={random_result['action_counts']}"
    )
    for planner_result in results["planner"]:
        print(
            f"planner h={planner_result['horizon']}: "
            f"success_rate={planner_result['success_rate']:.3f} "
            f"real_steps={planner_result['evaluation_real_steps']} "
            f"imagined_steps={planner_result['imagined_steps']} "
            f"actions={planner_result['action_counts']}"
        )
    print(f"Saved diagnostics to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diagnostics.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
