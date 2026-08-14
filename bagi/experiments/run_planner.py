"""Evaluate model-predictive planning in the real Workshop environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from bagi.models import WorldModel
from bagi.planning import ModelPredictivePlanner
from bagi.utils import seed_everything
from bagi.worlds import WorkshopEnv


def count_transitions(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config["seed"])
    seed_everything(seed)
    env_cfg = config["environment"]
    model_cfg = config["model"]
    plan_cfg = config["planning"]
    eval_cfg = config["evaluation"]

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
    checkpoint = Path(model_cfg["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing {checkpoint}. Run `make collect` and `make world-model` first."
        )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))

    planner = ModelPredictivePlanner(
        model=model,
        actions=int(model_cfg["actions"]),
        horizon=int(plan_cfg["horizon"]),
        candidates=int(plan_cfg["candidates"]),
        gamma=float(plan_cfg["gamma"]),
        seed=seed,
    )

    training_steps = count_transitions(Path(eval_cfg["training_transitions"]))
    output_dir = Path(eval_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    real_eval_steps = 0

    with metrics_path.open("w", encoding="utf-8") as metrics:
        for episode in range(int(eval_cfg["episodes"])):
            observation, _ = env.reset(seed=seed + episode)
            done = False
            reward_sum = 0.0
            episode_steps = 0
            imagined_before = planner.imagined_steps

            while not done:
                action = planner.select_action(observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                reward_sum += float(reward)
                episode_steps += 1
                real_eval_steps += 1
                done = terminated or truncated

            row = {
                "episode": episode + 1,
                "episode_reward": reward_sum,
                "episode_real_steps": episode_steps,
                "cumulative_eval_real_steps": real_eval_steps,
                "world_model_training_real_steps": training_steps,
                "total_real_steps": training_steps + real_eval_steps,
                "episode_imagined_steps": planner.imagined_steps - imagined_before,
                "cumulative_imagined_steps": planner.imagined_steps,
            }
            metrics.write(json.dumps(row) + "\n")
            print(
                f"episode={episode + 1} reward={reward_sum:.1f} "
                f"real={row['total_real_steps']} imagined={planner.imagined_steps}"
            )

    print(f"Saved planner metrics to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/planner.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
