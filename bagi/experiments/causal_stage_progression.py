"""Measure how far the planner progresses through Workshop v0.2's causal chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from bagi.models import WorldModel
from bagi.planning import ModelPredictivePlanner
from bagi.worlds import WorkshopEnv

STAGES = ("switch", "door", "key", "lock", "goal")


def evaluate(
    checkpoint: Path,
    env_cfg: dict,
    model_cfg: dict,
    plan_cfg: dict,
    planner_seed: int,
    episodes: int,
    evaluation_seed_base: int,
) -> dict[str, float]:
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

    reached = {stage: 0 for stage in STAGES}
    total_steps = 0
    for episode in range(episodes):
        observation, _ = env.reset(seed=evaluation_seed_base + episode)
        episode_reached = {stage: False for stage in STAGES}
        done = False
        while not done:
            action = planner.select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            events = set(info.get("events", []))
            position = np.asarray(info["position"])

            episode_reached["switch"] |= "switch_activated" in events
            episode_reached["door"] |= bool(np.array_equal(position, env._door))
            episode_reached["key"] |= "key_picked_up" in events
            episode_reached["lock"] |= "lock_opened" in events
            episode_reached["goal"] |= "goal_reached" in events or float(reward) > 0.0
            done = terminated or truncated

        for stage, value in episode_reached.items():
            reached[stage] += int(value)

    return {
        **{f"{stage}_rate": reached[stage] / max(1, episodes) for stage in STAGES},
        "mean_episode_steps": total_steps / max(1, episodes),
    }


def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    root = Path(config["run_dir"])
    budget = int(config["guided_budget"])
    checkpoint = root / f"guided_{budget:03d}" / "world_model" / "world_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    planner_seeds = [int(value) for value in config["planner_seeds"]]
    horizons = [int(value) for value in config["horizons"]]
    episodes = int(config["episodes"])
    evaluation_seed_base = int(config["evaluation_seed_base"])
    env_cfg = config["environment"]
    model_cfg = config["model"]
    base_plan_cfg = config["planning"]

    rows = []
    for horizon in horizons:
        per_seed = []
        for planner_seed in planner_seeds:
            result = evaluate(
                checkpoint,
                env_cfg,
                model_cfg,
                {**base_plan_cfg, "horizon": horizon},
                planner_seed,
                episodes,
                evaluation_seed_base,
            )
            per_seed.append(result)
            rows.append({"horizon": horizon, "planner_seed": planner_seed, **result})

        means = {
            key: sum(float(row[key]) for row in per_seed) / len(per_seed)
            for key in per_seed[0]
        }
        print(
            f"horizon={horizon} "
            f"switch={means['switch_rate']:.3f} "
            f"door={means['door_rate']:.3f} "
            f"key={means['key_rate']:.3f} "
            f"lock={means['lock_rate']:.3f} "
            f"goal={means['goal_rate']:.3f}"
        )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "question": "Where does the planner stop in Workshop v0.2's mandatory causal chain?",
        "guided_budget": budget,
        "planner_seeds": planner_seeds,
        "horizons": horizons,
        "episodes": episodes,
        "results": rows,
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved stage progression to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/causal_stage_progression.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
