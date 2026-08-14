"""Run a reproducible Workshop data-collection experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from bagi.experiments.logger import TransitionLogger
from bagi.utils import seed_everything
from bagi.worlds import WorkshopEnv


def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config["seed"])
    seed_everything(seed)

    env_cfg = config["environment"]
    exp_cfg = config["experiment"]
    env = WorkshopEnv(
        size=int(env_cfg["size"]),
        max_steps=int(env_cfg["max_steps"]),
        view_size=int(env_cfg["view_size"]),
    )

    output_dir = Path(exp_cfg["output_dir"])
    logger = TransitionLogger(output_dir / "transitions.jsonl")

    for episode in range(int(exp_cfg["episodes"])):
        obs, info = env.reset(seed=seed + episode)
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            if exp_cfg.get("log_transitions", True):
                logger.log(
                    episode=episode,
                    observation=obs,
                    action=int(action),
                    next_observation=next_obs,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=next_info,
                )
            obs = next_obs
            info = next_info
            done = terminated or truncated

    print(f"Completed {exp_cfg['episodes']} episodes")
    print(f"Transitions: {output_dir / 'transitions.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/workshop.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
