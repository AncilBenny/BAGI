"""Collect a mixture of causal curriculum and random Workshop experience."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from bagi.experiments.logger import TransitionLogger
from bagi.utils import seed_everything
from bagi.worlds import WorkshopEnv

# A compact curriculum trajectory that touches the Workshop's causal chain:
# start -> switch -> key -> lock -> goal.
GUIDED_ACTIONS = [1, 1, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2]


def log_step(logger, episode, obs, action, transition):
    next_obs, reward, terminated, truncated, info = transition
    logger.log(
        episode=episode,
        observation=obs,
        action=int(action),
        next_observation=next_obs,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )
    return next_obs, terminated or truncated, info


def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config["seed"])
    seed_everything(seed)
    env_cfg = config["environment"]
    collect_cfg = config["collection"]

    # Keep guided curriculum and random background experience on independent
    # seed ranges. This guarantees that changing guided_episodes does not also
    # change the random trajectories used in sample-efficiency comparisons.
    guided_seed_base = int(collect_cfg.get("guided_seed_base", seed))
    random_seed_base = int(collect_cfg.get("random_seed_base", seed + 100_000))

    env = WorkshopEnv(
        size=int(env_cfg["size"]),
        max_steps=int(env_cfg["max_steps"]),
        view_size=int(env_cfg["view_size"]),
    )

    output_dir = Path(collect_cfg["output_dir"])
    transitions_path = output_dir / "transitions.jsonl"
    if transitions_path.exists():
        transitions_path.unlink()
    logger = TransitionLogger(transitions_path)

    event_counts: dict[str, int] = {}
    episode_index = 0

    for guided_index in range(int(collect_cfg["guided_episodes"])):
        episode_seed = guided_seed_base + guided_index
        obs, _ = env.reset(seed=episode_seed)
        env.action_space.seed(episode_seed)
        done = False
        for action in GUIDED_ACTIONS:
            if done:
                break
            obs, done, info = log_step(logger, episode_index, obs, action, env.step(action))
            for event in info.get("events", []):
                event_counts[event] = event_counts.get(event, 0) + 1
        episode_index += 1

    for random_index in range(int(collect_cfg["random_episodes"])):
        episode_seed = random_seed_base + random_index
        obs, _ = env.reset(seed=episode_seed)
        env.action_space.seed(episode_seed)
        done = False
        while not done:
            action = env.action_space.sample()
            obs, done, info = log_step(logger, episode_index, obs, action, env.step(action))
            for event in info.get("events", []):
                event_counts[event] = event_counts.get(event, 0) + 1
        episode_index += 1

    print(
        f"Collected {collect_cfg['guided_episodes']} guided + "
        f"{collect_cfg['random_episodes']} random episodes"
    )
    print(
        f"Seed ranges: guided_base={guided_seed_base} "
        f"random_base={random_seed_base}"
    )
    print(f"Transitions: {transitions_path}")
    print("Causal event counts:")
    for event in ("switch_activated", "key_picked_up", "lock_opened", "goal_reached"):
        print(f"  {event}: {event_counts.get(event, 0)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/guided_collection.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
