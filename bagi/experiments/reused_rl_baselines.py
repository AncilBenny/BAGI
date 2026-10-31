"""Run existing PPO and recurrent PPO on standardized MiniGrid benchmarks.

The learning algorithms are imported from established SB3/SB3-Contrib libraries.
This is an integration smoke-test, NOT a new RL algorithm or a competence claim.
"""
from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401  # Registers standard MiniGrid environments.
import numpy as np
import torch
import yaml
from gymnasium.wrappers import FlattenObservation
from minigrid.wrappers import ImgObsWrapper

# Existing Farama benchmarks: tasks vary across distinct reset seeds.
TASKS = ("MiniGrid-MemoryS13Random-v0", "MiniGrid-DoorKey-6x6-v0")


def make_task(task: str):
    """Use only normal local image observations; no full-state or oracle signals."""
    if task not in TASKS:
        raise ValueError(f"Unapproved benchmark: {task}")
    return FlattenObservation(ImgObsWrapper(gym.make(task)))


def settings(config: dict) -> tuple[list[str], list[int], int, int]:
    tasks = [str(x) for x in config["tasks"]]
    seeds = [int(x) for x in config["training_seeds"]]
    train_steps = int(config["train_steps"])
    rollout_steps = int(config["rollout_steps"])
    eval_episodes = int(config["evaluation_episodes"])
    if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= set(TASKS):
        raise ValueError("Specify distinct registered standard MiniGrid tasks")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Specify distinct training seeds")
    if (train_steps < rollout_steps or rollout_steps < 1
            or train_steps % rollout_steps or eval_episodes < 1):
        raise ValueError("Require exact, equal rollout-divisible budgets and episodes")
    return tasks, seeds, train_steps, eval_episodes


def evaluate_agent(agent, task: str, evaluation_seeds: list[int],
                   env_factory=make_task) -> dict:
    """Count *real* evaluation interactions; reset LSTM state every episode."""
    results = []
    for seed in evaluation_seeds:
        env = env_factory(task)
        try:
            env.action_space.seed(seed)
            obs, _ = env.reset(seed=seed)
            lstm_states = None
            is_episode_start = np.array([True], dtype=bool)
            done = False
            steps = 0
            return_sum = 0.0
            while not done:
                action, lstm_states = agent.predict(
                    obs, state=lstm_states, episode_start=is_episode_start,
                    deterministic=True,
                )
                obs, reward, terminated, truncated, _ = env.step(
                    int(np.asarray(action).item())
                )
                steps += 1
                return_sum += float(reward)
                done = bool(terminated or truncated)
                is_episode_start[:] = done
            results.append({
                "seed": seed,
                "reward": return_sum,
                "goal_success": int(return_sum > 0),
                "evaluation_real_steps": steps,
            })
        finally:
            env.close()
    return {
        "episodes": len(results),
        "successes": sum(x["goal_success"] for x in results),
        "mean_reward": sum(x["reward"] for x in results) / len(results),
        "evaluation_real_steps": sum(x["evaluation_real_steps"] for x in results),
        "episode_results": results,
    }


def run(config_path: str) -> dict:
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    tasks, seeds, train_steps, eval_episodes = settings(config)
    torch.set_num_threads(1)
    rows = []
    for task in tasks:
        probe = make_task(task)
        try:
            check_env(probe, warn=True)
        finally:
            probe.close()
        for seed in seeds:
            evaluation_seeds = [
                int(config["evaluation_seed_base"]) + seed * 100 + i
                for i in range(eval_episodes)
            ]
            for name, implementation, policy in (
                ("PPO", PPO, "MlpPolicy"),
                ("RecurrentPPO", RecurrentPPO, "MlpLstmPolicy"),
            ):
                environment = make_task(task)
                try:
                    environment.action_space.seed(seed)
                    policy_kwargs = {"net_arch": {"pi": [32], "vf": [32]}}
                    if name == "RecurrentPPO":
                        policy_kwargs["lstm_hidden_size"] = 32
                    agent = implementation(
                        policy,
                        environment,
                        seed=seed,
                        device="cpu",
                        n_steps=int(config["rollout_steps"]),
                        batch_size=int(config["batch_size"]),
                        n_epochs=int(config["epochs_per_rollout"]),
                        policy_kwargs=policy_kwargs,
                        verbose=0,
                    )
                    started = time.monotonic()
                    agent.learn(total_timesteps=train_steps)
                    training_seconds = time.monotonic() - started
                    if int(agent.num_timesteps) != train_steps:
                        raise AssertionError("Training exceeded or undershot equal budget")
                    eval_results = evaluate_agent(agent, task, evaluation_seeds)
                    rows.append({
                        "task": task,
                        "algorithm": name,
                        "training_seed": seed,
                        "training_real_steps": int(agent.num_timesteps),
                        "training_wall_seconds": round(training_seconds, 3),
                        "evaluation": eval_results,
                    })
                    print(f"{task} {name} train={agent.num_timesteps} "
                          f"success={eval_results['successes']}/{eval_episodes} "
                          f"eval_steps={eval_results['evaluation_real_steps']}")
                finally:
                    environment.close()
    output = {
        "purpose": "Compatibility and budget-accuracy smoke test of EXISTING RL libraries on EXISTING randomized causal/memory benchmarks.",
        "contribution_type": "integration and reproduction, not a new algorithm",
        "package_versions": {
            "minigrid": version("minigrid"),
            "stable-baselines3": version("stable-baselines3"),
            "sb3-contrib": version("sb3-contrib"),
        },
        "config": config,
        "results": rows,
        "limitations": [
            "An intentionally tiny pilot cannot establish learning ability or sample-efficiency ranking.",
            "PPO and RecurrentPPO do not have identical parameter counts.",
            "Policy sees only local partial image observations. Human-language mission is excluded.",
            "Distinct MiniGrid reset seeds generate task instances; evaluation seeds are separate from training seeds.",
            "Training uses each library's unmodified sparse reward; diagnostic info is not given to policy.",
            "No hyperparameter search, privileged simulator state, expert demonstrations or custom RL algorithms.",
        ],
    }
    dest = Path(config["output_dir"]) / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved established-library integration results: {dest}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reused_rl_baselines.yaml")
    run(parser.parse_args().config)
