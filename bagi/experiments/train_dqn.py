"""Train the first model-free BAGI baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from bagi.agents.dqn import DQNAgent, Transition
from bagi.utils import seed_everything
from bagi.worlds import WorkshopEnv


def epsilon_for_episode(episode: int, start: float, end: float, decay_episodes: int) -> float:
    progress = min(1.0, episode / max(1, decay_episodes))
    return start + progress * (end - start)


def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config["seed"])
    seed_everything(seed)
    env_cfg = config["environment"]
    train_cfg = config["training"]
    env = WorkshopEnv(
        size=int(env_cfg["size"]),
        max_steps=int(env_cfg["max_steps"]),
        view_size=int(env_cfg["view_size"]),
    )
    agent = DQNAgent(
        observation_shape=env.observation_space.shape,
        actions=env.action_space.n,
        learning_rate=float(train_cfg["learning_rate"]),
        gamma=float(train_cfg["gamma"]),
    )

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    global_step = 0

    with metrics_path.open("w", encoding="utf-8") as metrics:
        for episode in range(int(train_cfg["episodes"])):
            observation, _ = env.reset(seed=seed + episode)
            epsilon = epsilon_for_episode(
                episode,
                float(train_cfg["epsilon_start"]),
                float(train_cfg["epsilon_end"]),
                int(train_cfg["epsilon_decay_episodes"]),
            )
            episode_reward = 0.0
            losses: list[float] = []
            done = False

            while not done:
                action = agent.select_action(observation, epsilon)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.buffer.add(
                    Transition(observation, action, reward, next_observation, done)
                )
                loss = agent.learn(int(train_cfg["batch_size"]))
                if loss is not None:
                    losses.append(loss)
                global_step += 1
                if global_step % int(train_cfg["target_sync_steps"]) == 0:
                    agent.sync_target()
                observation = next_observation
                episode_reward += reward

            row = {
                "episode": episode,
                "environment_steps": global_step,
                "reward": episode_reward,
                "epsilon": epsilon,
                "mean_loss": sum(losses) / len(losses) if losses else None,
            }
            metrics.write(json.dumps(row) + "\n")
            if (episode + 1) % 25 == 0:
                print(
                    f"episode={episode + 1} steps={global_step} "
                    f"reward={episode_reward:.2f} epsilon={epsilon:.3f}"
                )

    torch.save(agent.online.state_dict(), output_dir / "q_network.pt")
    print(f"Saved baseline metrics to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
