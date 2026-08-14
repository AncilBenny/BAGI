"""Train the first BAGI predictive world model from logged transitions."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from bagi.models import WorldModel
from bagi.utils import seed_everything


def load_transitions(path: Path) -> tuple[torch.Tensor, ...]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    next_observations: list[np.ndarray] = []
    rewards: list[float] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            observations.append(np.asarray(row["observation"], dtype=np.int64))
            actions.append(int(row["action"]))
            next_observations.append(np.asarray(row["next_observation"], dtype=np.int64))
            rewards.append(float(row["reward"]))

    if not observations:
        raise ValueError(f"No transitions found in {path}")

    return (
        torch.as_tensor(np.stack(observations), dtype=torch.long),
        torch.tensor(actions, dtype=torch.long),
        torch.as_tensor(np.stack(next_observations), dtype=torch.long),
        torch.tensor(rewards, dtype=torch.float32),
    )


def evaluate(model: WorldModel, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    correct = 0
    cells = 0
    reward_error = 0.0
    samples = 0
    with torch.no_grad():
        for observations, actions, targets, rewards in loader:
            logits, predicted_rewards = model(observations, actions)
            predictions = logits.argmax(dim=-1)
            correct += int((predictions == targets).sum().item())
            cells += int(targets.numel())
            reward_error += float(torch.abs(predicted_rewards - rewards).sum().item())
            samples += int(rewards.numel())
    return correct / max(1, cells), reward_error / max(1, samples)


def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config["seed"])
    seed_everything(seed)
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    observations, actions, next_observations, rewards = load_transitions(
        Path(data_cfg["transitions"])
    )

    indices = list(range(len(actions)))
    random.Random(seed).shuffle(indices)
    split = max(1, int(len(indices) * (1.0 - float(train_cfg["validation_fraction"]))))
    train_idx = indices[:split]
    val_idx = indices[split:] or indices[-1:]

    train_set = TensorDataset(
        observations[train_idx], actions[train_idx], next_observations[train_idx], rewards[train_idx]
    )
    val_set = TensorDataset(
        observations[val_idx], actions[val_idx], next_observations[val_idx], rewards[val_idx]
    )
    train_loader = DataLoader(train_set, batch_size=int(train_cfg["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=int(train_cfg["batch_size"]))

    model = WorldModel(
        observation_shape=tuple(observations.shape[1:]),
        actions=int(actions.max().item()) + 1,
        cell_values=int(model_cfg["cell_values"]),
        hidden=int(model_cfg["hidden"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]))

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    with metrics_path.open("w", encoding="utf-8") as metrics:
        for epoch in range(int(train_cfg["epochs"])):
            model.train()
            train_loss = 0.0
            batches = 0
            for obs_batch, action_batch, target_batch, reward_batch in train_loader:
                logits, predicted_rewards = model(obs_batch, action_batch)
                observation_loss = F.cross_entropy(
                    logits.reshape(-1, model.cell_values), target_batch.reshape(-1)
                )
                reward_loss = F.mse_loss(predicted_rewards, reward_batch)
                loss = observation_loss + reward_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                batches += 1

            cell_accuracy, reward_mae = evaluate(model, val_loader)
            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss / max(1, batches),
                "validation_cell_accuracy": cell_accuracy,
                "validation_reward_mae": reward_mae,
            }
            metrics.write(json.dumps(row) + "\n")
            print(
                f"epoch={epoch + 1} loss={row['train_loss']:.4f} "
                f"cell_acc={cell_accuracy:.3f} reward_mae={reward_mae:.4f}"
            )

    torch.save(model.state_dict(), output_dir / "world_model.pt")
    print(f"Saved world model metrics to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world_model.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
