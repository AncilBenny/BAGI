"""Train the BAGI predictive world model from logged Workshop transitions."""

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

EVENT_NAMES = ("switch_activated", "key_picked_up", "lock_opened", "goal_reached")


def load_transitions(path: Path) -> tuple[torch.Tensor, ...]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    next_observations: list[np.ndarray] = []
    rewards: list[float] = []
    events: list[list[float]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            observations.append(np.asarray(row["observation"], dtype=np.int64))
            actions.append(int(row["action"]))
            next_observations.append(np.asarray(row["next_observation"], dtype=np.int64))
            rewards.append(float(row["reward"]))
            observed_events = set(row.get("info", {}).get("events", []))
            events.append([float(name in observed_events) for name in EVENT_NAMES])

    if not observations:
        raise ValueError(f"No transitions found in {path}")

    return (
        torch.as_tensor(np.stack(observations), dtype=torch.long),
        torch.tensor(actions, dtype=torch.long),
        torch.as_tensor(np.stack(next_observations), dtype=torch.long),
        torch.tensor(rewards, dtype=torch.float32),
        torch.tensor(events, dtype=torch.float32),
    )


def evaluate(model: WorldModel, loader: DataLoader) -> tuple[float, float, list[float]]:
    model.eval()
    correct = 0
    cells = 0
    reward_error = 0.0
    samples = 0
    event_tp = torch.zeros(len(EVENT_NAMES), dtype=torch.float32)
    event_fp = torch.zeros(len(EVENT_NAMES), dtype=torch.float32)
    event_fn = torch.zeros(len(EVENT_NAMES), dtype=torch.float32)

    with torch.no_grad():
        for observations, actions, targets, rewards, event_targets in loader:
            logits, predicted_rewards, event_logits = model.forward_with_events(observations, actions)
            predictions = logits.argmax(dim=-1)
            correct += int((predictions == targets).sum().item())
            cells += int(targets.numel())
            reward_error += float(torch.abs(predicted_rewards - rewards).sum().item())
            samples += int(rewards.numel())

            event_predictions = torch.sigmoid(event_logits) >= 0.5
            truth = event_targets > 0.5
            event_tp += (event_predictions & truth).sum(dim=0).float()
            event_fp += (event_predictions & ~truth).sum(dim=0).float()
            event_fn += (~event_predictions & truth).sum(dim=0).float()

    f1 = (2 * event_tp) / torch.clamp(2 * event_tp + event_fp + event_fn, min=1.0)
    return correct / max(1, cells), reward_error / max(1, samples), f1.tolist()


def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config["seed"])
    seed_everything(seed)
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    observations, actions, next_observations, rewards, events = load_transitions(
        Path(data_cfg["transitions"])
    )

    indices = list(range(len(actions)))
    random.Random(seed).shuffle(indices)
    split = max(1, int(len(indices) * (1.0 - float(train_cfg["validation_fraction"]))))
    train_idx = indices[:split]
    val_idx = indices[split:] or indices[-1:]

    train_set = TensorDataset(
        observations[train_idx],
        actions[train_idx],
        next_observations[train_idx],
        rewards[train_idx],
        events[train_idx],
    )
    val_set = TensorDataset(
        observations[val_idx],
        actions[val_idx],
        next_observations[val_idx],
        rewards[val_idx],
        events[val_idx],
    )
    train_loader = DataLoader(train_set, batch_size=int(train_cfg["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=int(train_cfg["batch_size"]))

    model = WorldModel(
        observation_shape=tuple(observations.shape[1:]),
        actions=int(model_cfg["actions"]),
        cell_values=int(model_cfg["cell_values"]),
        hidden=int(model_cfg["hidden"]),
        event_count=len(EVENT_NAMES),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]))

    train_events = events[train_idx]
    positives = train_events.sum(dim=0)
    negatives = len(train_idx) - positives
    event_pos_weight = negatives / torch.clamp(positives, min=1.0)
    event_pos_weight = torch.clamp(event_pos_weight, max=float(train_cfg.get("max_event_pos_weight", 50.0)))

    event_loss_weight = float(train_cfg.get("event_loss_weight", 1.0))
    positive_reward_weight = float(train_cfg.get("positive_reward_weight", 20.0))

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    with metrics_path.open("w", encoding="utf-8") as metrics:
        for epoch in range(int(train_cfg["epochs"])):
            model.train()
            train_loss = 0.0
            batches = 0
            for obs_batch, action_batch, target_batch, reward_batch, event_batch in train_loader:
                logits, predicted_rewards, event_logits = model.forward_with_events(
                    obs_batch, action_batch
                )
                observation_loss = F.cross_entropy(
                    logits.reshape(-1, model.cell_values), target_batch.reshape(-1)
                )
                reward_weights = torch.where(
                    reward_batch > 0,
                    torch.full_like(reward_batch, positive_reward_weight),
                    torch.ones_like(reward_batch),
                )
                reward_loss = ((predicted_rewards - reward_batch) ** 2 * reward_weights).mean()
                event_loss = F.binary_cross_entropy_with_logits(
                    event_logits, event_batch, pos_weight=event_pos_weight
                )
                loss = observation_loss + reward_loss + event_loss_weight * event_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                batches += 1

            cell_accuracy, reward_mae, event_f1 = evaluate(model, val_loader)
            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss / max(1, batches),
                "validation_cell_accuracy": cell_accuracy,
                "validation_reward_mae": reward_mae,
                "validation_event_f1": dict(zip(EVENT_NAMES, event_f1, strict=True)),
            }
            metrics.write(json.dumps(row) + "\n")
            print(
                f"epoch={epoch + 1} loss={row['train_loss']:.4f} "
                f"cell_acc={cell_accuracy:.3f} reward_mae={reward_mae:.4f} "
                f"event_f1={sum(event_f1) / len(event_f1):.3f}"
            )

    torch.save(model.state_dict(), output_dir / "world_model.pt")
    print("Event F1 by class:")
    for name, score in zip(EVENT_NAMES, event_f1, strict=True):
        print(f"  {name}: {score:.3f}")
    print(f"Saved world model metrics to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world_model.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
