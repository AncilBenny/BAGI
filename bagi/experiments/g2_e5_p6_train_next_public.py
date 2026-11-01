"""G2-E5-P6: matched TRAIN-only learned next-public-state preflight.

Uses only already-published SHA-verified TRAIN transitions, never a native
simulator, evaluator state, held-out original TEST arrays, or policy rollout.
No model or policy is promoted by this diagnostic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bagi.experiments.g2_e5_p5_public_transition_identifiability import (
    _shift_within_episodes,
)
from bagi.experiments.g2_e5_train_history_audit import _safe_archive

ARMS = ("action_only", "reactive", "shuffled_history", "real_history")
TRAIN_SEEDS = (17, 29)
FOLDS = (0, 1)
EPOCHS = 12
BATCH = 128
LEARNING_RATE = 3e-4
HIDDEN = 192
HISTORY_LENGTH = 4


@dataclass(frozen=True)
class TrainTransitions:
    name: str
    observation: np.ndarray
    action: np.ndarray
    next_observation: np.ndarray
    history: np.ndarray
    fold: np.ndarray
    n_actions: int
    target_width: int


def _executed_history(actions: np.ndarray, episodes: np.ndarray,
                      n_actions: int) -> np.ndarray:
    """Last four actually executed commands, with episode-boundary zero pads."""
    if (actions.shape != episodes.shape or actions.ndim != 1 or n_actions < 2
            or np.any(actions < 0) or np.any(actions >= n_actions)):
        raise ValueError("Invalid executed action history")
    history = np.zeros((len(actions), HISTORY_LENGTH * n_actions), dtype=np.float32)
    for identifier in np.unique(episodes):
        positions = np.flatnonzero(episodes == identifier)
        if len(positions) and positions[-1] - positions[0] + 1 != len(positions):
            raise ValueError("Noncontiguous TRAIN episode")
        for t, position in enumerate(positions):
            for lag in range(1, min(HISTORY_LENGTH, t) + 1):
                history[position, (lag - 1) * n_actions + int(actions[positions[t - lag]])] = 1
    return history


def _dataset(name: str, obs: np.ndarray, action: np.ndarray, nxt: np.ndarray,
             groups: np.ndarray, history: np.ndarray, *, n_actions: int,
             target_width: int) -> TrainTransitions:
    if (obs.ndim != 2 or obs.shape != nxt.shape or len(obs) == 0
            or action.shape != (len(obs),) or groups.shape != (len(obs),)
            or history.shape[0] != len(obs) or history.ndim != 2
            or obs.dtype.kind not in "biuf" or nxt.dtype.kind not in "biuf"
            or action.dtype.kind not in "iu" or groups.dtype.kind not in "iu"
            or not np.isfinite(obs).all() or not np.isfinite(nxt).all()
            or np.any(obs < 0) or np.any(obs > 8)
            or np.any(nxt < 0) or np.any(nxt > 8)
            or np.any(action < 0) or np.any(action >= n_actions)
            or target_width <= 0 or target_width > obs.shape[1]
            or history.shape[1] not in (28, HISTORY_LENGTH * n_actions)
            or not np.isfinite(history).all() or not np.isin(history, (0, 1)).all()
            or set(np.unique(groups)) != {0, 1}):
        raise ValueError("Invalid or non-disjoint public TRAIN transition shape")
    return TrainTransitions(name, obs.astype(np.float32), action.astype(np.int64),
                            nxt.astype(np.float32), history.astype(np.float32),
                            groups.astype(np.int8), n_actions, target_width)


def _frozen_datasets(e1: Path, e2: Path) -> list[TrainTransitions]:
    """Only numeric TRAIN keys from the unchanged P0 hash-gated loader."""
    a = _safe_archive(e1, "e1")
    b = _safe_archive(e2, "e2")
    out = []
    for domain, n_actions in (("xland", 6), ("workshop", 4)):
        prefix = f"{domain}_train_"
        obs, actions, group = (a[prefix + key] for key in ("observations", "actions", "groups"))
        episode_ids = a[prefix + "episode_ids"]
        if domain == "workshop":
            # Exclude ten duplicated deterministic fixture episodes.
            keep = a[prefix + "behaviors"][group] == "uniform_random"
            obs, actions, group = obs[keep], actions[keep], group[keep]
        current, executed, after, episode = _shift_within_episodes(obs, actions, group)
        if domain == "xland":
            # Public XLand already supplies four executed actions, 7 bits each.
            history = current[:, -28:].copy()
            folds = np.asarray([int(episode_ids[int(g)].split("-")[1]) % 2
                                for g in episode], dtype=np.int8)
            target = current.shape[1] - 28
        else:
            history = _executed_history(executed, episode, n_actions)
            folds = np.asarray([int(episode_ids[int(g)].split("-")[-1]) % 2
                                for g in episode], dtype=np.int8)
            target = 25
        out.append(_dataset(f"e1_{domain}_train", current, executed, after,
                            folds, history, n_actions=n_actions, target_width=target))
    for si, seed in enumerate(TRAIN_SEEDS):
        mask = (b["seed_index"] == si) & (b["actor_index"] == 0)
        episode = b["episode_index"][mask]
        action = b["action"][mask]
        history = _executed_history(action, episode, 4)
        out.append(_dataset(f"e2_workshop_dqn_train_seed{seed}",
                            b["observation"][mask].reshape(-1, 25), action,
                            b["next_observation"][mask].reshape(-1, 25),
                            episode % 2, history, n_actions=4, target_width=25))
    if len(out) != 4:
        raise AssertionError("P6 source datasets changed")
    return out


def _features(data: TrainTransitions, indices: np.ndarray, arm: str,
              rng: np.random.Generator) -> np.ndarray:
    if arm not in ARMS:
        raise ValueError("Unregistered P6 arm")
    current = data.observation[indices, :data.target_width] / 8.0
    history = data.history[indices].copy()
    action = np.eye(data.n_actions, dtype=np.float32)[data.action[indices]]
    if arm == "action_only":
        current = np.zeros_like(current)
        history[:] = 0
    elif arm == "reactive":
        history[:] = 0
    elif arm == "shuffled_history":
        history = history[rng.permutation(len(indices))]
    return np.concatenate((current, history, action), axis=1).astype(np.float32)


def _new_model(input_width: int, output_width: int) -> nn.Sequential:
    """Reuse the repository WorldModel's 192x2 Linear/ReLU backbone pattern."""
    return nn.Sequential(nn.Linear(input_width, HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, output_width))


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray,
                 test_x: np.ndarray, *, seed: int) -> np.ndarray:
    """A single frozen supervised MSE recipe; no validation/test tuning."""
    torch.manual_seed(seed)
    model = _new_model(train_x.shape[1], train_y.shape[1]).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x = torch.from_numpy(train_x)
    y = torch.from_numpy(train_y)
    rng = np.random.default_rng(seed + 1000)
    model.train()
    for _ in range(EPOCHS):
        for idx in np.array_split(rng.permutation(len(x)),
                                  (len(x) + BATCH - 1) // BATCH):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(test_x)).numpy().clip(0, 1)


def _metrics(prediction: np.ndarray, expected: np.ndarray,
             current: np.ndarray) -> dict[str, float | int | None]:
    square = (prediction - expected) ** 2
    changed = np.abs(expected - current) > 1e-6
    return {
        "public_cell_mse": float(square.mean()),
        "changed_public_cells": int(changed.sum()),
        "changed_cell_mse": float(square[changed].mean()) if changed.any() else None,
        "evaluated_cells": int(square.size),
    }


def run(e1: Path, e2: Path) -> dict:
    torch.set_num_threads(1)
    rows = []
    for dataset in _frozen_datasets(e1, e2):
        for fold in FOLDS:
            train_idx = np.flatnonzero(dataset.fold != fold)
            test_idx = np.flatnonzero(dataset.fold == fold)
            if not len(train_idx) or not len(test_idx):
                raise ValueError("Empty or overlapping source-disjoint TRAIN fold")
            train_y = dataset.next_observation[train_idx, :dataset.target_width] / 8.0
            test_y = dataset.next_observation[test_idx, :dataset.target_width] / 8.0
            current = dataset.observation[test_idx, :dataset.target_width] / 8.0
            for seed in TRAIN_SEEDS:
                for arm in ARMS:
                    # Same split, weight seed, number of passes and minibatch order per arm.
                    train_x = _features(dataset, train_idx, arm,
                                        np.random.default_rng(9000 + seed + fold))
                    test_x = _features(dataset, test_idx, arm,
                                       np.random.default_rng(19000 + seed + fold))
                    pred = _fit_predict(train_x, train_y, test_x, seed=seed + 100 * fold)
                    rows.append({"dataset": dataset.name, "fold": fold,
                                 "model_seed": seed, "arm": arm,
                                 "train_rows": len(train_idx), "test_rows": len(test_idx),
                                 **_metrics(pred, test_y, current)})
                rows.append({"dataset": dataset.name, "fold": fold,
                             "model_seed": seed, "arm": "persistence_untrained",
                             "train_rows": len(train_idx), "test_rows": len(test_idx),
                             **_metrics(current, test_y, current)})
    return {"study": "G2-E5-P6", "status": "train_only_frozen_crossfit",
            "source": "SHA-verified E1/E2 TRAIN fields only", "train_model_fits": 64,
            "trained_policy_updates": 0, "native_actions": 0, "native_resets": 0,
            "teacher_queries": 0, "original_test_reads": 0,
            "no_learned_transfer_claim": True, "records": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-zip", type=Path, required=True)
    parser.add_argument("--e2-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.e1_zip, args.e2_zip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"study": result["study"], "status": result["status"],
                      "model_fits": result["train_model_fits"], "native_actions": 0}))


if __name__ == "__main__":
    main()
