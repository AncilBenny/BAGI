"""P8 prospective TRAIN-only public-visible-change probability comparison.

No native simulator, evaluator, original E2 mixed archive, reward or policy
updates. The registered source archives and all evaluation rules are frozen.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bagi.experiments.g2_e5_e2_train_isolation import load_isolated_train
from bagi.experiments.g2_e5_p5_public_transition_identifiability import (
    _shift_within_episodes,
)
from bagi.experiments.g2_e5_train_history_audit import _safe_archive
from bagi.models.world_model import WorldModel

ARMS = ("action_only", "reactive", "shuffled_train_action")
BASELINES = ("laplace_per_action", "global_train_rate", "always_no_change")
SEEDS = (17, 29)
FOLDS = (0, 1)
EPOCHS = 12
BATCH = 128
HIDDEN = 192
LR = 0.0003


@dataclass(frozen=True)
class PublicTransitions:
    name: str
    observation: np.ndarray
    action: np.ndarray
    changed: np.ndarray
    episode: np.ndarray
    fold: np.ndarray
    n_actions: int
    source_group: np.ndarray


def _dataset(name: str, observation: np.ndarray, action: np.ndarray,
             next_observation: np.ndarray, episode: np.ndarray,
             fold: np.ndarray, n_actions: int,
             source_group: np.ndarray | None = None) -> PublicTransitions:
    observation = np.asarray(observation)
    next_observation = np.asarray(next_observation)
    action = np.asarray(action)
    episode = np.asarray(episode)
    fold = np.asarray(fold)
    source_group = np.asarray(episode if source_group is None else source_group)
    if (observation.ndim != 2 or not len(observation)
            or observation.shape != next_observation.shape
            or action.shape != (len(observation),)
            or episode.shape != action.shape or fold.shape != action.shape
            or source_group.shape != action.shape
            or observation.dtype.kind not in "biuf"
            or next_observation.dtype.kind not in "biuf"
            or action.dtype.kind not in "iu" or episode.dtype.kind not in "iu"
            or fold.dtype.kind not in "iu" or source_group.dtype.kind not in "iu"
            or not np.isfinite(observation).all()
            or not np.isfinite(next_observation).all()
            or np.any(observation < 0) or np.any(observation > 8)
            or np.any(next_observation < 0) or np.any(next_observation > 8)
            or np.any(action < 0) or np.any(action >= n_actions)
            or set(np.unique(fold)) != {0, 1}):
        raise ValueError("Invalid public TRAIN transition contract")
    # The split unit is the entire source layout or native episode.
    for group in np.unique(episode):
        idx = np.flatnonzero(episode == group)
        if np.unique(fold[idx]).size != 1:
            raise ValueError("Source episode crosses TRAIN-only folds")
    for source in np.unique(source_group):
        if np.unique(fold[source_group == source]).size != 1:
            raise ValueError("Source layout crosses TRAIN-only folds")
    changed = np.any(observation != next_observation, axis=1).astype(np.float32)
    return PublicTransitions(name, observation.astype(np.float32),
                             action.astype(np.int64), changed,
                             episode.astype(np.int64), fold.astype(np.int8), n_actions,
                             source_group.astype(np.int64))


def _frozen_datasets(e1_zip: Path, e2_isolated_zip: Path) -> list[PublicTransitions]:
    """E1 reader materializes TRAIN-named keys; E2 reader only isolated actor0."""
    e1 = _safe_archive(e1_zip, "e1")
    e2 = load_isolated_train(e2_isolated_zip)
    datasets = []
    for domain, n_actions in (("xland", 6), ("workshop", 4)):
        p = f"{domain}_train_"
        obs = e1[p + "observations"]
        actions = e1[p + "actions"]
        episode = e1[p + "groups"]
        ids = e1[p + "episode_ids"]
        if domain == "workshop":
            # Exclude ten duplicate scripted fixtures, keep uniform random only.
            keep = e1[p + "behaviors"][episode] == "uniform_random"
            obs, actions, episode = obs[keep], actions[keep], episode[keep]
        cur, act, nxt, groups = _shift_within_episodes(obs, actions, episode)
        if domain == "xland":
            # Four executed actions already encoded in the last 28 legal input
            # features. P8 prospective task-neutral target ignores that suffix.
            cur, nxt = cur[:, :-28], nxt[:, :-28]
            layout = np.asarray([int(ids[int(g)].split("-")[1]) for g in groups])
            folds = layout % 2
        else:
            folds = np.asarray([int(ids[int(g)].split("-")[-1]) % 2 for g in groups])
        datasets.append(_dataset(f"e1_{domain}_train", cur, act, nxt,
                                 groups, folds, n_actions,
                                 source_group=layout if domain == "xland" else groups))
    for i, seed in enumerate(SEEDS):
        selected = e2["seed_index"] == i
        obs = e2["observation"][selected].reshape(-1, 25)
        nxt = e2["next_observation"][selected].reshape(-1, 25)
        episodes = e2["episode_index"][selected]
        datasets.append(_dataset(f"e2_workshop_dqn_train_seed{seed}", obs,
                                 e2["action"][selected], nxt, episodes,
                                 episodes % 2, 4))
    if [len(d.action) for d in datasets] != [3064, 190, 2048, 2048]:
        raise ValueError("Registered native TRAIN transition counts changed")
    return datasets


def _fit(data: PublicTransitions, train_idx: np.ndarray, test_idx: np.ndarray,
         arm: str, seed: int, fold: int) -> tuple[np.ndarray, int]:
    if arm not in ARMS or not len(train_idx) or not len(test_idx):
        raise ValueError("Unregistered P8 arm or empty fold")
    torch.manual_seed(seed + 100 * fold)
    cells = data.observation.shape[1]
    model = WorldModel(observation_shape=(1, cells), actions=data.n_actions,
                       cell_values=9, hidden=HIDDEN, event_count=1).cpu()
    # Update ONLY the existing WorldModel backbone and event head; the next-view
    # and reward heads have no optimizer parameters and no training targets.
    parameters = list(model.backbone.parameters()) + list(model.event_head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=LR)
    obs = data.observation[train_idx].copy()
    if arm == "action_only":
        obs[:] = 0
    actions = data.action[train_idx].copy()
    if arm == "shuffled_train_action":
        actions = np.random.default_rng(9000 + seed + fold).permutation(actions)
    x = torch.from_numpy(obs.reshape(-1, 1, cells))
    a = torch.from_numpy(actions)
    y = torch.from_numpy(data.changed[train_idx].reshape(-1, 1))
    rng = np.random.default_rng(seed + 1000)
    steps = 0
    model.train()
    for _ in range(EPOCHS):
        for idx in np.array_split(rng.permutation(len(x)),
                                  (len(x) + BATCH - 1) // BATCH):
            optimizer.zero_grad(set_to_none=True)
            logits = model.event_head(model._features(x[idx], a[idx]))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y[idx])
            loss.backward()
            optimizer.step()
            steps += 1
    test_obs = data.observation[test_idx].copy()
    if arm == "action_only":
        test_obs[:] = 0
    model.eval()
    with torch.no_grad():
        scores = model.event_head(model._features(
            torch.from_numpy(test_obs.reshape(-1, 1, cells)),
            torch.from_numpy(data.action[test_idx]),
        )).sigmoid().squeeze(1).numpy()
    return scores, steps


def _baseline(data: PublicTransitions, train_idx: np.ndarray,
              test_idx: np.ndarray, name: str) -> np.ndarray:
    actions = data.action[train_idx]
    y = data.changed[train_idx]
    if name == "laplace_per_action":
        counts = np.bincount(actions, minlength=data.n_actions)
        positives = np.bincount(actions, weights=y, minlength=data.n_actions)
        rates = (positives + 1) / (counts + 2)
        return rates[data.action[test_idx]]
    if name == "global_train_rate":
        return np.full(len(test_idx), y.mean(), dtype=np.float64)
    if name == "always_no_change":
        return np.zeros(len(test_idx), dtype=np.float64)
    raise ValueError("Unregistered P8 baseline")


def _metrics(prob: np.ndarray, label: np.ndarray, action: np.ndarray,
             episode: np.ndarray, n_actions: int) -> dict:
    if (prob.shape != label.shape or prob.shape != action.shape
            or prob.shape != episode.shape or not len(prob)
            or not np.isfinite(prob).all()
            or np.any(prob < 0) or np.any(prob > 1)):
        raise ValueError("Invalid held TRAIN-fold probabilities")
    square = (prob - label) ** 2
    safe = np.clip(prob, 1e-7, 1 - 1e-7)
    logloss = -(label * np.log(safe) + (1 - label) * np.log(1 - safe))
    bins = np.minimum((prob * 10).astype(np.int64), 9)
    ece = 0.0
    for b in range(10):
        keep = bins == b
        if keep.any():
            ece += float(keep.mean() * abs(prob[keep].mean() - label[keep].mean()))
    per_action = {}
    for a in range(n_actions):
        keep = action == a
        per_action[str(a)] = {"rows": int(keep.sum()),
                              "event_rate": float(label[keep].mean()) if keep.any() else None,
                              "brier": float(square[keep].mean()) if keep.any() else None}
    group_scores = [float(square[episode == g].mean()) for g in np.unique(episode)]
    return {"brier": float(square.mean()), "log_loss": float(logloss.mean()),
            "ece10": float(ece), "n_rows": len(prob), "positive_rows": int(label.sum()),
            "per_action": per_action, "n_held_groups": len(group_scores),
            "held_group_brier_std": float(np.std(group_scores)),
            "held_group_brier_min": min(group_scores),
            "held_group_brier_max": max(group_scores)}


def run(e1_zip: Path, e2_isolated_zip: Path) -> dict:
    torch.set_num_threads(1)
    rows = []
    training_steps = 0
    for data in _frozen_datasets(e1_zip, e2_isolated_zip):
        for fold in FOLDS:
            train_idx = np.flatnonzero(data.fold != fold)
            held_idx = np.flatnonzero(data.fold == fold)
            if not len(train_idx) or not len(held_idx):
                raise ValueError("Empty public TRAIN fold")
            for seed in SEEDS:
                for arm in (*ARMS, *BASELINES):
                    if arm in ARMS:
                        prob, steps = _fit(data, train_idx, held_idx, arm, seed, fold)
                        training_steps += steps
                    else:
                        prob = _baseline(data, train_idx, held_idx, arm)
                    row = {"dataset": data.name, "fold": fold, "model_seed": seed,
                           "arm": arm, "train_rows": len(train_idx),
                           **_metrics(prob, data.changed[held_idx],
                                      data.action[held_idx], data.source_group[held_idx],
                                      data.n_actions)}
                    rows.append(row)
    if len(rows) != 96:
        raise ValueError("P8 result group/arm accounting mismatch")
    return {"study": "G2-E5-P8", "status": "archived_train_only_crossfit",
            "registered_fits": 48, "completed_fits": 48,
            "supervised_optimizer_steps": training_steps,
            "model_hyperparameters": {"epochs": EPOCHS, "batch": BATCH,
                                       "hidden": HIDDEN, "learning_rate": LR},
            "native_actions": 0, "native_resets": 0, "teacher_queries": 0,
            "original_e2_eval_array_reads": 0, "policy_updates": 0,
            "records": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-zip", type=Path, required=True)
    parser.add_argument("--e2-isolated-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.e1_zip, args.e2_isolated_zip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"study": result["study"], "status": result["status"],
                      "fits": result["completed_fits"], "native_actions": 0}))


if __name__ == "__main__":
    main()
