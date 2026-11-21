"""Read-only G2-E5-P0 TRAIN-only history coverage audit of archived public traces.

No simulator imports, network fitting, or access to original TEST arrays. Predicts
whether the next public observation changes, NOT physical effect or task success.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

SOURCE_SHA256 = {
    "e1": "aaf8eb011763bf47e1d196771e257eb13c4eb35294bbeed2b71757ade525f073",
    "e2": "b000a02a2eacfb8aa07befa151e0bbfb2c5bf7fd430f8bd896bd11f90f57436a",
}
SEED = 20261026
PAD = -1


def _safe_archive(path: Path, kind: str) -> dict[str, np.ndarray]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256[kind]:
        raise ValueError(f"Unregistered {kind} archive digest")
    filename = "public_effect_numeric.npz" if kind == "e1" else "public_policy_numeric.npz"
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != {"summary.json", filename}:
            raise ValueError("Unregistered archive members")
        with np.load(io.BytesIO(archive.read(filename)), allow_pickle=False) as data:
            # Explicit whitelist: no TEST keys may be accessed, even if present.
            if kind == "e1":
                fields = ("observations", "actions", "labels", "groups",
                          "episode_ids", "behaviors")
                names = [f"{domain}_train_{field}" for domain in ("xland", "workshop")
                         for field in fields]
            else:
                names = ["observation", "next_observation", "action", "episode_index",
                         "seed_index", "actor_index"]
            return {key: data[key].copy() for key in names}


def _validate(obs: np.ndarray, action: np.ndarray, labels: np.ndarray,
              episode: np.ndarray, n_actions: int) -> None:
    n = len(action)
    if (obs.ndim != 2 or obs.shape[0] != n or labels.shape != (n,)
            or episode.shape != (n,) or not n or not np.isfinite(obs).all()
            or obs.dtype.kind not in "biuf" or action.dtype.kind not in "iu"
            or not np.all((action >= 0) & (action < n_actions))
            or not np.isin(labels, (0, 1)).all()):
        raise ValueError("Malformed public TRAIN data")
    if any(np.flatnonzero(episode == e).size !=
           (np.flatnonzero(episode == e)[-1] - np.flatnonzero(episode == e)[0] + 1)
           for e in np.unique(episode)):
        raise ValueError("Episode rows not contiguous")


def _history(action: np.ndarray, episode: np.ndarray, width: int,
             *, shuffle: bool = False) -> np.ndarray:
    """Only previous ACTUALLY executed commands within the same episode."""
    result = np.full((len(action), width), PAD, dtype=np.int16)
    rng = np.random.default_rng(SEED)
    for e in np.unique(episode):
        indices = np.flatnonzero(episode == e)
        a = action[indices].copy()
        if shuffle:
            a = rng.permutation(a)
        for t, index in enumerate(indices):
            for lag in range(1, min(width, t) + 1):
                result[index, lag - 1] = int(a[t - lag])
    return result


def _keys(obs: np.ndarray, action: np.ndarray, history: np.ndarray) -> list[bytes]:
    return [np.ascontiguousarray(obs[i]).tobytes() + int(action[i]).to_bytes(2, "little")
            + history[i].tobytes() for i in range(len(action))]


def _crossfit(obs: np.ndarray, action: np.ndarray, labels: np.ndarray,
              episode: np.ndarray, fold: np.ndarray, n_actions: int,
              history_width: int, shuffle: bool = False) -> dict:
    """Group-disjoint TRAIN crossfit with Laplace action-frequency fallback."""
    _validate(obs, action, labels, episode, n_actions)
    history = _history(action, episode, history_width, shuffle=shuffle)
    keys = _keys(obs, action, history)
    probs = np.full(len(action), np.nan, dtype=np.float64)
    fallback = np.full(len(action), np.nan, dtype=np.float64)
    matched = np.zeros(len(action), dtype=np.bool_)
    for group in np.unique(fold):
        held = np.flatnonzero(fold == group)
        train = np.flatnonzero(fold != group)
        if not len(train) or not len(held):
            raise ValueError("Need >=2 disjoint TRAIN folds")
        rates = (np.bincount(action[train], weights=labels[train], minlength=n_actions) + 1) / (
            np.bincount(action[train], minlength=n_actions) + 2)
        sums = Counter()
        counts = Counter()
        for j in train:
            sums[keys[j]] += int(labels[j])
            counts[keys[j]] += 1
        for j in held:
            fallback[j] = rates[int(action[j])]
            if keys[j] in counts:
                probs[j] = (sums[keys[j]] + 1) / (counts[keys[j]] + 2)
                matched[j] = True
            else:
                probs[j] = fallback[j]
    if not np.isfinite(probs).all() or not np.isfinite(fallback).all():
        raise AssertionError("Unscored TRAIN rows")
    return {
        "brier": round(float(np.mean((probs - labels)**2)), 8),
        "action_only_brier": round(float(np.mean((fallback - labels)**2)), 8),
        "lookup_covered_rows": int(matched.sum()),
        "rows": len(action),
        "lookup_coverage": round(float(matched.mean()), 8),
        "brier_on_covered": (round(float(np.mean((probs[matched] - labels[matched])**2)), 8)
                             if matched.any() else None),
    }


def _audit_dataset(name: str, obs: np.ndarray, action: np.ndarray,
                   labels: np.ndarray, episode: np.ndarray, fold: np.ndarray,
                   n_actions: int) -> dict:
    results = {}
    for width in (0, 1, 2, 4):
        results[f"real_history_{width}"] = _crossfit(
            obs, action, labels, episode, fold, n_actions, width)
    results["shuffled_history_4"] = _crossfit(
        obs, action, labels, episode, fold, n_actions, 4, shuffle=True)
    return {
        "name": name, "train_actions": len(action),
        "train_episodes": len(np.unique(episode)),
        "crossfit_folds": len(np.unique(fold)),
        "public_change_positives": int(labels.sum()),
        "results": results,
    }


def audit(e1: Path, e2: Path) -> dict:
    a = _safe_archive(e1, "e1")
    b = _safe_archive(e2, "e2")
    output = {"study": "G2-E5-P0", "status": "train_only_crossfit",
              "new_native_actions": 0, "new_policy_training_actions": 0,
              "new_teacher_queries": 0, "source_zip_sha256": SOURCE_SHA256,
              "datasets": []}
    for domain, n_actions in (("xland", 6), ("workshop", 4)):
        p = f"{domain}_train_"
        obs = a[p + "observations"]
        act = a[p + "actions"]
        y = a[p + "labels"]
        episode = a[p + "groups"]
        ids = a[p + "episode_ids"]
        if len(np.unique(episode)) != len(ids):
            raise ValueError("Invalid archived episode indices")
        # XLand 15 TRAIN layouts, group ALL source task/policy episodes by layout;
        # Workshop only ONE TRAIN layout, so crossfit by episode, never transfer.
        folds = np.array([ids[int(e)].split("-")[1] if domain == "xland" else str(e)
                          for e in episode])
        output["datasets"].append(_audit_dataset(f"e1_{domain}_public",
            obs, act, y, episode, folds, n_actions))
        behaviors = a[p + "behaviors"][episode]
        if domain == "xland":
            # Remove only already encoded last-four executed-action bits. This
            # is a separate descriptive ablation, not a new legal actor input.
            output["datasets"].append(_audit_dataset("e1_xland_without_encoded_history",
                obs[:, :-28], act, y, episode, folds, n_actions))
            for behavior in ("frozen_expanded_bc", "uniform_random"):
                mask = behaviors == behavior
                output["datasets"].append(_audit_dataset(f"e1_xland_{behavior}",
                    obs[mask], act[mask], y[mask], episode[mask], folds[mask], n_actions))
        else:
            # The other ten Workshop episodes are one duplicated fixed script;
            # crossfit those together would grossly overstate independent support.
            mask = behaviors == "uniform_random"
            output["datasets"].append(_audit_dataset("e1_workshop_random_only",
                obs[mask], act[mask], y[mask], episode[mask], folds[mask], n_actions))
    for seed_index, seed in enumerate((17, 29)):
        mask = (b["actor_index"] == 0) & (b["seed_index"] == seed_index)
        obs = b["observation"][mask].reshape(-1, 25)
        nxt = b["next_observation"][mask].reshape(-1, 25)
        act = b["action"][mask]
        y = np.any(obs != nxt, axis=1).astype(np.int8)
        episode = b["episode_index"][mask]
        output["datasets"].append(_audit_dataset(f"e2_workshop_dqn_seed{seed}",
            obs, act, y, episode, episode, 4))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1", type=Path, required=True)
    parser.add_argument("--e2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.e1, args.e2)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for dataset in result["datasets"]:
        print(dataset["name"], {key: (value["brier"], value["lookup_covered_rows"])
             for key, value in dataset["results"].items()})


if __name__ == "__main__":
    main()
