"""P10a frozen E1 TRAIN-only terminal-label selection; NO model fitting.

Pure public-data preflight for a prospective matched-size contrast. Never
imports a simulator, loads original TEST arrays or trains a policy/model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TerminalDataset:
    name: str
    observation: np.ndarray
    action: np.ndarray
    changed: np.ndarray
    episode: np.ndarray
    fold: np.ndarray
    source_group: np.ndarray
    tail: np.ndarray
    early_terminal: np.ndarray
    original_row: np.ndarray
    episode_id: np.ndarray
    n_actions: int


def _build_dataset(archive: dict, domain: str, *, strict: bool = True) -> TerminalDataset:
    """Verify original direct labels against every available interior next view."""
    if domain not in ("xland", "workshop"):
        raise ValueError("Unregistered P10a domain")
    prefix = f"{domain}_train_"
    observation = np.asarray(archive[prefix + "observations"])
    action = np.asarray(archive[prefix + "actions"])
    changed = np.asarray(archive[prefix + "labels"])
    episode = np.asarray(archive[prefix + "groups"])
    ids = np.asarray(archive[prefix + "episode_ids"])
    behaviors = np.asarray(archive[prefix + "behaviors"])
    original_row = np.arange(len(action), dtype=np.int64)
    if (observation.ndim != 2 or action.ndim != 1 or not len(action)
            or observation.shape[0] != len(action)
            or changed.shape != action.shape or episode.shape != action.shape
            or action.dtype.kind not in "iu" or episode.dtype.kind not in "iu"
            or changed.dtype.kind not in "biuf" or not np.isfinite(changed).all()
            or not np.isin(changed, (0, 1)).all()
            or ids.ndim != 1 or behaviors.ndim != 1
            or ids.dtype.kind not in "US" or behaviors.dtype.kind not in "US"
            or np.any(episode < 0) or np.any(episode >= len(ids))
            or np.any(episode >= len(behaviors))):
        raise ValueError("Malformed E1 TRAIN-only public arrays")
    if domain == "workshop":
        keep = behaviors[episode] == "uniform_random"
        observation, action, changed, episode, original_row = (
            array[keep] for array in (observation, action, changed, episode, original_row)
        )
    n_actions, width = (6, 115) if domain == "xland" else (4, 25)
    if (observation.shape != (len(action), width) or not len(action)
            or observation.dtype.kind not in "biuf"
            or not np.isfinite(observation).all()
            or np.any(observation < 0) or np.any(observation > 8)
            or np.any(action >= n_actions) or np.any(action < 0)
            or np.any(np.diff(episode) < 0)
            or len(np.unique(episode)) != np.count_nonzero(np.r_[True, episode[1:] != episode[:-1]])):
        raise ValueError("Invalid public features, action support or episode order")
    # Exactly the P8 public-effect target: exclude encoded executed-action history.
    public = observation[:, :-28] if domain == "xland" else observation
    tail = np.r_[episode[:-1] != episode[1:], True]
    interior = np.flatnonzero(~tail)
    direct = changed.astype(np.int8)
    inferred = np.any(public[interior] != public[interior + 1], axis=1)
    if not np.array_equal(direct[interior], inferred.astype(np.int8)):
        raise ValueError("E1 direct public labels disagree with next-view shift")
    episode_id = ids[episode]
    if domain == "xland":
        tokens = [str(value).split("-") for value in episode_id]
        if any(len(t) != 5 or t[0] != "xland" or not t[1].isdigit()
               for t in tokens):
            raise ValueError("Malformed XLand source-layout identity")
        source_group = np.asarray([int(t[1]) for t in tokens], dtype=np.int64)
        fold = source_group % 2
    else:
        tokens = [str(value).split("-") for value in episode_id]
        if any(len(t) != 4 or t[:3] != ["workshop", "default", "uniform_random"]
               or not t[3].isdigit() for t in tokens):
            raise ValueError("Malformed random Workshop episode identity")
        source_group = episode.copy()
        fold = np.asarray([int(t[3]) % 2 for t in tokens], dtype=np.int8)
    for group in np.unique(source_group):
        if np.unique(fold[source_group == group]).size != 1:
            raise ValueError("Source group crosses held folds")
    if set(np.unique(fold)) != {0, 1}:
        raise ValueError("Both independent held folds required")
    early_terminal = np.zeros(len(action), dtype=np.bool_)
    if domain == "xland":
        for group in np.unique(episode):
            idx = np.flatnonzero(episode == group)
            if len(idx) < 20:
                early_terminal[idx[-1]] = True
    if strict:
        if domain == "xland":
            if (len(action), int(tail.sum()), len(np.unique(episode)),
                    len(np.unique(source_group)), int(direct.sum()),
                    int(early_terminal.sum())) != (3244, 180, 180, 15, 2019, 30):
                raise ValueError("Registered XLand E1 TRAIN counts changed")
            if set(np.unique(source_group)) != set(range(11900, 11915)):
                raise ValueError("Registered XLand TRAIN layouts changed")
            if int(((action == 4) & (direct == 1)).sum()) != 4 or np.any(
                    (action == 4) & (direct == 1) & ~tail):
                raise ValueError("Registered XLand action-4 label coverage changed")
        elif (len(action), int(tail.sum()), len(np.unique(episode))) != (200, 10, 10):
            raise ValueError("Registered random Workshop TRAIN counts changed")
    return TerminalDataset(f"e1_{domain}_train", public.astype(np.float32),
                           action.astype(np.int64), direct, episode.astype(np.int64),
                           fold.astype(np.int8), source_group.astype(np.int64),
                           tail, early_terminal, original_row, episode_id, n_actions)


def _selection(data: TerminalDataset, held_fold: int) -> dict:
    """Hash-ranked TRAIN-only action-matched replacement, independent of labels."""
    if held_fold not in (0, 1):
        raise ValueError("Invalid held fold")
    train = data.fold != held_fold
    shifted = np.flatnonzero(train & ~data.tail)
    tails = np.flatnonzero(train & data.tail)
    held = np.flatnonzero(~train)
    if not len(shifted) or not len(tails) or not len(held):
        raise ValueError("Empty P10a fold/contract")
    removed = []
    for action in range(data.n_actions):
        n = int((data.action[tails] == action).sum())
        eligible = [int(i) for i in shifted if data.action[i] == action]
        if len(eligible) < n:
            raise ValueError("Infeasible TRAIN-only action-stratified tail swap")

        def rank(i: int) -> tuple[str, int]:
            message = (f"P10a|{data.name}|{held_fold}|{action}|"
                       f"{data.episode_id[i]}|{int(data.original_row[i])}")
            return hashlib.sha256(message.encode("utf-8")).hexdigest(), int(data.original_row[i])

        removed.extend(sorted(eligible, key=rank)[:n])
    removed_array = np.asarray(sorted(removed), dtype=np.int64)
    swap = np.asarray(sorted((set(map(int, shifted)) - set(removed))
                             | set(map(int, tails))), dtype=np.int64)
    if (len(swap) != len(shifted) or len(removed) != len(tails)
            or np.intersect1d(swap, held).size
            or np.intersect1d(removed_array, held).size
            or not np.array_equal(np.bincount(data.action[shifted], minlength=data.n_actions),
                                  np.bincount(data.action[swap], minlength=data.n_actions))):
        raise ValueError("P10a matched-size/action/held-fold contract failed")
    digest = hashlib.sha256(json.dumps(
        data.original_row[removed_array].tolist(), separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return {"shifted": shifted, "tail_swap": swap, "held": held,
            "removed": removed_array, "added_tails": tails,
            "removed_source_index_sha256": digest}


def preflight(e1_zip: Path) -> dict:
    """Read SHA-pinned E1 TRAIN whitelist only; return numeric selection ledger."""
    from bagi.experiments.g2_e5_train_history_audit import _safe_archive

    archive = _safe_archive(e1_zip, "e1")
    results = []
    for domain in ("xland", "workshop"):
        data = _build_dataset(archive, domain)
        folds = []
        for held_fold in (0, 1):
            selection = _selection(data, held_fold)
            folds.append({"held_fold": held_fold,
                          "shifted_train_rows": len(selection["shifted"]),
                          "tail_swap_train_rows": len(selection["tail_swap"]),
                          "added_train_tails": len(selection["added_tails"]),
                          "removed_train_interiors": len(selection["removed"]),
                          "held_rows": len(selection["held"]),
                          "removed_source_index_sha256":
                              selection["removed_source_index_sha256"]})
        results.append({"dataset": data.name, "all_rows": len(data.action),
                        "interior_rows": int((~data.tail).sum()),
                        "tail_rows": int(data.tail.sum()),
                        "early_terminal_tails": int(data.early_terminal.sum()),
                        "source_groups": len(np.unique(data.source_group)),
                        "folds": folds})
    return {"study": "G2-E5-P10a", "status": "selection_preflight_only",
            "native_actions": 0, "native_resets": 0,
            "model_fits": 0, "optimizer_steps": 0,
            "original_test_array_reads": 0, "datasets": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = preflight(args.e1_zip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"study": result["study"], "status": result["status"],
                      "model_fits": 0, "native_actions": 0}))


if __name__ == "__main__":
    main()
