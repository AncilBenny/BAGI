"""Retrospective G2-E5-P1 TRAIN-only native public next-view prediction audit.

Reuses P4 numeric TRAIN traces, not P4 TEST or any simulator. The exact-view
conditional-frequency estimator is a diagnostic, not a learned control policy.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SOURCE_SHA256 = "15b8976859385c7f620364ca50c42a7d952ec85d26f9ab1faf537b19dbc28759"
SEEDS = (17, 29)
SCALES = ("raw", "scaled")
NATIVE_TRAIN_ACTIONS = 16384
VIEW_WIDTH = 147
ACTION_COUNT = 7
FOLDS = 5
SHUFFLE_SEED = 20261026
TRAIN_FIELDS = ("public_codes", "actions", "episode_index", "terminated", "truncated")


def verify_registration() -> None:
    if (
        SEEDS != (17, 29)
        or SCALES != ("raw", "scaled")
        or NATIVE_TRAIN_ACTIONS != 16384
        or VIEW_WIDTH != 147
        or ACTION_COUNT != 7
        or FOLDS != 5
        or SHUFFLE_SEED != 20261026
    ):
        raise ValueError("Frozen TRAIN-only diagnostic registration changed")


def read_train_only(source: Path) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    """Verify the original archive and read ONLY whitelisted TRAIN numeric arrays."""
    verify_registration()
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Source P4 ZIP SHA256 mismatch")
    out = {}
    with zipfile.ZipFile(source) as archive:
        names = set(archive.namelist())
        if any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("Unsafe archive member")
        required = {f"train_seed{seed}_{scale}.npz" for seed in SEEDS for scale in SCALES}
        if not required.issubset(names) or "summary.json" not in names:
            raise ValueError("Missing registered TRAIN source")
        summary = json.loads(archive.read("summary.json"))
        if summary.get("study") != "G2-E4-P4" or summary.get("status") != "complete":
            raise ValueError("Wrong or incomplete source study")
        digests = {row["name"]: row["sha256"] for row in summary["numeric_artifacts"]}
        for seed in SEEDS:
            for scale in SCALES:
                name = f"train_seed{seed}_{scale}.npz"
                raw = archive.read(name)
                if hashlib.sha256(raw).hexdigest() != digests.get(name):
                    raise ValueError(f"Inner TRAIN NPZ checksum mismatch: {name}")
                with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                    # Never access a TEST array, even if present in the same ZIP.
                    out[(seed, scale)] = {field: data[field].copy() for field in TRAIN_FIELDS}
    return out


def transitions(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """Pair only adjacent, nonterminal TRAIN actions within the same episode."""
    view = arrays["public_codes"]
    action = arrays["actions"]
    episode = arrays["episode_index"]
    term = arrays["terminated"]
    trunc = arrays["truncated"]
    if (
        view.shape != (NATIVE_TRAIN_ACTIONS, VIEW_WIDTH)
        or view.dtype.kind != "f"
        or not np.isfinite(view).all()
        or action.shape != (NATIVE_TRAIN_ACTIONS,)
        or action.dtype.kind not in "iu"
        or np.any(action >= ACTION_COUNT)
        or np.any(action < 0)
        or episode.shape != action.shape
        or episode.dtype.kind not in "iu"
        or episode[0] != 0
        or np.any(np.diff(episode) < 0)
        or np.any(np.diff(episode) > 1)
        or term.shape != action.shape
        or trunc.shape != action.shape
        or not np.isin(term, (0, 1)).all()
        or not np.isin(trunc, (0, 1)).all()
    ):
        raise ValueError("Malformed registered TRAIN public-action ledger")
    # Every completed native episode must be followed by a new episode index;
    # the final partial episode may end without a terminal flag.
    boundaries = episode[:-1] != episode[1:]
    if np.any((term[:-1].astype(bool) | trunc[:-1].astype(bool)) != boundaries):
        raise ValueError("Terminal flags disagree with TRAIN episode boundaries")
    valid = np.flatnonzero(~boundaries)
    if np.any(term[valid]) or np.any(trunc[valid]):
        raise ValueError("Attempt to predict across terminal reset")
    return view[valid], action[valid], view[valid + 1], episode[valid]


def _modal(counter: Counter[bytes], fallback: bytes) -> bytes:
    if not counter:
        return fallback
    # Stable tie-break by target bytes; no evaluation-label information.
    return min(counter, key=lambda value: (-counter[value], value))


def crossfit(
    view: np.ndarray, action: np.ndarray, next_view: np.ndarray, episode: np.ndarray
) -> dict:
    """Episode-disjoint TRAIN folds; no layout-disjoint claim is possible."""
    n = len(action)
    if (
        not n
        or view.shape != next_view.shape
        or view.shape != (n, VIEW_WIDTH)
        or episode.shape != (n,)
        or len(np.unique(episode % FOLDS)) != FOLDS
    ):
        raise ValueError("Need registered nonempty five-fold TRAIN transition arrays")
    current = [np.ascontiguousarray(row).tobytes() for row in view]
    target = [np.ascontiguousarray(row).tobytes() for row in next_view]
    rng = np.random.default_rng(SHUFFLE_SEED)
    shuffled = [current[i] for i in rng.permutation(n)]
    changed = np.fromiter((current[i] != target[i] for i in range(n)), dtype=np.int8)
    conditions = ("action_only", "public_view_action", "shuffled_public_view_action")
    errors = {name: {"correct": 0, "mismatched_cells": 0, "brier_sum": 0.0,
                     "covered": 0} for name in conditions}
    for fold in range(FOLDS):
        train = np.flatnonzero(episode % FOLDS != fold)
        held = np.flatnonzero(episode % FOLDS == fold)
        if not len(train) or not len(held):
            raise ValueError("Empty TRAIN fold")
        counts: dict[str, dict[object, Counter[bytes]]] = {
            name: defaultdict(Counter) for name in conditions
        }
        changed_counts: dict[str, Counter[object]] = {
            name: Counter() for name in conditions
        }
        for j in train:
            j = int(j)
            keys = (
                int(action[j]),
                (current[j], int(action[j])),
                (shuffled[j], int(action[j])),
            )
            for name, key in zip(conditions, keys, strict=True):
                counts[name][key][target[j]] += 1
                changed_counts[name][key] += int(changed[j])
        global_default = _modal(Counter(target[j] for j in train), target[int(train[0])])
        for j in held:
            j = int(j)
            baseline = int(action[j])
            keys = (baseline, (current[j], baseline), (shuffled[j], baseline))
            for name, key in zip(conditions, keys, strict=True):
                table = counts[name]
                used_key = key if key in table else baseline
                used_table = table if key in table else counts["action_only"]
                pred = _modal(used_table[used_key], global_default)
                total = sum(used_table[used_key].values())
                positives = (changed_counts[name][key] if key in table
                             else changed_counts["action_only"][baseline])
                prob = (positives + 1) / (total + 2)
                errors[name]["correct"] += int(pred == target[j])
                errors[name]["mismatched_cells"] += int(np.count_nonzero(
                    np.frombuffer(pred, dtype=view.dtype) != next_view[j]
                ))
                errors[name]["brier_sum"] += (prob - int(changed[j])) ** 2
                errors[name]["covered"] += int(key in table)
    return {
        "train_only_rows": n,
        "train_episodes": len(np.unique(episode)),
        "distinct_public_views": len(set(current)),
        "distinct_public_next_views": len(set(target)),
        "distinct_public_view_action_pairs": len({
            (current[j], int(action[j])) for j in range(n)
        }),
        "public_change_positive_rows": int(changed.sum()),
        "fold_assignment": "episode_index_mod_5; TRAIN only; not layout-disjoint",
        "scores": {
            name: {
                "next_view_exact_accuracy": round(row["correct"] / n, 8),
                "next_view_cell_mismatch_rate": round(
                    row["mismatched_cells"] / (n * VIEW_WIDTH), 8
                ),
                "public_change_brier": round(row["brier_sum"] / n, 8),
                "exact_key_covered_rows": row["covered"],
            }
            for name, row in errors.items()
        },
    }



def cross_seed_train(
    source_view: np.ndarray, source_action: np.ndarray, source_next: np.ndarray,
    target_view: np.ndarray, target_action: np.ndarray, target_next: np.ndarray,
) -> dict:
    """Train on one model seed's TRAIN resets, score other seed's TRAIN resets.

    The two reset seed *ranges* are disjoint, but finite MiniGrid physical
    layouts can repeat. This is not heldout TEST or physical-layout transfer.
    """
    if (
        source_view.ndim != 2 or source_view.shape[1] != VIEW_WIDTH
        or target_view.ndim != 2 or target_view.shape[1] != VIEW_WIDTH
        or source_next.shape != source_view.shape
        or target_next.shape != target_view.shape
        or source_action.shape != (len(source_view),)
        or target_action.shape != (len(target_view),)
        or not len(source_view) or not len(target_view)
    ):
        raise ValueError("Malformed cross-seed TRAIN transition arrays")
    by_action: dict[int, Counter[bytes]] = defaultdict(Counter)
    by_view_action: dict[tuple[bytes, int], Counter[bytes]] = defaultdict(Counter)
    for view, action, next_view in zip(
        source_view, source_action, source_next, strict=True
    ):
        target = np.ascontiguousarray(next_view).tobytes()
        key = (np.ascontiguousarray(view).tobytes(), int(action))
        by_action[int(action)][target] += 1
        by_view_action[key][target] += 1
    global_default = _modal(
        Counter(np.ascontiguousarray(row).tobytes() for row in source_next),
        np.ascontiguousarray(source_next[0]).tobytes(),
    )
    count = len(target_view)
    correct_action = correct_view_action = covered = mismatched_cells = 0
    for view, action, next_view in zip(
        target_view, target_action, target_next, strict=True
    ):
        key = (np.ascontiguousarray(view).tobytes(), int(action))
        truth = np.ascontiguousarray(next_view).tobytes()
        baseline = _modal(by_action[int(action)], global_default)
        predicted = _modal(by_view_action.get(key, Counter()), baseline)
        correct_action += int(baseline == truth)
        correct_view_action += int(predicted == truth)
        covered += int(key in by_view_action)
        mismatched_cells += int(np.count_nonzero(
            np.frombuffer(predicted, dtype=target_view.dtype) != next_view
        ))
    return {
        "target_train_rows": count,
        "source_train_rows": len(source_view),
        "source_distinct_public_view_action_pairs": len(by_view_action),
        "target_exact_key_covered_rows": covered,
        "action_only_next_view_exact_accuracy": round(correct_action / count, 8),
        "view_action_next_view_exact_accuracy": round(correct_view_action / count, 8),
        "view_action_cell_mismatch_rate": round(
            mismatched_cells / (count * VIEW_WIDTH), 8
        ),
        "qualification": (
            "Different TRAIN reset-seed ranges, same 6x6 environment; "
            "finite layouts/views can recur; no heldout TEST or new domain"
        ),
    }

def audit(source: Path) -> dict:
    datasets = read_train_only(source)
    result = {
        "study": "G2-E5-P1",
        "status": "exploratory_train_only_retrospective",
        "source_study": "G2-E4-P4",
        "source_zip_sha256": SOURCE_SHA256,
        "new_native_actions": 0,
        "new_policy_updates": 0,
        "teacher_queries": 0,
        "test_arrays_accessed": False,
        "claim_limit": (
            "TRAIN episode-disjoint predictive lookup on repeated MiniGrid views; "
            "not heldout-layout generalization, online control, physical effects, "
            "Workshop competence, cross-domain transfer or a new algorithm"
        ),
        "datasets": [],
        "cross_model_seed_train_only": [],
    }
    transitions_by_condition = {}
    for seed in SEEDS:
        for scale in SCALES:
            view, action, next_view, episode = transitions(datasets[(seed, scale)])
            transitions_by_condition[(seed, scale)] = (view, action, next_view)
            result["datasets"].append({
                "model_seed": seed,
                "input_scale": scale,
                "original_native_train_actions": NATIVE_TRAIN_ACTIONS,
                "terminal_or_partial_end_rows_excluded": NATIVE_TRAIN_ACTIONS - len(action),
                **crossfit(view, action, next_view, episode),
            })
    for scale in SCALES:
        for source_seed, target_seed in ((17, 29), (29, 17)):
            source_view, source_action, source_next = transitions_by_condition[
                (source_seed, scale)
            ]
            target_view, target_action, target_next = transitions_by_condition[
                (target_seed, scale)
            ]
            result["cross_model_seed_train_only"].append({
                "source_model_seed": source_seed,
                "target_model_seed": target_seed,
                "input_scale": scale,
                "source_train_reset_seed_start": 2_000_000 if source_seed == 17 else 2_100_000,
                "target_train_reset_seed_start": 2_000_000 if target_seed == 17 else 2_100_000,
                **cross_seed_train(
                    source_view, source_action, source_next,
                    target_view, target_action, target_next,
                ),
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.source_zip)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["datasets"]:
        print(row["model_seed"], row["input_scale"], row["scores"])


if __name__ == "__main__":
    main()
