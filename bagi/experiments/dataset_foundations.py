"""Gate A: audit duplicate Workshop data, overlap, and partial-observation aliasing.

No models are trained. Privileged logger info is used for diagnosis, never
passed to a learned model or a controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

EVENTS = ("switch_activated", "key_picked_up", "lock_opened", "goal_reached")


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def model_input(row: dict) -> str:
    return canonical([row["observation"], int(row["action"])])


def target(row: dict) -> str:
    observed = set(row.get("info", {}).get("events", []))
    return canonical([row["next_observation"], float(row["reward"]),
                      [name in observed for name in EVENTS]])


def supervised_transition(row: dict) -> str:
    return canonical([model_input(row), target(row)])


def signature(rows: list[dict]) -> str:
    value = canonical([supervised_transition(row) for row in rows])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def overlap(rows: list[dict], train: list[int], valid: list[int]) -> dict:
    seen_exact = {supervised_transition(rows[i]) for i in train}
    seen_input = {model_input(rows[i]) for i in train}
    return {
        "train_transitions": len(train),
        "validation_transitions": len(valid),
        "validation_exact_duplicate_count": sum(
            supervised_transition(rows[i]) in seen_exact for i in valid
        ),
        "validation_exact_duplicate_fraction": sum(
            supervised_transition(rows[i]) in seen_exact for i in valid
        ) / len(valid),
        "validation_input_seen_fraction": sum(
            model_input(rows[i]) in seen_input for i in valid
        ) / len(valid),
        "validation_event_positive_counts": {
            name: sum(name in rows[i].get("info", {}).get("events", [])
                      for i in valid) for name in EVENTS
        },
    }


def splits(rows: list[dict], seed: int, fraction: float) -> dict:
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(rows) < 2:
        raise ValueError("Need at least two transitions")
    indices = list(range(len(rows)))
    # Identical to train_world_model.py: random transition split.
    random.Random(seed).shuffle(indices)
    cutoff = max(1, int(len(indices) * (1.0 - fraction)))
    actual = overlap(rows, indices[:cutoff], indices[cutoff:] or indices[-1:])

    by_episode: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_episode[int(row["episode"])].append(i)
    episode_ids = sorted(by_episode)
    if len(episode_ids) < 2:
        raise ValueError("Need at least two episodes for holdout")
    shuffled = episode_ids[:]
    random.Random(seed).shuffle(shuffled)
    n_holdout = min(len(shuffled) - 1, max(1, round(len(shuffled) * fraction)))
    held = set(shuffled[:n_holdout])
    episode = overlap(
        rows,
        [i for ep in episode_ids if ep not in held for i in by_episode[ep]],
        [i for ep in episode_ids if ep in held for i in by_episode[ep]],
    )

    by_trajectory: dict[str, list[int]] = defaultdict(list)
    for ep in episode_ids:
        sig = signature([rows[i] for i in by_episode[ep]])
        by_trajectory[sig].extend(by_episode[ep])
    unique = sorted(by_trajectory)
    grouped = None
    if len(unique) > 1:
        random.Random(seed).shuffle(unique)
        n_groups = min(len(unique) - 1, max(1, round(len(unique) * fraction)))
        held_sigs = set(unique[:n_groups])
        grouped = overlap(
            rows,
            [i for sig in unique if sig not in held_sigs for i in by_trajectory[sig]],
            [i for sig in unique if sig in held_sigs for i in by_trajectory[sig]],
        )
        grouped["training_trajectory_groups"] = len(unique) - n_groups
        grouped["validation_trajectory_groups"] = n_groups
    return {
        "actual_transition_random_split": actual,
        "episode_disjoint_counterfactual": episode,
        "trajectory_disjoint_counterfactual": grouped,
        "unique_episode_signatures": len(unique),
        "episode_count": len(episode_ids),
    }


def aliasing(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[model_input(row)].append(row)
    repeated = {k: v for k, v in groups.items() if len(v) > 1}
    different_targets = {
        k for k, v in repeated.items() if len({target(row) for row in v}) > 1
    }
    hidden_flags = {
        k for k, v in repeated.items()
        if len({
            (bool(row.get("info", {}).get("has_key", False)),
             bool(row.get("info", {}).get("door_open", False)))
            for row in v
        }) > 1
    }
    return {
        "unique_observation_action_pairs": len(groups),
        "repeated_observation_action_pairs": len(repeated),
        "ambiguous_outcome_pairs": len(different_targets),
        "mixed_hidden_flag_pairs": len(hidden_flags),
        "ambiguous_with_mixed_hidden_flags": len(different_targets & hidden_flags),
        "ambiguous_repeated_pair_fraction": (
            len(different_targets) / len(repeated) if repeated else 0
        ),
        "ambiguous_transition_rows": sum(len(repeated[k]) for k in different_targets),
        "example_pairs": [
            {"count": len(repeated[k]),
             "outcomes": len({target(row) for row in repeated[k]})}
            for k in sorted(different_targets)[:5]
        ],
    }


def audit(rows: list[dict], seed: int, fraction: float, guided: int) -> dict:
    if not rows:
        raise ValueError("No logged transitions to audit")
    episodes: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        episodes[int(row["episode"])].append(row)
    if any(i not in episodes for i in range(guided)):
        raise ValueError("Missing guided episode IDs")
    guided_signatures = [signature(episodes[i]) for i in range(guided)]
    exact_counts = Counter(supervised_transition(row) for row in rows)
    return {
        "transitions": len(rows),
        "episodes": len(episodes),
        "guided_episodes": guided,
        "unique_guided_episode_signatures": len(set(guided_signatures)),
        "duplicate_guided_episodes": guided - len(set(guided_signatures)),
        "unique_supervised_transitions": len(exact_counts),
        "rows_in_duplicated_supervised_transitions": sum(
            n for n in exact_counts.values() if n > 1
        ),
        "event_positive_counts": {
            name: sum(name in row.get("info", {}).get("events", []) for row in rows)
            for name in EVENTS
        },
        "splits": splits(rows, seed, fraction),
        "aliasing": aliasing(rows),
        "limits": [
            "Exact overlap identifies validation leakage risk, not measured model error.",
            "Episode-disjoint validation can still share identical guided trajectories.",
            "Hidden flags from logger metadata are used only for diagnostics.",
            "Environment geometry is fixed; distinct reset seeds do not imply transfer.",
        ],
    }


def run(config_path: str) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    dataset = Path(cfg["transitions"])
    if not dataset.is_file():
        raise FileNotFoundError(f"Missing training dataset: {dataset}")
    with dataset.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    result = audit(rows, int(cfg["training_seed"]),
                   float(cfg["validation_fraction"]), int(cfg["guided_episodes"]))
    marker = cfg.get("reconstruction_marker")
    result["dataset_origin"] = (
        "reconstructed_same_policy_and_seeds"
        if marker and Path(marker).exists() else "restored_checkpoint_cache"
    )
    result["dataset"] = str(dataset)
    path = Path(cfg["output_dir"]) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Gate A: {result['transitions']} transitions; "
          f"{result['duplicate_guided_episodes']}/{result['guided_episodes']} "
          "duplicate guided episodes")
    print("Actual validation exact-overlap fraction:",
          result["splits"]["actual_transition_random_split"][
              "validation_exact_duplicate_fraction"])
    print("Outcome-ambiguous observation/action pairs:",
          result["aliasing"]["ambiguous_outcome_pairs"])
    print(f"Saved audit to {path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset_foundations.yaml")
    run(parser.parse_args().config)
