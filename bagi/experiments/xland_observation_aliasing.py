"""REUSE-11: audit exact observed-action aliasing in the frozen REUSE-10 pool.

Only XLand's local image and direction are grouped. Skill labels are
diagnostic provenance, never learner features. No policies are trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from bagi.experiments.xland_existing_bc import TRAIN_TASKS, collect, validate_config


def observation_action_audit(observations, actions, tasks):
    """Empirical deterministic-lookup bound on the supplied exact expert rows."""
    obs = np.asarray(observations, dtype=np.float32)
    labels = np.asarray(actions, dtype=np.int64)
    task_names = list(tasks)
    if (obs.ndim != 2 or labels.ndim != 1 or len(labels) < 1
            or obs.shape[0] != len(labels) or len(task_names) != len(labels)
            or not np.isfinite(obs).all()):
        raise ValueError("Expected finite observation matrix and aligned nonempty labels")
    if set(task_names) - set(TRAIN_TASKS):
        raise ValueError("Only training-task provenance is permitted")
    if np.any((labels < 0) | (labels >= 6)):
        raise ValueError("Expected native XLand actions 0-5")

    # Hash exactly the same float32 features the BC model receives. Do not
    # include goal, task ID, grid, oracle position, or simulator state.
    groups = defaultdict(list)
    for i, row in enumerate(obs):
        groups[np.ascontiguousarray(row).tobytes()].append(i)

    ambiguous = []
    cross_task_conflicts = within_task_conflicts = task_resolved = 0
    max_correct = max_correct_given_task = 0
    per_action = {
        str(a): {"labeled_rows": int(np.count_nonzero(labels == a)),
                 "distinct_visible_observations": 0,
                 "rows_on_conflicting_observations": 0,
                 "observations_shared_with_other_actions": 0}
        for a in range(6)
    }
    unique_pairs = 0
    duplicates = sum(len(indices) > 1 for indices in groups.values())
    for key, indices in groups.items():
        counts = Counter(int(labels[i]) for i in indices)
        by_task = defaultdict(Counter)
        for i in indices:
            by_task[task_names[i]][int(labels[i])] += 1
        unique_pairs += len(counts)
        max_correct += max(counts.values())
        max_correct_given_task += sum(max(c.values()) for c in by_task.values())
        has_conflict = len(counts) > 1
        has_cross_task_conflict = has_conflict and len(by_task) > 1
        has_within_task_conflict = any(len(c) > 1 for c in by_task.values())
        cross_task_conflicts += int(has_cross_task_conflict)
        within_task_conflicts += int(has_within_task_conflict)
        task_resolved += int(has_conflict and not has_within_task_conflict)
        for action, frequency in counts.items():
            detail = per_action[str(action)]
            detail["distinct_visible_observations"] += 1
            if has_conflict:
                detail["rows_on_conflicting_observations"] += frequency
                detail["observations_shared_with_other_actions"] += 1
        if has_conflict:
            ambiguous.append({
                "visible_observation_sha256": hashlib.sha256(key).hexdigest(),
                "labeled_rows": len(indices),
                "action_counts": {str(a): n for a, n in sorted(counts.items())},
                "task_action_counts": {
                    task: {str(a): n for a, n in sorted(c.items())}
                    for task, c in sorted(by_task.items())
                },
                "cross_task": has_cross_task_conflict,
                "within_task": has_within_task_conflict,
            })

    ambiguous.sort(key=lambda g: (-g["labeled_rows"], g["visible_observation_sha256"]))
    total = len(labels)
    return {
        "labeled_rows": total,
        "observation_feature_width": int(obs.shape[1]),
        "distinct_visible_observations": len(groups),
        "distinct_visible_observation_action_pairs": unique_pairs,
        "repeated_visible_observation_groups": duplicates,
        "singleton_visible_observation_groups": len(groups) - duplicates,
        "ambiguous_visible_observation_groups": len(ambiguous),
        "rows_in_ambiguous_groups": sum(x["labeled_rows"] for x in ambiguous),
        "cross_task_conflict_groups": cross_task_conflicts,
        "within_task_conflict_groups": within_task_conflicts,
        "conflict_groups_fully_resolved_by_task_identity": task_resolved,
        "empirical_deterministic_lookup": {
            "max_correct_without_task_id": max_correct,
            "accuracy_without_task_id": max_correct / total,
            "irreducible_errors_without_task_id": total - max_correct,
            "max_correct_given_oracle_task_id": max_correct_given_task,
            "accuracy_given_oracle_task_id": max_correct_given_task / total,
            "irreducible_errors_given_task_id": total - max_correct_given_task,
        },
        "action_support": per_action,
        "conflicting_observation_examples": ambiguous[:12],
    }


def run(config_path: str) -> dict:
    import xminigrid

    settings = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    source_config = Path(settings["source_config"])
    source = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    budget, cap, _, _ = validate_config(source)
    if tuple(settings["training_tasks"]) != TRAIN_TASKS:
        raise ValueError("Audit must use the original three training tasks")
    env, defaults = xminigrid.make(source["environment"])
    data, provenance = collect(
        env, defaults, TRAIN_TASKS, budget=budget,
        seed_base=int(source["demonstration_seed_base"]), cap=cap,
        max_attempts=int(source["max_collection_attempts"]),
    )
    actual = observation_action_audit(
        data["obs"], data["acts"], provenance["used_label_skill_names"]
    )
    if actual["labeled_rows"] != 256:
        raise AssertionError("Frozen REUSE-10 dataset size changed")
    expected_counts = {0: 157, 1: 36, 2: 23, 3: 30, 4: 10}
    observed_counts = Counter(map(int, data["acts"]))
    if dict(observed_counts) != expected_counts:
        raise AssertionError("Recollected dataset is not the REUSE-10 pool")
    result = {
        "question": "Do identical learner-visible XLand observations demand conflicting expert actions?",
        "contribution": "Deterministic dataset information and coverage audit; no learning",
        "source_config": str(source_config),
        "heldout_demonstrations_used": 0,
        "policy_training_interactions": 0,
        "policy_training_runs": 0,
        "collection": provenance,
        "audit": actual,
        "limitations": [
            "Only these 256 training-only labeled decisions are audited; exact agreement does not prove generalization or memory is unnecessary.",
            "Task identity is diagnostic provenance only and was never an observed BC policy feature.",
            "A task-conditioned lookup bound is not a fair deployed policy or a learning result.",
            "We have not rerun the REUSE-10 trained policies or measured their off-expert rollout state coverage.",
            "Successful teacher generation interactions are counted; rejected-seed generator interactions remain uninstrumented as in REUSE-10.",
            "No held-out chain actions, labels, or oracle trajectories are used.",
        ],
    }
    dest = Path(settings["output_dir"]) / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("REUSE-11 exact visible observation/action audit")
    print(f"observations={actual['distinct_visible_observations']}/{budget}; "
          f"ambiguous_groups={actual['ambiguous_visible_observation_groups']}; "
          f"lookup_ceiling={actual['empirical_deterministic_lookup']['accuracy_without_task_id']:.4f}")
    print(f"Saved {dest}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xland_observation_aliasing.yaml")
    run(parser.parse_args().config)
