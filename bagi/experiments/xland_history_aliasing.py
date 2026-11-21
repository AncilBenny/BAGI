"""REUSE-12: test whether short preceding action history separates expert labels.

This is a deterministic exact-data information audit, NOT a trained memory
policy. No private task ID, hidden geometry, or held-out actions are inputs.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from bagi.experiments.xland_existing_bc import TRAIN_TASKS, collect, validate_config


def history_lookup_audit(observations, actions, dones, *, lookbacks=(0, 1, 4)):
    """Compare empirical lookup consistency given 0/1/4 *previous* actions."""
    obs = np.asarray(observations, dtype=np.float32)
    acts = np.asarray(actions, dtype=np.int64)
    ends = np.asarray(dones, dtype=bool)
    if (obs.ndim != 2 or len(obs) < 1 or acts.shape != (len(obs),)
            or ends.shape != (len(obs),) or not np.isfinite(obs).all()):
        raise ValueError("Finite observed rows must align with actions and episode ends")
    if np.any((acts < 0) | (acts > 5)):
        raise ValueError("Native XLand action IDs must be in range")
    if tuple(lookbacks) != (0, 1, 4):
        raise ValueError("Only preregistered 0/1/4 previous-action windows")

    history = []
    grouped = {window: defaultdict(Counter) for window in lookbacks}
    for i, row in enumerate(obs):
        features = np.ascontiguousarray(row).tobytes()
        for window in lookbacks:
            past_actions = tuple(history[-window:]) if window else ()
            grouped[window][(features, past_actions)][int(acts[i])] += 1
        # Include the action only for FUTURE decisions, never current label.
        history.append(int(acts[i]))
        if ends[i]:
            history.clear()

    report = {}
    for window, signatures in grouped.items():
        counts = list(signatures.values())
        max_correct = sum(max(c.values()) for c in counts)
        conflicted = [c for c in counts if len(c) > 1]
        report[str(window)] = {
            "previous_action_window": window,
            "distinct_observation_history_signatures": len(counts),
            "repeated_signature_groups": sum(sum(c.values()) > 1 for c in counts),
            "ambiguous_signature_groups": len(conflicted),
            "rows_in_ambiguous_groups": sum(sum(c.values()) for c in conflicted),
            "max_correct_exact_lookup": max_correct,
            "irreducible_errors_exact_lookup": len(acts) - max_correct,
            "empirical_lookup_accuracy": max_correct / len(acts),
        }
    return report


def run(config_path: str) -> dict:
    import xminigrid

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    source_path = Path(cfg["source_config"])
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    budget, cap, _, _ = validate_config(source)
    if tuple(cfg["training_tasks"]) != TRAIN_TASKS:
        raise ValueError("Held-out skill cannot enter the expert pool")
    env, defaults = xminigrid.make(source["environment"])
    dataset, collection = collect(
        env, defaults, TRAIN_TASKS, budget=budget,
        seed_base=int(source["demonstration_seed_base"]),
        cap=cap, max_attempts=int(source["max_collection_attempts"]),
    )
    expected_actions = {0: 157, 1: 36, 2: 23, 3: 30, 4: 10}
    if budget != 256 or dict(Counter(map(int, dataset["acts"]))) != expected_actions:
        raise AssertionError("REUSE-10/11 frozen source dataset changed")
    results = history_lookup_audit(
        dataset["obs"], dataset["acts"], dataset["dones"]
    )
    baseline = results["0"]
    if (baseline["distinct_observation_history_signatures"] != 180
            or baseline["max_correct_exact_lookup"] != 234
            or baseline["ambiguous_signature_groups"] != 16):
        raise AssertionError("Cannot reproduce the REUSE-11 memoryless audit")
    report = {
        "question": "Can 1 or 4 preceding expert actions resolve visible-state action aliasing?",
        "source_config": str(source_path),
        "used_heldout_chain_demonstrations": 0,
        "new_training": False,
        "expert_oracle_queries_outside_training_tasks": 0,
        "collection": collection,
        "history": results,
        "limitations": [
            "Empirical lookup consistency is not policy competence, learned memory, or generalization.",
            "Adding action history creates more unique keys, which can trivially improve training-set fit.",
            "Prior action histories come from expert trajectories, not learner rollouts.",
            "Action-only memory is narrower than full observation-action history.",
            "We do not measure off-expert state support or repeat the failed BC training pilots.",
            "Simulator positions, rulesets and privileged navigation are never learner features.",
        ],
    }
    dest = Path(cfg["output_dir"]) / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for k, row in results.items():
        print(f"previous_actions={k}: unique={row['distinct_observation_history_signatures']} "
              f"ambiguous={row['ambiguous_signature_groups']} "
              f"lookup={row['max_correct_exact_lookup']}/{budget}")
    print(f"Saved history audit: {dest}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xland_history_aliasing.yaml")
    run(parser.parse_args().config)
