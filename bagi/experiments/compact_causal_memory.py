"""Gate B3: a compact observation-only causal memory versus raw history keys.

Rule-based diagnostic control, NOT learned intelligence: persist two booleans
indicating that the actor stepped onto a visible SWITCH or KEY. The feature
never receives environment flags, position, event labels, or hidden state.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from bagi.experiments.history_ambiguity import encode, outcome, visible_input
from bagi.experiments.history_rollout_robustness import generate, lookup_score
from bagi.worlds import WorkshopEnv
from bagi.worlds.workshop import KEY, SWITCH

DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def sensed_causal_memory(history: list) -> tuple[bool, bool]:
    """Track events from visible target tiles before each executed move.

    This is an explicit *task-specific* rule-based belief/control baseline.
    The Workshop makes SWITCH and KEY traversable, allowing direct inference
    after an action toward their previously visible tile.
    """
    switch_seen = False
    key_seen = False
    for previous_observation, action in history:
        center = len(previous_observation) // 2
        delta_row, delta_col = DIRECTIONS[int(action)]
        tile = previous_observation[center + delta_row][center + delta_col]
        switch_seen |= tile == SWITCH
        key_seen |= tile == KEY
    return switch_seen, key_seen


def compact_feature(row: dict) -> str:
    return encode([sensed_causal_memory(row["history"]), row["observation"], row["action"]])


def lookup_with_compact_memory(train: list[dict], heldout: list[dict]) -> dict:
    if not train or not heldout:
        raise ValueError("Require training and heldout examples")
    tables = defaultdict(Counter)
    for row in train:
        tables[compact_feature(row)][outcome(row)] += 1
    majorities = {key: labels.most_common(1)[0][0] for key, labels in tables.items()}
    seen = 0
    correct = 0
    event_correct = 0
    for row in heldout:
        candidate = majorities.get(compact_feature(row))
        if candidate is None:
            continue
        seen += 1
        correct += int(candidate == outcome(row))
        event_correct += int(json.loads(candidate)[-1] == row["events"])
    return {
        "heldout_rows": len(heldout),
        "unique_training_keys": len(majorities),
        "recognized_heldout_rows": seen,
        "heldout_key_coverage": seen / len(heldout),
        "heldout_whole_accuracy_given_seen": correct / seen if seen else None,
        "heldout_event_accuracy_given_seen": event_correct / seen if seen else None,
        "heldout_whole_accuracy_with_unknown_as_error": correct / len(heldout),
    }


def known_reactive_ambiguity(train: list[dict]) -> set[str]:
    results = defaultdict(set)
    for row in train:
        results[visible_input(row, 0)].add(outcome(row))
    return {key for key, labels in results.items() if len(labels) > 1}


def run(config_path: str) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    env = WorkshopEnv(**config["environment"])
    cfg = config["evaluation"]
    rows, counts = generate(
        env=env,
        guided_actions=[int(a) for a in cfg["guided_actions"]],
        prefixes=[int(k) for k in cfg["prefix_steps"]],
        train_episodes=int(cfg["training_episodes_per_prefix"]),
        holdout_episodes=int(cfg["holdout_episodes_per_prefix"]),
        continuation_steps=int(cfg["continuation_steps"]),
        seed=int(cfg["seed"]),
        max_history=int(cfg["max_history"]),
    )
    train = [row for row in rows if row["split"] == "train"]
    held = [row for row in rows if row["split"] == "holdout"]
    reactive = lookup_score(train, held, 0)
    one_step = lookup_score(train, held, 1)
    eight_step = lookup_score(train, held, 8)
    compact = lookup_with_compact_memory(train, held)

    ambiguous = known_reactive_ambiguity(train)
    held_ambiguous = [r for r in held if visible_input(r, 0) in ambiguous]
    ambiguous_result = None
    if held_ambiguous:
        ambiguous_result = {
            "heldout_rows_with_ambiguous_training_observation_action": len(held_ambiguous),
            "reactive": lookup_score(train, held_ambiguous, 0),
            "compact": lookup_with_compact_memory(train, held_ambiguous),
        }
    result = {
        "question": "Can two hand-constructed observable causal-history bits cover unseen trajectories better than storing raw previous observations?",
        "evaluation": counts,
        "design": cfg,
        "baseline_reactive": reactive,
        "baseline_raw_history_one_step": one_step,
        "baseline_raw_history_eight_steps": eight_step,
        "two_bit_observed_causal_memory": compact,
        "ambiguous_reactive_input_subset": ambiguous_result,
        "limitations": [
            "The two-bit memory is explicitly hand-built from known SWITCH/KEY tile semantics, not learned causal knowledge.",
            "It only infers progress after a visible action into an expected traversable causal tile.",
            "All methods use the same fixed Workshop geometry, scripted prefixes and held-out action RNG trajectories.",
            "Exact feature lookup may abstain on unseen keys; conditional and unknown-as-error accuracy must both be reported.",
            "No claim about novel layouts, real learned-memory generalization, policy goal-reaching or sample efficiency is supported.",
        ],
    }
    path = Path(cfg["output_dir"]) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, row in [
        ("reactive", reactive),
        ("raw_1", one_step),
        ("raw_8", eight_step),
        ("two_bit", compact),
    ]:
        accuracy = row.get("heldout_whole_accuracy_with_unknown_as_error")
        if accuracy is None:
            accuracy = row["heldout_whole_outcome_accuracy_with_unknown_as_error"]
        print(f"{name}: coverage={row['heldout_key_coverage']:.4f}, "
              f"whole_accuracy_all={accuracy:.4f}")
    print(f"Ambiguous-reactive heldout rows: {len(held_ambiguous)}")
    print(f"Saved compact observed-memory comparison: {path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/compact_causal_memory.yaml")
    run(parser.parse_args().config)
