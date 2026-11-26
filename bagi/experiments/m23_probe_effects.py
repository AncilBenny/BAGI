"""M23 stage 6: learn from ACTUALLY executed public-action failed interactions.

Uses a matched cached-source-only control. Privileged native event witnesses
are evaluation-only; training labels remain public before/after observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np

from bagi.experiments.m1_goal_learning import admitted_source_commit
from bagi.experiments.m23_event_effects import (
    BATCH_SIZE,
    EPOCHS,
    EVENT_NAMES,
    LR,
    MIN_NEGATIVE,
    MIN_POSITIVE,
    encode_events,
    event_metrics,
    event_weights,
    fit_event,
    predict,
)
from bagi.experiments.m23_public_world_model import (
    SOURCE_ARTIFACT_ID,
    SOURCE_ARTIFACT_SHA256,
    TRAIN_SOURCE_COMMIT,
    load_source_artifact,
    save_numeric,
)
from bagi.experiments.m23_source_feasibility import SOURCE_ENV_IDS, run_source_episode
from bagi.experiments.m23_study_contract import (
    MODEL_SEEDS,
    PROBE_MONITOR_LAYOUTS,
    PROBE_TRAIN_LAYOUTS,
    validate_split_contract,
)

OUT = "runs/m23_probe_effects"


def source_probe_split(split: str, root: Path) -> tuple[dict, dict]:
    """Execute public-source controller + randomized real action interventions."""
    if split not in ("probe_train", "probe_monitor"):
        raise ValueError("M23 stage 6 allows only explicit source-probe splits")
    layout_map = PROBE_TRAIN_LAYOUTS if split == "probe_train" else PROBE_MONITOR_LAYOUTS
    batches = []
    summary = {}
    for task, task_id in SOURCE_ENV_IDS.items():
        rows: dict[str, list] = {}
        witnesses = []
        for seed in layout_map[task]:
            witness, episode = run_source_episode(task, seed, split=split)
            witnesses.append(witness)
            for key, values in episode.items():
                rows.setdefault(key, []).extend(values)
        source_actions = sum(w["native_actions"] for w in witnesses)
        histogram = [
            sum(w["action_histogram"][i] for w in witnesses) for i in range(7)
        ]
        if sum(histogram) != source_actions:
            raise ValueError("M23 source probe native actions inconsistent")
        effects = sum((Counter(w["events"]) for w in witnesses), Counter())
        path = root / f"{split}-{task}-native-public.npz"
        if path.exists():
            raise FileExistsError("Cannot overwrite executed source probe evidence")
        np.savez_compressed(path, **{
            key: np.asarray(values, dtype="U128" if key == "mission" else None)
            for key, values in rows.items()
        })
        summary[task] = {
            "environment": task_id, "episodes": len(witnesses),
            "actual_native_actions": source_actions,
            "executed_native_action_histogram": histogram,
            "probe_override_count": sum(
                w["probe_override_count"] for w in witnesses
            ),
            "probe_changed_action_count": sum(
                w["probe_different_from_controller_count"] for w in witnesses
            ),
            "evaluator_native_effects": dict(effects),
            "executed_action_sequence_sha256": [
                w["executed_action_sha256"] for w in witnesses
            ],
            "native_public_npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        batches.append(encode_events(rows))
    result = {
        k: np.concatenate([batch[k] for batch in batches], axis=0)
        for k in batches[0]
    }
    if split == "probe_train" and "evaluator_native_effect" in result:
        raise ValueError("M23 private native event witness leaked into TRAIN")
    if split == "probe_monitor" and "evaluator_native_effect" not in result:
        raise ValueError("M23 independent native event witness missing on MONITOR")
    return result, summary


def equal_exposure_success_only(original: dict, count: int,
                                seed: int) -> dict:
    """Repeat historical source interactions WITHOUT claiming new native steps."""
    size = len(original["proxy_labels"])
    if size == 0 or count < size:
        raise ValueError("M23 success-control requires original corpus and added data")
    rng = np.random.default_rng(seed + 2306000)
    sampled = rng.integers(0, size, size=count - size)
    idx = np.concatenate([np.arange(size), sampled])
    return {key: values[idx].copy() for key, values in original.items()}


def execute(source_input: str, output: str) -> dict:
    validate_split_contract()
    source_commit = admitted_source_commit()
    output_path = Path(output)
    if str(output_path) != f"{OUT}/summary.json" or output_path.exists():
        raise ValueError("M23 stage 6 output must be new in registered isolated path")
    root = output_path.parent
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    original_report, historical = load_source_artifact(Path(source_input))
    if original_report["actual_native_actions"] != 16832:
        raise ValueError("M23 historical source actual-action count mismatch")
    old_public = [encode_events(historical[t]) for t in SOURCE_ENV_IDS]
    original = {
        k: np.concatenate([batch[k] for batch in old_public])
        for k in old_public[0]
    }
    train_new, train_witness = source_probe_split("probe_train", root)
    monitor, monitor_witness = source_probe_split("probe_monitor", root)
    if "evaluator_native_effect" in train_new or "evaluator_native_effect" in original:
        raise ValueError("M23 native event truth cannot enter supervision")
    probe_train = {
        k: np.concatenate((original[k], train_new[k]))
        for k in original
    }
    weights = event_weights(probe_train["proxy_labels"])
    native = monitor["evaluator_native_effect"]
    positive = int((native > 0).sum())
    ineffective = int((native == 0).sum())
    public_agreement = event_metrics(native, monitor["proxy_labels"])
    rule = event_metrics(native, monitor["rule_predictions"])
    never = event_metrics(native, np.zeros_like(native))
    report = {
        "study": "m23_executed_intervention_source_effect_transfer",
        "source_commit": source_commit,
        "historical_source_commit": TRAIN_SOURCE_COMMIT,
        "historical_source_artifact_id": SOURCE_ARTIFACT_ID,
        "historical_source_zip_sha256": SOURCE_ARTIFACT_SHA256,
        "historical_cached_native_actions": original_report["actual_native_actions"],
        "historical_cached_interaction_examples": len(original["proxy_labels"]),
        "new_probe_train_native_actions": sum(
            w["actual_native_actions"] for w in train_witness.values()
        ),
        "new_probe_monitor_native_actions": sum(
            w["actual_native_actions"] for w in monitor_witness.values()
        ),
        "probe_train_interaction_examples": len(train_new["proxy_labels"]),
        "total_source_interaction_training_examples": len(probe_train["proxy_labels"]),
        "probe_monitor_interaction_steps": len(native),
        "native_probe_monitor_positive_events": positive,
        "native_probe_monitor_ineffective_actions": ineffective,
        "probe_train_witnesses": train_witness,
        "probe_monitor_witnesses": monitor_witness,
        "native_effect_public_proxy_agreement": public_agreement,
        "native_effect_precondition_rule": rule,
        "native_effect_always_none": never,
        "training_proxy_event_support": [
            int((probe_train["proxy_labels"] == i).sum())
            for i in range(len(EVENT_NAMES))
        ],
        "training_only_class_weights": [float(x) for x in weights],
        "target_unlockpickup_native_actions": 0,
        "target_final_episodes": 0, "imagined_planner_steps": 0,
        "new_fresh_model_fits": 0, "actual_optimizer_step_calls": 0,
        "cached_sample_presentations": 0,
        "model_seeds": list(MODEL_SEEDS), "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "learning_rate": LR,
        "packages": {
            key: version(key) for key in ("numpy", "torch", "minigrid", "gymnasium")
        },
        "per_seed": {},
    }
    for seed in MODEL_SEEDS:
        repeat_control = equal_exposure_success_only(
            original, len(probe_train["proxy_labels"]), seed
        )
        pair = {}
        for arm, corpus in (
            ("probe_trained", probe_train),
            ("success_only_matched", repeat_control),
        ):
            fitted, details = fit_event(corpus, seed, aware=True, weights=weights)
            guessed = predict(fitted, monitor, aware=True)
            score = event_metrics(native, guessed)
            digest = save_numeric(
                fitted, root / f"{arm}-seed{seed}.npz"
            )
            pair[arm] = {**details, **score, "checkpoint_sha256": digest}
            report["new_fresh_model_fits"] += 1
            report["actual_optimizer_step_calls"] += details[
                "actual_optimizer_step_calls"
            ]
            report["cached_sample_presentations"] += details[
                "cached_train_sample_presentations"
            ]
        report["per_seed"][str(seed)] = pair
        print(json.dumps({
            "seed": seed,
            "probe_native_f1": pair["probe_trained"]["macro_f1"],
            "success_only_native_f1": pair["success_only_matched"]["macro_f1"],
            "rule_native_f1": rule["macro_f1"],
            "native_positive": positive, "native_ineffective": ineffective,
            "optimizer_steps_per_model": pair["probe_trained"][
                "actual_optimizer_step_calls"
            ],
        }), flush=True)
    results = list(report["per_seed"].values())
    probe_scores = np.asarray([
        x["probe_trained"]["macro_f1"] for x in results
    ], dtype=np.float64)
    control_scores = np.asarray([
        x["success_only_matched"]["macro_f1"] for x in results
    ], dtype=np.float64)
    gains_control = probe_scores - control_scores
    gains_rule = probe_scores - rule["macro_f1"]
    positive_recalls = np.asarray([
        x["probe_trained"]["native_positive_event_recall"] or 0
        for x in results
    ], dtype=np.float64)
    rng = np.random.default_rng(230615)
    draws = rng.integers(0, len(MODEL_SEEDS), size=(20000, len(MODEL_SEEDS)))
    report["mean_probe_minus_success_only_native_f1"] = float(gains_control.mean())
    report["mean_probe_minus_public_rule_native_f1"] = float(gains_rule.mean())
    report["seeds_probe_over_control_gt_005"] = int((gains_control > 0.05).sum())
    report["seeds_probe_over_rule_gt_005"] = int((gains_rule > 0.05).sum())
    report["mean_correct_positive_native_effect_recall"] = float(
        positive_recalls.mean()
    )
    report["seed_bootstrap_95pct_probe_minus_control"] = [
        float(x) for x in np.quantile(
            gains_control[draws].mean(axis=1), (0.025, 0.975)
        )
    ]
    report["registered_probe_gate_pass"] = bool(
        positive >= MIN_POSITIVE and ineffective >= MIN_NEGATIVE
        and public_agreement["exact_event_accuracy"] >= 0.8
        and gains_control.mean() > 0.05
        and (gains_control > 0.05).sum() >= 6
        and gains_rule.mean() > 0.05
        and (gains_rule > 0.05).sum() >= 6
        and positive_recalls.mean() >= 0.6
    )
    report["elapsed_seconds"] = time.perf_counter() - t0
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_sha": source_commit,
        "new_probe_train_native_actions": report["new_probe_train_native_actions"],
        "new_probe_monitor_native_actions": report["new_probe_monitor_native_actions"],
        "native_probe_monitor_positive_events": positive,
        "native_probe_monitor_ineffective_actions": ineffective,
        "new_fresh_model_fits": report["new_fresh_model_fits"],
        "actual_optimizer_step_calls": report["actual_optimizer_step_calls"],
        "mean_probe_minus_success_only_native_f1":
            report["mean_probe_minus_success_only_native_f1"],
        "mean_probe_minus_public_rule_native_f1":
            report["mean_probe_minus_public_rule_native_f1"],
        "registered_probe_gate_pass": report["registered_probe_gate_pass"],
    }), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=f"{OUT}/input")
    parser.add_argument("--output", default=f"{OUT}/summary.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("M23 source intervention study requires --execute")
    execute(args.input_root, args.output)


if __name__ == "__main__":
    main()
