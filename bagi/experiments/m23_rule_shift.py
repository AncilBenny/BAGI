"""M23 stage 7: no-retraining policy-shift test of stage-6 native-effect models.

All model weights are frozen numeric-only NPZ, SHA256 verified against the
successful source experiment. Only NEW, independently seeded source MiniGrid
actions provide outcomes; no target UnlockPickup states or rewards are read.
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
import torch
from minigrid.core.constants import OBJECT_TO_IDX

from bagi.experiments.m1_goal_learning import admitted_source_commit
from bagi.experiments.m23_event_effects import (
    BATCH_SIZE,
    PublicActionEffectPredictor,
    _front,
    encode_events,
    event_metrics,
    predict,
    public_effect_proxy,
)
from bagi.experiments.m23_source_feasibility import SOURCE_ENV_IDS, run_source_episode
from bagi.experiments.m23_study_contract import (
    MODEL_SEEDS,
    RULE_SHIFT_MONITOR_LAYOUTS,
    validate_split_contract,
)

OUT = "runs/m23_rule_shift"
PREVIOUS_SOURCE_SHA = "472ed53f37af64ae6c7e2ed9c12f74412f7ef34f"
PREVIOUS_ARTIFACT_ID = 14134639518
PREVIOUS_ARTIFACT_DIGEST = (
    "de2270383c7de3d52dea54f4be208e0383b83d52e33001a27ba85cbff512e82e"
)
MIN_ELIGIBLE_POSITIVE = 16
MIN_ELIGIBLE_INEFFECTIVE = 16
_CARRYABLE = {OBJECT_TO_IDX[k] for k in ("key", "ball", "box")}


def numeric_checkpoint(root: Path, expected_sha: str,
                       arm: str, seed: int) -> PublicActionEffectPredictor:
    """Load only numerical torch.state_dict tensors, never pickle/model code."""
    if arm not in ("probe_trained", "success_only_matched"):
        raise ValueError("Unregistered M23 frozen checkpoint arm")
    found = list(root.rglob(f"{arm}-seed{seed}.npz"))
    if len(found) != 1:
        raise ValueError("Missing or ambiguous M23 source model checkpoint")
    path = found[0]
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError("Frozen M23 model checkpoint SHA256 mismatch")
    model = PublicActionEffectPredictor()
    original = model.state_dict()
    keys = [f"tensor_{i:03d}" for i in range(len(original))]
    with np.load(path, allow_pickle=False) as contents:
        if set(contents.files) != set(keys):
            raise ValueError("M23 checkpoint numeric tensor manifest mismatch")
        state = {}
        for i, (name, existing) in enumerate(original.items()):
            array = contents[keys[i]]
            if tuple(array.shape) != tuple(existing.shape):
                raise ValueError("M23 frozen tensor shape mismatch")
            if array.dtype.kind not in "fiub":
                raise ValueError("Nonnumeric M23 weights are forbidden")
            state[name] = torch.from_numpy(array.copy()).to(existing.dtype)
    model.load_state_dict(state, strict=True)
    return model.eval()


def source_checkpoint_manifest(root: Path) -> dict:
    """Requires exact stage-6 provenance and its preregistered PASS marker."""
    matches = list(root.rglob("m23_probe_effects-summary.json"))
    if len(matches) != 1:
        raise ValueError("Stage-6 source report missing/ambiguous in artifact")
    manifest = json.loads(matches[0].read_text(encoding="utf-8"))
    if (manifest.get("source_commit") != PREVIOUS_SOURCE_SHA
            or manifest.get("registered_probe_gate_pass") is not True
            or manifest.get("new_fresh_model_fits") != 16
            or manifest.get("actual_optimizer_step_calls") != 98560
            or manifest.get("target_final_episodes") != 0):
        raise ValueError("Frozen stage-6 model provenance is inconsistent")
    if set(manifest.get("per_seed", {})) != {str(x) for x in MODEL_SEEDS}:
        raise ValueError("Frozen M23 model seed manifest changed")
    return manifest


def public_history_effect_rule(rows: dict) -> np.ndarray:
    """Publicly inferred inventory state, NOT a native carrying-state oracle.

    Predictions use current public observation, attempted action, and only
    PREVIOUS executed visible changes; updates occur AFTER each native action.
    """
    carried: int | None = None
    predictions = []
    for before, action, after in zip(
        rows["observation_image"], rows["action"], rows["next_image"], strict=True
    ):
        action = int(action)
        front, state = _front(np.asarray(before))
        predicted = 0
        if action == 3 and front in _CARRYABLE and carried is None:
            predicted = 1
        elif action == 4 and front in (
            OBJECT_TO_IDX["empty"], OBJECT_TO_IDX["floor"]
        ) and carried is not None:
            predicted = 2
        elif (action == 5 and front == OBJECT_TO_IDX["door"]
              and state == 2 and carried == OBJECT_TO_IDX["key"]):
            predicted = 3
        predictions.append(predicted)
        # This is self-observed past effect, not the independent native truth.
        witnessed_publicly = public_effect_proxy(np.asarray(before), action,
                                                 np.asarray(after))
        if witnessed_publicly == 1:
            carried = front
        elif witnessed_publicly == 2:
            carried = None
    return np.asarray(predictions, dtype=np.int64)


def collect_shifted_source(root: Path) -> tuple[dict, dict]:
    """Source-MONITOR-only native actions, independently witnessed and archived."""
    merged = []
    provenance = {}
    for task, env_id in SOURCE_ENV_IDS.items():
        episodes = []
        native = []
        for seed in RULE_SHIFT_MONITOR_LAYOUTS[task]:
            witness, rows = run_source_episode(task, seed, split="rule_shift_monitor")
            if "evaluator_native_effect" not in rows:
                raise ValueError("Missing evaluator-native source effect witness")
            if not (rows["terminated"][-1] or rows["truncated"][-1]):
                raise ValueError("Incomplete native episode; carrying memory unsafe")
            history_decisions = public_history_effect_rule(rows)
            encoded = encode_events(rows)
            executed = np.isin(np.asarray(rows["action"]), (3, 4, 5))
            encoded["public_history_rule"] = history_decisions[executed]
            episodes.append(encoded)
            native.append(witness)
        all_rows = {}
        for field in episodes[0]:
            all_rows[field] = np.concatenate([ep[field] for ep in episodes])
        # Save numeric-only independent source monitor data; not a trained model.
        output = root / f"{task}-shifted-public-native-monitor.npz"
        if output.exists():
            raise FileExistsError("Refusing to overwrite prior source evaluation")
        np.savez_compressed(output, **all_rows)
        effects = sum((Counter(w["events"]) for w in native), Counter())
        n_actions = sum(w["native_actions"] for w in native)
        histogram = [
            sum(w["action_histogram"][i] for w in native) for i in range(7)
        ]
        if sum(histogram) != n_actions:
            raise ValueError("Source native executed-action count mismatch")
        provenance[task] = {
            "env_id": env_id, "episodes": len(native),
            "actual_native_actions": n_actions,
            "action_histogram": histogram,
            "probe_override_count": sum(w["probe_override_count"] for w in native),
            "probe_changed_action_count": sum(
                w["probe_different_from_controller_count"] for w in native
            ),
            "native_effects": dict(effects),
            "native_reward_sum": sum(w["native_reward_sum"] for w in native),
            "executed_action_hashes": [
                w["executed_action_sha256"] for w in native
            ],
            "independent_monitor_sha256": hashlib.sha256(
                output.read_bytes()
            ).hexdigest(),
        }
        merged.append(all_rows)
    total = {
        field: np.concatenate([part[field] for part in merged])
        for field in merged[0]
    }
    return total, provenance


def scoped_metrics(native: np.ndarray, guesses: np.ndarray) -> dict:
    metrics = event_metrics(native, guesses)
    no_effect = native == 0
    true_effect = native > 0
    return {
        **metrics,
        "native_no_effect_specificity": float(
            np.mean(guesses[no_effect] == 0)
        ) if no_effect.any() else None,
        "correct_native_effect_positive_recall": float(
            np.mean(guesses[true_effect] == native[true_effect])
        ) if true_effect.any() else None,
    }


def execute(input_root: str, output: str) -> dict:
    validate_split_contract()
    source_sha = admitted_source_commit()
    target = Path(output)
    if str(target) != f"{OUT}/summary.json" or target.exists():
        raise ValueError("M23 shifted-source output must be new and isolated")
    target.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    manifest = source_checkpoint_manifest(Path(input_root))
    torch.set_num_threads(1)
    monitor, source = collect_shifted_source(target.parent)
    truth = monitor["evaluator_native_effect"]
    eligible = monitor["rule_predictions"] > 0
    pos = int(((truth > 0) & eligible).sum())
    neg = int(((truth == 0) & eligible).sum())
    report = {
        "study": "m23_frozen_native_effect_policy_shift",
        "evaluation_source_commit": source_sha,
        "historical_trained_source_commit": PREVIOUS_SOURCE_SHA,
        "historical_trained_artifact_id": PREVIOUS_ARTIFACT_ID,
        "historical_trained_artifact_sha256": PREVIOUS_ARTIFACT_DIGEST,
        "new_actual_source_monitor_actions": sum(
            x["actual_native_actions"] for x in source.values()
        ),
        "source_witnesses": source,
        "new_source_interaction_attempts": int(len(truth)),
        "rule_eligible_source_attempts": int(eligible.sum()),
        "rule_eligible_native_effect_positive": pos,
        "rule_eligible_native_ineffective": neg,
        "total_native_effect_positive": int((truth > 0).sum()),
        "total_native_ineffective": int((truth == 0).sum()),
        "no_new_training_native_actions": True,
        "fresh_model_fits": 0,
        "optimizer_step_calls": 0,
        "new_training_presentations": 0,
        "target_unlockpickup_actions": 0,
        "target_final_episodes": 0,
        "imagined_planner_steps": 0,
        "torch_forward_batches": 0,
        "versions": {
            k: version(k) for k in ("numpy", "torch", "minigrid", "gymnasium")
        },
        "full_interaction_reference": {},
        "eligible_subset": {},
        "frozen_seed_results": {},
    }
    for name, predictions in (
        ("public_rule", monitor["rule_predictions"]),
        ("public_history_rule", monitor["public_history_rule"]),
        ("never_effect", np.zeros_like(truth)),
        ("public_posthoc_proxy_diagnostic", monitor["proxy_labels"]),
    ):
        report["full_interaction_reference"][name] = scoped_metrics(truth, predictions)
        report["eligible_subset"][name] = (
            scoped_metrics(truth[eligible], predictions[eligible])
            if eligible.any() else None
        )
    for seed in MODEL_SEEDS:
        scored = {}
        for arm in ("probe_trained", "success_only_matched"):
            record = manifest["per_seed"][str(seed)][arm]
            model = numeric_checkpoint(
                Path(input_root), record["checkpoint_sha256"], arm, seed
            )
            guess = predict(model, monitor, aware=True)
            report["torch_forward_batches"] += (
                (len(truth) + BATCH_SIZE - 1) // BATCH_SIZE
            )
            scored[arm] = {
                "model_checksum_sha256": record["checkpoint_sha256"],
                "all_interactions": scoped_metrics(truth, guess),
                "public_rule_eligible": scoped_metrics(
                    truth[eligible], guess[eligible]
                ) if eligible.any() else None,
            }
        report["frozen_seed_results"][str(seed)] = scored
        print(json.dumps({
            "seed": seed,
            "probe_eligible_f1": scored["probe_trained"][
                "public_rule_eligible"
            ]["macro_f1"] if eligible.any() else None,
            "success_control_eligible_f1": scored["success_only_matched"][
                "public_rule_eligible"
            ]["macro_f1"] if eligible.any() else None,
            "rule_eligible_positive": pos, "rule_eligible_ineffective": neg,
            "new_optimizer_steps": 0,
        }), flush=True)
    values = list(report["frozen_seed_results"].values())
    probe = np.asarray([
        x["probe_trained"]["public_rule_eligible"]["macro_f1"]
        for x in values
    ]) if eligible.any() else np.zeros(len(MODEL_SEEDS))
    control = np.asarray([
        x["success_only_matched"]["public_rule_eligible"]["macro_f1"]
        for x in values
    ]) if eligible.any() else np.zeros(len(MODEL_SEEDS))
    front_f1 = report["eligible_subset"]["public_rule"]["macro_f1"]
    history_f1 = report["eligible_subset"]["public_history_rule"]["macro_f1"]
    rule_gains = probe - front_f1
    control_gains = probe - control
    recalls = [
        x["probe_trained"]["public_rule_eligible"][
            "correct_native_effect_positive_recall"
        ] or 0 for x in values
    ] if eligible.any() else []
    rng = np.random.default_rng(230710)
    draw = rng.integers(0, len(MODEL_SEEDS), size=(20000, len(MODEL_SEEDS)))
    report["mean_probe_eligible_native_macro_f1"] = float(probe.mean())
    report["mean_probe_minus_rule_eligible_f1"] = float(rule_gains.mean())
    report["mean_probe_minus_success_control_eligible_f1"] = float(control_gains.mean())
    report["public_history_rule_eligible_native_macro_f1"] = float(history_f1)
    report["seeds_probe_over_rule_by_0_05"] = int((rule_gains > 0.05).sum())
    report["seeds_probe_over_success_control_by_0_05"] = int(
        (control_gains > 0.05).sum()
    )
    report["mean_probe_eligible_effect_positive_recall"] = (
        float(np.mean(recalls)) if recalls else None
    )
    report["seed_bootstrap_95pct_probe_minus_rule"] = [
        float(v) for v in np.quantile(rule_gains[draw].mean(axis=1), [0.025, 0.975])
    ]
    report["registered_shift_robustness_gate_pass"] = bool(
        pos >= MIN_ELIGIBLE_POSITIVE and neg >= MIN_ELIGIBLE_INEFFECTIVE
        and probe.mean() > 0.75
        and rule_gains.mean() > 0.05 and (rule_gains > 0.05).sum() >= 6
        and control_gains.mean() > 0.05 and (control_gains > 0.05).sum() >= 6
        and len(recalls) == len(MODEL_SEEDS) and np.mean(recalls) >= 0.70
    )
    report["registered_public_history_reuse_preferred"] = bool(
        pos >= MIN_ELIGIBLE_POSITIVE and neg >= MIN_ELIGIBLE_INEFFECTIVE
        and history_f1 >= 0.95 and history_f1 >= probe.mean()
    )
    report["wall_seconds"] = time.perf_counter() - start
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_sha": source_sha,
        "new_actual_source_monitor_actions": report["new_actual_source_monitor_actions"],
        "rule_eligible_native_effect_positive": pos,
        "rule_eligible_native_ineffective": neg,
        "mean_probe_eligible_native_macro_f1": report["mean_probe_eligible_native_macro_f1"],
        "public_history_rule_eligible_native_macro_f1": history_f1,
        "registered_shift_robustness_gate_pass": report["registered_shift_robustness_gate_pass"],
        "registered_public_history_reuse_preferred": report[
            "registered_public_history_reuse_preferred"
        ],
        "optimizer_step_calls": 0, "fresh_model_fits": 0,
    }), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=f"{OUT}/input")
    parser.add_argument("--output", default=f"{OUT}/summary.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("M23 frozen-policy shift requires explicit --execute")
    execute(args.input_root, args.output)


if __name__ == "__main__":
    main()
