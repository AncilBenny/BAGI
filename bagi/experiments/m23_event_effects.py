"""M23 stage 5: public-input causal-effect prediction against native witnesses.

Never feeds evaluator-only native fields into the fitted models. Uses native
witnesses only on newly registered source MONITOR episodes for scoring.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from minigrid.core.constants import OBJECT_TO_IDX
from torch import nn

from bagi.experiments.m1_goal_learning import admitted_source_commit
from bagi.experiments.m23_public_world_model import (
    SOURCE_ARTIFACT_ID,
    SOURCE_ARTIFACT_SHA256,
    TRAIN_SOURCE_COMMIT,
    WIDTH,
    encode_public_rows,
    load_source_artifact,
    save_numeric,
)
from bagi.experiments.m23_source_feasibility import SOURCE_ENV_IDS, run_source_episode
from bagi.experiments.m23_study_contract import (
    EFFECT_MONITOR_LAYOUTS,
    MODEL_SEEDS,
    validate_split_contract,
)

OUT = "runs/m23_event_effects"
EVENT_NAMES = ("none", "pickup", "drop", "unlock")
N_EVENTS = len(EVENT_NAMES)
BATCH_SIZE = 128
EPOCHS = 40
LR = 0.001
MIN_POSITIVE = 8
MIN_NEGATIVE = 8

_PICKABLE = {OBJECT_TO_IDX[key] for key in ("key", "ball", "box")}
_EMPTY = {OBJECT_TO_IDX[key] for key in ("empty", "floor")}
_DOOR = OBJECT_TO_IDX["door"]


def _front(observation: np.ndarray) -> tuple[int, int]:
    img = np.asarray(observation)
    if img.shape != (7, 7, 3):
        raise ValueError("M23 requires standard public 7x7x3 observation")
    return int(img[3, 5, 0]), int(img[3, 5, 2])


def public_effect_proxy(before: np.ndarray, action: int, after: np.ndarray) -> int:
    """TRAIN labels only: observable front-cell change AFTER executed action."""
    obj, state = _front(before)
    following, next_state = _front(after)
    if action == 3 and obj in _PICKABLE and following != obj:
        return 1
    if action == 4 and obj in _EMPTY and following in _PICKABLE:
        return 2
    if action == 5 and obj == following == _DOOR and state == 2 and next_state == 0:
        return 3
    return 0


def public_eligibility_rule(before: np.ndarray, action: int) -> int:
    """No-training physical precondition baseline; never reads next observation."""
    obj, state = _front(before)
    if action == 3 and obj in _PICKABLE:
        return 1
    if action == 4 and obj in _EMPTY:
        return 2
    if action == 5 and obj == _DOOR and state == 2:
        return 3
    return 0


def encode_events(rows: dict) -> dict[str, np.ndarray]:
    """Encode public source transitions; discard any evaluator-native field."""
    base = encode_public_rows(rows)
    actions = np.asarray(rows["action"], dtype=np.int64)
    use = np.isin(actions, (3, 4, 5))
    if not use.any():
        raise ValueError("M23 source split contains no executed interaction actions")
    labels = np.array([
        public_effect_proxy(before, int(action), after)
        for before, action, after in zip(
            rows["observation_image"], actions, rows["next_image"], strict=True
        )
    ], dtype=np.int64)
    heuristics = np.array([
        public_eligibility_rule(before, int(action))
        for before, action in zip(rows["observation_image"], actions, strict=True)
    ], dtype=np.int64)
    result = {
        "features": base["features"][use].copy(),
        "proxy_labels": labels[use],
        "rule_predictions": heuristics[use],
        "native_actions": actions[use],
    }
    if "evaluator_native_effect" in rows:
        # STRICTLY evaluation, never exposed to the neural model/optimizer.
        actual = np.asarray(rows["evaluator_native_effect"], dtype=np.int64)
        if len(actual) != len(actions) or not np.all((0 <= actual) & (actual < N_EVENTS)):
            raise ValueError("M23 evaluator-only native effects malformed")
        result["evaluator_native_effect"] = actual[use].copy()
    return result


class PublicActionEffectPredictor(nn.Module):
    """Action-conditioned public inputs, no physics oracle or future image."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(WIDTH, 192), nn.ReLU(),
            nn.Linear(192, 192), nn.ReLU(),
            nn.Linear(192, N_EVENTS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != WIDTH:
            raise ValueError("M23 event model requires public feature width")
        return self.net(x)


def event_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=N_EVENTS)
    if len(counts) != N_EVENTS or np.any(labels < 0) or np.any(labels >= N_EVENTS):
        raise ValueError("M23 source public event label out of range")
    if counts.sum() == 0:
        raise ValueError("M23 no public effect training labels")
    weights = np.zeros(N_EVENTS, dtype=np.float64)
    present = counts > 0
    weights[present] = np.sqrt(counts.sum() / counts[present])
    norm = np.dot(counts, weights) / counts.sum()
    weights[present] = np.clip(weights[present] / norm, 0.25, 8.0)
    return weights.astype(np.float32)


def fit_event(rows: dict[str, np.ndarray], seed: int, *, aware: bool,
              weights: np.ndarray) -> tuple[PublicActionEffectPredictor, dict]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    model = PublicActionEffectPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    features = rows["features"].astype(np.float32, copy=True)
    if not aware:
        # Only current action is removed; own prior real actions stay public.
        features[:, -7:] = 0.0
    x = torch.from_numpy(features)
    y = torch.from_numpy(rows["proxy_labels"].copy()).long()
    w = torch.as_tensor(weights, dtype=torch.float32)
    updates = 0
    last_loss = None
    model.train()
    for _ in range(EPOCHS):
        shuffled = rng.permutation(len(x))
        for start in range(0, len(x), BATCH_SIZE):
            index = torch.as_tensor(shuffled[start:start + BATCH_SIZE])
            if not len(index):
                continue
            logits = model(x[index])
            loss = nn.functional.cross_entropy(logits, y[index], weight=w)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
            last_loss = float(loss.detach().cpu())
    return model.eval(), {
        "actual_optimizer_step_calls": updates,
        "cached_train_sample_presentations": len(x) * EPOCHS,
        "last_loss": last_loss,
    }


@torch.no_grad()
def predict(model: PublicActionEffectPredictor, rows: dict, *, aware: bool) -> np.ndarray:
    features = rows["features"].astype(np.float32, copy=True)
    if not aware:
        features[:, -7:] = 0
    parts = []
    for start in range(0, len(features), BATCH_SIZE):
        parts.append(model(torch.from_numpy(features[start:start + BATCH_SIZE]))
                     .argmax(dim=-1).cpu().numpy())
    return np.concatenate(parts)


def event_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    if not len(truth) or len(truth) != len(prediction):
        raise ValueError("M23 empty/mismatched effect assessment")
    if np.any(truth < 0) or np.any(truth >= N_EVENTS):
        raise ValueError("M23 invalid evaluator-only effect labels")
    scores = []
    per_class = {}
    for code, name in enumerate(EVENT_NAMES):
        tp = int(((truth == code) & (prediction == code)).sum())
        fp = int(((truth != code) & (prediction == code)).sum())
        fn = int(((truth == code) & (prediction != code)).sum())
        support = int((truth == code).sum())
        predicted = int((prediction == code).sum())
        if support or predicted:
            scores.append(2 * tp / (2 * tp + fp + fn))
        per_class[name] = {
            "support": support, "predicted_support": predicted,
            "true_positive": tp, "precision": tp / predicted if predicted else None,
            "recall": tp / support if support else None,
        }
    truth_positive = truth > 0
    tp_positive = int(((prediction == truth) & truth_positive).sum())
    return {
        "macro_f1": float(np.mean(scores)) if scores else 0.0,
        "exact_event_accuracy": float(np.mean(truth == prediction)),
        "native_positive_event_recall": tp_positive / int(truth_positive.sum())
        if truth_positive.any() else None,
        "per_class": per_class,
    }


def collect_fresh_source_monitor() -> tuple[dict, dict]:
    public_parts = []
    records = {}
    for task, identifier in SOURCE_ENV_IDS.items():
        aggregate: dict[str, list] = {}
        witnesses = []
        for seed in EFFECT_MONITOR_LAYOUTS[task]:
            summary, rows = run_source_episode(task, seed, split="effect_monitor")
            witnesses.append(summary)
            for key, values in rows.items():
                aggregate.setdefault(key, []).extend(values)
        records[task] = {
            "env_id": identifier,
            "native_actions": sum(x["native_actions"] for x in witnesses),
            "action_histogram": [
                sum(w["action_histogram"][i] for w in witnesses) for i in range(7)
            ],
            "physical_effects": dict(sum(
                (Counter(w["events"]) for w in witnesses), Counter()
            )),
            "executed_action_hashes": [
                w["executed_action_sha256"] for w in witnesses
            ],
        }
        if sum(records[task]["action_histogram"]) != records[task]["native_actions"]:
            raise ValueError("M23 monitor physical action histogram mismatch")
        public_parts.append(encode_events(aggregate))
    result = {}
    for name in public_parts[0]:
        result[name] = np.concatenate([part[name] for part in public_parts])
    return result, records


def execute(input_root: str, output: str) -> dict:
    validate_split_contract()
    source_sha = admitted_source_commit()
    path = Path(output)
    if str(path) != f"{OUT}/summary.json" or path.exists():
        raise ValueError("M23 effect output must be new registered isolated file")
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    original, data = load_source_artifact(Path(input_root))
    if original["actual_native_actions"] != 16832:
        raise ValueError("M23 cached source action total mismatch")
    source_batches = [encode_events(data[t]) for t in SOURCE_ENV_IDS]
    if any("evaluator_native_effect" in batch for batch in source_batches):
        raise ValueError("M23 native effect oracle leaked into TRAIN")
    train = {
        k: np.concatenate([a[k] for a in source_batches])
        for k in source_batches[0]
    }
    weights = event_weights(train["proxy_labels"])
    monitor, recorded = collect_fresh_source_monitor()
    native_truth = monitor["evaluator_native_effect"]
    rule = event_metrics(native_truth, monitor["rule_predictions"])
    proxy = event_metrics(native_truth, monitor["proxy_labels"])
    positives = int((native_truth > 0).sum())
    negatives = int((native_truth == 0).sum())
    report = {
        "stage": "m23_native_effect_source_event_prediction",
        "source_commit": source_sha,
        "cached_train_source_commit": TRAIN_SOURCE_COMMIT,
        "cached_train_artifact_id": SOURCE_ARTIFACT_ID,
        "cached_train_artifact_sha256": SOURCE_ARTIFACT_SHA256,
        "cached_train_native_actions": original["actual_native_actions"],
        "cached_train_executed_interactions": len(train["proxy_labels"]),
        "new_source_monitor_native_actions": int(sum(
            x["native_actions"] for x in recorded.values()
        )),
        "new_source_monitor_interaction_actions": len(native_truth),
        "monitor_native_positive_effects": positives,
        "monitor_native_no_effect_actions": negatives,
        "monitor_native_source_witnesses": recorded,
        "monitor_public_proxy_agreement_with_native": proxy,
        "monitor_public_precondition_baseline": rule,
        "monitor_no_effect_baseline": event_metrics(
            native_truth, np.zeros_like(native_truth)
        ),
        "target_unlockpickup_actions": 0, "target_final_episodes": 0,
        "fresh_model_fits": 0, "actual_optimizer_step_calls": 0,
        "cached_sample_presentations": 0, "model_seeds": list(MODEL_SEEDS),
        "training_proxy_class_support": [
            int((train["proxy_labels"] == c).sum()) for c in range(N_EVENTS)
        ],
        "training_only_class_weights": [float(w) for w in weights],
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LR,
        "libraries": {
            k: version(k) for k in ("numpy", "torch", "gymnasium", "minigrid")
        },
        "paired_results": {},
    }
    for seed in MODEL_SEEDS:
        pair = {}
        for arm in ("aware", "masked"):
            model, trained = fit_event(
                train, seed, aware=arm == "aware", weights=weights
            )
            guess = predict(model, monitor, aware=arm == "aware")
            metrics = event_metrics(native_truth, guess)
            digest = save_numeric(
                model, path.parent / f"{arm}-source-effect-seed{seed}.npz"
            )
            pair[arm] = {
                **trained, **metrics, "checkpoint_sha256": digest,
            }
            report["fresh_model_fits"] += 1
            report["actual_optimizer_step_calls"] += trained["actual_optimizer_step_calls"]
            report["cached_sample_presentations"] += trained[
                "cached_train_sample_presentations"
            ]
        report["paired_results"][str(seed)] = pair
        print(json.dumps({
            "seed": seed, "aware_native_macro_f1": pair["aware"]["macro_f1"],
            "masked_native_macro_f1": pair["masked"]["macro_f1"],
            "rule_native_macro_f1": rule["macro_f1"],
            "native_positive_effects": positives,
            "optimizer_updates_this_seed": sum(
                x["actual_optimizer_step_calls"] for x in pair.values()
            ),
        }), flush=True)
    vals = list(report["paired_results"].values())
    aware_scores = np.asarray([v["aware"]["macro_f1"] for v in vals])
    masked_scores = np.asarray([v["masked"]["macro_f1"] for v in vals])
    differences = aware_scores - masked_scores
    advantage = aware_scores - rule["macro_f1"]
    recalls = np.asarray([
        v["aware"]["native_positive_event_recall"] or 0 for v in vals
    ])
    rng = np.random.default_rng(230515)
    draws = rng.integers(0, len(vals), size=(20000, len(vals)))
    report["mean_aware_minus_masked_f1"] = float(differences.mean())
    report["mean_aware_minus_rule_f1"] = float(advantage.mean())
    report["seeds_aware_over_masked_by_005"] = int((differences > 0.05).sum())
    report["seeds_aware_over_rule_by_005"] = int((advantage > 0.05).sum())
    report["mean_aware_true_native_effect_recall"] = float(recalls.mean())
    report["seed_bootstrap_95pct_aware_minus_rule"] = [
        float(x) for x in np.quantile(advantage[draws].mean(axis=1), [0.025, 0.975])
    ]
    report["registered_effect_gate_pass"] = bool(
        positives >= MIN_POSITIVE and negatives >= MIN_NEGATIVE
        and proxy["exact_event_accuracy"] >= 0.8
        and advantage.mean() > 0.05 and (advantage > 0.05).sum() >= 6
        and differences.mean() > 0.05 and (differences > 0.05).sum() >= 6
        and recalls.mean() >= 0.6
    )
    report["elapsed_seconds"] = time.perf_counter() - t0
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_sha": source_sha,
        "new_source_monitor_native_actions": report["new_source_monitor_native_actions"],
        "new_source_monitor_interaction_actions": len(native_truth),
        "fresh_model_fits": report["fresh_model_fits"],
        "actual_optimizer_step_calls": report["actual_optimizer_step_calls"],
        "mean_aware_minus_rule_f1": report["mean_aware_minus_rule_f1"],
        "mean_aware_minus_masked_f1": report["mean_aware_minus_masked_f1"],
        "registered_effect_gate_pass": report["registered_effect_gate_pass"],
    }), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=f"{OUT}/input")
    parser.add_argument("--output", default=f"{OUT}/summary.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("M23 source effects study requires explicit --execute")
    execute(args.input_root, args.output)


if __name__ == "__main__":
    main()
