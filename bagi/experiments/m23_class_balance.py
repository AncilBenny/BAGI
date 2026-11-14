"""M23 stage 3: source-only rare-class loss ablation, no composition target access.

Compare the original next-frame model loss with a TRAIN-derived weighted
object cross entropy. Source TRAIN data is immutable and cached; independent
source balance MONITOR executes new actual actions exactly once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from minigrid.core.constants import OBJECT_TO_IDX
from torch import nn

from bagi.experiments.m1_goal_learning import admitted_source_commit
from bagi.experiments.m23_public_world_model import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    N_OBJECT,
    SOURCE_ARTIFACT_ID,
    SOURCE_ARTIFACT_SHA256,
    TRAIN_SOURCE_COMMIT,
    PublicNextFramePredictor,
    combine,
    encode_public_rows,
    evaluate,
    fit_one,
    load_source_artifact,
    save_numeric,
)
from bagi.experiments.m23_source_feasibility import SOURCE_ENV_IDS, run_source_episode
from bagi.experiments.m23_study_contract import (
    BALANCE_MONITOR_LAYOUTS,
    MODEL_SEEDS,
    validate_split_contract,
)

OUT = "runs/m23_class_balance"
MIN_INTERACTION_STEPS = 16


def training_object_weights(rows: dict[str, np.ndarray]) -> np.ndarray:
    """Inverse-sqrt frequency, train-only, count-weight-mean normalized."""
    obj = np.asarray(rows["next_obj"], dtype=np.int64)
    if obj.ndim != 2 or obj.shape[1] != 49 or len(obj) == 0:
        raise ValueError("M23 object-weight input must be real 7x7 labels")
    if np.any(obj < 0) or np.any(obj >= N_OBJECT):
        raise ValueError("M23 object label out of range")
    counts = np.bincount(obj.ravel(), minlength=N_OBJECT).astype(np.float64)
    present = counts > 0
    if not np.any(present):
        raise ValueError("M23 empty object distribution")
    raw = np.zeros(N_OBJECT, dtype=np.float64)
    raw[present] = np.sqrt(counts.sum() / counts[present])
    scale = (raw * counts).sum() / counts.sum()
    raw[present] = np.clip(raw[present] / scale, 0.25, 8.0)
    return raw.astype(np.float32)


def fit_weighted(rows: dict[str, np.ndarray], seed: int,
                 object_weights: np.ndarray) -> tuple[PublicNextFramePredictor, dict]:
    """Exact stage-2 training loop, changing ONLY the next-object CE weights."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    model = PublicNextFramePredictor()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    labels = {
        k: torch.from_numpy(v.copy()).to(
            torch.float32 if k in ("features", "next_reward") else torch.int64
        )
        for k, v in rows.items() if k not in ("prior_obj", "action")
    }
    weight = torch.as_tensor(object_weights, dtype=torch.float32)
    updates = 0
    final_loss = None
    n = len(rows["features"])
    model.train()
    for _ in range(EPOCHS):
        shuffled = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            indices = torch.as_tensor(shuffled[start:start + BATCH_SIZE])
            outputs = model(labels["features"][indices])
            losses = (
                torch.nn.functional.cross_entropy(
                    outputs[0].flatten(0, 1),
                    labels["next_obj"][indices].flatten(), weight=weight,
                ),
                torch.nn.functional.cross_entropy(
                    outputs[1].flatten(0, 1), labels["next_color"][indices].flatten(),
                ),
                torch.nn.functional.cross_entropy(
                    outputs[2].flatten(0, 1), labels["next_state"][indices].flatten(),
                ),
                torch.nn.functional.cross_entropy(
                    outputs[3], labels["next_direction"][indices],
                ),
                torch.nn.functional.mse_loss(
                    outputs[4], labels["next_reward"][indices],
                ),
            )
            loss = sum(losses[:3]) + 0.2 * losses[3] + 0.05 * losses[4]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            updates += 1
            final_loss = float(loss.detach().cpu())
    return model.eval(), {
        "optimizer_step_calls": updates,
        "training_sample_presentations": EPOCHS * n,
        "last_loss": final_loss,
    }


def source_balance_monitor() -> tuple[dict[str, np.ndarray], dict]:
    """Real public source MONITOR data, never target, frozen per preregistration."""
    batches = []
    report = {}
    for task in SOURCE_ENV_IDS:
        cumulative = {}
        summaries = []
        for layout in BALANCE_MONITOR_LAYOUTS[task]:
            scored, observed = run_source_episode(task, layout, split="balance_monitor")
            summaries.append(scored)
            for k, values in observed.items():
                cumulative.setdefault(k, []).extend(values)
        total = sum(x["native_actions"] for x in summaries)
        action_histogram = [
            sum(x["action_histogram"][i] for x in summaries) for i in range(7)
        ]
        if sum(action_histogram) != total:
            raise ValueError("M23 source balance native action count inconsistent")
        events = {}
        for scored in summaries:
            for name, count in scored["events"].items():
                events[name] = events.get(name, 0) + int(count)
        report[task] = {
            "episodes": len(summaries),
            "actual_native_actions": total,
            "action_histogram": action_histogram,
            "real_physical_effects": events,
            "executed_action_hashes": [s["executed_action_sha256"] for s in summaries],
        }
        batches.append(encode_public_rows(cumulative))
    return combine(batches), report


@torch.no_grad()
def per_class_interaction(model: nn.Module, rows: dict[str, np.ndarray]) -> dict:
    """Additional fixed analysis; no tuning decisions or evaluator state."""
    pieces = []
    for start in range(0, len(rows["features"]), BATCH_SIZE):
        outputs = model(torch.from_numpy(rows["features"][start:start + BATCH_SIZE]))
        pieces.append(outputs[0].argmax(-1).numpy())
    predictions = np.concatenate(pieces)
    actual = rows["next_obj"]
    eligible = np.isin(rows["action"], [3, 4, 5])
    out = {}
    for name in ("door", "key", "box", "ball", "wall"):
        i = OBJECT_TO_IDX[name]
        t = actual[eligible] == i
        p = predictions[eligible] == i
        tp = int((t & p).sum())
        fp = int((~t & p).sum())
        fn = int((t & ~p).sum())
        out[name] = {
            "support": int(t.sum()), "predicted_support": int(p.sum()),
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": (tp / (tp + fp) if tp + fp else None),
            "recall": (tp / (tp + fn) if tp + fn else None),
        }
    return out


def execute(input_root: str, output: str) -> dict:
    validate_split_contract()
    source_sha = admitted_source_commit()
    path = Path(output)
    if str(path) != f"{OUT}/summary.json" or path.exists():
        raise ValueError("M23 balance output must be new at the isolated registered path")
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    original, data = load_source_artifact(Path(input_root))
    if original["actual_native_actions"] != 16832:
        raise ValueError("M23 training artifact source-action count mismatch")
    train = combine([encode_public_rows(data[task]) for task in SOURCE_ENV_IDS])
    monitor, physical = source_balance_monitor()
    weight = training_object_weights(train)
    interactions = int(np.isin(monitor["action"], [3, 4, 5]).sum())
    report = {
        "study": "m23_train_only_balanced_next_object_ablation",
        "source_commit": source_sha, "source_dataset_commit": TRAIN_SOURCE_COMMIT,
        "source_artifact_id": SOURCE_ARTIFACT_ID,
        "source_artifact_sha256": SOURCE_ARTIFACT_SHA256,
        "cached_training_native_actions": original["actual_native_actions"],
        "new_source_monitor_native_actions": sum(
            v["actual_native_actions"] for v in physical.values()
        ),
        "source_monitor_effects": physical,
        "native_interaction_action_steps": interactions,
        "target_env_actions": 0, "target_final_episodes": 0,
        "model_seeds": list(MODEL_SEEDS), "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LR, "fresh_model_fits": 0, "optimizer_step_calls": 0,
        "training_sample_presentations": 0,
        "training_only_object_weights": [float(x) for x in weight],
        "library_versions": {k: version(k)
                             for k in ("numpy", "torch", "minigrid", "gymnasium")},
        "pairs": {},
    }
    for seed in MODEL_SEEDS:
        pair = {}
        for arm in ("ordinary", "balanced_object"):
            model, training = (
                fit_one(train, seed) if arm == "ordinary"
                else fit_weighted(train, seed, weight)
            )
            metrics = evaluate(model, monitor)
            detailed = per_class_interaction(model, monitor)
            checkpoint = path.parent / f"{arm}-seed{seed}.npz"
            digest = save_numeric(model, checkpoint)
            pair[arm] = {
                **training, **metrics,
                "per_class_interaction": detailed,
                "checkpoint_sha256": digest,
            }
            report["fresh_model_fits"] += 1
            report["optimizer_step_calls"] += training["optimizer_step_calls"]
            report["training_sample_presentations"] += training["training_sample_presentations"]
        report["pairs"][str(seed)] = pair
        print(json.dumps({
            "model_seed": seed,
            "ordinary_f1": pair["ordinary"]["interaction_macro_f1"],
            "balanced_f1": pair["balanced_object"]["interaction_macro_f1"],
            "persistence_f1": pair["ordinary"]["persistence_interaction_macro_f1"],
            "optimizer_updates_per_arm": pair["ordinary"]["optimizer_step_calls"],
            "native_interaction_steps": interactions,
        }), flush=True)
    advantages = np.array([
        v["balanced_object"]["interaction_macro_f1"]
        - v["ordinary"]["interaction_macro_f1"]
        for v in report["pairs"].values()
    ])
    acc_diffs = np.array([
        v["balanced_object"]["next_object_cell_accuracy"]
        - v["ordinary"]["next_object_cell_accuracy"]
        for v in report["pairs"].values()
    ])
    rng = np.random.default_rng(230316)
    draw = rng.integers(0, len(advantages), size=(20000, len(advantages)))
    boot = advantages[draw].mean(axis=1)
    report["mean_balanced_minus_ordinary_macro_f1"] = float(advantages.mean())
    report["positive_seed_count_above_0_05"] = int((advantages > 0.05).sum())
    report["mean_balanced_minus_ordinary_cell_accuracy"] = float(acc_diffs.mean())
    report["seed_bootstrap_95pct_f1_difference"] = [
        float(x) for x in np.quantile(boot, [0.025, 0.975])
    ]
    report["registered_balance_gate_pass"] = bool(
        interactions >= MIN_INTERACTION_STEPS
        and advantages.mean() > 0.05
        and (advantages > 0.05).sum() >= 6
        and acc_diffs.mean() >= -0.05
    )
    report["elapsed_seconds"] = time.perf_counter() - start
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_sha": source_sha,
        "fresh_model_fits": report["fresh_model_fits"],
        "optimizer_step_calls": report["optimizer_step_calls"],
        "new_source_monitor_native_actions": report["new_source_monitor_native_actions"],
        "mean_balanced_minus_ordinary_macro_f1":
            report["mean_balanced_minus_ordinary_macro_f1"],
        "registered_balance_gate_pass": report["registered_balance_gate_pass"],
    }), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=f"{OUT}/input")
    parser.add_argument("--output", default=f"{OUT}/summary.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("M23 class-balance requires explicit --execute")
    execute(args.input_root, args.output)


if __name__ == "__main__":
    main()
