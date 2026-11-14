"""M23 stage 4: source-only observation-history GRU world-model ablation.

Frozen comparison: class-balanced public MLP, one-frame GRU, eight-frame GRU.
Reuses original verified source TRAIN native actions. New MONITOR actions only
on registered source tasks; no UnlockPickup target episode is ever opened.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from torch import nn

from bagi.experiments.m1_goal_learning import admitted_source_commit
from bagi.experiments.m23_class_balance import (
    fit_weighted,
    per_class_interaction,
    training_object_weights,
)
from bagi.experiments.m23_public_world_model import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    N_ACTIONS,
    N_CELLS,
    SOURCE_ARTIFACT_ID,
    SOURCE_ARTIFACT_SHA256,
    TRAIN_SOURCE_COMMIT,
    WIDTH,
    combine,
    encode_public_rows,
    evaluate,
    load_source_artifact,
    save_numeric,
)
from bagi.experiments.m23_source_feasibility import SOURCE_ENV_IDS, run_source_episode
from bagi.experiments.m23_study_contract import (
    MEMORY_MONITOR_LAYOUTS,
    MODEL_SEEDS,
    validate_split_contract,
)

OUT = "runs/m23_belief_memory"
HISTORY_FRAMES = 8
HIDDEN = 96
MIN_INTERACTION_STEPS = 16


def encode_history(rows: dict) -> dict[str, np.ndarray]:
    """Build episode-safe public-view windows with RIGHT zero-padding.

    Last real (observation, executed action) of each window predicts the
    successor observation. No post-transition observation or unexecuted
    teacher action ever appears in a model input. Episode starts reset memory.
    """
    base = encode_public_rows(rows)
    n = len(base["features"])
    frames = np.zeros((n, HISTORY_FRAMES, WIDTH), dtype=np.float32)
    history: deque[np.ndarray] = deque(maxlen=HISTORY_FRAMES)
    for i, current in enumerate(base["features"]):
        history.append(current)
        window = np.asarray(history, dtype=np.float32)
        frames[i, :len(window), :] = window
        if bool(rows["terminated"][i]) or bool(rows["truncated"][i]):
            history.clear()
    if not np.array_equal(frames[:, 0, :], base["features"]):
        # Only the earliest step of an episode has current in slot zero.
        # This prevents treating padded windows as if all had length one.
        if np.any(np.count_nonzero(frames, axis=(1, 2)) == 0):
            raise ValueError("M23 sequence contained no public observation")
    return {**base, "sequence": frames}


class PublicGRUNextFrame(nn.Module):
    """Maintained torch.nn.GRU over current and previous public observations."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(WIDTH, HIDDEN), nn.ReLU())
        self.gru = nn.GRU(input_size=HIDDEN, hidden_size=HIDDEN, batch_first=True)
        self.obj_head = nn.Linear(HIDDEN, N_CELLS * len(OBJECT_TO_IDX))
        self.color_head = nn.Linear(HIDDEN, N_CELLS * len(COLOR_TO_IDX))
        self.state_head = nn.Linear(HIDDEN, N_CELLS * len(STATE_TO_IDX))
        self.direction_head = nn.Linear(HIDDEN, 4)
        self.reward_head = nn.Linear(HIDDEN, 1)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if sequence.ndim != 3 or sequence.shape[1] not in (1, HISTORY_FRAMES):
            raise ValueError("M23 GRU input must have one or eight public frames")
        if sequence.shape[-1] != WIDTH:
            raise ValueError("M23 GRU public feature dimensions changed")
        # RIGHT padding is ignored at the position of the last real input:
        # GRU is causal so later padding cannot affect that hidden state.
        present = (sequence.abs().sum(dim=-1) > 0)
        lengths = present.sum(dim=-1)
        if torch.any(lengths == 0):
            raise ValueError("M23 GRU cannot predict without a real observation")
        expected = torch.arange(sequence.shape[1], device=sequence.device)[None, :]
        if not torch.all(present == (expected < lengths[:, None])):
            raise ValueError("M23 GRU windows must be contiguous, right-padded")
        encoded = self.projection(sequence)
        outputs, _ = self.gru(encoded)
        selected = outputs[
            torch.arange(len(lengths), device=outputs.device), lengths - 1
        ]
        return (
            self.obj_head(selected).reshape(-1, N_CELLS, len(OBJECT_TO_IDX)),
            self.color_head(selected).reshape(-1, N_CELLS, len(COLOR_TO_IDX)),
            self.state_head(selected).reshape(-1, N_CELLS, len(STATE_TO_IDX)),
            self.direction_head(selected),
            self.reward_head(selected).flatten(),
        )


def fit_gru(data: dict[str, np.ndarray], seed: int, weights: np.ndarray,
            frames: int) -> tuple[PublicGRUNextFrame, dict]:
    """One-step GRU versus eight-step GRU: identical optimizer and initialization."""
    if frames not in (1, HISTORY_FRAMES):
        raise ValueError("M23 GRU study registers only 1- or 8-frame history")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    model = PublicGRUNextFrame()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    sequence = data["features"][:, None, :] if frames == 1 else data["sequence"]
    features = torch.from_numpy(np.asarray(sequence, dtype=np.float32).copy())
    targets = {
        key: torch.from_numpy(data[key].copy()).to(
            torch.float32 if key == "next_reward" else torch.int64
        )
        for key in ("next_obj", "next_color", "next_state",
                    "next_direction", "next_reward")
    }
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32)
    n = len(features)
    calls = 0
    last_loss = None
    model.train()
    for _ in range(EPOCHS):
        shuffled = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = torch.as_tensor(shuffled[start:start + BATCH_SIZE])
            predictions = model(features[idx])
            losses = (
                torch.nn.functional.cross_entropy(
                    predictions[0].flatten(0, 1),
                    targets["next_obj"][idx].flatten(), weight=weight_tensor
                ),
                torch.nn.functional.cross_entropy(
                    predictions[1].flatten(0, 1),
                    targets["next_color"][idx].flatten(),
                ),
                torch.nn.functional.cross_entropy(
                    predictions[2].flatten(0, 1),
                    targets["next_state"][idx].flatten(),
                ),
                torch.nn.functional.cross_entropy(
                    predictions[3], targets["next_direction"][idx],
                ),
                torch.nn.functional.mse_loss(
                    predictions[4], targets["next_reward"][idx],
                ),
            )
            loss = sum(losses[:3]) + 0.2 * losses[3] + 0.05 * losses[4]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            calls += 1
            last_loss = float(loss.detach().cpu())
    return model.eval(), {
        "actual_optimizer_step_calls": calls,
        "training_sample_presentations": EPOCHS * n,
        "last_loss": last_loss,
    }


def source_memory_monitor() -> tuple[dict[str, np.ndarray], dict]:
    """Execute only new source MONITOR seeds, with external physical auditing."""
    batches = []
    scored = {}
    for task, task_id in SOURCE_ENV_IDS.items():
        public = {}
        witnesses = []
        for seed in MEMORY_MONITOR_LAYOUTS[task]:
            witness, rows = run_source_episode(task, seed, split="memory_monitor")
            if not bool(rows["terminated"][-1] or rows["truncated"][-1]):
                raise ValueError("M23 source MONITOR episode missing terminal marker")
            witnesses.append(witness)
            for name, values in rows.items():
                public.setdefault(name, []).extend(values)
        native_actions = sum(w["native_actions"] for w in witnesses)
        histogram = [
            sum(w["action_histogram"][j] for w in witnesses)
            for j in range(N_ACTIONS)
        ]
        if sum(histogram) != native_actions:
            raise ValueError("M23 native action tally mismatch")
        effects = {}
        for record in witnesses:
            for k, count in record["events"].items():
                effects[k] = effects.get(k, 0) + int(count)
        scored[task] = {
            "native_task_id": task_id,
            "episodes": len(witnesses),
            "new_native_actions": native_actions,
            "native_action_histogram": histogram,
            "physical_effects": effects,
            "executed_action_hashes": [
                w["executed_action_sha256"] for w in witnesses
            ],
        }
        batches.append(encode_history(public))
    return combine(batches), scored


def _without_sequence(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: v for k, v in data.items() if k != "sequence"}


def _f1_differences(pairs: dict, numerator: str, denominator: str) -> np.ndarray:
    return np.asarray([
        v[numerator]["interaction_macro_f1"]
        - v[denominator]["interaction_macro_f1"]
        for v in pairs.values()
    ], dtype=np.float64)


def execute(input_root: str, output: str) -> dict:
    validate_split_contract()
    sha = admitted_source_commit()
    path = Path(output)
    if str(path) != f"{OUT}/summary.json" or path.exists():
        raise ValueError("M23 belief output must be new and isolated")
    path.parent.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    original, trajectories = load_source_artifact(Path(input_root))
    if original["actual_native_actions"] != 16832:
        raise ValueError("M23 cached native source action count mismatch")
    for task in SOURCE_ENV_IDS:
        if not bool(trajectories[task]["terminated"][-1]
                    or trajectories[task]["truncated"][-1]):
            raise ValueError("M23 source TRAIN episode boundary incomplete")
    train = combine([encode_history(trajectories[t]) for t in SOURCE_ENV_IDS])
    monitor, source_witnesses = source_memory_monitor()
    without_seq_train = _without_sequence(train)
    without_seq_monitor = _without_sequence(monitor)
    weights = training_object_weights(without_seq_train)
    interaction_steps = int(np.isin(monitor["action"], [3, 4, 5]).sum())
    report = {
        "stage": "m23_source_observation_history_ablation",
        "research_commit": sha,
        "historical_training_commit": TRAIN_SOURCE_COMMIT,
        "historical_training_artifact_id": SOURCE_ARTIFACT_ID,
        "historical_training_artifact_sha256": SOURCE_ARTIFACT_SHA256,
        "cached_train_executed_actions": original["actual_native_actions"],
        "new_native_monitor_actions": sum(
            v["new_native_actions"] for v in source_witnesses.values()
        ),
        "new_native_monitor_witnesses": source_witnesses,
        "new_native_interaction_steps": interaction_steps,
        "target_unlockpickup_actions": 0,
        "target_final_episodes": 0,
        "actual_imagined_planner_actions": 0,
        "fresh_model_fits": 0, "optimizer_step_calls": 0,
        "training_sample_presentations": 0,
        "source_only_object_class_weights": [float(w) for w in weights],
        "history_window_length": HISTORY_FRAMES,
        "gru_hidden_dim": HIDDEN, "model_seeds": list(MODEL_SEEDS),
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LR,
        "versions": {
            k: version(k) for k in ("numpy", "torch", "minigrid", "gymnasium")
        },
        "paired_seeds": {},
    }
    for seed in MODEL_SEEDS:
        outcomes = {}
        for mode in ("balanced_mlp", "gru1", "gru8"):
            if mode == "balanced_mlp":
                model, train_info = fit_weighted(without_seq_train, seed, weights)
                eval_batch = without_seq_monitor
                updates = train_info["optimizer_step_calls"]
            else:
                length = 1 if mode == "gru1" else HISTORY_FRAMES
                model, train_info = fit_gru(train, seed, weights, length)
                eval_batch = {
                    **without_seq_monitor,
                    "features": (
                        monitor["features"][:, None, :] if length == 1
                        else monitor["sequence"]
                    ),
                }
                updates = train_info["actual_optimizer_step_calls"]
            score = evaluate(model, eval_batch)
            per_class = per_class_interaction(model, eval_batch)
            checkpoint = path.parent / f"{mode}-seed{seed}.npz"
            digest = save_numeric(model, checkpoint)
            outcomes[mode] = {
                **train_info, **score, "checkpoint_sha256": digest,
                "per_class_interaction": per_class,
            }
            report["fresh_model_fits"] += 1
            report["optimizer_step_calls"] += updates
            report["training_sample_presentations"] += (
                train_info["training_sample_presentations"]
            )
        report["paired_seeds"][str(seed)] = outcomes
        print(json.dumps({
            "seed": seed,
            "balanced_mlp_f1": outcomes["balanced_mlp"]["interaction_macro_f1"],
            "gru1_f1": outcomes["gru1"]["interaction_macro_f1"],
            "gru8_f1": outcomes["gru8"]["interaction_macro_f1"],
            "persistence_f1": outcomes["gru8"]["persistence_interaction_macro_f1"],
            "updates_this_seed": sum(
                (v.get("actual_optimizer_step_calls")
                 or v.get("optimizer_step_calls")) for v in outcomes.values()
            ),
        }), flush=True)
    pairs = report["paired_seeds"]
    gains_memory = _f1_differences(pairs, "gru8", "gru1")
    gains_mlp = _f1_differences(pairs, "gru8", "balanced_mlp")
    baseline_gains = np.asarray([
        v["gru8"]["interaction_macro_f1"]
        - v["gru8"]["persistence_interaction_macro_f1"]
        for v in pairs.values()
    ])
    report["mean_gru8_minus_gru1_f1"] = float(gains_memory.mean())
    report["mean_gru8_minus_mlp_f1"] = float(gains_mlp.mean())
    report["mean_gru8_minus_persistence_f1"] = float(baseline_gains.mean())
    report["seeds_gru8_over_gru1_by_gt_005"] = int((gains_memory > 0.05).sum())
    report["seeds_gru8_over_mlp_by_gt_005"] = int((gains_mlp > 0.05).sum())
    report["seeds_gru8_over_persistence"] = int((baseline_gains > 0).sum())
    rng = np.random.default_rng(230408)
    resamples = rng.integers(0, len(MODEL_SEEDS), size=(20000, len(MODEL_SEEDS)))
    report["seed_bootstrap_95pct_gru8_minus_gru1"] = [
        float(v) for v in np.quantile(gains_memory[resamples].mean(axis=1), [0.025, 0.975])
    ]
    report["registered_memory_gate_pass"] = bool(
        interaction_steps >= MIN_INTERACTION_STEPS
        and gains_memory.mean() > 0.05
        and (gains_memory > 0.05).sum() >= 6
        and gains_mlp.mean() > 0.05
        and (gains_mlp > 0.05).sum() >= 6
        and (baseline_gains > 0).sum() >= 6
    )
    report["elapsed_seconds"] = time.perf_counter() - began
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "research_commit": sha,
        "fresh_model_fits": report["fresh_model_fits"],
        "optimizer_step_calls": report["optimizer_step_calls"],
        "cached_train_executed_actions": report["cached_train_executed_actions"],
        "new_native_monitor_actions": report["new_native_monitor_actions"],
        "new_native_interaction_steps": interaction_steps,
        "mean_gru8_minus_gru1_f1": report["mean_gru8_minus_gru1_f1"],
        "mean_gru8_minus_mlp_f1": report["mean_gru8_minus_mlp_f1"],
        "registered_memory_gate_pass": report["registered_memory_gate_pass"],
    }), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=f"{OUT}/input")
    parser.add_argument("--output", default=f"{OUT}/summary.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("M23 source memory study requires explicit --execute")
    execute(args.input_root, args.output)


if __name__ == "__main__":
    main()
