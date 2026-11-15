"""M23 Stage11: native-source public vision and rare-action balanced BC factorial.

Only native PUBLIC observations and own executed-action-derived public memory may
enter the policies. Private physical state stays inside independent auditors.
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
from torch import nn

from bagi.experiments.m1_goal_learning import admitted_source_commit
from bagi.experiments.m13_public_interaction_coverage import Controller
from bagi.experiments.m15_public_action_bc import (
    N_COLOR,
    N_OBJECT,
    N_STATE,
    class_weights,
    features,
    save_numeric,
)
from bagi.experiments.m23_source_mission_bc import make_native_audit, make_source_actor
from bagi.experiments.m23_source_feasibility import HORIZON
from bagi.experiments.m23_study_contract import (
    VISION_MONITOR_LAYOUTS,
    VISION_TRAIN_LAYOUTS,
    validate_split_contract,
)

OUT = "runs/m23_source_public_vision_bc"
TASKS = ("doorkey", "fetch")
ARMS = ("front_unweighted", "front_balanced", "image_unweighted", "image_balanced")
MODEL_SEEDS = (941, 947, 953)
EPOCHS = 12
BATCH = 128
LR = 0.003
CELL = N_OBJECT + N_COLOR + N_STATE
IMAGE_WIDTH = 49 * CELL
WIDTH = IMAGE_WIDTH + 64 + 4 + 1
FRONT_INDEX = 3 * 7 + 5
BOOTSTRAP_SEED = 231101
MIN_GAIN = 12
ACTOR_MARGIN = 24
MIN_DOOR_OPEN_EPISODES = 16


def public_visual_features(obs: dict, memory: bool) -> np.ndarray:
    """One-hot native egocentric 7x7 PUBLIC image; not the hidden world grid."""
    if not isinstance(obs, dict) or set(obs) != {"image", "mission", "direction"}:
        raise ValueError("Stage11 feature input must have exact native public fields")
    image = np.asarray(obs["image"])
    if image.shape != (7, 7, 3) or image.dtype.kind not in "iu":
        raise ValueError("Stage11 public image must have native integer shape")
    if (np.any(image[:, :, 0] >= N_OBJECT)
            or np.any(image[:, :, 1] >= N_COLOR)
            or np.any(image[:, :, 2] >= N_STATE)):
        raise ValueError("Stage11 native public image category out of range")
    onehot = np.concatenate([
        np.eye(N_OBJECT, dtype=np.float32)[image[:, :, 0]],
        np.eye(N_COLOR, dtype=np.float32)[image[:, :, 1]],
        np.eye(N_STATE, dtype=np.float32)[image[:, :, 2]],
    ], axis=-1).reshape(-1)
    # M15's suffix is exactly the 64-dimensional PUBLIC mission encoding,
    # four-dimensional public heading and one-bit visible pickup memory.
    suffix = features(obs, memory)[CELL:]
    vector = np.concatenate([onehot, suffix]).astype(np.float32, copy=False)
    if vector.shape != (WIDTH,) or not np.isfinite(vector).all():
        raise ValueError("Stage11 public feature dimension invalid")
    return vector


def restrict_to_front(x: np.ndarray) -> np.ndarray:
    """Ablate all but public front cell while preserving identical model width."""
    if x.ndim not in (1, 2) or x.shape[-1] != WIDTH:
        raise ValueError("Stage11 front ablation requires registered feature width")
    output = np.array(x, dtype=np.float32, copy=True)
    grid = output[..., :IMAGE_WIDTH].reshape(*output.shape[:-1], 49, CELL)
    grid[..., :FRONT_INDEX, :] = 0
    grid[..., FRONT_INDEX + 1:, :] = 0
    return output


class VisualBC(nn.Module):
    """Maintained M15 two-layer tanh MLP, width fixed across four arms."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(WIDTH, 64), nn.Tanh(), nn.Linear(64, 7)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != WIDTH:
            raise ValueError("Stage11 policy saw wrong public input width")
        return self.layers(x)


def collect_episode(task: str, layout: int) -> dict:
    if task not in TASKS or layout not in VISION_TRAIN_LAYOUTS[task]:
        raise ValueError("Stage11 source teacher restricted to fresh TRAIN layouts")
    actor = make_source_actor(task)
    rng = np.random.default_rng(8_800_000 + layout)
    audit = make_native_audit(task)
    xs: list[np.ndarray] = []
    actions: list[int] = []
    finished = False
    try:
        obs, _ = audit.reset(seed=layout)
        for _ in range(HORIZON):
            xs.append(public_visual_features(obs, actor.key_observed_after_pickup))
            action = int(actor.choose(obs, rng))
            if action not in range(7):
                raise ValueError("Stage11 source teacher illegal native action")
            successor, _reward, terminated, truncated, _ = audit.step(action)
            actor.update(obs, action, successor)
            actions.append(action)
            obs = successor
            if terminated or truncated:
                finished = True
                break
        truth = audit.summary()
        if (not finished or len(actions) != truth["native_actions"]
                or len(xs) != len(actions)
                or sum(truth["action_counts"].values()) != len(actions)):
            raise ValueError("Stage11 source examples not matched to native steps")
        return {
            "task": task, "layout_seed": layout,
            "x": np.stack(xs).astype(np.float32),
            "y": np.asarray(actions, dtype=np.int64),
            "actual_native_actions": len(actions),
            "actor_rng_seed": 8_800_000 + layout,
            "native_action_histogram": [truth["action_counts"][str(i)] for i in range(7)],
            "executed_action_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
            "external_physical_full_goal": bool(truth["full_physical_goal"]),
        }
    finally:
        audit.env.close()


def acquire_train() -> tuple[np.ndarray, np.ndarray, dict]:
    validate_split_contract()
    episodes = [collect_episode(task, layout)
                for task in TASKS for layout in VISION_TRAIN_LAYOUTS[task]]
    xs = np.concatenate([row["x"] for row in episodes])
    ys = np.concatenate([row["y"] for row in episodes])
    if len(xs) != len(ys) or not len(ys) or not np.isfinite(xs).all():
        raise ValueError("Stage11 actual native source feature/label error")
    if len(ys) != sum(row["actual_native_actions"] for row in episodes):
        raise ValueError("Stage11 actual source native actions not conserved")
    return xs, ys, {
        "native_train_episodes": len(episodes),
        "actual_native_train_actions": len(ys),
        "actual_native_action_histogram": np.bincount(ys, minlength=7).tolist(),
        "public_full_x_sha256": hashlib.sha256(xs.tobytes()).hexdigest(),
        "executed_action_y_sha256": hashlib.sha256(ys.tobytes()).hexdigest(),
        "episodes": [
            {key: value for key, value in row.items() if key not in ("x", "y")}
            for row in episodes
        ],
    }


def fit_model(x: np.ndarray, y: np.ndarray, arm: str,
              seed: int) -> tuple[VisualBC, dict]:
    if arm not in ARMS or seed not in MODEL_SEEDS:
        raise ValueError("Stage11 fit arm/model seed not preregistered")
    if (x.ndim != 2 or x.shape[1] != WIDTH or len(x) != len(y) or not len(y)
            or set(np.unique(y)) - set(range(7))):
        raise ValueError("Stage11 native public TRAIN input/labels invalid")
    inputs = restrict_to_front(x) if arm.startswith("front_") else x.copy()
    weights = class_weights(y) if arm.endswith("_balanced") else None
    torch.manual_seed(seed)
    model = VisualBC()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    source = torch.from_numpy(inputs)
    labels = torch.from_numpy(y)
    generator = torch.Generator().manual_seed(10_000 + seed)
    updates = 0
    presentations = 0
    losses = 0.0
    started = time.perf_counter()
    for _ in range(EPOCHS):
        for indices in torch.randperm(len(y), generator=generator).split(BATCH):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(
                model(source[indices]), labels[indices], weight=weights
            )
            loss.backward()
            opt.step()
            updates += 1
            presentations += len(indices)
            losses += float(loss.detach())
    if presentations != EPOCHS * len(y):
        raise ValueError("Stage11 cached presentations not conserved")
    model.eval()
    return model, {
        "arm": arm, "seed": seed, "epochs": EPOCHS, "batch_size": BATCH,
        "adam_learning_rate": LR, "architecture": "M15_two_layer_Tanh_hidden64",
        "input_width": WIDTH, "public_image_cells": 1 if arm.startswith("front_") else 49,
        "class_balanced": weights is not None,
        "class_weights": weights.tolist() if weights is not None else None,
        "actual_optimizer_step_calls": updates,
        "cached_native_source_label_presentations": presentations,
        "mean_minibatch_loss": losses / updates,
        "elapsed_fit_seconds": time.perf_counter() - started,
        "public_train_x_sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
    }


def monitor_episode(task: str, layout: int, arm: str,
                    seed: int | None, model: VisualBC | None) -> dict:
    if task not in TASKS or layout not in VISION_MONITOR_LAYOUTS[task]:
        raise ValueError("Stage11 MONITOR only allows fresh registered source layouts")
    if arm == "actor_only":
        if seed is not None or model is not None:
            raise ValueError("Stage11 actor_only must run once per source layout")
        actor = make_source_actor(task)
    elif arm in ARMS and seed in MODEL_SEEDS and model is not None:
        actor = Controller("public_memory")
    else:
        raise ValueError("Stage11 source policy/evaluation mode invalid")
    rng = np.random.default_rng(8_800_000 + layout)
    audit = make_native_audit(task)
    actions: list[int] = []
    reward_sum = 0.0
    finished = False
    try:
        obs, _ = audit.reset(seed=layout)
        for _ in range(HORIZON):
            if model is None:
                action = int(actor.choose(obs, rng))
            else:
                vector = public_visual_features(obs, actor.key_observed_after_pickup)
                if arm.startswith("front_"):
                    vector = restrict_to_front(vector)
                with torch.no_grad():
                    action = int(torch.argmax(
                        model(torch.from_numpy(vector).unsqueeze(0)), dim=1
                    ).item())
            if action not in range(7):
                raise ValueError("Stage11 policy proposed illegal native action")
            next_obs, reward, terminated, truncated, _ = audit.step(action)
            actor.update(obs, action, next_obs)
            actions.append(action)
            reward_sum += float(reward)
            obs = next_obs
            if terminated or truncated:
                finished = True
                break
        truth = audit.summary()
        if (not finished or len(actions) != truth["native_actions"]
                or sum(truth["action_counts"].values()) != len(actions)):
            raise ValueError("Stage11 actual native MONITOR steps not conserved")
        return {
            "task": task, "layout_seed": layout, "arm": arm, "model_seed": seed,
            "actual_native_actions": len(actions),
            "native_action_histogram": [truth["action_counts"][str(i)] for i in range(7)],
            "executed_action_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
            "actual_reward_sum": reward_sum,
            "external_physical_full_goal": bool(truth["full_physical_goal"]),
            "external_rewarded_terminations": truth["rewarded_terminations"],
            "external_effective_pickup_events": truth["effective_pickup_events"],
            "external_effective_door_open_events": (
                truth["effective_door_open_events"] if task == "doorkey" else 0
            ),
            "external_effective_forward_movements":
                truth.get("effective_forward_movements", 0),
            "external_wrong_target_pickup_events":
                truth.get("wrong_target_pickup_events", 0),
        }
    finally:
        audit.env.close()


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Stage11 must have native MONITOR rows")
    count = sum(row["actual_native_actions"] for row in rows)
    hist = [sum(row["native_action_histogram"][i] for row in rows) for i in range(7)]
    if sum(hist) != count:
        raise ValueError("Stage11 native MONITOR histogram mismatch")
    return {
        "episodes": len(rows), "actual_native_actions": count,
        "native_action_histogram": hist,
        "strict_full_physical_goals": sum(
            int(row["external_physical_full_goal"]) for row in rows
        ),
        "reward_positive_episodes": sum(
            int(row["external_rewarded_terminations"] > 0) for row in rows
        ),
        "reward_only_nonqualifying_episodes": sum(
            int(row["external_rewarded_terminations"] > 0
                and not row["external_physical_full_goal"]) for row in rows
        ),
        "effective_door_open_episodes": sum(
            int(row["external_effective_door_open_events"] > 0) for row in rows
        ),
        "actual_effective_pickup_events": sum(
            row["external_effective_pickup_events"] for row in rows
        ),
    }


def acquire_monitor(models: dict[tuple[str, int], VisualBC]) -> dict:
    if set(models) != {(a, s) for a in ARMS for s in MODEL_SEEDS}:
        raise ValueError("Stage11 requires twelve trained source policies")
    groups = {}
    for task in TASKS:
        layouts = VISION_MONITOR_LAYOUTS[task]
        actor = [monitor_episode(task, layout, "actor_only", None, None)
                 for layout in layouts]
        groups[f"{task}:actor_only"] = {
            "episodes": actor, "summary": summarize(actor)
        }
        for arm in ARMS:
            for seed in MODEL_SEEDS:
                rows = [
                    monitor_episode(task, layout, arm, seed, models[(arm, seed)])
                    for layout in layouts
                ]
                groups[f"{task}:{arm}:{seed}"] = {
                    "episodes": rows, "summary": summarize(rows)
                }
    return groups


def decide(monitor: dict) -> dict:
    required = {f"{task}:actor_only" for task in TASKS} | {
        f"{task}:{arm}:{seed}"
        for task in TASKS for arm in ARMS for seed in MODEL_SEEDS
    }
    if set(monitor) != required:
        raise ValueError("Stage11 independent paired source modes incomplete")
    paired = {}
    task_gains = {}
    seed_gains = {seed: 0 for seed in MODEL_SEEDS}
    contrasts = {
        "image_unweighted_minus_front_unweighted": 0,
        "front_balanced_minus_front_unweighted": 0,
        "image_balanced_minus_image_unweighted": 0,
        "factorial_interaction": 0,
    }
    changed_actor, changed_front = {}, {}
    actor_goals = treated_goals = opens = 0
    for task in TASKS:
        layouts = list(VISION_MONITOR_LAYOUTS[task])
        actor = monitor[f"{task}:actor_only"]["episodes"]
        if [row["layout_seed"] for row in actor] != layouts:
            raise ValueError("Stage11 source actor layout pairing mismatch")
        actor_goals += sum(int(row["external_physical_full_goal"]) for row in actor)
        changed_actor[task] = {}
        changed_front[task] = {}
        task_rows = []
        for seed in MODEL_SEEDS:
            rows = {}
            for arm in ARMS:
                rows[arm] = monitor[f"{task}:{arm}:{seed}"]["episodes"]
                if [row["layout_seed"] for row in rows[arm]] != layouts:
                    raise ValueError("Stage11 learned policy layout pairing mismatch")
            a, b, c, d = [
                np.asarray(
                    [int(row["external_physical_full_goal"]) for row in rows[arm]],
                    dtype=np.int64,
                )
                for arm in ARMS
            ]
            diff = d - a
            task_rows.append(diff)
            seed_gains[seed] += int(diff.sum())
            treated_goals += int(d.sum())
            contrasts["image_unweighted_minus_front_unweighted"] += int((c - a).sum())
            contrasts["front_balanced_minus_front_unweighted"] += int((b - a).sum())
            contrasts["image_balanced_minus_image_unweighted"] += int((d - c).sum())
            contrasts["factorial_interaction"] += int((d - c - b + a).sum())
            if task == "doorkey":
                opens += sum(
                    int(row["external_effective_door_open_events"] > 0)
                    for row in rows["image_balanced"]
                )
            changed_actor[task][str(seed)] = sum(
                x["executed_action_sha256"] != y["executed_action_sha256"]
                for x, y in zip(rows["image_balanced"], actor, strict=True)
            )
            changed_front[task][str(seed)] = sum(
                x["executed_action_sha256"] != y["executed_action_sha256"]
                for x, y in zip(
                    rows["image_balanced"], rows["front_unweighted"], strict=True
                )
            )
        paired[task] = np.asarray(task_rows, dtype=np.int64)
        task_gains[task] = int(paired[task].sum())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.zeros(20000, dtype=np.int64)
    for task in TASKS:
        values = paired[task]
        seed_ix = rng.integers(0, 3, size=(20000, 3))
        layout_ix = rng.integers(0, values.shape[1],
                                 size=(20000, 3, values.shape[1]))
        draws += values[seed_ix[:, :, None], layout_ix].sum(axis=(1, 2))
    ci = [float(v) for v in np.quantile(draws, (0.025, 0.975))]
    distinct = all(
        sum(
            all(sets[task][str(seed)] >= 16 for task in TASKS)
            for seed in MODEL_SEEDS
        ) >= 2
        for sets in (changed_actor, changed_front)
    )
    gain = int(sum(task_gains.values()))
    noninferior = treated_goals >= 3 * actor_goals - ACTOR_MARGIN
    numeric = bool(
        gain >= MIN_GAIN
        and all(task_gains[task] > 0 for task in TASKS)
        and sum(seed_gains[s] > 0 for s in MODEL_SEEDS) >= 2
        and noninferior and opens >= MIN_DOOR_OPEN_EPISODES
    )
    passed = distinct and numeric and ci[0] > 0
    return {
        "registered_decision": (
            "INSUFFICIENT" if not distinct else "PASS" if passed else "FAIL"
        ),
        "necessary_native_action_distinction": distinct,
        "registered_numeric_physical_gate": numeric,
        "95pct_bootstrap_lower_bound_above_zero": ci[0] > 0,
        "paired_image_balanced_minus_front_unweighted_physical_goals": gain,
        "per_task_physical_gain": task_gains,
        "per_seed_physical_gain": {str(s): int(v) for s, v in seed_gains.items()},
        "factorial_physical_goal_contrasts": contrasts,
        "image_balanced_actor_action_hash_mismatches": changed_actor,
        "image_balanced_front_action_hash_mismatches": changed_front,
        "physical_image_balanced_goals": treated_goals,
        "actual_actor_only_goals_measured_once": actor_goals,
        "actor_goals_repeated_three_times_arithmetic_only": 3 * actor_goals,
        "actor_noninferiority_margin": ACTOR_MARGIN,
        "image_balanced_noninferior_with_margin": noninferior,
        "image_balanced_effective_door_open_episodes": opens,
        "seed_task_paired_bootstrap_95pct_interval": ci,
        "target_unlockpickup_evaluated": False,
    }


def execute(output: str) -> dict:
    if output != f"{OUT}/summary.json" or Path(output).exists():
        raise ValueError("Stage11 output must be fresh registered isolated path")
    validate_split_contract()
    sha = admitted_source_commit()
    folder = Path(output).parent
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    start = time.perf_counter()
    xs, ys, train = acquire_train()
    models: dict[tuple[str, int], VisualBC] = {}
    fits = []
    for seed in MODEL_SEEDS:
        for arm in ARMS:
            model, profile = fit_model(xs, ys, arm, seed)
            profile["numeric_checkpoint"] = save_numeric(
                model, folder / f"{arm}-{seed}.npz"
            )
            models[(arm, seed)] = model
            fits.append(profile)
    monitor = acquire_monitor(models)
    verdict = decide(monitor)
    monitor_actions = sum(group["summary"]["actual_native_actions"]
                          for group in monitor.values())
    result = {
        "study": "m23_stage11_source_full_public_image_x_class_balance_bc",
        "source_commit": sha,
        "train_layouts": {t: list(v) for t, v in VISION_TRAIN_LAYOUTS.items()},
        "monitor_layouts": {t: list(v) for t, v in VISION_MONITOR_LAYOUTS.items()},
        "arms": list(ARMS), "model_seeds": list(MODEL_SEEDS),
        "epochs": EPOCHS, "batch_size": BATCH, "adam_lr": LR,
        "common_public_input_width": WIDTH, "public_image_cells": 49,
        "native_train": train,
        "actual_new_native_train_actions": train["actual_native_train_actions"],
        "actual_new_native_monitor_actions": monitor_actions,
        "actual_new_native_total_actions":
            train["actual_native_train_actions"] + monitor_actions,
        "actual_model_fits": len(fits),
        "actual_optimizer_step_calls": sum(
            row["actual_optimizer_step_calls"] for row in fits
        ),
        "cached_label_presentations_not_native_steps": sum(
            row["cached_native_source_label_presentations"] for row in fits
        ),
        "imagined_planner_actions": 0,
        "target_unlockpickup_native_actions": 0, "target_FINAL_episodes": 0,
        "fits_and_numeric_checkpoints": fits,
        "independent_native_source_monitor": monitor,
        "registered_scientific_decision": verdict,
        "versions": {
            name: version(name) for name in ("numpy", "torch", "gymnasium", "minigrid")
        },
        "actual_elapsed_seconds": time.perf_counter() - start,
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_sha": sha,
        "native_train_actions": result["actual_new_native_train_actions"],
        "native_monitor_actions": monitor_actions,
        "model_fits": len(fits),
        "optimizer_updates": result["actual_optimizer_step_calls"],
        "per_task_goal_gain": verdict["per_task_physical_gain"],
        "native_action_distinction": verdict["necessary_native_action_distinction"],
        "registered_decision": verdict["registered_decision"],
    }), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=f"{OUT}/summary.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Stage11 real source training requires --execute")
    execute(args.output)


if __name__ == "__main__":
    main()
