"""Stage 10: source-only public-mission BC on native executed source actions.

Trained policies never see the native auditor's full state or hidden labels.
A MONITOR numerical gain without distinct native actions is not scientific PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
from torch import nn

from bagi.experiments.m1_goal_learning import admitted_source_commit
from bagi.experiments.m10_native_audit import NativeDoorKeyAudit
from bagi.experiments.m13_public_interaction_coverage import Controller
from bagi.experiments.m14_public_open_door import TraversableOpenDoorController
from bagi.experiments.m15_public_action_bc import (
    BCPolicy,
    N_COLOR,
    N_OBJECT,
    N_STATE,
    features,
    save_numeric,
)
from bagi.experiments.m20_fetch_physics import NativeFetchAudit, PublicFetchTeacher
from bagi.experiments.m23_source_feasibility import HORIZON, SOURCE_ENV_IDS
from bagi.experiments.m23_study_contract import (
    BC_MONITOR_LAYOUTS,
    BC_TRAIN_LAYOUTS,
    validate_split_contract,
)

OUT = "runs/m23_source_mission_bc"
TASKS = ("doorkey", "fetch")
MODEL_SEEDS = (941, 947, 953)
LEARNED_ARMS = ("mission_bc", "mission_blind_bc")
EPOCHS = 12
BATCH = 128
LR = 0.003
MIN_MISSION_GAIN = 8
MISSION_START = N_OBJECT + N_COLOR + N_STATE
BOOTSTRAP_SEED = 231001


def make_source_actor(task: str) -> Controller:
    if task == "doorkey":
        return TraversableOpenDoorController()
    if task == "fetch":
        return PublicFetchTeacher()
    raise ValueError("Stage10 source task not registered")


def make_native_audit(task: str):
    if task not in TASKS:
        raise ValueError("Stage10 native evaluator is source-only")
    wrapped = gym.wrappers.TimeLimit(
        gym.make(SOURCE_ENV_IDS[task]), max_episode_steps=HORIZON
    )
    return NativeDoorKeyAudit(wrapped) if task == "doorkey" else NativeFetchAudit(wrapped)


def mask_public_mission(x: np.ndarray) -> np.ndarray:
    """Ablate only the 64 mission coordinates of maintained public M15 features."""
    if x.ndim not in (1, 2) or x.shape[-1] != MISSION_START + 64 + 4 + 1:
        raise ValueError("Stage10 maintained public feature width changed")
    out = np.array(x, copy=True, dtype=np.float32)
    out[..., MISSION_START:MISSION_START + 64] = 0
    return out


def collect_episode(task: str, layout: int) -> dict:
    if task not in TASKS or layout not in BC_TRAIN_LAYOUTS[task]:
        raise ValueError("Stage10 data collection restricted to registered TRAIN")
    actor = make_source_actor(task)
    rng = np.random.default_rng(8_800_000 + layout)
    audit = make_native_audit(task)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    finished = False
    try:
        obs, _ = audit.reset(seed=layout)
        for _ in range(HORIZON):
            xs.append(features(obs, actor.key_observed_after_pickup))
            action = int(actor.choose(obs, rng))
            if action not in range(7):
                raise ValueError("Stage10 source actor proposed invalid native action")
            next_obs, _reward, terminated, truncated, _ = audit.step(action)
            ys.append(action)
            actor.update(obs, action, next_obs)
            obs = next_obs
            if terminated or truncated:
                finished = True
                break
        truth = audit.summary()
        if not finished or truth["native_actions"] != len(ys) or len(xs) != len(ys):
            raise ValueError("Stage10 collected labels not paired with real native steps")
        if sum(truth["action_counts"].values()) != len(ys):
            raise ValueError("Stage10 native source train action histogram mismatch")
        return {
            "task": task, "layout_seed": layout,
            "x": np.stack(xs).astype(np.float32),
            "y": np.asarray(ys, dtype=np.int64),
            "actual_native_actions": len(ys),
            "source_actor_seed": 8_800_000 + layout,
            "action_histogram": [truth["action_counts"][str(i)] for i in range(7)],
            "actual_executed_action_sha256": hashlib.sha256(bytes(ys)).hexdigest(),
            "external_physical_full_goal": bool(truth["full_physical_goal"]),
            "external_rewarded_terminations": truth["rewarded_terminations"],
        }
    finally:
        audit.env.close()


def acquire_train() -> tuple[np.ndarray, np.ndarray, dict]:
    validate_split_contract()
    episodes = [
        collect_episode(task, seed)
        for task in TASKS for seed in BC_TRAIN_LAYOUTS[task]
    ]
    xs = np.concatenate([row["x"] for row in episodes])
    ys = np.concatenate([row["y"] for row in episodes])
    if len(xs) != len(ys) or not len(ys) or not np.isfinite(xs).all():
        raise ValueError("Stage10 native public TRAIN input/label corruption")
    if len(ys) != sum(row["actual_native_actions"] for row in episodes):
        raise ValueError("Stage10 native training actions not conserved")
    return xs, ys, {
        "real_native_train_episodes": len(episodes),
        "new_actual_native_train_actions": len(ys),
        "source_label_histogram": np.bincount(ys, minlength=7).tolist(),
        "public_train_x_sha256": hashlib.sha256(xs.tobytes()).hexdigest(),
        "executed_train_actions_y_sha256": hashlib.sha256(ys.tobytes()).hexdigest(),
        "per_episode": [
            {k: v for k, v in row.items() if k not in ("x", "y")}
            for row in episodes
        ],
    }


def fit_source_policy(x: np.ndarray, y: np.ndarray,
                      arm: str, seed: int) -> tuple[BCPolicy, dict]:
    if arm not in LEARNED_ARMS or seed not in MODEL_SEEDS:
        raise ValueError("Stage10 fit requires preregistered source model seed and arm")
    if not len(y) or len(x) != len(y) or set(np.unique(y)) - set(range(7)):
        raise ValueError("Stage10 requires actual native public TRAIN labels")
    inputs = mask_public_mission(x) if arm == "mission_blind_bc" else x.copy()
    torch.manual_seed(seed)
    model = BCPolicy()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    source = torch.from_numpy(inputs)
    labels = torch.from_numpy(y)
    generator = torch.Generator().manual_seed(10_000 + seed)
    updates = 0
    presentations = 0
    loss_sum = 0.0
    start = time.perf_counter()
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(y), generator=generator).split(BATCH):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(source[idx]), labels[idx])
            loss.backward()
            optimizer.step()
            updates += 1
            presentations += len(idx)
            loss_sum += float(loss.detach())
    if presentations != EPOCHS * len(y):
        raise ValueError("Stage10 model optimization exposures not conserved")
    model.eval()
    return model, {
        "arm": arm, "model_seed": seed, "epochs": EPOCHS,
        "batch_size": BATCH, "adam_lr": LR,
        "actual_optimizer_step_calls": updates,
        "cached_native_action_label_presentations": presentations,
        "mean_minibatch_loss": loss_sum / updates,
        "fit_elapsed_seconds": time.perf_counter() - start,
        "masked_public_mission": arm == "mission_blind_bc",
        "architecture": "maintained_m15_BCPolicy_MLP_64_hidden",
        "source_train_x_sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
    }


def evaluate_episode(task: str, layout: int, arm: str,
                     seed: int | None, model: BCPolicy | None) -> dict:
    if task not in TASKS or layout not in BC_MONITOR_LAYOUTS[task]:
        raise ValueError("Stage10 physical evaluation only permits fresh source MONITOR")
    if (arm == "actor_only") != (model is None):
        raise ValueError("Stage10 actor/model contract violated")
    if arm == "actor_only":
        if seed is not None:
            raise ValueError("Actor control is measured once per source layout")
    elif arm not in LEARNED_ARMS or seed not in MODEL_SEEDS:
        raise ValueError("Stage10 learned policy arm/seed is unregistered")
    actor = make_source_actor(task) if arm == "actor_only" else Controller("public_memory")
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
                row = features(obs, actor.key_observed_after_pickup)
                if arm == "mission_blind_bc":
                    row = mask_public_mission(row)
                with torch.no_grad():
                    logits = model(torch.from_numpy(row).unsqueeze(0))
                    action = int(torch.argmax(logits, dim=1).item())
            if action not in range(7):
                raise ValueError("Stage10 learned native action outside interface")
            successor, reward, terminated, truncated, _ = audit.step(action)
            actor.update(obs, action, successor)
            reward_sum += float(reward)
            actions.append(action)
            obs = successor
            if terminated or truncated:
                finished = True
                break
        truth = audit.summary()
        if not finished or len(actions) != truth["native_actions"]:
            raise ValueError("Stage10 source policy did not finish valid native episode")
        if sum(truth["action_counts"].values()) != len(actions):
            raise ValueError("Stage10 source native action count mismatch")
        return {
            "task": task, "layout_seed": layout, "arm": arm, "model_seed": seed,
            "actual_native_actions": len(actions),
            "native_action_histogram": [truth["action_counts"][str(i)] for i in range(7)],
            "actual_executed_action_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
            "actual_reward_sum": reward_sum,
            "external_physical_full_goal": bool(truth["full_physical_goal"]),
            "external_rewarded_terminations": truth["rewarded_terminations"],
            "external_effective_pickup_events": truth["effective_pickup_events"],
            "external_effective_door_open_events": truth["effective_door_open_events"],
            "external_wrong_target_pickup_events": (
                truth["wrong_target_pickup_events"] if task == "fetch" else 0
            ),
            "external_effective_forward_movements":
                truth.get("effective_forward_movements"),
        }
    finally:
        audit.env.close()


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Stage10 needs physical episodes for summaries")
    actions = sum(r["actual_native_actions"] for r in rows)
    hist = [sum(r["native_action_histogram"][i] for r in rows) for i in range(7)]
    if sum(hist) != actions:
        raise ValueError("Stage10 native action summary inconsistent")
    successes = sum(int(r["external_physical_full_goal"]) for r in rows)
    rewarded = sum(int(r["external_rewarded_terminations"] > 0) for r in rows)
    return {
        "episodes": len(rows), "native_actions": actions,
        "native_action_histogram": hist, "strict_full_physical_goals": successes,
        "reward_positive_episodes": rewarded,
        "reward_only_nonqualifying_episodes": sum(
            int(r["external_rewarded_terminations"] > 0
                and not r["external_physical_full_goal"]) for r in rows
        ),
    }


def acquire_monitor(models: dict[tuple[str, int], BCPolicy]) -> dict:
    expected = {(arm, seed) for arm in LEARNED_ARMS for seed in MODEL_SEEDS}
    if set(models) != expected:
        raise ValueError("Stage10 requires all six independently trained policies")
    groups = {}
    for task in TASKS:
        layouts = BC_MONITOR_LAYOUTS[task]
        actor_rows = [evaluate_episode(task, layout, "actor_only", None, None)
                      for layout in layouts]
        groups[f"{task}:actor_only"] = {
            "episodes": actor_rows, "summary": summarize(actor_rows)
        }
        for arm in LEARNED_ARMS:
            for seed in MODEL_SEEDS:
                rows = [
                    evaluate_episode(task, layout, arm, seed, models[(arm, seed)])
                    for layout in layouts
                ]
                groups[f"{task}:{arm}:{seed}"] = {
                    "episodes": rows, "summary": summarize(rows)
                }
    return groups


def decide(monitor: dict) -> dict:
    expected = {
        f"{task}:actor_only" for task in TASKS
    } | {
        f"{task}:{arm}:{seed}"
        for task in TASKS for arm in LEARNED_ARMS for seed in MODEL_SEEDS
    }
    if set(monitor) != expected:
        raise ValueError("Stage10 requires all paired independent source arms")
    paired = {}
    per_task_gains = {}
    changed_actor = {}
    changed_blind = {}
    per_seed_gain = {seed: 0 for seed in MODEL_SEEDS}
    observed_actor = 0
    observed_mission = 0
    for task in TASKS:
        actor = monitor[f"{task}:actor_only"]["episodes"]
        layouts = list(BC_MONITOR_LAYOUTS[task])
        if [r["layout_seed"] for r in actor] != layouts:
            raise ValueError("Stage10 source actor layout pairing broken")
        observed_actor += sum(int(r["external_physical_full_goal"]) for r in actor)
        by_seed = []
        changed_actor[task] = {}
        changed_blind[task] = {}
        for seed in MODEL_SEEDS:
            mission = monitor[f"{task}:mission_bc:{seed}"]["episodes"]
            blind = monitor[f"{task}:mission_blind_bc:{seed}"]["episodes"]
            if ([r["layout_seed"] for r in mission] != layouts
                    or [r["layout_seed"] for r in blind] != layouts):
                raise ValueError("Stage10 paired mission/blind layouts broken")
            differences = [
                int(m["external_physical_full_goal"])
                - int(b["external_physical_full_goal"])
                for m, b in zip(mission, blind, strict=True)
            ]
            by_seed.append(differences)
            per_seed_gain[seed] += sum(differences)
            changed_actor[task][str(seed)] = sum(
                m["actual_executed_action_sha256"] != a["actual_executed_action_sha256"]
                for m, a in zip(mission, actor, strict=True)
            )
            changed_blind[task][str(seed)] = sum(
                m["actual_executed_action_sha256"] != b["actual_executed_action_sha256"]
                for m, b in zip(mission, blind, strict=True)
            )
            observed_mission += sum(int(r["external_physical_full_goal"]) for r in mission)
        paired[task] = np.asarray(by_seed, dtype=np.int64)
        per_task_gains[task] = int(paired[task].sum())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.zeros(20000, dtype=np.float64)
    for task in TASKS:
        values = paired[task]
        seed_idx = rng.integers(0, len(MODEL_SEEDS), size=(len(draws), 3))
        layout_idx = rng.integers(0, values.shape[1],
                                  size=(len(draws), 3, values.shape[1]))
        draws += values[seed_idx[:, :, None], layout_idx].sum(axis=(1, 2))
    ci = [float(v) for v in np.quantile(draws, (0.025, 0.975))]
    support = all(
        sum(changed_actor[task][str(seed)] >= 16 for seed in MODEL_SEEDS) >= 2
        and sum(changed_blind[task].values()) >= 16 for task in TASKS
    )
    total_gain = int(sum(per_task_gains.values()))
    noninferior = observed_mission >= 3 * observed_actor
    pass_numeric = bool(
        support and total_gain >= MIN_MISSION_GAIN
        and all(per_task_gains[t] > 0 for t in TASKS)
        and sum(per_seed_gain[s] > 0 for s in MODEL_SEEDS) >= 2
        and noninferior
    )
    return {
        "per_task_mission_minus_blind_physical_goal_gain": per_task_gains,
        "per_model_seed_goal_gain": {str(k): int(v) for k, v in per_seed_gain.items()},
        "mission_minus_actor_native_action_hash_mismatches": changed_actor,
        "mission_minus_blind_native_action_hash_mismatches": changed_blind,
        "physical_mission_goals": observed_mission,
        "physical_actor_goals_measured_once": observed_actor,
        "actor_goals_repeated_three_times_for_arithmetic_only": 3 * observed_actor,
        "physical_mission_noninferior_to_source_actor": noninferior,
        "total_paired_mission_minus_blind_physical_goal_gain": total_gain,
        "necessary_native_action_distinction": support,
        "seed_task_stratified_paired_bootstrap_95pct_total_gain": ci,
        "registered_numerical_gate_pass": pass_numeric,
        "reliable_mission_gain_over_blind": pass_numeric and ci[0] > 0,
        "registered_decision": (
            "INSUFFICIENT" if not support else "PASS" if pass_numeric else "FAIL"
        ),
        "is_target_zero_shot_transfer": False,
    }


def execute(output: str) -> dict:
    validate_split_contract()
    source_sha = admitted_source_commit()
    path = Path(output)
    if str(path) != f"{OUT}/summary.json" or path.exists():
        raise ValueError("Stage10 must use fresh isolated output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    x, y, training = acquire_train()
    models = {}
    fits = []
    for seed in MODEL_SEEDS:
        for arm in LEARNED_ARMS:
            model, fit = fit_source_policy(x, y, arm, seed)
            checkpoint_path = path.parent / f"{arm}-{seed}.npz"
            fit["numeric_checkpoint"] = save_numeric(model, checkpoint_path)
            fits.append(fit)
            models[(arm, seed)] = model
    monitor = acquire_monitor(models)
    decision = decide(monitor)
    real_monitor_actions = sum(
        group["summary"]["native_actions"] for group in monitor.values()
    )
    result = {
        "study": "m23_stage10_source_only_public_mission_conditioned_bc",
        "source_commit": source_sha,
        "train_layouts": {k: list(v) for k, v in BC_TRAIN_LAYOUTS.items()},
        "monitor_layouts": {k: list(v) for k, v in BC_MONITOR_LAYOUTS.items()},
        "model_seeds": list(MODEL_SEEDS), "learned_arms": list(LEARNED_ARMS),
        "architecture": "maintained_m15_BCPolicy",
        "train_epochs": EPOCHS, "batch_size": BATCH, "adam_learning_rate": LR,
        "public_mission_feature_mask_width": 64,
        "true_native_training": training,
        "new_actual_native_train_actions": training["new_actual_native_train_actions"],
        "new_actual_native_monitor_actions": real_monitor_actions,
        "new_actual_native_total_actions": (
            training["new_actual_native_train_actions"] + real_monitor_actions
        ),
        "new_model_fits": len(fits),
        "actual_optimizer_step_calls": sum(
            fit["actual_optimizer_step_calls"] for fit in fits
        ),
        "cached_label_presentations": sum(
            fit["cached_native_action_label_presentations"] for fit in fits
        ),
        "actual_imagined_planner_actions": 0,
        "target_unlockpickup_native_actions": 0,
        "target_final_episodes": 0,
        "fits_and_numeric_checkpoints": fits,
        "independent_source_monitor": monitor,
        "registered_scientific_decision": decision,
        "versions": {
            name: version(name) for name in ("numpy", "torch", "gymnasium", "minigrid")
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_sha": source_sha,
        "new_native_train_actions": training["new_actual_native_train_actions"],
        "new_native_monitor_actions": real_monitor_actions,
        "new_fits": len(fits),
        "optimizer_updates": result["actual_optimizer_step_calls"],
        "per_task_goal_gain": decision["per_task_mission_minus_blind_physical_goal_gain"],
        "distinct_native_actions": decision["necessary_native_action_distinction"],
        "registered_decision": decision["registered_decision"],
    }), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=f"{OUT}/summary.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Stage10 source behavioral cloning requires --execute")
    execute(args.output)


if __name__ == "__main__":
    main()
