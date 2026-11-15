"""M23 Stage 12: source-only public corrective demonstrations versus expert extras.

DAgger-like teacher queries are advisory, NOT native executed actions. Learner
uses only publicly observable MiniGrid image, mission, heading and own executed
pickup history. Independent native auditors alone see physical world witnesses.
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
from bagi.experiments.m15_public_action_bc import class_weights, save_numeric
from bagi.experiments.m20_fetch_physics import NativeFetchAudit
from bagi.experiments.m23_source_feasibility import SOURCE_ENV_IDS
from bagi.experiments.m23_source_mission_bc import make_native_audit, make_source_actor
from bagi.experiments.m23_source_public_vision_bc import (
    VisualBC,
    public_visual_features,
)
from bagi.experiments.m23_study_contract import (
    CORRECT_ACQUIRE_LAYOUTS,
    CORRECT_MONITOR_LAYOUTS,
    CORRECT_TRAIN_LAYOUTS,
    validate_split_contract,
)

OUT = "runs/m23_source_corrective_bc"
TASKS = ("doorkey", "fetch")
MODEL_SEEDS = (941, 947, 953)
ARMS = ("base_bc", "extra_expert", "corrective_onpolicy")
ACQUIRE_ARMS = ("extra_expert", "corrective_onpolicy")
EPOCHS = 12
BATCH = 128
LR = 0.003
HORIZON = 256
ACQUIRE_HORIZON = 128
BOOTSTRAP_SEED = 231201
MIN_GAIN = 12
ACTOR_MARGIN = 24
MIN_DOOR_GOALS = 16
MIN_QUERY_DISAGREEMENTS = 128


def native_acquire_audit(task: str):
    """Only evaluator, never the policy or teacher, sees private native world."""
    if task not in TASKS:
        raise ValueError("Stage12 source-only TRAIN acquisition task required")
    native = gym.wrappers.TimeLimit(
        gym.make(SOURCE_ENV_IDS[task]), max_episode_steps=ACQUIRE_HORIZON
    )
    return NativeDoorKeyAudit(native) if task == "doorkey" else NativeFetchAudit(native)


def freeze_predict(model: VisualBC, obs: dict, memory: bool) -> int:
    """Native deterministic argmax from legitimate public observations only."""
    if model.training:
        raise ValueError("Stage12 acquisition policy must be frozen and eval-only")
    with torch.no_grad():
        x = torch.from_numpy(public_visual_features(obs, memory)).unsqueeze(0)
        result = int(torch.argmax(model(x), dim=1).item())
    if result not in range(7):
        raise ValueError("Stage12 learner native action outside official interface")
    return result


def collect_base_episode(task: str, layout: int) -> dict:
    if task not in TASKS or layout not in CORRECT_TRAIN_LAYOUTS[task]:
        raise ValueError("Stage12 initial expert restricted to fresh TRAIN layouts")
    actor = make_source_actor(task)
    rng = np.random.default_rng(8_800_000 + layout)
    audit = make_native_audit(task)
    x, y = [], []
    finished = False
    try:
        obs, _ = audit.reset(seed=layout)
        for _ in range(HORIZON):
            x.append(public_visual_features(obs, actor.key_observed_after_pickup))
            action = int(actor.choose(obs, rng))
            if action not in range(7):
                raise ValueError("Stage12 initial expert illegal native action")
            nxt, _reward, terminated, truncated, _ = audit.step(action)
            y.append(action)
            actor.update(obs, action, nxt)
            obs = nxt
            if terminated or truncated:
                finished = True
                break
        truth = audit.summary()
        if (not finished or len(x) != len(y)
                or len(y) != truth["native_actions"]
                or sum(truth["action_counts"].values()) != len(y)):
            raise ValueError("Stage12 initial public labels not actual executed actions")
        return {
            "task": task, "layout_seed": layout,
            "x": np.stack(x).astype(np.float32),
            "y": np.asarray(y, dtype=np.int64),
            "actual_native_steps": len(y),
            "public_teacher_queries": len(y),
            "teacher_recommendations_not_executed": 0,
            "executed_actions_sha256": hashlib.sha256(bytes(y)).hexdigest(),
            "public_labels_sha256": hashlib.sha256(np.asarray(y, dtype=np.int64).tobytes()).hexdigest(),
            "native_action_histogram": [truth["action_counts"][str(i)] for i in range(7)],
            "external_strict_physical_goal": bool(truth["full_physical_goal"]),
        }
    finally:
        audit.env.close()


def collect_base() -> tuple[np.ndarray, np.ndarray, dict]:
    validate_split_contract()
    rows = [collect_base_episode(task, seed)
            for task in TASKS for seed in CORRECT_TRAIN_LAYOUTS[task]]
    x = np.concatenate([row["x"] for row in rows])
    y = np.concatenate([row["y"] for row in rows])
    if len(x) != len(y) or not len(y) or not np.isfinite(x).all():
        raise ValueError("Stage12 initial real TRAIN features/labels invalid")
    if len(y) != sum(row["actual_native_steps"] for row in rows):
        raise ValueError("Stage12 initial TRAIN native steps not conserved")
    return x, y, {
        "actual_base_native_episodes": len(rows),
        "actual_base_native_steps": len(y),
        "seven_native_action_histogram": np.bincount(y, minlength=7).tolist(),
        "public_source_x_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "true_executed_y_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
        "per_episode": [{k: v for k, v in row.items() if k not in ("x", "y")} for row in rows],
    }


def fit_from_scratch(
    x: np.ndarray, y: np.ndarray, seed: int, arm: str,
    fixed_weights: torch.Tensor,
) -> tuple[VisualBC, dict]:
    if seed not in MODEL_SEEDS or arm not in ARMS:
        raise ValueError("Stage12 source model seed/arm not preregistered")
    if (x.ndim != 2 or x.shape[1] != public_visual_features_width()
            or len(x) != len(y) or not len(y)
            or set(np.unique(y)) - set(range(7))):
        raise ValueError("Stage12 public source-label fit is malformed")
    torch.manual_seed(seed)
    model = VisualBC()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    inputs = torch.from_numpy(np.asarray(x, dtype=np.float32))
    labels = torch.from_numpy(np.asarray(y, dtype=np.int64))
    generator = torch.Generator().manual_seed(10_000 + seed)
    count, presented, losses = 0, 0, 0.0
    started = time.perf_counter()
    for _ in range(EPOCHS):
        for indices in torch.randperm(len(y), generator=generator).split(BATCH):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(
                model(inputs[indices]), labels[indices], weight=fixed_weights
            )
            loss.backward()
            optimizer.step()
            count += 1
            presented += len(indices)
            losses += float(loss.detach())
    if presented != EPOCHS * len(y):
        raise ValueError("Stage12 reused source label presentations not conserved")
    model.eval()
    return model, {
        "seed": seed, "arm": arm,
        "architecture": "maintained_M23_Stage11_VisualBC_full_public_input",
        "training_epochs": EPOCHS, "batch_size": BATCH, "adam_lr": LR,
        "common_class_weights": fixed_weights.tolist(),
        "actual_optimizer_step_calls": count,
        "cached_source_label_presentations": presented,
        "actual_fitting_wall_seconds": time.perf_counter() - started,
        "mean_minibatch_cross_entropy": losses / count,
        "train_public_x_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "train_source_y_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
    }


def public_visual_features_width() -> int:
    """Stage11 fixed public encoding; never infer input from private states."""
    from bagi.experiments.m23_source_public_vision_bc import WIDTH
    return WIDTH


def acquire_episode(task: str, layout: int, seed: int,
                    arm: str, base_model: VisualBC) -> dict:
    if (task not in TASKS or layout not in CORRECT_ACQUIRE_LAYOUTS[task]
            or seed not in MODEL_SEEDS or arm not in ACQUIRE_ARMS):
        raise ValueError("Stage12 TRAIN correction episode outside registered scope")
    if base_model.training:
        raise ValueError("Stage12 corrective query requires frozen base model")
    public_teacher = make_source_actor(task)
    teacher_rng = np.random.default_rng(8_800_000 + layout)
    audit = native_acquire_audit(task)
    features, labels, executed = [], [], []
    disagreements = 0
    try:
        obs, _ = audit.reset(seed=layout)
        for _ in range(ACQUIRE_HORIZON):
            x = public_visual_features(obs, public_teacher.key_observed_after_pickup)
            advised = int(public_teacher.choose(obs, teacher_rng))
            if advised not in range(7):
                raise ValueError("Stage12 teacher recommendation not a native action")
            actual = (
                advised if arm == "extra_expert"
                else freeze_predict(
                    base_model, obs, public_teacher.key_observed_after_pickup
                )
            )
            nxt, _reward, terminated, truncated, _ = audit.step(actual)
            public_teacher.update(obs, actual, nxt)
            features.append(x)
            labels.append(advised)
            executed.append(actual)
            disagreements += int(advised != actual)
            obs = nxt
            if terminated or truncated:
                break
        truth = audit.summary()
        if (len(features) != len(labels) or len(labels) != len(executed)
                or len(executed) != truth["native_actions"]
                or sum(truth["action_counts"].values()) != len(executed)):
            raise ValueError("Stage12 public corrective queries do not align to real states")
        y = np.asarray(labels, dtype=np.int64)
        return {
            "task": task, "layout_seed": layout, "model_seed": seed, "acquisition_arm": arm,
            "x": np.stack(features).astype(np.float32),
            "y": y,
            "actual_native_executed_actions": len(executed),
            "actual_native_action_histogram":
                [truth["action_counts"][str(i)] for i in range(7)],
            "public_teacher_advice_queries": len(labels),
            "public_advised_native_action_histogram": np.bincount(y, minlength=7).tolist(),
            "teacher_advice_vs_executed_action_disagreements": disagreements,
            "teacher_advice_not_physically_executed": (
                len(labels) if arm == "corrective_onpolicy" else 0
            ),
            "executed_actions_sha256": hashlib.sha256(bytes(executed)).hexdigest(),
            "advisory_labels_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
            "external_strict_full_physical_goal": bool(truth["full_physical_goal"]),
            "external_effective_door_open_events":
                truth["effective_door_open_events"] if task == "doorkey" else 0,
        }
    finally:
        audit.env.close()


def acquire_matched(base_models: dict[int, VisualBC]) -> tuple[dict, dict]:
    if set(base_models) != set(MODEL_SEEDS):
        raise ValueError("Stage12 needs three separately trained frozen base models")
    extra = {}
    summary = {}
    for seed in MODEL_SEEDS:
        task_rows = {}
        all_x = {arm: [] for arm in ACQUIRE_ARMS}
        all_y = {arm: [] for arm in ACQUIRE_ARMS}
        for task in TASKS:
            episode_rows = {}
            for arm in ACQUIRE_ARMS:
                episode_rows[arm] = [
                    acquire_episode(task, layout, seed, arm, base_models[seed])
                    for layout in CORRECT_ACQUIRE_LAYOUTS[task]
                ]
            pairs = {}
            for arm in ACQUIRE_ARMS:
                rows = episode_rows[arm]
                pairs[arm] = (
                    np.concatenate([row["x"] for row in rows]),
                    np.concatenate([row["y"] for row in rows]),
                )
            retained = min(len(pairs[a][1]) for a in ACQUIRE_ARMS)
            if not retained:
                raise ValueError("Stage12 no matched source corrective labels")
            for arm in ACQUIRE_ARMS:
                x, y = pairs[arm]
                all_x[arm].append(x[:retained].copy())
                all_y[arm].append(y[:retained].copy())
            task_rows[task] = {
                "matched_retained_public_teacher_labels_per_arm": retained,
                "raw_actual_native_steps_per_arm": {
                    arm: sum(r["actual_native_executed_actions"] for r in episode_rows[arm])
                    for arm in ACQUIRE_ARMS
                },
                "raw_public_teacher_queries_per_arm": {
                    arm: sum(r["public_teacher_advice_queries"] for r in episode_rows[arm])
                    for arm in ACQUIRE_ARMS
                },
                "discarded_unmatched_advisory_labels_per_arm": {
                    arm: len(pairs[arm][1]) - retained for arm in ACQUIRE_ARMS
                },
                "public_teacher_vs_actual_native_action_disagreements": sum(
                    r["teacher_advice_vs_executed_action_disagreements"]
                    for r in episode_rows["corrective_onpolicy"]
                ),
                "source_teacher_advice_action_counts_per_arm": {
                    arm: np.bincount(pairs[arm][1], minlength=7).tolist()
                    for arm in ACQUIRE_ARMS
                },
                "per_episode": {
                    arm: [
                        {k: v for k, v in row.items() if k not in ("x", "y")}
                        for row in episode_rows[arm]
                    ] for arm in ACQUIRE_ARMS
                },
            }
        extra[seed] = {
            arm: (np.concatenate(all_x[arm]), np.concatenate(all_y[arm]))
            for arm in ACQUIRE_ARMS
        }
        sizes = {arm: len(extra[seed][arm][1]) for arm in ACQUIRE_ARMS}
        if len(set(sizes.values())) != 1:
            raise ValueError("Stage12 public TRAIN extra label budgets mismatch")
        summary[str(seed)] = {
            "task_data": task_rows,
            "total_matched_extra_source_labels_per_arm": sizes,
            "source_action_budget_is_matched": False,
        }
    return extra, summary


def monitor_episode(task: str, layout: int, arm: str,
                    seed: int | None, model: VisualBC | None) -> dict:
    if task not in TASKS or layout not in CORRECT_MONITOR_LAYOUTS[task]:
        raise ValueError("Stage12 MONITOR only permits fresh source native layouts")
    if arm == "actor_only":
        if seed is not None or model is not None:
            raise ValueError("Stage12 actor baseline must be one real run per layout")
        actor = make_source_actor(task)
    elif arm in ARMS and seed in MODEL_SEEDS and model is not None:
        if model.training:
            raise ValueError("Stage12 source model must be eval-only on MONITOR")
        actor = Controller("public_memory")
    else:
        raise ValueError("Stage12 MONITOR source policy mode not registered")
    rng = np.random.default_rng(8_800_000 + layout)
    audit = make_native_audit(task)
    actions, reward_sum = [], 0.0
    finished = False
    try:
        obs, _ = audit.reset(seed=layout)
        for _ in range(HORIZON):
            action = (
                int(actor.choose(obs, rng)) if model is None
                else freeze_predict(model, obs, actor.key_observed_after_pickup)
            )
            nxt, reward, terminated, truncated, _ = audit.step(action)
            actor.update(obs, action, nxt)
            actions.append(action)
            reward_sum += float(reward)
            obs = nxt
            if terminated or truncated:
                finished = True
                break
        truth = audit.summary()
        if (not finished or len(actions) != truth["native_actions"]
                or sum(truth["action_counts"].values()) != len(actions)):
            raise ValueError("Stage12 MONITOR actual native action accounting invalid")
        return {
            "task": task, "layout_seed": layout, "mode": arm, "model_seed": seed,
            "actual_native_actions": len(actions),
            "native_action_histogram": [truth["action_counts"][str(i)] for i in range(7)],
            "actual_executed_action_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
            "actual_reward_total": reward_sum,
            "external_strict_full_physical_goal": bool(truth["full_physical_goal"]),
            "external_rewarded_terminations": truth["rewarded_terminations"],
            "external_effective_pickup_events": truth["effective_pickup_events"],
            "external_effective_door_open_events":
                truth["effective_door_open_events"] if task == "doorkey" else 0,
            "external_effective_forward_movements":
                truth.get("effective_forward_movements", 0),
            "external_wrong_target_pickup_events":
                truth.get("wrong_target_pickup_events", 0),
        }
    finally:
        audit.env.close()


def monitor_summary(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Stage12 physical evaluation must contain real native runs")
    counts = [sum(row["native_action_histogram"][i] for row in rows)
              for i in range(7)]
    total = sum(row["actual_native_actions"] for row in rows)
    if sum(counts) != total:
        raise ValueError("Stage12 native MONITOR steps and histogram disagree")
    return {
        "physical_episode_count": len(rows), "actual_native_actions": total,
        "native_action_histogram": counts,
        "strict_full_physical_goals": sum(
            int(row["external_strict_full_physical_goal"]) for row in rows
        ),
        "reward_positive_episodes": sum(
            int(row["external_rewarded_terminations"] > 0) for row in rows
        ),
        "reward_only_nonqualifying_episodes": sum(
            int(row["external_rewarded_terminations"] > 0
                and not row["external_strict_full_physical_goal"]) for row in rows
        ),
        "effective_door_open_episodes": sum(
            int(row["external_effective_door_open_events"] > 0) for row in rows
        ),
        "effective_pickup_events": sum(
            row["external_effective_pickup_events"] for row in rows
        ),
    }


def acquire_monitor(models: dict[tuple[str, int], VisualBC]) -> dict:
    if set(models) != {(arm, seed) for arm in ARMS for seed in MODEL_SEEDS}:
        raise ValueError("Stage12 needs all nine real trained source policy checkpoints")
    groups = {}
    for task in TASKS:
        layouts = CORRECT_MONITOR_LAYOUTS[task]
        actor_rows = [monitor_episode(task, layout, "actor_only", None, None)
                      for layout in layouts]
        groups[f"{task}:actor_only"] = {
            "episodes": actor_rows, "summary": monitor_summary(actor_rows)
        }
        for arm in ARMS:
            for seed in MODEL_SEEDS:
                rows = [
                    monitor_episode(task, layout, arm, seed, models[(arm, seed)])
                    for layout in layouts
                ]
                groups[f"{task}:{arm}:{seed}"] = {
                    "episodes": rows, "summary": monitor_summary(rows)
                }
    return groups


def decide(monitor: dict, acquisition_summary: dict) -> dict:
    required = {f"{task}:actor_only" for task in TASKS} | {
        f"{task}:{arm}:{seed}"
        for task in TASKS for arm in ARMS for seed in MODEL_SEEDS
    }
    if set(monitor) != required or set(acquisition_summary) != {
        str(seed) for seed in MODEL_SEEDS
    }:
        raise ValueError("Stage12 registered MONITOR/real TRAIN acquisition incomplete")
    by_task, by_seed = {}, {s: 0 for s in MODEL_SEEDS}
    changed_expert, changed_actor, changed_base = {}, {}, {}
    actor_goals = corrected_goals = door_goals = 0
    diffs = {}
    for task in TASKS:
        layouts = list(CORRECT_MONITOR_LAYOUTS[task])
        actor = monitor[f"{task}:actor_only"]["episodes"]
        if [r["layout_seed"] for r in actor] != layouts:
            raise ValueError("Stage12 actor MONITOR layout pairing broken")
        actor_goals += sum(int(r["external_strict_full_physical_goal"]) for r in actor)
        changed_expert[task] = {}
        changed_actor[task] = {}
        changed_base[task] = {}
        pairs = []
        for seed in MODEL_SEEDS:
            rows = {}
            for arm in ARMS:
                rows[arm] = monitor[f"{task}:{arm}:{seed}"]["episodes"]
                if [r["layout_seed"] for r in rows[arm]] != layouts:
                    raise ValueError("Stage12 learned MONITOR layout pairing broken")
            extra = rows["extra_expert"]
            corrected = rows["corrective_onpolicy"]
            d = np.asarray([
                int(c["external_strict_full_physical_goal"])
                - int(e["external_strict_full_physical_goal"])
                for c, e in zip(corrected, extra, strict=True)
            ], dtype=np.int64)
            pairs.append(d)
            by_seed[seed] += int(d.sum())
            corrected_goals += sum(
                int(r["external_strict_full_physical_goal"]) for r in corrected
            )
            if task == "doorkey":
                door_goals += sum(
                    int(r["external_strict_full_physical_goal"]) for r in corrected
                )
            for dest, other in (
                (changed_expert, extra), (changed_actor, actor),
                (changed_base, rows["base_bc"]),
            ):
                dest[task][str(seed)] = sum(
                    c["actual_executed_action_sha256"] != o[
                        "actual_executed_action_sha256"
                    ] for c, o in zip(corrected, other, strict=True)
                )
        diffs[task] = np.asarray(pairs, dtype=np.int64)
        by_task[task] = int(diffs[task].sum())
    teacher_query_differences = sum(
        info["public_teacher_vs_actual_native_action_disagreements"]
        for seed_info in acquisition_summary.values()
        for info in seed_info["task_data"].values()
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.zeros(20000, dtype=np.int64)
    for task in TASKS:
        values = diffs[task]
        seed_indices = rng.integers(0, 3, size=(20000, 3))
        layout_indices = rng.integers(0, values.shape[1], size=(20000, 3, values.shape[1]))
        draws += values[seed_indices[:, :, None], layout_indices].sum(axis=(1, 2))
    ci = [float(x) for x in np.quantile(draws, (0.025, 0.975))]
    query_support = teacher_query_differences >= MIN_QUERY_DISAGREEMENTS
    path_support = sum(
        all(changed_expert[task][str(seed)] >= 16 for task in TASKS)
        for seed in MODEL_SEEDS
    ) >= 2
    physical_margin = corrected_goals >= 3 * actor_goals - ACTOR_MARGIN
    gain = int(sum(by_task.values()))
    numeric = bool(
        gain >= MIN_GAIN
        and all(by_task[task] > 0 for task in TASKS)
        and sum(by_seed[s] > 0 for s in MODEL_SEEDS) >= 2
        and physical_margin
        and door_goals >= MIN_DOOR_GOALS
    )
    passed = query_support and path_support and numeric and ci[0] > 0
    return {
        "registered_decision": (
            "INSUFFICIENT" if not query_support or not path_support
            else "PASS" if passed else "FAIL"
        ),
        "live_advisory_query_disagreements_not_native_steps": teacher_query_differences,
        "training_teacher_query_nonvacuity": query_support,
        "native_evaluation_action_distinction": path_support,
        "real_corrective_minus_expert_native_action_hash_mismatches": changed_expert,
        "real_corrective_minus_actor_native_action_hash_mismatches": changed_actor,
        "real_corrective_minus_base_native_action_hash_mismatches": changed_base,
        "strict_physical_gain_corrective_minus_expert": gain,
        "per_task_strict_physical_gain": by_task,
        "per_seed_strict_physical_gain": {str(k): int(v) for k, v in by_seed.items()},
        "corrective_physical_goals": corrected_goals,
        "actor_physical_goals_measured_once": actor_goals,
        "actor_goals_repeated_for_comparison_arithmetic_only": 3 * actor_goals,
        "corrective_within_preregistered_actor_margin": physical_margin,
        "corrective_doorkey_strict_goals": door_goals,
        "registered_numeric_physical_goal_gate": numeric,
        "paired_task_seed_stratified_95pct_bootstrap_gain_interval": ci,
        "95pct_lower_bound_strictly_positive": ci[0] > 0,
        "is_target_zero_shot_transfer": False,
    }


def execute(output: str) -> dict:
    if output != f"{OUT}/summary.json" or Path(output).exists():
        raise ValueError("Stage12 requires fresh registered isolated summary output")
    validate_split_contract()
    source_sha = admitted_source_commit()
    directory = Path(output).parent
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    x, y, initial = collect_base()
    fixed_weights = class_weights(y)
    models: dict[tuple[str, int], VisualBC] = {}
    fits = []
    bases = {}
    for seed in MODEL_SEEDS:
        base, profile = fit_from_scratch(x, y, seed, "base_bc", fixed_weights)
        profile["numeric_checkpoint"] = save_numeric(
            base, directory / f"base_bc-{seed}.npz"
        )
        fits.append(profile)
        bases[seed] = base
        models[("base_bc", seed)] = base
    additions, acquisition = acquire_matched(bases)
    for seed in MODEL_SEEDS:
        for arm in ACQUIRE_ARMS:
            more_x, more_y = additions[seed][arm]
            # Strictly equal retained samples and matched optimizer schedule.
            final_x = np.concatenate([x, more_x])
            final_y = np.concatenate([y, more_y])
            model, profile = fit_from_scratch(
                final_x, final_y, seed, arm, fixed_weights
            )
            profile["new_advisory_labels_retained"] = len(more_y)
            profile["numeric_checkpoint"] = save_numeric(
                model, directory / f"{arm}-{seed}.npz"
            )
            models[(arm, seed)] = model
            fits.append(profile)
    for seed in MODEL_SEEDS:
        if not all(
            fits_lookup["new_advisory_labels_retained"]
            == acquisition[str(seed)]["total_matched_extra_source_labels_per_arm"][arm]
            for arm in ACQUIRE_ARMS
            for fits_lookup in fits
            if fits_lookup["seed"] == seed and fits_lookup["arm"] == arm
        ):
            raise ValueError("Stage12 trained additions not equal to registered budgets")
    monitor = acquire_monitor(models)
    decision = decide(monitor, acquisition)
    real_acquisition_actions = sum(
        metrics["raw_actual_native_steps_per_arm"][arm]
        for row in acquisition.values() for metrics in row["task_data"].values()
        for arm in ACQUIRE_ARMS
    )
    monitor_actions = sum(
        group["summary"]["actual_native_actions"] for group in monitor.values()
    )
    result = {
        "study": "m23_stage12_source_public_onpolicy_corrective_teacher_bc",
        "source_commit": source_sha,
        "initial_train_layouts": {k: list(v) for k, v in CORRECT_TRAIN_LAYOUTS.items()},
        "source_acquisition_train_layouts":
            {k: list(v) for k, v in CORRECT_ACQUIRE_LAYOUTS.items()},
        "fresh_monitor_layouts": {k: list(v) for k, v in CORRECT_MONITOR_LAYOUTS.items()},
        "model_seeds": list(MODEL_SEEDS), "learned_arms": list(ARMS),
        "base_epochs": EPOCHS, "retrain_epochs": EPOCHS,
        "acquisition_native_horizon": ACQUIRE_HORIZON,
        "monitor_native_horizon": HORIZON,
        "fixed_class_weights_from_base_train_only": fixed_weights.tolist(),
        "true_initial_expert_source_native_training": initial,
        "real_source_corrective_acquisitions_and_label_budget": acquisition,
        "real_independent_source_monitor": monitor,
        "registered_scientific_decision": decision,
        "fits_and_numeric_checkpoints": fits,
        "new_actual_native_initial_train_steps": initial["actual_base_native_steps"],
        "new_actual_native_acquisition_steps": real_acquisition_actions,
        "new_actual_native_monitor_steps": monitor_actions,
        "new_actual_native_total_steps":
            initial["actual_base_native_steps"] + real_acquisition_actions + monitor_actions,
        "new_actual_model_fits": len(fits),
        "new_actual_optimizer_steps": sum(row["actual_optimizer_step_calls"] for row in fits),
        "cached_label_presentations_not_native_steps": sum(
            row["cached_source_label_presentations"] for row in fits
        ),
        "new_actual_imagined_actions": 0,
        "target_unlockpickup_native_steps": 0, "target_FINAL_episodes": 0,
        "versions": {
            name: version(name) for name in ("numpy", "torch", "gymnasium", "minigrid")
        },
        "real_wall_seconds": time.perf_counter() - started,
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_sha": source_sha,
        "native_base_steps": result["new_actual_native_initial_train_steps"],
        "native_acquisition_steps": real_acquisition_actions,
        "native_monitor_steps": monitor_actions,
        "genuine_fits": len(fits),
        "optimizer_steps": result["new_actual_optimizer_steps"],
        "teacher_query_disagreements":
            decision["live_advisory_query_disagreements_not_native_steps"],
        "physical_goal_gain": decision["strict_physical_gain_corrective_minus_expert"],
        "native_action_distinct": decision["native_evaluation_action_distinction"],
        "registered_decision": decision["registered_decision"],
    }), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=f"{OUT}/summary.json")
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    if not options.execute:
        raise SystemExit("Stage12 source corrective native training needs --execute")
    execute(options.output)


if __name__ == "__main__":
    main()
