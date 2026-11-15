"""M23 Stage13: equal native TRAIN acquisition budgets, actually executed teacher help.

All teacher actions and learner decisions use native PUBLIC observation only;
native physical witnesses remain isolated in evaluator audit wrappers.
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

from bagi.experiments.m1_goal_learning import admitted_source_commit
from bagi.experiments.m13_public_interaction_coverage import Controller
from bagi.experiments.m15_public_action_bc import class_weights, save_numeric
from bagi.experiments.m23_source_corrective_bc import (
    fit_from_scratch,
    freeze_predict,
    native_acquire_audit,
)
from bagi.experiments.m23_source_mission_bc import make_native_audit, make_source_actor
from bagi.experiments.m23_source_public_vision_bc import VisualBC, public_visual_features
from bagi.experiments.m23_study_contract import (
    INTERVENE_ACQUIRE_LAYOUTS,
    INTERVENE_MONITOR_LAYOUTS,
    INTERVENE_TRAIN_LAYOUTS,
    validate_split_contract,
)

OUT = "runs/m23_source_intervention_bc"
TASKS = ("doorkey", "fetch")
MODEL_SEEDS = (941, 947, 953)
ARMS = ("base_bc", "extra_expert", "intervention_mixed")
ACQUIRE_ARMS = ("extra_expert", "intervention_mixed")
EPOCHS = 12
BATCH = 128
LR = 0.003
HORIZON = 256
ACQUIRE_HORIZON = 128
ACQUIRE_BUDGET = 2304
BOOTSTRAP_SEED = 231301
MIN_GAIN = 12
MIN_TEACHER_EXECUTED = 2500
MIN_EFFECTIVE_ACQUIRE_DOOR_OPEN = 16
MIN_MONITOR_DOOR_GOALS = 16
ACTOR_MARGIN = 24


def collect_base_episode(task: str, layout: int) -> dict:
    if task not in TASKS or layout not in INTERVENE_TRAIN_LAYOUTS[task]:
        raise ValueError("Stage13 initial TRAIN actor layout not registered")
    actor = make_source_actor(task)
    rng = np.random.default_rng(8_800_000 + layout)
    audit = make_native_audit(task)
    xs, actions = [], []
    terminated_by_env = False
    try:
        obs, _ = audit.reset(seed=layout)
        for _ in range(HORIZON):
            xs.append(public_visual_features(obs, actor.key_observed_after_pickup))
            action = int(actor.choose(obs, rng))
            if action not in range(7):
                raise ValueError("Stage13 source teacher provided illegal native action")
            nxt, _reward, done, timeout, _ = audit.step(action)
            actor.update(obs, action, nxt)
            actions.append(action)
            obs = nxt
            if done or timeout:
                terminated_by_env = True
                break
        truth = audit.summary()
        if (not terminated_by_env or not len(actions) or len(xs) != len(actions)
                or truth["native_actions"] != len(actions)
                or sum(truth["action_counts"].values()) != len(actions)):
            raise ValueError("Stage13 native initial TRAIN accounting mismatch")
        y = np.asarray(actions, dtype=np.int64)
        return {
            "task": task, "layout_seed": layout,
            "x": np.stack(xs).astype(np.float32), "y": y,
            "actual_native_steps": len(actions),
            "actual_executed_action_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
            "native_action_histogram": [truth["action_counts"][str(i)] for i in range(7)],
            "external_physical_full_goal": bool(truth["full_physical_goal"]),
        }
    finally:
        audit.env.close()


def collect_base() -> tuple[np.ndarray, np.ndarray, dict]:
    rows = [
        collect_base_episode(task, layout)
        for task in TASKS for layout in INTERVENE_TRAIN_LAYOUTS[task]
    ]
    xs = np.concatenate([row["x"] for row in rows])
    ys = np.concatenate([row["y"] for row in rows])
    if len(xs) != len(ys) or not len(ys) or not np.isfinite(xs).all():
        raise ValueError("Stage13 initial public TRAIN features/executed labels invalid")
    return xs, ys, {
        "actual_episodes": len(rows),
        "actual_native_steps": int(len(ys)),
        "native_executed_action_histogram": np.bincount(ys, minlength=7).tolist(),
        "public_x_sha256": hashlib.sha256(xs.tobytes()).hexdigest(),
        "executed_y_sha256": hashlib.sha256(ys.tobytes()).hexdigest(),
        "episodes": [
            {k: v for k, v in row.items() if k not in ("x", "y")} for row in rows
        ],
    }


def acquire_one(task: str, layout: int, visit: int, model_seed: int,
                arm: str, base: VisualBC, remaining: int) -> dict:
    """Execute exactly min(remaining, native episode length) real source actions."""
    if (task not in TASKS or layout not in INTERVENE_ACQUIRE_LAYOUTS[task]
            or model_seed not in MODEL_SEEDS or arm not in ACQUIRE_ARMS
            or not 1 <= remaining <= ACQUIRE_BUDGET or base.training):
        raise ValueError("Stage13 native TRAIN acquisition outside registered conditions")
    teacher = make_source_actor(task)
    rng = np.random.default_rng(8_800_000 + layout)
    audit = native_acquire_audit(task)
    public_x, advised_labels, native_executed = [], [], []
    selected_interventions = 0
    query_disagreements = 0
    teacher_selected_rare = 0
    native_finished = False
    try:
        obs, _ = audit.reset(seed=layout)
        for i in range(min(remaining, ACQUIRE_HORIZON)):
            public_x.append(
                public_visual_features(obs, teacher.key_observed_after_pickup)
            )
            advice = int(teacher.choose(obs, rng))
            if advice not in range(7):
                raise ValueError("Stage13 public actor advice outside 0..6")
            selected = arm == "extra_expert" or advice in (3, 5) or i % 4 == 0
            executed = (
                advice if selected
                else freeze_predict(base, obs, teacher.key_observed_after_pickup)
            )
            nxt, _reward, done, timeout, _ = audit.step(executed)
            teacher.update(obs, executed, nxt)
            advised_labels.append(advice)
            native_executed.append(executed)
            selected_interventions += int(selected and arm == "intervention_mixed")
            teacher_selected_rare += int(
                selected and arm == "intervention_mixed" and advice in (3, 5)
            )
            query_disagreements += int(advice != executed)
            obs = nxt
            if done or timeout:
                native_finished = True
                break
        truth = audit.summary()
        count = len(native_executed)
        if (count != len(public_x) or count != len(advised_labels)
                or count != truth["native_actions"]
                or sum(truth["action_counts"].values()) != count
                or not 1 <= count <= remaining):
            raise ValueError("Stage13 public advice/native action step reconciliation failed")
        labels = np.asarray(advised_labels, dtype=np.int64)
        return {
            "task": task, "layout_seed": layout, "visit_index": visit,
            "model_seed": model_seed, "arm": arm,
            "x": np.stack(public_x).astype(np.float32), "y": labels,
            "actual_native_actions": count, "public_teacher_query_count": count,
            "actual_teacher_selected_native_steps": selected_interventions,
            "actual_teacher_selected_rare_3_or_5": teacher_selected_rare,
            "queried_but_not_executed_teacher_advice": query_disagreements,
            "native_episode_finished": native_finished,
            "budget_cut_partial": (
                not native_finished and count == remaining
            ),
            "actual_native_action_histogram":
                [truth["action_counts"][str(i)] for i in range(7)],
            "public_teacher_label_histogram": np.bincount(labels, minlength=7).tolist(),
            "native_executed_action_sha256":
                hashlib.sha256(bytes(native_executed)).hexdigest(),
            "advised_label_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            "external_effective_door_open_events":
                truth["effective_door_open_events"] if task == "doorkey" else 0,
            "external_effective_pickup_events": truth["effective_pickup_events"],
            "external_full_physical_goal": bool(truth["full_physical_goal"]),
        }
    finally:
        audit.env.close()


def acquire_budgeted(task: str, seed: int, arm: str,
                     base: VisualBC) -> tuple[np.ndarray, np.ndarray, dict]:
    """Exactly 2,304 real native actions; serial native episodes, no padding."""
    if task not in TASKS or seed not in MODEL_SEEDS or arm not in ACQUIRE_ARMS:
        raise ValueError("Stage13 acquisition arm/task/seed not preregistered")
    layouts = INTERVENE_ACQUIRE_LAYOUTS[task]
    rows = []
    tally = 0
    while tally < ACQUIRE_BUDGET:
        iteration = len(rows)
        layout = layouts[iteration % len(layouts)]
        row = acquire_one(
            task, layout, iteration // len(layouts), seed, arm,
            base, ACQUIRE_BUDGET - tally,
        )
        tally += row["actual_native_actions"]
        rows.append(row)
    if tally != ACQUIRE_BUDGET or not rows:
        raise ValueError("Stage13 true native action quota not exactly conserved")
    xs = np.concatenate([row["x"] for row in rows])
    ys = np.concatenate([row["y"] for row in rows])
    if len(ys) != ACQUIRE_BUDGET or len(xs) != len(ys):
        raise ValueError("Stage13 label and real source acquisition budget mismatch")
    return xs, ys, {
        "task": task, "model_seed": seed, "arm": arm,
        "actual_new_native_actions": tally,
        "actual_new_public_teacher_queries": tally,
        "actual_teacher_executed_intervention_steps": sum(
            row["actual_teacher_selected_native_steps"] for row in rows
        ),
        "actual_teacher_selected_rare_action_steps": sum(
            row["actual_teacher_selected_rare_3_or_5"] for row in rows
        ),
        "queried_advice_disagreements_not_native_actions": sum(
            row["queried_but_not_executed_teacher_advice"] for row in rows
        ),
        "external_effective_door_open_events": sum(
            row["external_effective_door_open_events"] for row in rows
        ),
        "external_effective_pickup_events": sum(
            row["external_effective_pickup_events"] for row in rows
        ),
        "public_teacher_label_histogram": np.bincount(ys, minlength=7).tolist(),
        "actually_executed_native_action_histogram": [
            sum(row["actual_native_action_histogram"][i] for row in rows)
            for i in range(7)
        ],
        "acquisition_native_episode_count": len(rows),
        "distinct_layout_seeds_used": len({row["layout_seed"] for row in rows}),
        "native_acquired_x_sha256": hashlib.sha256(xs.tobytes()).hexdigest(),
        "public_teacher_labels_y_sha256": hashlib.sha256(ys.tobytes()).hexdigest(),
        "native_episode_records": [
            {k: v for k, v in row.items() if k not in ("x", "y")} for row in rows
        ],
    }


def acquire_all(bases: dict[int, VisualBC]) -> tuple[dict, dict]:
    if set(bases) != set(MODEL_SEEDS):
        raise ValueError("Stage13 requires exactly three frozen base models")
    additions = {}
    audit_rows = {}
    for seed in MODEL_SEEDS:
        by_task = {}
        concat = {arm: {"x": [], "y": []} for arm in ACQUIRE_ARMS}
        for task in TASKS:
            by_task[task] = {}
            for arm in ACQUIRE_ARMS:
                x, y, audit = acquire_budgeted(task, seed, arm, bases[seed])
                concat[arm]["x"].append(x)
                concat[arm]["y"].append(y)
                by_task[task][arm] = audit
        additions[seed] = {
            arm: (
                np.concatenate(concat[arm]["x"]),
                np.concatenate(concat[arm]["y"]),
            )
            for arm in ACQUIRE_ARMS
        }
        if {len(additions[seed][arm][1]) for arm in ACQUIRE_ARMS} != {
            ACQUIRE_BUDGET * len(TASKS)
        }:
            raise ValueError("Stage13 equal source native/label budgets violated")
        audit_rows[str(seed)] = by_task
    return additions, audit_rows


def monitor_episode(task: str, layout: int, arm: str, seed: int | None,
                    policy: VisualBC | None) -> dict:
    if task not in TASKS or layout not in INTERVENE_MONITOR_LAYOUTS[task]:
        raise ValueError("Stage13 source MONITOR seed outside fresh registered split")
    if arm == "actor_only":
        if seed is not None or policy is not None:
            raise ValueError("Stage13 native actor control must run once/layout")
        actor = make_source_actor(task)
    elif arm in ARMS and seed in MODEL_SEEDS and policy is not None:
        if policy.training:
            raise ValueError("Stage13 MONITOR learned model must be eval-only")
        actor = Controller("public_memory")
    else:
        raise ValueError("Stage13 MONITOR model/arm/seed not registered")
    rng = np.random.default_rng(8_800_000 + layout)
    audit = make_native_audit(task)
    actions = []
    reward_sum = 0.0
    finished = False
    try:
        obs, _ = audit.reset(seed=layout)
        for _ in range(HORIZON):
            action = (
                int(actor.choose(obs, rng)) if policy is None
                else freeze_predict(policy, obs, actor.key_observed_after_pickup)
            )
            nxt, reward, done, timeout, _ = audit.step(action)
            actor.update(obs, action, nxt)
            actions.append(action)
            reward_sum += float(reward)
            obs = nxt
            if done or timeout:
                finished = True
                break
        truth = audit.summary()
        if (not finished or len(actions) != truth["native_actions"]
                or sum(truth["action_counts"].values()) != len(actions)):
            raise ValueError("Stage13 actual MONITOR native action audit mismatch")
        return {
            "task": task, "layout_seed": layout, "arm": arm, "model_seed": seed,
            "actual_native_actions": len(actions),
            "actual_native_action_histogram":
                [truth["action_counts"][str(i)] for i in range(7)],
            "actual_executed_action_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
            "actual_reward_sum": reward_sum,
            "external_full_physical_goal": bool(truth["full_physical_goal"]),
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


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Stage13 source MONITOR rows missing")
    hist = [
        sum(row["actual_native_action_histogram"][i] for row in rows)
        for i in range(7)
    ]
    steps = sum(row["actual_native_actions"] for row in rows)
    if sum(hist) != steps:
        raise ValueError("Stage13 native MONITOR action histogram invalid")
    return {
        "native_episode_count": len(rows),
        "actual_native_actions": steps,
        "actual_native_action_histogram": hist,
        "strict_physical_goals": sum(int(row["external_full_physical_goal"]) for row in rows),
        "reward_positive_episodes": sum(
            int(row["external_rewarded_terminations"] > 0) for row in rows
        ),
        "reward_only_not_strict_physical_episodes": sum(
            int(row["external_rewarded_terminations"] > 0
                and not row["external_full_physical_goal"]) for row in rows
        ),
        "effective_door_open_episodes": sum(
            int(row["external_effective_door_open_events"] > 0) for row in rows
        ),
    }


def acquire_monitor(models: dict[tuple[str, int], VisualBC]) -> dict:
    if set(models) != {(arm, s) for arm in ARMS for s in MODEL_SEEDS}:
        raise ValueError("Stage13 requires all nine independently fitted source models")
    groups = {}
    for task in TASKS:
        layouts = INTERVENE_MONITOR_LAYOUTS[task]
        actor_rows = [
            monitor_episode(task, layout, "actor_only", None, None) for layout in layouts
        ]
        groups[f"{task}:actor_only"] = {
            "episodes": actor_rows, "summary": summarize(actor_rows)
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


def decide(monitor: dict, acquired: dict) -> dict:
    required = {f"{task}:actor_only" for task in TASKS} | {
        f"{task}:{arm}:{seed}"
        for task in TASKS for arm in ARMS for seed in MODEL_SEEDS
    }
    if set(monitor) != required or set(acquired) != {str(s) for s in MODEL_SEEDS}:
        raise ValueError("Stage13 native source audit must include all registered arms")
    effective_open = sum(
        acquired[str(seed)]["doorkey"]["intervention_mixed"][
            "external_effective_door_open_events"
        ] for seed in MODEL_SEEDS
    )
    actual_teacher_steps = sum(
        acquired[str(seed)][task]["intervention_mixed"][
            "actual_teacher_executed_intervention_steps"
        ] for seed in MODEL_SEEDS for task in TASKS
    )
    for seed in MODEL_SEEDS:
        for task in TASKS:
            for arm in ACQUIRE_ARMS:
                row = acquired[str(seed)][task][arm]
                if (row["actual_new_native_actions"] != ACQUIRE_BUDGET
                        or row["actual_new_public_teacher_queries"] != ACQUIRE_BUDGET):
                    raise ValueError("Stage13 registered real acquisition budget missing")
    support_acquisition = (
        actual_teacher_steps >= MIN_TEACHER_EXECUTED
        and effective_open >= MIN_EFFECTIVE_ACQUIRE_DOOR_OPEN
    )
    task_gains, seed_gains = {}, {seed: 0 for seed in MODEL_SEEDS}
    mismatches, changed_actor, changed_base = {}, {}, {}
    pairs = {}
    actor_goals = treated_goals = doorkey_goals = 0
    for task in TASKS:
        layouts = list(INTERVENE_MONITOR_LAYOUTS[task])
        actor = monitor[f"{task}:actor_only"]["episodes"]
        if [row["layout_seed"] for row in actor] != layouts:
            raise ValueError("Stage13 once-measured actor source layouts misaligned")
        actor_goals += sum(int(r["external_full_physical_goal"]) for r in actor)
        mismatches[task], changed_actor[task], changed_base[task] = {}, {}, {}
        model_pairs = []
        for seed in MODEL_SEEDS:
            rows = {
                arm: monitor[f"{task}:{arm}:{seed}"]["episodes"] for arm in ARMS
            }
            if any(
                [r["layout_seed"] for r in arr] != layouts for arr in rows.values()
            ):
                raise ValueError("Stage13 native trained policy layout mismatch")
            expert = rows["extra_expert"]
            intervention = rows["intervention_mixed"]
            diff = np.asarray([
                int(i["external_full_physical_goal"])
                - int(e["external_full_physical_goal"])
                for i, e in zip(intervention, expert, strict=True)
            ], dtype=np.int64)
            model_pairs.append(diff)
            seed_gains[seed] += int(diff.sum())
            treated_goals += sum(int(r["external_full_physical_goal"]) for r in intervention)
            if task == "doorkey":
                doorkey_goals += sum(
                    int(r["external_full_physical_goal"]) for r in intervention
                )
            for dest, other in (
                (mismatches, expert), (changed_actor, actor),
                (changed_base, rows["base_bc"]),
            ):
                dest[task][str(seed)] = sum(
                    x["actual_executed_action_sha256"]
                    != y["actual_executed_action_sha256"]
                    for x, y in zip(intervention, other, strict=True)
                )
        pairs[task] = np.asarray(model_pairs, dtype=np.int64)
        task_gains[task] = int(pairs[task].sum())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.zeros(20000, dtype=np.int64)
    for task in TASKS:
        values = pairs[task]
        model_ix = rng.integers(0, len(MODEL_SEEDS), size=(20000, 3))
        layout_ix = rng.integers(
            0, values.shape[1], size=(20000, 3, values.shape[1])
        )
        draws += values[model_ix[:, :, None], layout_ix].sum(axis=(1, 2))
    interval = [float(v) for v in np.quantile(draws, (0.025, 0.975))]
    support_path = sum(
        all(mismatches[t][str(seed)] >= 16 for t in TASKS)
        for seed in MODEL_SEEDS
    ) >= 2
    gain = int(sum(task_gains.values()))
    noninferior = treated_goals >= 3 * actor_goals - ACTOR_MARGIN
    numeric = bool(
        gain >= MIN_GAIN
        and all(task_gains[t] > 0 for t in TASKS)
        and sum(seed_gains[s] > 0 for s in MODEL_SEEDS) >= 2
        and noninferior
        and doorkey_goals >= MIN_MONITOR_DOOR_GOALS
    )
    passed = support_acquisition and support_path and numeric and interval[0] > 0
    return {
        "registered_decision": (
            "INSUFFICIENT" if not support_acquisition or not support_path
            else "PASS" if passed else "FAIL"
        ),
        "equal_actual_native_acquisition_budget_per_seed_task_arm": ACQUIRE_BUDGET,
        "intervention_actual_teacher_executed_native_steps": actual_teacher_steps,
        "intervention_actual_effective_acquired_door_open_events": effective_open,
        "necessary_real_teacher_intervention_coverage": support_acquisition,
        "necessary_learned_native_action_distinction": support_path,
        "intervention_minus_expert_paired_native_action_hash_mismatches": mismatches,
        "intervention_minus_once_actor_native_action_hash_mismatches": changed_actor,
        "intervention_minus_base_native_action_hash_mismatches": changed_base,
        "strict_physical_gain_intervention_minus_expert": gain,
        "per_task_strict_physical_gain": task_gains,
        "per_model_seed_strict_physical_gain": {
            str(k): int(v) for k, v in seed_gains.items()
        },
        "intervention_strict_physical_goals": treated_goals,
        "actor_only_goals_measured_once": actor_goals,
        "actor_goals_repeated_three_times_for_arithmetic_only": 3 * actor_goals,
        "within_preregistered_actor_goal_margin": noninferior,
        "intervention_doorkey_strict_goals": doorkey_goals,
        "preregistered_numerical_physical_gate": numeric,
        "paired_seed_task_stratified_bootstrap_95pct_interval": interval,
        "bootstrap_lower_95pct_strictly_positive": interval[0] > 0,
        "target_unlockpickup_evaluated": False,
    }


def execute(output: str) -> dict:
    if output != f"{OUT}/summary.json" or Path(output).exists():
        raise ValueError("Stage13 output must be a fresh isolated registered path")
    validate_split_contract()
    source_sha = admitted_source_commit()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    x, y, base_audit = collect_base()
    weights = class_weights(y)
    bases, models, fits = {}, {}, []
    for seed in MODEL_SEEDS:
        model, profile = fit_from_scratch(x, y, seed, "base_bc", weights)
        profile["numeric_checkpoint"] = save_numeric(
            model, path.parent / f"base_bc-{seed}.npz"
        )
        bases[seed] = model
        models[("base_bc", seed)] = model
        fits.append(profile)
    acquired, acquisition_audit = acquire_all(bases)
    for seed in MODEL_SEEDS:
        for arm in ACQUIRE_ARMS:
            new_x, new_y = acquired[seed][arm]
            final_x = np.concatenate([x, new_x])
            final_y = np.concatenate([y, new_y])
            # Stage12 maintained fitting routine is identical for both arms.
            # 'corrective_onpolicy' is a local fitting API alias ONLY;
            # recorded Stage13 arm remains 'intervention_mixed'.
            fit_alias = "extra_expert" if arm == "extra_expert" else "corrective_onpolicy"
            model, profile = fit_from_scratch(
                final_x, final_y, seed, fit_alias, weights
            )
            profile["arm"] = arm
            profile["new_train_native_steps"] = len(new_y)
            profile["new_public_teacher_advisory_labels"] = len(new_y)
            profile["numeric_checkpoint"] = save_numeric(
                model, path.parent / f"{arm}-{seed}.npz"
            )
            models[(arm, seed)] = model
            fits.append(profile)
    monitor = acquire_monitor(models)
    decision = decide(monitor, acquisition_audit)
    native_acquisition_steps = sum(
        acquisition_audit[str(seed)][task][arm]["actual_new_native_actions"]
        for seed in MODEL_SEEDS for task in TASKS for arm in ACQUIRE_ARMS
    )
    native_monitor_steps = sum(
        group["summary"]["actual_native_actions"] for group in monitor.values()
    )
    if native_acquisition_steps != (
        len(MODEL_SEEDS) * len(TASKS) * len(ACQUIRE_ARMS) * ACQUIRE_BUDGET
    ):
        raise ValueError("Stage13 registered native TRAIN acquisition budget was not met")
    result = {
        "study": "m23_stage13_matched_native_action_source_public_intervention_bc",
        "exact_execution_source_sha": source_sha,
        "initial_train_layouts": {
            t: list(v) for t, v in INTERVENE_TRAIN_LAYOUTS.items()
        },
        "acquisition_train_layouts": {
            t: list(v) for t, v in INTERVENE_ACQUIRE_LAYOUTS.items()
        },
        "fresh_monitor_layouts": {
            t: list(v) for t, v in INTERVENE_MONITOR_LAYOUTS.items()
        },
        "arms": list(ARMS), "model_seeds": list(MODEL_SEEDS),
        "acquisition_native_steps_per_arm_seed_task": ACQUIRE_BUDGET,
        "acquisition_episode_max_native_steps": ACQUIRE_HORIZON,
        "initial_train_native_expert_audit": base_audit,
        "actual_native_acquisition_with_public_advice_and_physical_witnesses":
            acquisition_audit,
        "independent_strict_native_source_monitor": monitor,
        "registered_scientific_decision": decision,
        "fits_numeric_checkpoint_manifests": fits,
        "actual_new_native_initial_source_steps": base_audit["actual_native_steps"],
        "actual_new_native_acquisition_source_steps": native_acquisition_steps,
        "actual_new_native_monitor_source_steps": native_monitor_steps,
        "actual_new_native_total_source_steps":
            base_audit["actual_native_steps"]
            + native_acquisition_steps + native_monitor_steps,
        "actual_fresh_model_fits": len(fits),
        "actual_optimizer_step_calls": sum(
            p["actual_optimizer_step_calls"] for p in fits
        ),
        "cached_label_presentations_not_new_native_actions": sum(
            p["cached_source_label_presentations"] for p in fits
        ),
        "imagined_planner_actions": 0,
        "target_unlockpickup_actual_native_steps": 0,
        "target_FINAL_episodes": 0,
        "platform_versions": {
            name: version(name) for name in ("numpy", "torch", "gymnasium", "minigrid")
        },
        "actual_study_wall_seconds": time.perf_counter() - started,
    }
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_sha": source_sha,
        "native_initial_steps": result["actual_new_native_initial_source_steps"],
        "native_acquisition_steps": native_acquisition_steps,
        "native_monitor_steps": native_monitor_steps,
        "actual_model_fits": len(fits),
        "actual_optimizer_steps": result["actual_optimizer_step_calls"],
        "real_teacher_intervention_actions":
            decision["intervention_actual_teacher_executed_native_steps"],
        "real_acquired_door_open_events":
            decision["intervention_actual_effective_acquired_door_open_events"],
        "paired_strict_goal_gain":
            decision["strict_physical_gain_intervention_minus_expert"],
        "registered_decision": decision["registered_decision"],
    }), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=f"{OUT}/summary.json")
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    if not options.execute:
        raise SystemExit("Stage13 real native source research needs --execute")
    execute(options.output)


if __name__ == "__main__":
    main()
