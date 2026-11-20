"""Stage23: teacher-free recurrent recovery learning from executed expert suffixes.

Physically execute one shared frozen learner->public expert TRAIN acquisition.
Compare equal optimizer/label budgets for original source replay versus genuine
native post-open recovery. Evaluate ONLY learned GRUs on fresh held-out sources.
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
from bagi.experiments.m23_executed_recovery_acquisition import (
    ACQUIRE_BUDGET,
    TASKS,
    collect_acquired,
    collect_teacher,
)
from bagi.experiments.m23_executed_recovery_train import (
    ARMS,
    FOLLOWUP_UPDATES,
    TOKENS_PER_UPDATE,
    fit_followup,
)
from bagi.experiments.m23_source_feasibility import HORIZON
from bagi.experiments.m23_source_interaction_context_bc import (
    MODEL_SEEDS,
    InteractionContextGRU,
    fit_model,
    public_action_features,
)
from bagi.experiments.m23_source_mission_bc import make_native_audit, make_source_actor
from bagi.experiments.m23_study_contract import (
    RECOVERY_ACQUIRE_LAYOUTS,
    RECOVERY_MONITOR_LAYOUTS,
    RECOVERY_TRAIN_LAYOUTS,
    validate_split_contract,
)

OUT = "runs/m23_source_executed_recovery_learning"
BOOTSTRAP_SEED = 232301
MIN_RECOVERY_NATIVE_LABELS_PER_SEED = 128
MIN_PUBLIC_TAKEOVERS_PER_SEED = 8
MIN_DISTINCT_DOORKEY_TRAJECTORIES = 16
MIN_GOAL_GAIN = 12
MIN_MONITOR_DOOR_GOALS = 16
ACTOR_MARGIN = 24


def monitor_episode(
    task: str, layout: int, arm: str, seed: int | None,
    model: InteractionContextGRU | None,
) -> dict:
    if task not in TASKS or layout not in RECOVERY_MONITOR_LAYOUTS[task]:
        raise ValueError("Stage23 MONITOR must use independent unopened native split")
    if arm == "actor_only":
        if seed is not None or model is not None:
            raise ValueError("Stage23 original actor measured once per layout")
        actor = make_source_actor(task)
    elif arm in ARMS and seed in MODEL_SEEDS and model is not None:
        if model.training:
            raise ValueError("Stage23 learned source MONITOR model must be frozen")
        actor = Controller("public_memory")
    else:
        raise ValueError("Stage23 source learned arm/seed not registered")
    rng = np.random.default_rng(8_800_000 + layout)
    audit = make_native_audit(task)
    previous: int | None = None
    hidden = None
    trace: list[int] = []
    reward_sum = 0.0
    finished = False
    try:
        obs, _ = audit.reset(seed=layout)
        for _ in range(HORIZON):
            if model is None:
                action = int(actor.choose(obs, rng))
            else:
                vector = public_action_features(
                    obs, actor.key_observed_after_pickup, previous,
                    arm="interaction_context",
                )
                action, hidden = model.native_step(vector, hidden)
            if action not in range(7):
                raise ValueError("Stage23 model selected illegal native action")
            nxt, reward, done, truncated, _ = audit.step(action)
            actor.update(obs, action, nxt)
            trace.append(action)
            previous = action
            reward_sum += float(reward)
            obs = nxt
            if done or truncated:
                finished = True
                break
        truth = audit.summary()
        if (not finished or not trace or truth["native_actions"] != len(trace)
                or sum(truth["action_counts"].values()) != len(trace)):
            raise ValueError("Stage23 independent MONITOR true native actions inconsistent")
        return {
            "task": task, "layout_seed": layout, "arm": arm, "model_seed": seed,
            "actual_native_steps": len(trace),
            "native_true_action_histogram": [
                truth["action_counts"][str(i)] for i in range(7)
            ],
            "full_native_executed_action_sha256":
                hashlib.sha256(bytes(trace)).hexdigest(),
            "external_strict_full_physical_goal": bool(truth["full_physical_goal"]),
            "external_rewarded_terminations": truth["rewarded_terminations"],
            "external_effective_pickup_events": truth["effective_pickup_events"],
            "external_effective_open_events": (
                truth["effective_door_open_events"] if task == "doorkey" else 0
            ),
            "external_wrong_target_pickups": truth.get("wrong_target_pickup_events", 0),
            "actual_reward_sum": reward_sum,
        }
    finally:
        audit.env.close()


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Stage23 MONITOR source native episodes missing")
    total = sum(r["actual_native_steps"] for r in rows)
    hist = [
        sum(r["native_true_action_histogram"][i] for r in rows)
        for i in range(7)
    ]
    if total != sum(hist):
        raise ValueError("Stage23 MONITOR true native seven-action histogram mismatch")
    return {
        "actual_native_episodes": len(rows),
        "actual_native_steps": total,
        "native_true_action_histogram": hist,
        "strict_full_physical_goals": sum(
            int(r["external_strict_full_physical_goal"]) for r in rows
        ),
        "physical_effective_open_episodes": sum(
            int(r["external_effective_open_events"] > 0) for r in rows
        ),
        "reward_only_nonqualifying_episodes": sum(
            int(r["external_rewarded_terminations"] > 0
                and not r["external_strict_full_physical_goal"]) for r in rows
        ),
    }


def acquire_monitor(
    models: dict[tuple[str, int], InteractionContextGRU],
) -> dict:
    if (set(models) != {(a, s) for a in ARMS for s in MODEL_SEEDS}
            or any(m.training for m in models.values())):
        raise ValueError("Stage23 exactly six teacher-free final learned models required")
    result = {}
    for task in TASKS:
        layouts = RECOVERY_MONITOR_LAYOUTS[task]
        actor = [
            monitor_episode(task, layout, "actor_only", None, None)
            for layout in layouts
        ]
        result[f"{task}:actor_only"] = {
            "episodes": actor, "summary": summarize(actor)
        }
        for arm in ARMS:
            for seed in MODEL_SEEDS:
                rows = [
                    monitor_episode(task, layout, arm, seed, models[(arm, seed)])
                    for layout in layouts
                ]
                result[f"{task}:{arm}:{seed}"] = {
                    "episodes": rows, "summary": summarize(rows)
                }
    return result


def decide(monitor: dict, acquisition: dict) -> dict:
    expected = {f"{task}:actor_only" for task in TASKS} | {
        f"{task}:{arm}:{seed}"
        for task in TASKS for arm in ARMS for seed in MODEL_SEEDS
    }
    if set(monitor) != expected:
        raise ValueError("Stage23 source MONITOR groups incomplete")
    actor_goals = treatment_goals = treatment_door_goals = 0
    by_seed = {s: 0 for s in MODEL_SEEDS}
    door_seed = {s: 0 for s in MODEL_SEEDS}
    by_task = {}
    native_path_diffs = {}
    pairs = {}
    for task in TASKS:
        layouts = list(RECOVERY_MONITOR_LAYOUTS[task])
        actor = monitor[f"{task}:actor_only"]["episodes"]
        if [r["layout_seed"] for r in actor] != layouts:
            raise ValueError("Stage23 source actor MONITOR not paired")
        actor_goals += sum(int(x["external_strict_full_physical_goal"]) for x in actor)
        task_pairs = []
        native_path_diffs[task] = {}
        for seed in MODEL_SEEDS:
            ctrl = monitor[f"{task}:source_replay:{seed}"]["episodes"]
            trt = monitor[f"{task}:executed_rescue:{seed}"]["episodes"]
            if any([r["layout_seed"] for r in rows] != layouts
                   for rows in (ctrl, trt)):
                raise ValueError("Stage23 source control/treatment independent layout mismatch")
            diffs = np.asarray([
                int(b["external_strict_full_physical_goal"])
                - int(a["external_strict_full_physical_goal"])
                for a, b in zip(ctrl, trt, strict=True)
            ], dtype=np.int64)
            task_pairs.append(diffs)
            by_seed[seed] += int(diffs.sum())
            treatment_goals += sum(
                int(r["external_strict_full_physical_goal"]) for r in trt
            )
            if task == "doorkey":
                door_seed[seed] += int(diffs.sum())
                treatment_door_goals += sum(
                    int(r["external_strict_full_physical_goal"]) for r in trt
                )
            native_path_diffs[task][str(seed)] = sum(
                a["full_native_executed_action_sha256"]
                != b["full_native_executed_action_sha256"]
                for a, b in zip(ctrl, trt, strict=True)
            )
        pairs[task] = np.asarray(task_pairs, dtype=np.int64)
        by_task[task] = int(pairs[task].sum())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.zeros(20_000, dtype=np.int64)
    for task in TASKS:
        values = pairs[task]
        seeds = rng.integers(0, 3, size=(20_000, 3))
        layouts = rng.integers(0, 40, size=(20_000, 3, 40))
        bootstrap += values[seeds[:, :, None], layouts].sum(axis=(1, 2))
    ci = [float(x) for x in np.quantile(bootstrap, (.025, .975))]
    native_label_counts = {
        str(seed): acquisition["per_seed_task"][f"doorkey:{seed}"][
            "actually_executed_teacher_suffix_native_actions"
        ] for seed in MODEL_SEEDS
    }
    native_takeover_counts = {
        str(seed): acquisition["per_seed_task"][f"doorkey:{seed}"][
            "public_witnessed_effectful_door_open_episodes"
        ] for seed in MODEL_SEEDS
    }
    nonvacuous = bool(
        all(
            native_label_counts[str(s)] >= MIN_RECOVERY_NATIVE_LABELS_PER_SEED
            and native_takeover_counts[str(s)] >= MIN_PUBLIC_TAKEOVERS_PER_SEED
            for s in MODEL_SEEDS
        )
        and sum(
            native_path_diffs["doorkey"][str(s)]
            >= MIN_DISTINCT_DOORKEY_TRAJECTORIES for s in MODEL_SEEDS
        ) >= 2
    )
    gain = int(sum(by_task.values()))
    noninferior = treatment_goals >= 3 * actor_goals - ACTOR_MARGIN
    numeric = bool(
        gain >= MIN_GOAL_GAIN
        and by_task["doorkey"] >= MIN_GOAL_GAIN
        and by_task["fetch"] >= -4
        and sum(door_seed[s] > 0 for s in MODEL_SEEDS) >= 2
        and treatment_door_goals >= MIN_MONITOR_DOOR_GOALS
        and noninferior
    )
    return {
        "registered_decision": (
            "INSUFFICIENT" if not nonvacuous
            else "PASS" if numeric and ci[0] > 0 else "FAIL"
        ),
        "genuine_native_recovery_and_behavior_nonvacuity": nonvacuous,
        "registered_physical_numeric_gate": numeric,
        "paired_executed_rescue_minus_source_replay_strict_goal_gain": gain,
        "paired_strict_goal_gain_by_task": by_task,
        "paired_strict_goal_gain_by_model_seed": {
            str(s): int(by_seed[s]) for s in MODEL_SEEDS
        },
        "paired_doorkey_goal_gain_by_model_seed": {
            str(s): int(door_seed[s]) for s in MODEL_SEEDS
        },
        "native_full_action_sha256_differences": native_path_diffs,
        "real_teacher_executed_suffix_action_counts": native_label_counts,
        "real_public_open_handoff_episode_counts": native_takeover_counts,
        "paired_seed_task_20000_bootstrap_95pct_interval": ci,
        "bootstrap_lower_strictly_positive": ci[0] > 0,
        "once_measured_source_actor_strict_goals": actor_goals,
        "source_actor_arithmetic_three_times": 3 * actor_goals,
        "source_actor_margin_goal_threshold": 3 * actor_goals - ACTOR_MARGIN,
        "source_actor_noninferiority": noninferior,
        "executed_recovery_learned_full_goals": treatment_goals,
        "executed_recovery_learned_doorkey_full_goals": treatment_door_goals,
        "evaluation_teacher_free": True,
        "original_target_unlockpickup_final_touched": False,
    }


def enough_executed_recovery_evidence(acquisition: dict) -> bool:
    """Gate MONITOR access on genuine native rescue exposure, before any fitting."""
    counts = acquisition["per_seed_task"]
    if set(counts) != {
        f"{task}:{seed}" for task in TASKS for seed in MODEL_SEEDS
    }:
        raise ValueError("Stage23 missing actual TRAIN acquisition source/task strata")
    return all(
        counts[f"doorkey:{seed}"]["actually_executed_teacher_suffix_native_actions"]
        >= MIN_RECOVERY_NATIVE_LABELS_PER_SEED
        and counts[f"doorkey:{seed}"]["public_witnessed_effectful_door_open_episodes"]
        >= MIN_PUBLIC_TAKEOVERS_PER_SEED
        for seed in MODEL_SEEDS
    )


def execute(output: str) -> dict:
    if output != f"{OUT}/summary.json" or Path(output).exists():
        raise ValueError("Stage23 scientific output must be fresh and isolated")
    validate_split_contract()
    source_sha = admitted_source_commit()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    teacher, real_source_y, teacher_meta = collect_teacher()
    weights = class_weights(real_source_y)
    frozen = {}
    frozen_profiles = []
    for seed in MODEL_SEEDS:
        model, profile = fit_model(teacher, seed, "interaction_context", weights)
        profile["numeric_checkpoint"] = save_numeric(
            model, path.parent / f"frozen-source-{seed}.npz"
        )
        frozen[seed] = model
        frozen_profiles.append(profile)
    if len(frozen_profiles) != 3 or len({
        p["num_parameters"] for p in frozen_profiles
    }) != 1:
        raise ValueError("Stage23 frozen source original policy capacity mismatch")
    acquired, acquired_meta = collect_acquired(frozen)
    if acquired_meta["real_shared_native_acquisition_steps"] != 6 * ACQUIRE_BUDGET:
        raise ValueError("Stage23 exact fixed shared native TRAIN acquisition broken")
    if not enough_executed_recovery_evidence(acquired_meta):
        # The precommitted real-event gates forbid opening independent held-out
        # MONITOR if the actual TRAIN acquisition could not support recovery.
        report = {
            "study": "m23_stage23_source_true_executed_post_open_recovery_learning",
            "execution_source_commit": source_sha,
            "registered_scientific_decision": {
                "registered_decision": "INSUFFICIENT",
                "reason": "genuine executed recovery TRAIN action/open threshold not met",
                "independent_monitor_not_opened": True,
            },
            "actual_new_native_source_teacher_steps": teacher_meta[
                "real_teacher_native_steps"
            ],
            "actual_new_native_shared_acquisition_steps": acquired_meta[
                "real_shared_native_acquisition_steps"
            ],
            "actual_native_monitor_steps": 0,
            "actual_total_real_native_source_steps": (
                teacher_meta["real_teacher_native_steps"]
                + acquired_meta["real_shared_native_acquisition_steps"]
            ),
            "real_source_teacher_train": teacher_meta,
            "actual_shared_native_learner_teacher_acquisition": acquired_meta,
            "frozen_source_numeric_checkpoints": frozen_profiles,
            "real_optimizer_updates": sum(
                p["actual_optimizer_step_calls"] for p in frozen_profiles
            ),
            "original_target_unlockpickup_native_steps": 0,
            "original_target_final_episodes": 0,
        }
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "source": source_sha, "decision": "INSUFFICIENT",
            "monitor_native_steps": 0,
        }), flush=True)
        return report
    models = {}
    final_profiles = []
    for seed in MODEL_SEEDS:
        for arm in ARMS:
            model, profile = fit_followup(
                teacher, acquired[seed], frozen[seed], seed, arm, weights
            )
            profile["numeric_checkpoint"] = save_numeric(
                model, path.parent / f"{arm}-{seed}.npz"
            )
            models[(arm, seed)] = model
            final_profiles.append(profile)
    if (len(final_profiles) != 6
            or len({p["model_parameters"] for p in final_profiles}) != 1
            or any(
                a["frozen_base_model_sha256"] != b["frozen_base_model_sha256"]
                or a["real_optimizer_updates"] != b["real_optimizer_updates"]
                or a["actual_cached_label_presentations_not_native_steps"]
                != b["actual_cached_label_presentations_not_native_steps"]
                for a, b in zip(final_profiles[::2], final_profiles[1::2], strict=True)
            )
            or any(
                p["real_optimizer_updates"] != FOLLOWUP_UPDATES
                or p["fixed_supervised_native_labels_per_update"] != TOKENS_PER_UPDATE
                for p in final_profiles
            )):
        raise ValueError("Stage23 control/treatment initial weights and budget unmatched")
    monitor = acquire_monitor(models)
    verdict = decide(monitor, acquired_meta)
    monitor_steps = sum(
        group["summary"]["actual_native_steps"] for group in monitor.values()
    )
    total = (
        teacher_meta["real_teacher_native_steps"]
        + acquired_meta["real_shared_native_acquisition_steps"]
        + monitor_steps
    )
    result = {
        "study": "m23_stage23_source_true_executed_post_open_recovery_learning",
        "execution_source_commit": source_sha,
        "fresh_source_teacher_train_layouts": {
            task: list(xs) for task, xs in RECOVERY_TRAIN_LAYOUTS.items()
        },
        "fresh_true_native_acquisition_layouts": {
            task: list(xs) for task, xs in RECOVERY_ACQUIRE_LAYOUTS.items()
        },
        "fresh_independent_native_monitor_layouts": {
            task: list(xs) for task, xs in RECOVERY_MONITOR_LAYOUTS.items()
        },
        "model_seeds": list(MODEL_SEEDS), "final_learned_arms": list(ARMS),
        "source_teacher_real_train": teacher_meta,
        "actual_shared_native_learner_teacher_acquisition": acquired_meta,
        "source_independent_native_monitor": monitor,
        "actual_new_native_source_teacher_steps": teacher_meta["real_teacher_native_steps"],
        "actual_new_native_shared_acquisition_steps":
            acquired_meta["real_shared_native_acquisition_steps"],
        "actual_native_monitor_steps": monitor_steps,
        "actual_total_real_native_source_steps": total,
        "actual_frozen_source_model_fits": len(frozen_profiles),
        "actual_final_model_fits": len(final_profiles),
        "actual_optimizer_step_calls": sum(
            p["actual_optimizer_step_calls"] for p in frozen_profiles
        ) + sum(p["real_optimizer_updates"] for p in final_profiles),
        "total_cached_label_presentations_not_native_actions": sum(
            p["cached_source_label_presentations_not_native_steps"]
            for p in frozen_profiles
        ) + sum(p["actual_cached_label_presentations_not_native_steps"]
                for p in final_profiles),
        "frozen_source_numeric_checkpoints": frozen_profiles,
        "final_learned_numeric_checkpoints": final_profiles,
        "registered_scientific_decision": verdict,
        "original_target_unlockpickup_native_steps": 0,
        "original_target_final_episodes": 0,
        "new_imagined_native_steps": 0,
        "actual_study_wall_seconds": time.perf_counter() - started,
        "versions": {
            name: version(name) for name in ("numpy", "torch", "gymnasium", "minigrid")
        },
    }
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source": source_sha,
        "teacher_native_steps": result["actual_new_native_source_teacher_steps"],
        "acquired_native_steps": result["actual_new_native_shared_acquisition_steps"],
        "monitor_native_steps": monitor_steps,
        "true_model_fits": len(frozen_profiles) + len(final_profiles),
        "source_native_goal_gain":
            verdict["paired_executed_rescue_minus_source_replay_strict_goal_gain"],
        "decision": verdict["registered_decision"],
    }), flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=f"{OUT}/summary.json")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if not args.execute:
        raise SystemExit("Stage23 true native recovery experiment requires --execute")
    execute(args.output)


if __name__ == "__main__":
    main()
