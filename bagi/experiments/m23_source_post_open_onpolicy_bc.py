"""M23 Stage20 equal-real-action post-open on-policy public corrective GRU BC.

Exactly one shared frozen-learner visitation stream per model seed; two matched
final objective arms differ ONLY in temporal location of boosted public expert
ADVISORY training labels, with no private evaluator signals during acting.
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
from bagi.experiments.m23_post_open_acquisition import (
    ACQUIRE_BUDGET,
    TASKS,
    collect_acquired,
    collect_teacher,
)
from bagi.experiments.m23_post_open_corrected_train import (
    ARMS,
    fit_corrected,
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
    POST_CORRECT_ACQUIRE_LAYOUTS,
    POST_CORRECT_MONITOR_LAYOUTS,
    POST_CORRECT_TRAIN_LAYOUTS,
    validate_split_contract,
)

OUT = "runs/m23_source_post_open_onpolicy_bc"
BOOTSTRAP_SEED = 232001
MIN_GAIN = 12
MIN_MONITOR_DOOR_GOALS = 16
ACTOR_MARGIN = 24
MIN_ACQUIRED_EFFECTFUL_OPEN_EPISODES = 16
MIN_PUBLIC_POST_OPEN_ADVICE_DISAGREEMENTS = 128
MIN_DOORKEY_NATIVE_ACTION_DIFFERENCES = 16


def monitor_episode(
    task: str, layout: int, arm: str, seed: int | None,
    policy: InteractionContextGRU | None,
) -> dict:
    if task not in TASKS or layout not in POST_CORRECT_MONITOR_LAYOUTS[task]:
        raise ValueError("Stage20 MONITOR source must be independently fresh")
    if arm == "actor_only":
        if seed is not None or policy is not None:
            raise ValueError("Stage20 source actor measured once, no learned fit")
        actor = make_source_actor(task)
    elif arm in ARMS and seed in MODEL_SEEDS and policy is not None:
        if policy.training:
            raise ValueError("Stage20 MONITOR learned GRU must be eval only")
        actor = Controller("public_memory")
    else:
        raise ValueError("Stage20 MONITOR unregistered model/seed/task")
    rng = np.random.default_rng(8_800_000 + layout)
    audit = make_native_audit(task)
    previous: int | None = None
    hidden = None
    trace: list[int] = []
    reward_total = 0.0
    finished = False
    try:
        obs, _ = audit.reset(seed=layout)
        for _ in range(HORIZON):
            if policy is None:
                action = int(actor.choose(obs, rng))
            else:
                vector = public_action_features(
                    obs, actor.key_observed_after_pickup, previous,
                    arm="interaction_context",
                )
                action, hidden = policy.native_step(vector, hidden)
            if action not in range(7):
                raise ValueError("Stage20 MONITOR attempted invalid native action")
            nxt, reward, done, truncated, _ = audit.step(action)
            actor.update(obs, action, nxt)
            trace.append(action)
            previous = action
            reward_total += float(reward)
            obs = nxt
            if done or truncated:
                finished = True
                break
        truth = audit.summary()
        if (not finished or not trace or truth["native_actions"] != len(trace)
                or sum(truth["action_counts"].values()) != len(trace)):
            raise ValueError("Stage20 strict MONITOR native action accounting failed")
        return {
            "task": task, "layout_seed": layout, "arm": arm, "model_seed": seed,
            "actual_native_steps": len(trace),
            "actual_native_executed_action_sha256":
                hashlib.sha256(bytes(trace)).hexdigest(),
            "real_native_action_histogram": [
                truth["action_counts"][str(i)] for i in range(7)
            ],
            "external_strict_full_physical_goal": bool(truth["full_physical_goal"]),
            "external_rewarded_terminations": truth["rewarded_terminations"],
            "external_effective_door_open_events": (
                truth["effective_door_open_events"] if task == "doorkey" else 0
            ),
            "external_effective_pickup_events": truth["effective_pickup_events"],
            "external_effective_forward_movements":
                truth.get("effective_forward_movements", 0),
            "external_wrong_target_pickup_events":
                truth.get("wrong_target_pickup_events", 0),
            "total_actual_reward": reward_total,
        }
    finally:
        audit.env.close()


def summarize(rows: list[dict]) -> dict:
    if not rows or any(r["task"] != rows[0]["task"] for r in rows):
        raise ValueError("Stage20 real MONITOR episode group invalid")
    actual = sum(r["actual_native_steps"] for r in rows)
    counts = [
        sum(r["real_native_action_histogram"][i] for r in rows)
        for i in range(7)
    ]
    if sum(counts) != actual:
        raise ValueError("Stage20 MONITOR real action histogram mismatch")
    return {
        "real_native_episodes": len(rows), "real_native_steps": actual,
        "real_native_action_histogram": counts,
        "strict_full_physical_goals": sum(
            int(r["external_strict_full_physical_goal"]) for r in rows
        ),
        "effective_door_open_episodes": sum(
            int(r["external_effective_door_open_events"] > 0) for r in rows
        ),
        "reward_only_nonphysical_episodes": sum(
            int(r["external_rewarded_terminations"] > 0
                and not r["external_strict_full_physical_goal"]) for r in rows
        ),
    }


def acquire_monitor(
    models: dict[tuple[str, int], InteractionContextGRU],
) -> dict:
    if set(models) != {(arm, seed) for arm in ARMS for seed in MODEL_SEEDS}:
        raise ValueError("Stage20 exactly six true final policy fits required")
    monitor = {}
    for task in TASKS:
        layouts = POST_CORRECT_MONITOR_LAYOUTS[task]
        original = [
            monitor_episode(task, layout, "actor_only", None, None)
            for layout in layouts
        ]
        monitor[f"{task}:actor_only"] = {
            "episodes": original, "summary": summarize(original)
        }
        for arm in ARMS:
            for seed in MODEL_SEEDS:
                rows = [
                    monitor_episode(task, layout, arm, seed, models[(arm, seed)])
                    for layout in layouts
                ]
                monitor[f"{task}:{arm}:{seed}"] = {
                    "episodes": rows, "summary": summarize(rows)
                }
    return monitor


def decide(monitor: dict, acquisition: dict) -> dict:
    expected = {f"{task}:actor_only" for task in TASKS} | {
        f"{task}:{arm}:{seed}"
        for task in TASKS for arm in ARMS for seed in MODEL_SEEDS
    }
    if set(monitor) != expected:
        raise ValueError("Stage20 true source MONITOR missing paired models")
    per_seed = {s: 0 for s in MODEL_SEEDS}
    door_seed = {s: 0 for s in MODEL_SEEDS}
    task_gains = {}
    path_diffs = {}
    paired = {}
    actor_goals = treatment_goals = treatment_door_goals = 0
    for task in TASKS:
        layouts = list(POST_CORRECT_MONITOR_LAYOUTS[task])
        actor = monitor[f"{task}:actor_only"]["episodes"]
        if [r["layout_seed"] for r in actor] != layouts:
            raise ValueError("Stage20 actor MONITOR layouts invalid")
        actor_goals += sum(int(r["external_strict_full_physical_goal"]) for r in actor)
        paths = {}
        all_diffs = []
        for seed in MODEL_SEEDS:
            a = monitor[f"{task}:uniform_corrected:{seed}"]["episodes"]
            b = monitor[f"{task}:post_open_corrected:{seed}"]["episodes"]
            if ([r["layout_seed"] for r in a] != layouts
                    or [r["layout_seed"] for r in b] != layouts):
                raise ValueError("Stage20 native learner MONITOR layouts unpaired")
            diffs = np.asarray([
                int(y["external_strict_full_physical_goal"])
                - int(x["external_strict_full_physical_goal"])
                for x, y in zip(a, b, strict=True)
            ], dtype=np.int64)
            all_diffs.append(diffs)
            per_seed[seed] += int(diffs.sum())
            treatment_goals += sum(int(r["external_strict_full_physical_goal"]) for r in b)
            if task == "doorkey":
                door_seed[seed] += int(diffs.sum())
                treatment_door_goals += sum(
                    int(r["external_strict_full_physical_goal"]) for r in b
                )
            paths[str(seed)] = sum(
                x["actual_native_executed_action_sha256"]
                != y["actual_native_executed_action_sha256"]
                for x, y in zip(a, b, strict=True)
            )
        paired[task] = np.asarray(all_diffs, dtype=np.int64)
        task_gains[task] = int(paired[task].sum())
        path_diffs[task] = paths
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.zeros(20000, dtype=np.int64)
    for task in TASKS:
        arr = paired[task]
        seed_ix = rng.integers(0, 3, size=(20000, 3))
        layout_ix = rng.integers(0, 40, size=(20000, 3, 40))
        draws += arr[seed_ix[:, :, None], layout_ix].sum(axis=(1, 2))
    interval = [float(v) for v in np.quantile(draws, (.025, .975))]
    gain = sum(task_gains.values())
    actor_noninferior = treatment_goals >= 3 * actor_goals - ACTOR_MARGIN
    acquisition_nonvacuous = bool(
        acquisition["public_door_open_episode_count"]
        >= MIN_ACQUIRED_EFFECTFUL_OPEN_EPISODES
        and acquisition["post_open_public_teacher_advice_disagreements"]
        >= MIN_PUBLIC_POST_OPEN_ADVICE_DISAGREEMENTS
    )
    learned_distinct = sum(
        path_diffs["doorkey"][str(seed)] >= MIN_DOORKEY_NATIVE_ACTION_DIFFERENCES
        for seed in MODEL_SEEDS
    ) >= 2
    numeric = bool(
        gain >= MIN_GAIN
        and task_gains["doorkey"] >= MIN_GAIN
        and task_gains["fetch"] >= -4
        and sum(door_seed[s] > 0 for s in MODEL_SEEDS) >= 2
        and treatment_door_goals >= MIN_MONITOR_DOOR_GOALS
        and actor_noninferior
    )
    sufficient = acquisition_nonvacuous and learned_distinct
    verdict = "INSUFFICIENT" if not sufficient else (
        "PASS" if numeric and interval[0] > 0 else "FAIL"
    )
    return {
        "registered_decision": verdict,
        "registered_physical_numeric_gate": numeric,
        "real_acquisition_public_advice_nonvacuity": acquisition_nonvacuous,
        "real_native_learned_action_distinctness": learned_distinct,
        "true_native_action_hash_differences": path_diffs,
        "physical_goal_gain_post_open_minus_uniform": gain,
        "per_task_physical_goal_gain": task_gains,
        "per_model_seed_goal_gain": {str(s): int(v) for s, v in per_seed.items()},
        "per_model_seed_doorkey_goal_gain": {
            str(s): int(v) for s, v in door_seed.items()
        },
        "actor_once_source_physical_goals": actor_goals,
        "actor_tripled_arithmetic_only": 3 * actor_goals,
        "actor_noninferiority_goal_threshold": 3 * actor_goals - ACTOR_MARGIN,
        "post_open_corrected_strict_full_goals": treatment_goals,
        "post_open_corrected_doorkey_full_goals": treatment_door_goals,
        "actor_noninferior": actor_noninferior,
        "paired_seed_task_bootstrap_95pct_goal_gain_interval": interval,
        "bootstrap_lower_bound_above_zero": interval[0] > 0,
        "original_unlockpickup_target_final_touched": False,
    }


def execute(output: str) -> dict:
    if output != f"{OUT}/summary.json" or Path(output).exists():
        raise ValueError("Stage20 requires fresh isolated output path")
    validate_split_contract()
    commit = admitted_source_commit()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    start = time.perf_counter()
    teacher, source_y, teacher_meta = collect_teacher()
    class_w = class_weights(source_y)
    frozen = {}
    base_manifests = []
    for seed in MODEL_SEEDS:
        model, profile = fit_model(teacher, seed, "interaction_context", class_w)
        profile["numeric_checkpoint"] = save_numeric(
            model, path.parent / f"frozen-base-{seed}.npz"
        )
        frozen[seed] = model
        base_manifests.append(profile)
    if len(base_manifests) != 3 or len({
        m["num_parameters"] for m in base_manifests
    }) != 1:
        raise ValueError("Stage20 source frozen base architecture mismatch")
    acquired, acq_meta = collect_acquired(frozen)
    if acq_meta["actual_new_acquisition_native_actions"] != 6 * ACQUIRE_BUDGET:
        raise ValueError("Stage20 real shared native source acquisition budget mismatch")
    final_models = {}
    final_profiles = []
    for seed in MODEL_SEEDS:
        for arm in ARMS:
            model, profile = fit_corrected(teacher, acquired[seed], seed, arm, class_w)
            profile["numeric_checkpoint"] = save_numeric(
                model, path.parent / f"{arm}-{seed}.npz"
            )
            final_models[(arm, seed)] = model
            final_profiles.append(profile)
    if (len(final_profiles) != 6
            or len({p["num_parameters"] for p in final_profiles}) != 1
            or any(
                a["actual_optimizer_step_calls"] != b["actual_optimizer_step_calls"]
                or a["cached_label_presentations_not_new_native_steps"]
                != b["cached_label_presentations_not_new_native_steps"]
                or a["public_advice_4x_boost_mask_counts"]
                != b["public_advice_4x_boost_mask_counts"]
                for a, b in zip(final_profiles[::2], final_profiles[1::2], strict=True)
            )):
        raise ValueError("Stage20 paired final fit capacity/label exposure invalid")
    monitored = acquire_monitor(final_models)
    result_gate = decide(monitored, acq_meta)
    monitor_steps = sum(
        group["summary"]["real_native_steps"] for group in monitored.values()
    )
    real_total = (
        teacher_meta["real_teacher_executed_native_steps"]
        + acq_meta["actual_new_acquisition_native_actions"]
        + monitor_steps
    )
    result = {
        "study": "m23_stage20_post_open_onpolicy_source_recurrent_corrective_bc",
        "execution_source_commit": commit,
        "fresh_teacher_train_source_layouts": {
            task: list(ids) for task, ids in POST_CORRECT_TRAIN_LAYOUTS.items()
        },
        "fresh_acquisition_train_source_layouts": {
            task: list(ids) for task, ids in POST_CORRECT_ACQUIRE_LAYOUTS.items()
        },
        "independent_monitor_source_layouts": {
            task: list(ids) for task, ids in POST_CORRECT_MONITOR_LAYOUTS.items()
        },
        "fitted_model_seeds": list(MODEL_SEEDS), "final_arms": list(ARMS),
        "teacher_real_train": teacher_meta,
        "real_shared_learner_native_acquisition": acq_meta,
        "real_native_monitor": monitored,
        "real_original_teacher_native_train_steps":
            teacher_meta["real_teacher_executed_native_steps"],
        "real_shared_actual_native_acquisition_steps":
            acq_meta["actual_new_acquisition_native_actions"],
        "real_native_monitor_steps": monitor_steps,
        "total_real_native_source_env_steps": real_total,
        "frozen_base_model_fits": len(base_manifests),
        "final_paired_model_fits": len(final_profiles),
        "real_optimizer_step_calls": sum(
            m["actual_optimizer_step_calls"]
            for m in base_manifests + final_profiles
        ),
        "cached_label_presentations_not_new_native_steps": sum(
            m.get(
                "cached_label_presentations_not_new_native_steps",
                m.get("cached_source_label_presentations_not_native_steps", 0),
            ) for m in base_manifests + final_profiles
        ),
        "frozen_base_numeric_checkpoints": base_manifests,
        "final_numeric_checkpoints": final_profiles,
        "registered_scientific_decision": result_gate,
        "original_unlockpickup_target_native_actions": 0,
        "original_target_final_episodes": 0,
        "new_imagined_native_steps": 0,
        "actual_study_wall_seconds": time.perf_counter() - start,
        "versions": {
            name: version(name)
            for name in ("numpy", "torch", "gymnasium", "minigrid")
        },
    }
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source": commit, "train_real_steps": result["real_original_teacher_native_train_steps"],
        "acquisition_real_steps": result["real_shared_actual_native_acquisition_steps"],
        "monitor_real_steps": monitor_steps,
        "real_fits": len(base_manifests) + len(final_profiles),
        "actual_optimizer_steps": result["real_optimizer_step_calls"],
        "paired_strict_goal_gain": result_gate["physical_goal_gain_post_open_minus_uniform"],
        "decision": result_gate["registered_decision"],
    }), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=f"{OUT}/summary.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Stage20 source native real acquisition requires --execute")
    execute(args.output)


if __name__ == "__main__":
    main()
