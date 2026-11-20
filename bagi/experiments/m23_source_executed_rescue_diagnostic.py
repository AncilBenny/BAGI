"""Stage22 real native source diagnostic: public teacher rescues after learner-opened door.

Paired runs have identical frozen learned prefixes. The expert takes control only
after a genuinely executed action5 visibly opens a door. A positive diagnostic
is teacher-assisted, NOT autonomous learned competence or target transfer.
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
from bagi.experiments.m23_source_feasibility import HORIZON
from bagi.experiments.m23_source_interaction_context_bc import (
    MODEL_SEEDS,
    InteractionContextGRU,
    fit_model,
    public_action_features,
)
from bagi.experiments.m23_source_mission_bc import make_native_audit, make_source_actor
from bagi.experiments.m23_source_post_open_continuation_bc import (
    public_door_open_transition,
)
from bagi.experiments.m23_study_contract import (
    RESCUE_MONITOR_LAYOUTS,
    RESCUE_TRAIN_LAYOUTS,
    validate_split_contract,
)

OUT = "runs/m23_source_executed_rescue_diagnostic"
TASKS = ("doorkey", "fetch")
ARMS = ("learner_continues", "teacher_after_open")
BOOTSTRAP_SEED = 232201
MIN_OBSERVED_OPENS = 16
MIN_EXECUTED_TEACHER_EPISODES = 16
MIN_DISTINCT_DOORKEY_PATHS_PER_SEED = 16
MIN_RESCUE_GOAL_GAIN = 12
MIN_RESCUE_DOORKEY_GOALS = 16


def live_teacher_action(
    arm: str, switched: bool, learned_action: int,
    teacher, observation: dict, rng: np.random.Generator,
) -> tuple[int, bool]:
    """No teacher.choose in learner control; never treat advice as executed."""
    if arm not in ARMS or type(switched) is not bool or learned_action not in range(7):
        raise ValueError("Stage22 illegal rescue arm, phase, or learner action")
    if arm == "teacher_after_open" and switched:
        chosen = int(teacher.choose(observation, rng))
        if chosen not in range(7):
            raise ValueError("Stage22 public teacher proposed invalid native action")
        return chosen, True
    return learned_action, False


def actual_public_open(
    task: str, already_open: bool, before: dict, executed: int, after: dict,
) -> bool:
    """Public-only witness from an action actually passed to native env.step."""
    if task not in TASKS or type(already_open) is not bool:
        raise ValueError("Stage22 invalid native source task or phase")
    if task == "fetch":
        if already_open:
            raise ValueError("Stage22 Fetch must never switch to door rescue")
        return False
    return already_open or public_door_open_transition(before, executed, after)


def train_episode(task: str, layout: int) -> dict:
    if task not in TASKS or layout not in RESCUE_TRAIN_LAYOUTS[task]:
        raise ValueError("Stage22 source TRAIN layout unregistered")
    actor = make_source_actor(task)
    rng = np.random.default_rng(8_800_000 + layout)
    audit = make_native_audit(task)
    xs: list[np.ndarray] = []
    actions: list[int] = []
    finished = False
    try:
        obs, _ = audit.reset(seed=layout)
        previous: int | None = None
        for _ in range(HORIZON):
            xs.append(public_action_features(
                obs, actor.key_observed_after_pickup, previous,
                arm="interaction_context",
            ))
            action = int(actor.choose(obs, rng))
            if action not in range(7):
                raise ValueError("Stage22 source actor illegal native action")
            nxt, _reward, done, truncated, _ = audit.step(action)
            actor.update(obs, action, nxt)
            actions.append(action)
            previous = action
            obs = nxt
            if done or truncated:
                finished = True
                break
        truth = audit.summary()
        if (not finished or not actions or len(xs) != len(actions)
                or truth["native_actions"] != len(actions)
                or sum(truth["action_counts"].values()) != len(actions)):
            raise ValueError("Stage22 TRAIN native executed steps inconsistent")
        x = np.stack(xs).astype(np.float32, copy=False)
        y = np.asarray(actions, dtype=np.int64)
        return {
            "task": task, "layout_seed": layout, "x": x, "y": y,
            "actual_native_steps": len(actions),
            "actual_native_executed_action_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
            "public_train_x_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            "real_train_y_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
            "native_action_histogram": [
                truth["action_counts"][str(i)] for i in range(7)
            ],
            "external_strict_full_physical_goal": bool(truth["full_physical_goal"]),
        }
    finally:
        audit.env.close()


def collect_train() -> tuple[list[dict], np.ndarray, dict]:
    validate_split_contract()
    rows = [
        train_episode(task, layout)
        for task in TASKS for layout in RESCUE_TRAIN_LAYOUTS[task]
    ]
    if len(rows) != 192:
        raise ValueError("Stage22 requires 192 physically executed TRAIN episodes")
    true_y = np.concatenate([r["y"] for r in rows])
    steps = sum(r["actual_native_steps"] for r in rows)
    histogram = np.bincount(true_y, minlength=7).tolist()
    if steps != len(true_y) or sum(histogram) != steps:
        raise ValueError("Stage22 TRAIN action accounting mismatch")
    xsha = hashlib.sha256()
    for row in rows:
        xsha.update(row["x"].tobytes())
    return rows, true_y, {
        "real_native_train_episodes": len(rows),
        "actual_native_train_steps": steps,
        "native_executed_action_histogram": histogram,
        "real_source_public_x_sha256": xsha.hexdigest(),
        "real_source_executed_y_sha256": hashlib.sha256(true_y.tobytes()).hexdigest(),
        "per_episode": [
            {k: v for k, v in r.items() if k not in ("x", "y")}
            for r in rows
        ],
    }


def monitor_episode(
    task: str, layout: int, arm: str, seed: int | None,
    model: InteractionContextGRU | None,
) -> dict:
    if task not in TASKS or layout not in RESCUE_MONITOR_LAYOUTS[task]:
        raise ValueError("Stage22 MONITOR outside fresh independent native source split")
    if arm == "actor_only":
        if seed is not None or model is not None:
            raise ValueError("Stage22 original source actor measured once only")
        actor = make_source_actor(task)
        learner_witness = None
        shadow_teacher = None
    elif arm in ARMS and seed in MODEL_SEEDS and model is not None:
        if model.training:
            raise ValueError("Stage22 learned model must be frozen in eval mode")
        actor = None
        learner_witness = Controller("public_memory")
        shadow_teacher = make_source_actor(task)
    else:
        raise ValueError("Stage22 unregistered learned rescue arm/model")
    audit = make_native_audit(task)
    rng = np.random.default_rng(8_800_000 + layout)
    rescue_rng = np.random.default_rng(44_000_000 + layout + int(seed or 0) * 37)
    actions: list[int] = []
    hidden: torch.Tensor | None = None
    previous: int | None = None
    first_open: int | None = None
    switched = False
    teacher_executed = 0
    reward_total = 0.0
    finished = False
    try:
        obs, _ = audit.reset(seed=layout)
        for _ in range(HORIZON):
            if model is None:
                action = int(actor.choose(obs, rng))
                teacher_chosen = False
            else:
                if arm == "teacher_after_open" and switched:
                    # The learner's unused proposal is NOT queried or env.step'd.
                    proposed = 0
                else:
                    vector = public_action_features(
                        obs, learner_witness.key_observed_after_pickup, previous,
                        arm="interaction_context",
                    )
                    proposed, hidden = model.native_step(vector, hidden)
                action, teacher_chosen = live_teacher_action(
                    arm, switched, proposed, shadow_teacher, obs, rescue_rng
                )
            if action not in range(7):
                raise ValueError("Stage22 selected native action outside 0..6")
            nxt, reward, done, truncated, _ = audit.step(action)
            next_phase = actual_public_open(task, switched, obs, action, nxt)
            if next_phase and not switched:
                first_open = len(actions)
            switched = next_phase
            if actor is not None:
                actor.update(obs, action, nxt)
            else:
                learner_witness.update(obs, action, nxt)
                # Shadow observes ONLY truly executed public before/action/after,
                # never private audit state or hypothetical learner proposals.
                shadow_teacher.update(obs, action, nxt)
            teacher_executed += int(teacher_chosen)
            actions.append(action)
            previous = action
            reward_total += float(reward)
            obs = nxt
            if done or truncated:
                finished = True
                break
        truth = audit.summary()
        if (not finished or not actions
                or truth["native_actions"] != len(actions)
                or sum(truth["action_counts"].values()) != len(actions)):
            raise ValueError("Stage22 MONITOR true native actions not conserved")
        effective_open = truth["effective_door_open_events"] > 0 if task == "doorkey" else False
        if (first_open is not None) != effective_open:
            raise ValueError("Stage22 public observed open disagrees with independent physics")
        if model is None and teacher_executed:
            raise ValueError("Stage22 source actor measured independently from takeover")
        if arm == "learner_continues" and teacher_executed:
            raise ValueError("Stage22 learner-only control executed expert actions")
        if task == "fetch" and (first_open is not None or teacher_executed):
            raise ValueError("Stage22 Fetch must have no door rescue")
        prefix = actions if first_open is None else actions[:first_open + 1]
        return {
            "task": task, "layout_seed": layout, "arm": arm, "model_seed": seed,
            "actual_native_steps": len(actions),
            "actual_native_action_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
            "native_action_histogram": [
                truth["action_counts"][str(i)] for i in range(7)
            ],
            "first_public_open_executed_action_index": first_open,
            "actual_common_prefix_steps": len(prefix),
            "actual_common_prefix_sha256": hashlib.sha256(bytes(prefix)).hexdigest(),
            "teacher_took_over_next_native_step": teacher_executed > 0,
            "actually_executed_teacher_post_open_actions": teacher_executed,
            "public_witnessed_open_episode": first_open is not None,
            "external_physical_effective_open_episode": effective_open,
            "external_strict_full_physical_goal": bool(truth["full_physical_goal"]),
            "external_reward_positive_episodes": int(truth["rewarded_terminations"] > 0),
            "external_effective_pickup_events": truth["effective_pickup_events"],
            "external_wrong_target_pickup_events": truth.get(
                "wrong_target_pickup_events", 0
            ),
            "actual_reward_sum": reward_total,
        }
    finally:
        audit.env.close()


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Stage22 MONITOR must have real source episodes")
    steps = sum(int(r["actual_native_steps"]) for r in rows)
    hist = [
        sum(int(r["native_action_histogram"][i]) for r in rows)
        for i in range(7)
    ]
    if steps != sum(hist):
        raise ValueError("Stage22 MONITOR true action histogram mismatch")
    return {
        "real_native_episodes": len(rows), "actual_native_steps": steps,
        "native_action_histogram": hist,
        "strict_native_full_physical_goals": sum(
            int(r["external_strict_full_physical_goal"]) for r in rows
        ),
        "public_first_open_episodes": sum(
            int(r["public_witnessed_open_episode"]) for r in rows
        ),
        "actual_expert_took_over_episodes": sum(
            int(r["teacher_took_over_next_native_step"]) for r in rows
        ),
        "actual_executed_expert_post_open_actions": sum(
            int(r["actually_executed_teacher_post_open_actions"]) for r in rows
        ),
        "reward_only_nonqualifying_episodes": sum(
            int(r["external_reward_positive_episodes"] > 0
                and not r["external_strict_full_physical_goal"]) for r in rows
        ),
    }


def acquire_monitor(models: dict[int, InteractionContextGRU]) -> dict:
    if set(models) != set(MODEL_SEEDS) or any(m.training for m in models.values()):
        raise ValueError("Stage22 needs exactly three trained frozen source GRUs")
    result = {}
    for task in TASKS:
        layouts = RESCUE_MONITOR_LAYOUTS[task]
        actor_rows = [
            monitor_episode(task, layout, "actor_only", None, None)
            for layout in layouts
        ]
        result[f"{task}:actor_only"] = {
            "episodes": actor_rows, "summary": summarize(actor_rows)
        }
        for seed in MODEL_SEEDS:
            for arm in ARMS:
                rows = [
                    monitor_episode(task, layout, arm, seed, models[seed])
                    for layout in layouts
                ]
                result[f"{task}:{arm}:{seed}"] = {
                    "episodes": rows, "summary": summarize(rows)
                }
    return result


def decide(monitor: dict) -> dict:
    required = {f"{task}:actor_only" for task in TASKS} | {
        f"{task}:{arm}:{seed}"
        for task in TASKS for seed in MODEL_SEEDS for arm in ARMS
    }
    if set(monitor) != required:
        raise ValueError("Stage22 independent native MONITOR groups missing")
    differences = {}
    path_differences = {}
    gains_by_seed = {s: 0 for s in MODEL_SEEDS}
    total_actor_goals = 0
    learner_goals = rescue_goals = 0
    rescue_door_goals = 0
    public_opens = expert_episodes = expert_actions = 0
    conditional_control_goals = conditional_rescue_goals = 0
    for task in TASKS:
        layouts = list(RESCUE_MONITOR_LAYOUTS[task])
        actor = monitor[f"{task}:actor_only"]["episodes"]
        if [r["layout_seed"] for r in actor] != layouts:
            raise ValueError("Stage22 actor native episode layouts unpaired")
        total_actor_goals += sum(int(x["external_strict_full_physical_goal"]) for x in actor)
        differences[task] = []
        path_differences[task] = {}
        for seed in MODEL_SEEDS:
            control = monitor[f"{task}:learner_continues:{seed}"]["episodes"]
            rescue = monitor[f"{task}:teacher_after_open:{seed}"]["episodes"]
            if any([r["layout_seed"] for r in rows] != layouts
                   for rows in (control, rescue)):
                raise ValueError("Stage22 native learned layouts not paired")
            paired_values = []
            action_diffs = 0
            for left, right in zip(control, rescue, strict=True):
                left_event = left["first_public_open_executed_action_index"]
                right_event = right["first_public_open_executed_action_index"]
                if (left_event != right_event
                        or left["actual_common_prefix_steps"]
                        != right["actual_common_prefix_steps"]
                        or left["actual_common_prefix_sha256"]
                        != right["actual_common_prefix_sha256"]):
                    raise ValueError("Stage22 paired real learned prefixes differed BEFORE rescue")
                if right["teacher_took_over_next_native_step"] != (
                    right["actually_executed_teacher_post_open_actions"] > 0
                ):
                    raise ValueError("Stage22 teacher takeover is not true native action")
                same = (
                    left["actual_native_action_sha256"]
                    == right["actual_native_action_sha256"]
                    and left["actual_native_steps"] == right["actual_native_steps"]
                    and left["external_strict_full_physical_goal"]
                    == right["external_strict_full_physical_goal"]
                )
                if (task == "fetch" or left_event is None) and not same:
                    raise ValueError("Stage22 non-triggered native actions must be identical")
                action_diffs += int(
                    left["actual_native_action_sha256"]
                    != right["actual_native_action_sha256"]
                )
                success_control = int(left["external_strict_full_physical_goal"])
                success_rescue = int(right["external_strict_full_physical_goal"])
                paired_values.append(success_rescue - success_control)
                learner_goals += success_control
                rescue_goals += success_rescue
                if task == "doorkey":
                    rescue_door_goals += success_rescue
                    public_opens += int(left_event is not None)
                    expert_episodes += int(right["teacher_took_over_next_native_step"])
                    expert_actions += right["actually_executed_teacher_post_open_actions"]
                    if left_event is not None:
                        conditional_control_goals += success_control
                        conditional_rescue_goals += success_rescue
            path_differences[task][str(seed)] = action_diffs
            gains_by_seed[seed] += sum(paired_values)
            differences[task].append(paired_values)
    if any(path_differences["fetch"][str(seed)] for seed in MODEL_SEEDS):
        raise ValueError("Stage22 Fetch rescue/control real actions must be identical")
    paired = {task: np.asarray(values, dtype=np.int64)
              for task, values in differences.items()}
    by_task = {task: int(paired[task].sum()) for task in TASKS}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.zeros(20_000, dtype=np.int64)
    # Fetch is matched action-identical and contributes exactly zero, but retain
    # the independent task's paired resampling for audit consistency.
    for task in TASKS:
        values = paired[task]
        seed_ix = rng.integers(0, 3, size=(20_000, 3))
        layout_ix = rng.integers(0, 40, size=(20_000, 3, 40))
        bootstrap += values[seed_ix[:, :, None], layout_ix].sum(axis=(1, 2))
    ci = [float(x) for x in np.quantile(bootstrap, (0.025, 0.975))]
    nonvacuity = bool(
        public_opens >= MIN_OBSERVED_OPENS
        and expert_episodes >= MIN_EXECUTED_TEACHER_EPISODES
        and sum(
            path_differences["doorkey"][str(seed)]
            >= MIN_DISTINCT_DOORKEY_PATHS_PER_SEED
            for seed in MODEL_SEEDS
        ) >= 2
    )
    effect = bool(
        by_task["doorkey"] >= MIN_RESCUE_GOAL_GAIN
        and by_task["fetch"] == 0
        and sum(gains_by_seed[seed] > 0 for seed in MODEL_SEEDS) >= 2
        and rescue_door_goals >= MIN_RESCUE_DOORKEY_GOALS
        and ci[0] > 0
    )
    return {
        "registered_decision": (
            "INSUFFICIENT" if not nonvacuity
            else "DIAGNOSTIC_SUPPORT" if effect else "NOT_SUPPORTED"
        ),
        "necessary_native_rescue_nonvacuity": nonvacuity,
        "diagnostic_physical_effect_gate": effect,
        "paired_rescue_minus_learner_strict_goals_by_task": by_task,
        "paired_rescue_minus_learner_strict_goals_total": int(sum(by_task.values())),
        "paired_rescue_minus_learner_strict_goals_by_model_seed": {
            str(s): int(gains_by_seed[s]) for s in MODEL_SEEDS
        },
        "paired_bootstrap_95pct_rescue_minus_learner_interval": ci,
        "full_native_action_hash_differences": path_differences,
        "first_public_open_paired_door_episodes": public_opens,
        "actual_expert_took_over_door_episodes": expert_episodes,
        "actual_expert_executed_post_open_native_actions": expert_actions,
        "matched_triggered_control_strict_goals": conditional_control_goals,
        "matched_triggered_rescue_strict_goals": conditional_rescue_goals,
        "learner_continues_native_strict_goals": learner_goals,
        "teacher_assisted_native_strict_goals": rescue_goals,
        "teacher_assisted_doorkey_strict_goals": rescue_door_goals,
        "original_source_actor_once_strict_goals": total_actor_goals,
        "original_actor_tripled_arithmetic_only": 3 * total_actor_goals,
        "descriptive_original_actor_24_goal_margin": 3 * total_actor_goals - 24,
        "expert_assisted_not_autonomous_learning": True,
        "fetch_rescue_and_learner_action_identical": True,
        "original_unlockpickup_final_touched": False,
    }


def execute(output: str) -> dict:
    if output != f"{OUT}/summary.json" or Path(output).exists():
        raise ValueError("Stage22 output path must be isolated and fresh")
    validate_split_contract()
    source_sha = admitted_source_commit()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    episodes, true_y, training = collect_train()
    weights = class_weights(true_y)
    models: dict[int, InteractionContextGRU] = {}
    profiles = []
    for seed in MODEL_SEEDS:
        model, profile = fit_model(episodes, seed, "interaction_context", weights)
        profile["numeric_checkpoint"] = save_numeric(
            model, path.parent / f"source-learner-{seed}.npz"
        )
        models[seed] = model
        profiles.append(profile)
    if (len(profiles) != 3
            or len({p["num_parameters"] for p in profiles}) != 1
            or len({p["actual_optimizer_step_calls"] for p in profiles}) != 1
            or len({p["cached_source_label_presentations_not_native_steps"]
                    for p in profiles}) != 1):
        raise ValueError("Stage22 frozen model capacity/optimizer exposure inconsistent")
    monitor = acquire_monitor(models)
    outcome = decide(monitor)
    monitor_steps = sum(
        group["summary"]["actual_native_steps"] for group in monitor.values()
    )
    report = {
        "study": "m23_stage22_source_real_post_open_expert_rescue_diagnostic",
        "execution_source_commit": source_sha,
        "train_source_layouts": {
            task: list(ids) for task, ids in RESCUE_TRAIN_LAYOUTS.items()
        },
        "monitor_source_layouts": {
            task: list(ids) for task, ids in RESCUE_MONITOR_LAYOUTS.items()
        },
        "model_seeds": list(MODEL_SEEDS),
        "arms": list(ARMS),
        "actual_fitted_learned_models": len(models),
        "real_native_training": training,
        "actual_native_source_train_steps": training["actual_native_train_steps"],
        "real_native_monitor": monitor,
        "actual_native_source_monitor_steps": monitor_steps,
        "actual_native_source_total_steps":
            training["actual_native_train_steps"] + monitor_steps,
        "real_optimizer_updates": sum(p["actual_optimizer_step_calls"] for p in profiles),
        "cached_source_label_presentations_not_native_steps": sum(
            p["cached_source_label_presentations_not_native_steps"] for p in profiles
        ),
        "source_model_checkpoint_manifests": profiles,
        "registered_scientific_decision": outcome,
        "original_target_unlockpickup_native_steps": 0,
        "original_target_final_episodes": 0,
        "new_imagined_native_steps": 0,
        "actual_study_wall_seconds": time.perf_counter() - started,
        "versions": {
            name: version(name)
            for name in ("numpy", "torch", "gymnasium", "minigrid")
        },
    }
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source": source_sha, "train_steps": report["actual_native_source_train_steps"],
        "monitor_steps": monitor_steps,
        "frozen_models": len(models),
        "rescue_door_goals_gain":
            outcome["paired_rescue_minus_learner_strict_goals_by_task"]["doorkey"],
        "decision": outcome["registered_decision"],
    }), flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=f"{OUT}/summary.json")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if not args.execute:
        raise SystemExit("Stage22 needs explicit --execute after current CI")
    execute(args.output)


if __name__ == "__main__":
    main()
