"""M23 stage 8: reuse source public-effect memory for executed task decisions.

Zero training: compare real source tasks under a shared public-only action
override policy, with native physical goal audits held strictly outside policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from bagi.experiments.m1_goal_learning import admitted_source_commit
from bagi.experiments.m10_native_audit import NativeDoorKeyAudit
from bagi.experiments.m14_public_open_door import TraversableOpenDoorController
from bagi.experiments.m20_fetch_physics import NativeFetchAudit, PublicFetchTeacher
from bagi.experiments.m23_event_effects import public_eligibility_rule
from bagi.experiments.m23_rule_shift import PublicEffectMemory
from bagi.experiments.m23_source_feasibility import (
    HORIZON,
    SOURCE_ENV_IDS,
    choose_public_rule_shift_probe,
)
from bagi.experiments.m23_study_contract import (
    ACTION_MONITOR_LAYOUTS,
    ACTION_TRAIN_LAYOUTS,
    validate_split_contract,
)

OUT = "runs/m23_effect_action_selection"
TASKS = ("doorkey", "fetch")
MODES = ("unfiltered", "front_rule_gate", "history_rule_gate")
OVERRIDE_PROBABILITY = 0.65
MIN_RULE_DISAGREEMENTS = 16


def accept_public_override(mode: str, observation: dict, proposal: int,
                           memory: PublicEffectMemory) -> bool:
    """No simulator state, reward, native-effect witness or future observation."""
    if mode not in MODES or type(proposal) is not int or proposal not in (3, 4, 5):
        raise ValueError("M23 only registers real native interaction overrides")
    if set(observation) != {"image", "direction", "mission"}:
        raise ValueError("M23 action-gate requires standard public observation")
    if mode == "unfiltered":
        return True
    if mode == "front_rule_gate":
        return public_eligibility_rule(observation["image"], proposal) != 0
    return memory.predict_before(observation["image"], proposal) != 0


def run_source_goal(task: str, seed: int, mode: str, split: str) -> dict:
    """Perform actions in real MiniGrid; native goal audit never influences action."""
    registered = {
        "action_train": ACTION_TRAIN_LAYOUTS,
        "action_monitor": ACTION_MONITOR_LAYOUTS,
    }
    if task not in TASKS or mode not in MODES or split not in registered:
        raise ValueError("M23 source goal mode/task/split is unregistered")
    if seed not in registered[split][task]:
        raise ValueError("M23 native task seed outside frozen action split")
    import gymnasium as gym
    import minigrid  # noqa: F401

    source_rng = np.random.default_rng(8_800_000 + seed)
    override_rng = np.random.default_rng(23_800_000 + seed)
    controller = (
        TraversableOpenDoorController() if task == "doorkey" else PublicFetchTeacher()
    )
    memory = PublicEffectMemory()
    wrapped = gym.wrappers.TimeLimit(
        gym.make(SOURCE_ENV_IDS[task]), max_episode_steps=HORIZON
    )
    audit = NativeDoorKeyAudit(wrapped) if task == "doorkey" else NativeFetchAudit(wrapped)
    actions: list[int] = []
    override_attempts = 0
    accepted = 0
    rejected = 0
    override_changed_from_controller = 0
    disagreement = 0
    reward_sum = 0.0
    finished = False
    try:
        observation, _ = audit.reset(seed=seed)
        for _ in range(HORIZON):
            if set(observation) != {"image", "direction", "mission"}:
                raise ValueError("Source action policy received nonpublic state")
            source_action = int(controller.choose(observation, source_rng))
            executed = source_action
            if override_rng.random() < OVERRIDE_PROBABILITY:
                proposed = choose_public_rule_shift_probe(observation, override_rng)
                override_attempts += 1
                override_changed_from_controller += int(proposed != source_action)
                front_predicts = (
                    public_eligibility_rule(observation["image"], proposed) != 0
                )
                memory_predicts = (
                    memory.predict_before(observation["image"], proposed) != 0
                )
                disagreement += int(front_predicts != memory_predicts)
                if accept_public_override(mode, observation, proposed, memory):
                    executed = proposed
                    accepted += 1
                else:
                    rejected += 1
            if executed not in range(7):
                raise ValueError("Invalid native executed source action")
            following, reward, terminated, truncated, _ = audit.step(executed)
            actions.append(executed)
            reward_sum += float(reward)
            memory.update_after(
                observation["image"], executed, following["image"]
            )
            controller.update(observation, executed, following)
            observation = following
            if terminated or truncated:
                finished = True
                break
        result = audit.summary()
        if not finished or result["native_actions"] != len(actions):
            raise ValueError("M23 source physical episode did not terminate")
        if sum(result["action_counts"].values()) != len(actions):
            raise ValueError("M23 source task action histogram mismatch")
        return {
            "task": task, "seed": seed, "mode": mode, "split": split,
            "public_teacher_seed": 8_800_000 + seed,
            "public_override_seed": 23_800_000 + seed,
            "actual_native_actions": len(actions),
            "actual_executed_action_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
            "override_attempts": override_attempts,
            "accepted_overrides": accepted,
            "rejected_overrides": rejected,
            "override_changed_from_controller": override_changed_from_controller,
            "front_history_rule_disagreement_attempts": disagreement,
            "native_reward_sum": reward_sum,
            "external_physical_full_goal": bool(result["full_physical_goal"]),
            "external_rewarded_terminations": result["rewarded_terminations"],
            "external_effective_pickup_events": result["effective_pickup_events"],
            "external_effective_door_open_events": result["effective_door_open_events"],
            "external_effective_forward_movements": result.get("effective_forward_movements"),
            "external_wrong_target_pickup_events": (
                result["wrong_target_pickup_events"] if task == "fetch" else 0
            ),
            "native_action_histogram": [
                result["action_counts"][str(i)] for i in range(7)
            ],
        }
    finally:
        audit.env.close()


def summarize_task(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No registered physical source goal episodes")
    actions = sum(row["actual_native_actions"] for row in rows)
    histogram = [sum(row["native_action_histogram"][i] for row in rows)
                 for i in range(7)]
    if sum(histogram) != actions:
        raise ValueError("M23 true native task action totals inconsistent")
    successes = [row for row in rows if row["external_physical_full_goal"]]
    return {
        "episodes": len(rows),
        "physical_full_goal_episodes": len(successes),
        "rewarded_terminations": sum(
            int(row["external_rewarded_terminations"] > 0) for row in rows
        ),
        "reward_only_without_full_physical_goal_episodes": sum(
            int(row["external_rewarded_terminations"] > 0
                and not row["external_physical_full_goal"]) for row in rows
        ),
        "native_actions": actions,
        "native_action_histogram": histogram,
        "mean_actual_actions_to_physical_goal": (
            sum(row["actual_native_actions"] for row in successes) / len(successes)
            if successes else None
        ),
        "actual_accepted_overrides": sum(row["accepted_overrides"] for row in rows),
        "actual_rejected_overrides": sum(row["rejected_overrides"] for row in rows),
        "override_attempts": sum(row["override_attempts"] for row in rows),
        "front_history_rule_disagreements": sum(
            row["front_history_rule_disagreement_attempts"] for row in rows
        ),
        "native_effective_pickup_events": sum(
            row["external_effective_pickup_events"] for row in rows
        ),
        "native_effective_door_open_events": sum(
            row["external_effective_door_open_events"] for row in rows
        ),
        "native_reward_sum": sum(row["native_reward_sum"] for row in rows),
    }


def acquire(split: str) -> dict:
    groups = {
        "action_train": ACTION_TRAIN_LAYOUTS,
        "action_monitor": ACTION_MONITOR_LAYOUTS,
    }
    if split not in groups:
        raise ValueError("M23 source policy study permits only registered splits")
    output = {}
    for task in TASKS:
        for mode in MODES:
            rows = [run_source_goal(task, seed, mode, split)
                    for seed in groups[split][task]]
            output[f"{task}:{mode}"] = {
                "summary": summarize_task(rows),
                "episodes": rows,
            }
    return output


def decide(monitor: dict) -> dict:
    expected = {f"{t}:{m}" for t in TASKS for m in MODES}
    if set(monitor) != expected:
        raise ValueError("M23 requires all paired source task conditions")
    gains = {}
    full_history = 0
    full_unfiltered = 0
    diffs = []
    disagreements = 0
    per_task = {}
    for task in TASKS:
        front = monitor[f"{task}:front_rule_gate"]["episodes"]
        history = monitor[f"{task}:history_rule_gate"]["episodes"]
        unfiltered = monitor[f"{task}:unfiltered"]["episodes"]
        if not (len(front) == len(history) == len(unfiltered)):
            raise ValueError("M23 source task paired native episode count changed")
        paired = []
        for a, b, c in zip(front, history, unfiltered, strict=True):
            if a["seed"] != b["seed"] or b["seed"] != c["seed"]:
                raise ValueError("M23 paired source-native layout seeds mismatch")
            paired.append(int(b["external_physical_full_goal"])
                          - int(a["external_physical_full_goal"]))
        gains[task] = sum(paired)
        per_task[task] = paired
        diffs.extend(paired)
        full_history += sum(int(r["external_physical_full_goal"]) for r in history)
        full_unfiltered += sum(int(r["external_physical_full_goal"]) for r in unfiltered)
        disagreements += sum(r["front_history_rule_disagreement_attempts"]
                             for r in history)
    rng = np.random.default_rng(230816)
    draws = np.zeros(20000, dtype=np.float64)
    for task in TASKS:
        values = np.asarray(per_task[task], dtype=np.float64)
        idx = rng.integers(0, len(values), size=(20000, len(values)))
        draws += values[idx].sum(axis=1)
    ci = [float(v) for v in np.quantile(draws, (0.025, 0.975))]
    return {
        "physical_full_goal_gain_memory_minus_front": gains,
        "total_paired_goal_gain_memory_minus_front": sum(diffs),
        "history_vs_unfiltered_physical_goal_noninferior":
            full_history >= full_unfiltered,
        "front_history_rule_disagreement_attempts_in_monitor": disagreements,
        "two_source_task_clustered_paired_bootstrap_95pct_goal_gain": ci,
        "registered_source_action_selection_gate_pass": bool(
            sum(diffs) >= 8
            and all(gains[t] >= 1 for t in TASKS)
            and full_history >= full_unfiltered
            and disagreements >= MIN_RULE_DISAGREEMENTS
        ),
        "scope": "source_task_physical_goal_under_public_action_interventions_only",
        "not_compositional_zero_shot_transfer": True,
    }


def execute(output: str) -> dict:
    validate_split_contract()
    source_sha = admitted_source_commit()
    path = Path(output)
    if str(path) != f"{OUT}/summary.json" or path.exists():
        raise ValueError("M23 source action output must be new in isolated location")
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pilot = acquire("action_train")
    monitor = acquire("action_monitor")
    gate = decide(monitor)
    fresh_train = sum(x["summary"]["native_actions"] for x in pilot.values())
    fresh_monitor = sum(x["summary"]["native_actions"] for x in monitor.values())
    result = {
        "study": "m23_physical_source_goal_public_memory_action_selection",
        "source_commit": source_sha,
        "training_condition": "none: reused public source controllers only",
        "new_native_train_pilot_actions": fresh_train,
        "new_native_monitor_actions": fresh_monitor,
        "total_new_actual_native_actions": fresh_train + fresh_monitor,
        "new_model_fits": 0, "actual_optimizer_step_calls": 0,
        "actual_imagined_planner_steps": 0,
        "target_unlockpickup_native_actions": 0,
        "target_final_episodes": 0,
        "public_action_override_probability": OVERRIDE_PROBABILITY,
        "modes": list(MODES), "tasks": list(TASKS),
        "train_layouts": {k: list(v) for k, v in ACTION_TRAIN_LAYOUTS.items()},
        "monitor_layouts": {k: list(v) for k, v in ACTION_MONITOR_LAYOUTS.items()},
        "versions": {
            k: version(k) for k in ("numpy", "gymnasium", "minigrid")
        },
        "train_pilot": pilot, "independent_source_monitor": monitor,
        "registered_scientific_decision": gate,
        "elapsed_seconds": time.perf_counter() - started,
    }
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_sha": source_sha,
        "new_native_train_pilot_actions": fresh_train,
        "new_native_monitor_actions": fresh_monitor,
        "total_new_actual_native_actions": fresh_train + fresh_monitor,
        "new_model_fits": 0, "optimizer_step_calls": 0,
        "physical_paired_goal_gain": gate["total_paired_goal_gain_memory_minus_front"],
        "front_history_rule_disagreements":
            gate["front_history_rule_disagreement_attempts_in_monitor"],
        "registered_source_action_selection_gate_pass":
            gate["registered_source_action_selection_gate_pass"],
    }), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=f"{OUT}/summary.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("M23 source real-task action selection requires --execute")
    execute(args.output)


if __name__ == "__main__":
    main()
