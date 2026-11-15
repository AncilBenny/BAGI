"""M23 Stage 9: prospective mission-relevance filtering of real source actions.

A hand-authored public-mission baseline, NOT learned policy selection or
compositional transfer. Physical success witnesses are evaluator-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX

from bagi.experiments.m1_goal_learning import admitted_source_commit
from bagi.experiments.m10_native_audit import NativeDoorKeyAudit
from bagi.experiments.m14_public_open_door import TraversableOpenDoorController
from bagi.experiments.m20_fetch_physics import (
    NativeFetchAudit,
    PublicFetchTeacher,
    fetch_request,
)
from bagi.experiments.m23_effect_action_selection import summarize_task
from bagi.experiments.m23_event_effects import public_eligibility_rule
from bagi.experiments.m23_rule_shift import PublicEffectMemory
from bagi.experiments.m23_source_feasibility import (
    HORIZON,
    SOURCE_ENV_IDS,
    choose_public_rule_shift_probe,
)
from bagi.experiments.m23_study_contract import (
    SKILL_MONITOR_LAYOUTS,
    SKILL_TRAIN_LAYOUTS,
    validate_split_contract,
)

OUT = "runs/m23_mission_skill_selection"
TASKS = ("doorkey", "fetch")
MODES = (
    "actor_only",
    "front_rule_gate",
    "history_rule_gate",
    "mission_skill_gate",
)
PROBABILITY = 0.65
MIN_DISAGREEMENTS = 32
MIN_GAIN = 12
BOOTSTRAP_SEED = 230917


def mission_kind(observation: dict) -> tuple[str, tuple[str, str] | None]:
    """Accept only native public mission text; no task ID or oracle leakage."""
    if set(observation) != {"image", "direction", "mission"}:
        raise ValueError("Stage9 policy requires exact public observation contract")
    mission = observation["mission"]
    if not isinstance(mission, str):
        raise ValueError("Native mission must be public text")
    normal = " ".join(mission.strip().lower().split())
    if normal.startswith("use the key to open the door") and "goal" in normal:
        return "doorkey", None
    return "fetch", fetch_request(observation)


def mission_relevant_override(
    observation: dict, proposal: int, memory: PublicEffectMemory,
) -> bool:
    """Public feasible effect AND public mission relevance; no native audit."""
    if type(proposal) is not int or proposal not in (3, 4, 5):
        raise ValueError("Stage9 only accepts native interaction override 3/4/5")
    kind, target = mission_kind(observation)
    image = np.asarray(observation["image"])
    if image.shape != (7, 7, 3):
        raise ValueError("Stage9 requires maintained 7x7x3 public MiniGrid image")
    if memory.predict_before(image, proposal) == 0:
        return False
    front = image[3, 5]
    front_obj, front_color = int(front[0]), int(front[1])
    if proposal == 4:
        return False
    if kind == "doorkey":
        return (
            front_obj == OBJECT_TO_IDX["key"] if proposal == 3 else proposal == 5
        )
    if proposal != 3:
        return False
    if target is None:
        raise ValueError("Fetch target missing from public mission")
    color, item = target
    return (
        front_obj == OBJECT_TO_IDX[item]
        and front_color == COLOR_TO_IDX[color]
    )


def accept_override(
    mode: str, observation: dict, proposal: int, memory: PublicEffectMemory,
) -> bool:
    if mode not in MODES or type(proposal) is not int or proposal not in (3, 4, 5):
        raise ValueError("Stage9 mode/action not registered")
    mission_kind(observation)  # rejects private state / unfamiliar task mission
    if mode == "actor_only":
        return False
    if mode == "front_rule_gate":
        return public_eligibility_rule(observation["image"], proposal) != 0
    if mode == "history_rule_gate":
        return memory.predict_before(observation["image"], proposal) != 0
    return mission_relevant_override(observation, proposal, memory)


def run_source_goal(task: str, seed: int, mode: str, split: str) -> dict:
    groups = {"skill_train": SKILL_TRAIN_LAYOUTS,
              "skill_monitor": SKILL_MONITOR_LAYOUTS}
    if task not in TASKS or mode not in MODES or split not in groups:
        raise ValueError("Stage9 unregistered task/mode/split")
    if seed not in groups[split][task]:
        raise ValueError("Stage9 source native layout outside registered split")
    import gymnasium as gym
    import minigrid  # noqa: F401

    source_rng = np.random.default_rng(8_800_000 + seed)
    override_rng = np.random.default_rng(23_800_000 + seed)
    controller = (
        TraversableOpenDoorController() if task == "doorkey"
        else PublicFetchTeacher()
    )
    memory = PublicEffectMemory()
    wrapped = gym.wrappers.TimeLimit(
        gym.make(SOURCE_ENV_IDS[task]), max_episode_steps=HORIZON
    )
    audit = (
        NativeDoorKeyAudit(wrapped) if task == "doorkey"
        else NativeFetchAudit(wrapped)
    )
    actions: list[int] = []
    proposed_count = 0
    accepted = 0
    rejected = 0
    history_mission_disagreements = 0
    front_history_disagreements = 0
    reward_sum = 0.0
    finished = False
    try:
        observation, _ = audit.reset(seed=seed)
        for _ in range(HORIZON):
            registered_kind, _ = mission_kind(observation)
            if registered_kind != task:
                raise ValueError("Stage9 public mission conflicts with source task")
            actor_action = int(controller.choose(observation, source_rng))
            executed = actor_action
            if mode != "actor_only" and override_rng.random() < PROBABILITY:
                proposed = choose_public_rule_shift_probe(observation, override_rng)
                proposed_count += 1
                history_ok = (
                    memory.predict_before(observation["image"], proposed) != 0
                )
                mission_ok = mission_relevant_override(
                    observation, proposed, memory
                )
                front_ok = (
                    public_eligibility_rule(observation["image"], proposed) != 0
                )
                history_mission_disagreements += int(history_ok != mission_ok)
                front_history_disagreements += int(front_ok != history_ok)
                if accept_override(mode, observation, proposed, memory):
                    executed = proposed
                    accepted += 1
                else:
                    rejected += 1
            if executed not in range(7):
                raise ValueError("Stage9 invalid native executed source action")
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
            raise ValueError("Stage9 missing native episode completion")
        if sum(result["action_counts"].values()) != len(actions):
            raise ValueError("Stage9 action histogram disagrees with execution")
        return {
            "task": task, "seed": seed, "mode": mode, "split": split,
            "public_teacher_seed": 8_800_000 + seed,
            "public_override_seed": 23_800_000 + seed,
            "actual_native_actions": len(actions),
            "actual_executed_action_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
            "override_attempts": proposed_count,
            "accepted_overrides": accepted,
            "rejected_overrides": rejected,
            "override_changed_from_controller": 0,
            "front_history_rule_disagreement_attempts":
                front_history_disagreements,
            "history_mission_rule_disagreement_attempts":
                history_mission_disagreements,
            "native_reward_sum": reward_sum,
            "external_physical_full_goal": bool(result["full_physical_goal"]),
            "external_rewarded_terminations": result["rewarded_terminations"],
            "external_effective_pickup_events": result["effective_pickup_events"],
            "external_effective_door_open_events": result["effective_door_open_events"],
            "external_effective_forward_movements":
                result.get("effective_forward_movements"),
            "external_wrong_target_pickup_events": (
                result["wrong_target_pickup_events"] if task == "fetch" else 0
            ),
            "native_action_histogram": [
                result["action_counts"][str(i)] for i in range(7)
            ],
        }
    finally:
        audit.env.close()


def acquire(split: str) -> dict:
    groups = {"skill_train": SKILL_TRAIN_LAYOUTS,
              "skill_monitor": SKILL_MONITOR_LAYOUTS}
    if split not in groups:
        raise ValueError("Stage9 may run only registered source layouts")
    output = {}
    for task in TASKS:
        for mode in MODES:
            rows = [
                run_source_goal(task, seed, mode, split)
                for seed in groups[split][task]
            ]
            output[f"{task}:{mode}"] = {
                "summary": summarize_task(rows), "episodes": rows
            }
    return output


def decide(monitor: dict) -> dict:
    expected = {f"{task}:{mode}" for task in TASKS for mode in MODES}
    if set(monitor) != expected:
        raise ValueError("Stage9 missing an independent registered source arm")
    per_task: dict[str, list[int]] = {}
    gains: dict[str, int] = {}
    successes: dict[str, dict[str, int]] = {}
    disagreements = 0
    for task in TASKS:
        arms = {
            mode: monitor[f"{task}:{mode}"]["episodes"] for mode in MODES
        }
        for mode, rows in arms.items():
            if [row["seed"] for row in rows] != list(SKILL_MONITOR_LAYOUTS[task]):
                raise ValueError("Stage9 paired source monitor layouts mismatch")
        diffs = [
            int(m["external_physical_full_goal"])
            - int(h["external_physical_full_goal"])
            for m, h in zip(
                arms["mission_skill_gate"], arms["history_rule_gate"], strict=True
            )
        ]
        per_task[task] = diffs
        gains[task] = sum(diffs)
        successes[task] = {
            mode: sum(int(row["external_physical_full_goal"])
                      for row in arms[mode]) for mode in MODES
        }
        disagreements += sum(
            row["history_mission_rule_disagreement_attempts"]
            for row in arms["mission_skill_gate"]
        )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.zeros(20000)
    for task in TASKS:
        values = np.asarray(per_task[task], dtype=np.float64)
        idx = rng.integers(0, len(values), size=(len(boot), len(values)))
        boot += values[idx].sum(axis=1)
    ci = [float(x) for x in np.quantile(boot, [0.025, 0.975])]
    gain = sum(gains.values())
    mission = sum(successes[t]["mission_skill_gate"] for t in TASKS)
    actor = sum(successes[t]["actor_only"] for t in TASKS)
    sufficient = disagreements >= MIN_DISAGREEMENTS
    gate = bool(
        sufficient and gain >= MIN_GAIN
        and all(gains[t] > 0 for t in TASKS)
        and mission >= actor
    )
    return {
        "per_task_full_physical_goals": successes,
        "mission_minus_history_paired_goal_gains": gains,
        "pooled_paired_full_goal_gain": gain,
        "mission_vs_actor_only_total_noninferior": mission >= actor,
        "history_mission_disagreements_in_monitor": disagreements,
        "necessary_nonvacuous_support": sufficient,
        "task_stratified_paired_bootstrap_95pct_total_gain": ci,
        "registered_numerical_gate_pass": gate,
        "reliable_physical_improvement": gate and ci[0] > 0,
        "registered_decision": (
            "INSUFFICIENT" if not sufficient else "PASS" if gate else "FAIL"
        ),
        "scope": "public_mission_heuristic_source_task_physical_completion_only",
        "is_learned_zero_shot_transfer": False,
    }


def execute(output: str) -> dict:
    validate_split_contract()
    source_sha = admitted_source_commit()
    path = Path(output)
    if str(path) != f"{OUT}/summary.json" or path.exists():
        raise ValueError("Stage9 output must be new in isolated registered location")
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pilot = acquire("skill_train")
    monitor = acquire("skill_monitor")
    gate = decide(monitor)
    train_actions = sum(x["summary"]["native_actions"] for x in pilot.values())
    monitor_actions = sum(x["summary"]["native_actions"] for x in monitor.values())
    result = {
        "study": "m23_mission_conditioned_public_source_skill_selection",
        "source_commit": source_sha,
        "training_condition": "none: hand-authored public mission relevance baseline",
        "new_native_train_pilot_actions": train_actions,
        "new_native_monitor_actions": monitor_actions,
        "total_new_actual_native_actions": train_actions + monitor_actions,
        "new_model_fits": 0, "actual_optimizer_step_calls": 0,
        "actual_imagined_planner_steps": 0,
        "target_unlockpickup_native_actions": 0,
        "target_final_episodes": 0,
        "public_action_override_probability": PROBABILITY,
        "task_names": list(TASKS), "modes": list(MODES),
        "train_layouts": {k: list(v) for k, v in SKILL_TRAIN_LAYOUTS.items()},
        "monitor_layouts": {k: list(v) for k, v in SKILL_MONITOR_LAYOUTS.items()},
        "versions": {
            name: version(name) for name in ("numpy", "gymnasium", "minigrid")
        },
        "train_pilot": pilot, "independent_source_monitor": monitor,
        "registered_scientific_decision": gate,
        "elapsed_seconds": time.perf_counter() - started,
    }
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_sha": source_sha,
        "new_native_train_pilot_actions": train_actions,
        "new_native_monitor_actions": monitor_actions,
        "registered_decision": gate["registered_decision"],
        "pooled_paired_full_goal_gain": gate["pooled_paired_full_goal_gain"],
        "new_fits": 0, "optimizer_updates": 0,
    }), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=f"{OUT}/summary.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Stage9 native source task action selection needs --execute")
    execute(args.output)


if __name__ == "__main__":
    main()
