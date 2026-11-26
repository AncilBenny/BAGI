"""M14: open-door traversability replication with public-only finite memory.

Minimal correction to M13's source-visible blocked-open-door control bug.
This is a physical non-learning reference-controller experiment, not PPO.
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
import yaml

from bagi.experiments.m1_goal_learning import admitted_source_commit, save_progress
from bagi.experiments.m10_native_audit import NativeDoorKeyAudit, snapshot_doorkey
from bagi.experiments.m13_public_interaction_coverage import (
    DOOR,
    HORIZON,
    TASKS,
    Controller,
    front,
)

MODES = ("blocked_open_door", "traversable_open_door")
MONITOR = (tuple(range(191001, 191013)), tuple(range(201001, 201013)))
FINAL = (tuple(range(192001, 192037)), tuple(range(202001, 202037)))
OUT = "runs/m14_public_open_door"


def validate_config(config: dict) -> dict:
    expected = {
        "registered_study": "m14", "tasks": list(TASKS),
        "controllers": list(MODES),
        "monitor_seeds": [list(x) for x in MONITOR],
        "final_seeds": [list(x) for x in FINAL],
        "episode_horizon": HORIZON, "open_door_state": 0,
        "front_cell": [3, 5], "minimum_pooled_goal_gain": 2,
        "minimum_open_door_entries_per_task": 12,
        "output_dir": OUT,
    }
    if config != expected:
        raise ValueError("M14 preregistered public traversal protocol changed")
    all_seeds = sum(MONITOR + FINAL, ())
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("M14 monitor/final layout collision")
    return config


class TraversableOpenDoorController(Controller):
    """One observable-only change to the frozen M13 memory controller."""

    def __init__(self):
        super().__init__("public_memory")

    def choose(self, observation: dict, rng: np.random.Generator) -> int:
        obj, state = front(observation)
        if obj == DOOR and state == 0:
            return 2  # native forward into an OBSERVABLY open door
        return super().choose(observation, rng)


def rollout(task: str, seed: int, mode: str, split: str) -> dict:
    groups = MONITOR if split == "monitor" else FINAL if split == "final" else None
    if task not in TASKS or mode not in MODES or groups is None:
        raise ValueError("M14 task/mode/split not registered")
    if seed not in groups[TASKS.index(task)]:
        raise ValueError("M14 may access only preregistered native layouts")
    # Same PRNG stream across modes until their first open-door decision.
    rng_seed = 12_000_000 + seed
    rng = np.random.default_rng(rng_seed)
    controller = (
        Controller("public_memory")
        if mode == "blocked_open_door" else TraversableOpenDoorController()
    )
    audit = NativeDoorKeyAudit(
        gym.wrappers.TimeLimit(gym.make(task), max_episode_steps=HORIZON)
    )
    prefix_actions = []
    opened = False
    crossed_open_door = False
    forward_to_open_door = 0
    reward_sum = 0.0
    try:
        observation, _ = audit.reset(seed=seed)
        finished = False
        for _ in range(HORIZON):
            obj, state = front(observation)
            action = controller.choose(observation, rng)
            if not opened:
                prefix_actions.append(action)
            if obj == DOOR and state == 0 and action == 2:
                forward_to_open_door += 1
            following, reward, terminated, truncated, _ = audit.step(action)
            controller.update(observation, action, following)
            # EVALUATOR ONLY. Private position/door state never changes actions.
            external = snapshot_doorkey(audit.env)
            if external.physical.door_open:
                opened = True
                world = audit.env.unwrapped
                pos = tuple(int(x) for x in world.agent_pos)
                crossed_open_door |= pos == external.door_cell
            reward_sum += float(reward)
            observation = following
            if terminated or truncated:
                finished = True
                break
        summary = audit.summary()
        if not finished or summary["native_actions"] > HORIZON:
            raise ValueError("M14 native physical episode not complete")
        if sum(summary["action_counts"].values()) != summary["native_actions"]:
            raise ValueError("M14 actual native action counts differ")
        if summary["full_physical_goal"] and not crossed_open_door:
            raise ValueError("M14 physical full goal without verified door traversal")
        return {
            "task": task, "seed": seed, "mode": mode, "split": split,
            "controller_rng_seed": rng_seed,
            "pre_first_open_action_sha256": hashlib.sha256(
                bytes(prefix_actions)
            ).hexdigest(),
            "pre_first_open_action_count": len(prefix_actions),
            "public_forward_to_open_door_actions": forward_to_open_door,
            "external_open_door_cell_traversed": bool(crossed_open_door),
            "actual_reward_sum": reward_sum,
            **summary,
        }
    finally:
        audit.env.close()


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("M14 cannot summarize an empty native condition")
    histogram = [
        sum(row["action_counts"][str(a)] for row in rows) for a in range(7)
    ]
    total = sum(row["native_actions"] for row in rows)
    if sum(histogram) != total:
        raise ValueError("M14 seven-action accounting mismatch")
    return {
        "episodes": len(rows), "native_actions": total,
        "native_action_histogram": histogram,
        "key_pickup_episodes": sum(
            int(row["effective_pickup_events"] > 0) for row in rows
        ),
        "effective_door_opening_episodes": sum(
            int(row["effective_door_open_events"] > 0) for row in rows
        ),
        "actual_door_cell_entry_episodes": sum(
            int(row["external_open_door_cell_traversed"]) for row in rows
        ),
        "public_forward_to_open_door_actions": sum(
            row["public_forward_to_open_door_actions"] for row in rows
        ),
        "physical_full_goal_episodes": sum(
            int(row["full_physical_goal"]) for row in rows
        ),
        "native_rewarded_goal_episodes": sum(
            int(row["rewarded_terminations"] > 0) for row in rows
        ),
        "valid_toggle_attempts": sum(
            row["valid_toggle_attempts"] for row in rows
        ),
        "all_toggle_attempts": histogram[5],
    }


def acquire(split: str) -> dict:
    groups = MONITOR if split == "monitor" else FINAL if split == "final" else None
    if groups is None:
        raise ValueError("M14 unregistered split")
    output = {}
    for task, seeds in zip(TASKS, groups):
        for mode in MODES:
            rows = [rollout(task, seed, mode, split) for seed in seeds]
            totals = summarize(rows)
            if totals["episodes"] != len(seeds):
                raise ValueError("M14 missing native episodes")
            output[f"{task}:{mode}"] = {"totals": totals, "episodes": rows}
    return output


def judge(data: dict) -> dict:
    expected = {f"{task}:{mode}" for task in TASKS for mode in MODES}
    if set(data) != expected:
        raise ValueError("M14 requires all independent native conditions")
    goals = {}
    entered = {}
    prefix_checks = {}
    for task in TASKS:
        old = data[f"{task}:blocked_open_door"]
        new = data[f"{task}:traversable_open_door"]
        left = old["episodes"]
        right = new["episodes"]
        if len(left) != len(right):
            raise ValueError("M14 paired layouts missing")
        checks = [
            before["seed"] == after["seed"]
            and before["controller_rng_seed"] == after["controller_rng_seed"]
            and before["pre_first_open_action_sha256"]
            == after["pre_first_open_action_sha256"]
            for before, after in zip(left, right)
        ]
        prefix_checks[task] = all(checks)
        goals[task] = (
            new["totals"]["physical_full_goal_episodes"]
            - old["totals"]["physical_full_goal_episodes"]
        )
        entered[task] = new["totals"]["actual_door_cell_entry_episodes"]
    return {
        "paired_pre_opening_actions_equal": prefix_checks,
        "physical_full_goal_episode_gain": goals,
        "open_door_cell_entry_episodes": entered,
        "pilot_gate_pass": (
            all(prefix_checks.values())
            and all(entered[task] >= 12 for task in TASKS)
            and all(goals[task] >= 1 for task in TASKS)
            and sum(goals.values()) >= 2
        ),
        "not_learned_policy": True,
        "not_task_independent_transfer": True,
    }


def run(config: Path, output: Path) -> dict:
    if output.exists() or (output.parent / "progress.json").exists():
        raise FileExistsError("M14 experiment evidence already exists")
    source_bytes = config.read_bytes()
    validate_config(yaml.safe_load(source_bytes))
    if output.parent.as_posix() != OUT:
        raise ValueError("M14 output directory not registered")
    source = admitted_source_commit()
    started = time.monotonic()
    stage = "monitor"
    monitor, final = {}, {}
    try:
        monitor = acquire("monitor")
        save_progress(output.parent / "progress.json", {
            "status": "partial_not_completed", "source_commit": source,
            "stage": stage, "monitor_native_actions": sum(
                x["totals"]["native_actions"] for x in monitor.values()
            ), "optimizer_step_calls": 0, "new_model_fits": 0,
        })
        if len(monitor) != len(TASKS) * len(MODES):
            raise ValueError("M14 heldout locked until monitor complete")
        stage = "independent_final"
        final = acquire("final")
        primary = judge(final)
    except BaseException as exc:
        save_progress(output.parent / "progress.json", {
            "status": "interrupted_not_completed",
            "source_commit": source, "stage": stage,
            "error_type": type(exc).__name__,
            "monitor_conditions": len(monitor),
            "final_conditions": len(final),
            "optimizer_step_calls": 0, "new_model_fits": 0,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        raise
    monitor_steps = sum(x["totals"]["native_actions"] for x in monitor.values())
    independent_steps = sum(x["totals"]["native_actions"] for x in final.values())
    result = {
        "registered_study": "m14",
        "study": "M14 public open-door traversal native physical goal replication",
        "status": "completed_public_only_native_task_goal_diagnostic",
        "source_commit": source,
        "config_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "versions": {
            p: version(p) for p in ("gymnasium", "minigrid", "numpy")
        },
        "fresh_native_actions": monitor_steps + independent_steps,
        "monitor_native_actions": monitor_steps,
        "independent_native_actions": independent_steps,
        "independent_episodes": sum(
            x["totals"]["episodes"] for x in final.values()
        ),
        "optimizer_step_calls": 0, "new_model_fits": 0, "teacher_queries": 0,
        "monitor": monitor, "independent_final": final, "pilot": primary,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "limitations": [
            "Hand-specified public DoorKey controller, not trained RL.",
            "Two DoorKey sizes share mechanics, not cross-domain transfer.",
            "Different post-opening actions make goal comparisons behavioral.",
            "Actual physical goal requires key pickup, opening, and native goal.",
            "Private agent position is read only for external crossing audit.",
            "No model learning, optimizer updates, or continual transfer measured.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/m14_public_open_door.yaml")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/m14_public_open_door/summary.json"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("M14 requires explicit --execute for native actions")
    result = run(args.config, args.output)
    print(json.dumps({
        "status": result["status"],
        "fresh_native_actions": result["fresh_native_actions"],
        "pilot": result["pilot"],
    }, indent=2))


if __name__ == "__main__":
    main()
