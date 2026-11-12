"""M13 public-only physical interaction coverage; evaluator-only private audit.

Three fixed controllers use only PUBLIC current observations and own actual
executed-action history. This experiment tests event coverage before learning.
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
from minigrid.core.constants import OBJECT_TO_IDX

from bagi.experiments.m1_goal_learning import admitted_source_commit, save_progress
from bagi.experiments.m10_native_audit import NativeDoorKeyAudit, snapshot_doorkey

TASKS = ("MiniGrid-DoorKey-6x6-v0", "MiniGrid-DoorKey-8x8-v0")
MODES = ("uniform", "reactive", "public_memory")
MONITOR = (tuple(range(171001, 171025)), tuple(range(181001, 181025)))
FINAL = (tuple(range(172001, 172037)), tuple(range(182001, 182037)))
HORIZON = 256
OUT = "runs/m13_public_interaction_coverage"
KEY = OBJECT_TO_IDX["key"]
DOOR = OBJECT_TO_IDX["door"]
WALL = OBJECT_TO_IDX["wall"]


def registered(config: dict) -> dict:
    expected = {
        "registered_study": "m13", "tasks": list(TASKS),
        "controllers": list(MODES),
        "monitor_seeds": [list(x) for x in MONITOR],
        "final_seeds": [list(x) for x in FINAL],
        "episode_horizon": HORIZON,
        "reactive_toggle_probability": 0.5,
        "navigate_forward_probability": 0.7,
        "front_cell": [3, 5],
        "opening_gain_gate": 2,
        "output_dir": OUT,
    }
    if config != expected:
        raise ValueError("M13 registered physical interaction protocol changed")
    seeds = sum(MONITOR + FINAL, ())
    if len(seeds) != len(set(seeds)):
        raise ValueError("M13 reused diagnostic or independent layout")
    return config


def front(observation: dict) -> tuple[int, int]:
    if not isinstance(observation, dict):
        raise TypeError("M13 acting observation must be public dict")
    if set(observation) != {"image", "mission", "direction"}:
        raise ValueError("M13 actor received unregistered observation fields")
    image = np.asarray(observation["image"])
    if image.shape != (7, 7, 3) or not isinstance(observation["mission"], str):
        raise ValueError("M13 public MiniGrid observation contract changed")
    return int(image[3, 5, 0]), int(image[3, 5, 2])


class Controller:
    """Minimal public evidence memory; neither a learned policy nor oracle."""

    def __init__(self, mode: str):
        if mode not in MODES:
            raise ValueError("Unknown M13 native control")
        self.mode = mode
        self.key_observed_after_pickup = False
        self.public_pickup_evidence = 0

    def choose(self, observation: dict, rng: np.random.Generator) -> int:
        obj, state = front(observation)
        if self.mode == "uniform":
            return int(rng.integers(7))
        if obj == KEY and not self.key_observed_after_pickup:
            return 3
        if obj == DOOR and state == 2:
            if self.mode == "public_memory" and self.key_observed_after_pickup:
                return 5
            if self.mode == "reactive" and rng.random() < 0.5:
                return 5
        if obj not in (KEY, DOOR, WALL) and rng.random() < 0.7:
            return 2
        return int(rng.integers(2))

    def update(self, before: dict, executed: int, after: dict) -> None:
        """Use ONLY the public next image after an actually executed action."""
        prior, _ = front(before)
        successor, _ = front(after)
        if executed == 3 and prior == KEY and successor != KEY:
            self.key_observed_after_pickup = True
            self.public_pickup_evidence += 1


def episode(task: str, seed: int, mode: str, split: str) -> dict:
    groups = MONITOR if split == "monitor" else FINAL if split == "final" else None
    if task not in TASKS or mode not in MODES or groups is None:
        raise ValueError("Unregistered M13 native task/controller/split")
    if seed not in groups[TASKS.index(task)]:
        raise ValueError("M13 attempted unregistered layout seed")
    rng_seed = 8_000_000 + seed * 10 + MODES.index(mode)
    rng = np.random.default_rng(rng_seed)
    actor = Controller(mode)
    audit = NativeDoorKeyAudit(
        gym.wrappers.TimeLimit(gym.make(task), max_episode_steps=HORIZON)
    )
    reward_total = 0.0
    disagreements = {"false_public_key": 0, "missed_public_key": 0}
    try:
        obs, _ = audit.reset(seed=seed)
        finished = False
        for _ in range(HORIZON):
            action = actor.choose(obs, rng)
            successor, reward, terminated, truncated, _ = audit.step(action)
            actor.update(obs, action, successor)
            # Private state belongs exclusively to independent outcome scoring.
            # It never enters Controller.choose or Controller.update.
            physical = snapshot_doorkey(audit.env).physical
            belief = actor.key_observed_after_pickup
            disagreements["false_public_key"] += int(
                belief and not physical.has_key
            )
            disagreements["missed_public_key"] += int(
                not belief and physical.has_key
            )
            reward_total += float(reward)
            obs = successor
            if terminated or truncated:
                finished = True
                break
        recorded = audit.summary()
        if not finished or recorded["native_actions"] > HORIZON:
            raise ValueError("Incomplete or overlength M13 native episode")
        if sum(recorded["action_counts"].values()) != recorded["native_actions"]:
            raise ValueError("M13 native executed action accounting mismatch")
        return {
            "task": task, "mode": mode, "split": split, "seed": seed,
            "action_rng_seed": rng_seed, "reward_sum": reward_total,
            "public_pickup_evidence_count": actor.public_pickup_evidence,
            "external_belief_disagreement": disagreements, **recorded,
        }
    finally:
        audit.env.close()


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("M13 cannot summarize missing native episodes")
    histogram = [
        sum(r["action_counts"][str(i)] for r in rows) for i in range(7)
    ]
    native = sum(r["native_actions"] for r in rows)
    if sum(histogram) != native:
        raise ValueError("M13 physical auditor action counts disagree")
    valid = sum(r["valid_toggle_attempts"] for r in rows)
    if valid > histogram[5]:
        raise ValueError("M13 valid toggle attempts exceed toggle actions")
    return {
        "episodes": len(rows), "native_actions": native,
        "native_action_histogram": histogram,
        "key_pickup_episodes": sum(
            int(r["effective_pickup_events"] > 0) for r in rows
        ),
        "valid_door_opportunity_episodes": sum(
            int(r["valid_toggle_opportunities"] > 0) for r in rows
        ),
        "valid_door_opportunity_steps": sum(
            r["valid_toggle_opportunities"] for r in rows
        ),
        "valid_toggle_attempts": valid,
        "all_toggle_attempts": histogram[5],
        "unqualified_toggle_attempts": histogram[5] - valid,
        "effective_door_opening_episodes": sum(
            int(r["effective_door_open_events"] > 0) for r in rows
        ),
        "physical_full_goal_episodes": sum(
            int(r["full_physical_goal"]) for r in rows
        ),
        "native_rewarded_goal_episodes": sum(
            int(r["rewarded_terminations"] > 0) for r in rows
        ),
        "public_key_evidence_episodes": sum(
            int(r["public_pickup_evidence_count"] > 0) for r in rows
        ),
        "public_evidence_false_positive_steps": sum(
            r["external_belief_disagreement"]["false_public_key"] for r in rows
        ),
        "public_evidence_false_negative_steps": sum(
            r["external_belief_disagreement"]["missed_public_key"] for r in rows
        ),
    }


def acquire(split: str) -> dict:
    group = MONITOR if split == "monitor" else FINAL if split == "final" else None
    if group is None:
        raise ValueError("M13 has no such native acquisition split")
    result = {}
    for task, seeds in zip(TASKS, group):
        for mode in MODES:
            rows = [episode(task, seed, mode, split) for seed in seeds]
            stats = summarize(rows)
            if stats["episodes"] != len(seeds):
                raise ValueError("M13 missing a registered real episode")
            result[f"{task}:{mode}"] = {"totals": stats, "episodes": rows}
    return result


def evaluate_gate(final: dict) -> dict:
    expected = {f"{task}:{mode}" for task in TASKS for mode in MODES}
    if set(final) != expected:
        raise ValueError("M13 final physics requires all matched conditions")
    gains = {}
    eligible = {}
    for task in TASKS:
        memory = final[f"{task}:public_memory"]["totals"]
        baseline = final[f"{task}:reactive"]["totals"]
        gains[task] = (
            memory["effective_door_opening_episodes"]
            - baseline["effective_door_opening_episodes"]
        )
        eligible[task] = memory["key_pickup_episodes"] >= 4
    return {
        "effective_door_opening_episode_gain": gains,
        "public_memory_key_pickup_coverage": eligible,
        "pilot_gate_pass": (
            all(eligible.values()) and all(g >= 0 for g in gains.values())
            and sum(gains.values()) >= 2
        ),
        "not_causal_identification": True,
        "not_learned_policy": True,
    }


def run(config: Path, output: Path) -> dict:
    if output.exists() or (output.parent / "progress.json").exists():
        raise FileExistsError("Never overwrite existing M13 evidence")
    raw = config.read_bytes()
    registered(yaml.safe_load(raw))
    if output.parent.as_posix() != OUT:
        raise ValueError("M13 unregistered result directory")
    source = admitted_source_commit()
    started = time.monotonic()
    phase = "monitor"
    monitor, final = {}, {}
    try:
        monitor = acquire("monitor")
        save_progress(output.parent / "progress.json", {
            "status": "partial_not_completed", "phase": phase,
            "source_commit": source,
            "monitor_native_actions": sum(
                x["totals"]["native_actions"] for x in monitor.values()
            ), "optimizer_steps": 0, "new_fits": 0,
        })
        if len(monitor) != len(TASKS) * len(MODES):
            raise ValueError("M13 independent split locked on incomplete monitors")
        phase = "independent_final"
        final = acquire("final")
    except BaseException as error:
        save_progress(output.parent / "progress.json", {
            "status": "interrupted_not_completed", "phase": phase,
            "source_commit": source, "failure_type": type(error).__name__,
            "completed_monitor_conditions": len(monitor),
            "completed_final_conditions": len(final),
            "optimizer_steps": 0, "new_fits": 0,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        raise
    monitor_actions = sum(x["totals"]["native_actions"] for x in monitor.values())
    final_actions = sum(x["totals"]["native_actions"] for x in final.values())
    report = {
        "registered_study": "m13",
        "study": "M13 public-observation native physical manipulation coverage",
        "status": "completed_public_control_physical_diagnostic",
        "source_commit": source,
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "versions": {
            pkg: version(pkg) for pkg in ("gymnasium", "minigrid", "numpy")
        },
        "fresh_native_actions": monitor_actions + final_actions,
        "train_only_monitor_actions": monitor_actions,
        "independent_native_actions": final_actions,
        "independent_episodes": sum(x["totals"]["episodes"] for x in final.values()),
        "optimizer_step_calls": 0, "new_model_fits": 0, "teacher_queries": 0,
        "monitor": monitor, "independent_final": final,
        "pilot": evaluate_gate(final),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "limitations": [
            "All controllers are hand-specified public baselines, not learned.",
            "6x6 and 8x8 DoorKey share mechanics; not unrelated domains.",
            "History of key disappearance is not a probabilistic belief state.",
            "Different controller actions alter navigation and confound causality.",
            "Private snapshots used only by external evaluator after native steps.",
            "No model fit, optimizer step or task-independent learning was tested.",
            "Native reward alone is not physical key-door-goal completion.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/m13_public_interaction_coverage.yaml"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/m13_public_interaction_coverage/summary.json"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("M13 needs explicit --execute for new simulator actions")
    report = run(args.config, args.output)
    print(json.dumps({
        "status": report["status"], "source_commit": report["source_commit"],
        "fresh_native_actions": report["fresh_native_actions"],
        "pilot": report["pilot"],
    }, indent=2))


if __name__ == "__main__":
    main()
