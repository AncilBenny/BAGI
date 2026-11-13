"""M5: preregistered task retention with upstream PPO and public MiniGrid inputs.

Blocked, interleaved and task-only policies use fresh on-policy rollouts.
Private DoorKey state is read ONLY by a separate outcome evaluator, never the
policy observation, reward, info or action-selection interface.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import deque
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
import yaml
from gymnasium.spaces import Box
from minigrid.core.world_object import Door, Key

from bagi.experiments.m1_goal_learning import (
    CountNativeSteps,
    admitted_source_commit,
    count_optimizer_steps,
    numeric_state,
    save_progress,
)

MEMORY = "MiniGrid-MemoryS7-v0"
DOORKEY = "MiniGrid-DoorKey-8x8-v0"
TASKS = (MEMORY, DOORKEY)
ARMS = ("blocked", "interleaved", "memory_only", "doorkey_only")
SEEDS = (71, 83)
MEMORY_MONITOR = tuple(range(104001, 104007))
DOORKEY_MONITOR = tuple(range(105001, 105007))
MEMORY_TEST = tuple(range(102001, 102013))
DOORKEY_TEST = tuple(range(103001, 103013))
STEPS_PER_TASK = 65536
ROLLOUT = 128
BATCH = 64
EPOCHS = 2
EVAL_HORIZON = 256
MISSION_WIDTH = 64
ACTION_HISTORY = 4


def mission_vector(mission: str) -> np.ndarray:
    """Fixed hashed public word bag; no task-specific vocabulary or oracle features."""
    if not isinstance(mission, str):
        raise TypeError("Policy mission must be the publicly observed string")
    words = re.findall(r"[a-z0-9]+", mission.lower())
    result = np.zeros(MISSION_WIDTH, dtype=np.float32)
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        result[int.from_bytes(digest[:4], "big") % MISSION_WIDTH] += 1
    if words:
        result /= len(words)
    return result


class PublicGoalActions(gym.Wrapper):
    """Only local public image, mission words and actually executed actions."""

    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError("MiniGrid public observation must be a Dict")
        image = env.observation_space["image"]
        if image.shape != (7, 7, 3) or not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("M5 local public image and native action contract changed")
        self.n_actions = int(env.action_space.n)
        if self.n_actions != 7:
            raise ValueError("M5 requires native seven-action MiniGrid control")
        self.previous = deque([self.n_actions] * ACTION_HISTORY, maxlen=ACTION_HISTORY)
        width = int(np.prod(image.shape)) + MISSION_WIDTH + ACTION_HISTORY * (
            self.n_actions + 1
        )
        self.observation_space = Box(0.0, 255.0, shape=(width,), dtype=np.float32)

    def _encode(self, raw):
        if set(raw) != {"image", "direction", "mission"}:
            raise ValueError("Unexpected task observation fields")
        pixels = np.asarray(raw["image"], dtype=np.float32)
        if pixels.shape != (7, 7, 3):
            raise ValueError("Unexpected local public image dimensions")
        actions = np.zeros((ACTION_HISTORY, self.n_actions + 1), dtype=np.float32)
        for i, action in enumerate(self.previous):
            actions[i, action] = 1
        return np.concatenate((pixels.ravel(), mission_vector(raw["mission"]),
                               actions.ravel()))

    def reset(self, *, seed=None, options=None):
        self.previous = deque([self.n_actions] * ACTION_HISTORY, maxlen=ACTION_HISTORY)
        observation, info = self.env.reset(seed=seed, options=options)
        return self._encode(observation), info

    def step(self, action):
        executed = int(action)
        if not 0 <= executed < self.n_actions:
            raise ValueError("Unregistered native action")
        observation, reward, terminated, truncated, info = self.env.step(executed)
        self.previous.append(executed)
        return self._encode(observation), reward, terminated, truncated, info


class SeededTrainingNative(CountNativeSteps):
    """Task-local deterministic native training layouts, independent of test splits."""

    def __init__(self, env, *, seed: int, task: str):
        super().__init__(env, max_native_steps=STEPS_PER_TASK)
        if seed not in SEEDS or task not in TASKS:
            raise ValueError("Unregistered M5 model seed or training task")
        self.source_seed = seed
        self.source_task = task
        self.seed_hash = hashlib.sha256()
        self.first_reset_seed = None
        self.last_reset_seed = None

    def reset(self, **kwargs):
        task_index = TASKS.index(self.source_task)
        layout_seed = (1_000_000 + self.source_seed * 500_000
                       + task_index * 200_000 + self.native_resets)
        # Ignore SB3's one-time seeded reset: all training episode layouts use
        # only this registered task-local stream, never monitor/heldout seeds.
        kwargs["seed"] = layout_seed
        self.seed_hash.update(f"{layout_seed}\n".encode())
        if self.first_reset_seed is None:
            self.first_reset_seed = layout_seed
        self.last_reset_seed = layout_seed
        return super().reset(**kwargs)

    def provenance(self):
        return {
            "native_actions": self.native_steps,
            "native_resets": self.native_resets,
            "positive_reward_steps": self.native_positive_reward_steps,
            "rewarded_goal_terminations": self.native_rewarded_goal_terminations,
            "public_reward_sum": self.native_public_reward_sum,
            "first_reset_seed": self.first_reset_seed,
            "last_reset_seed": self.last_reset_seed,
            "reset_seed_stream_sha256": self.seed_hash.hexdigest(),
        }


class DoorKeyEffectAudit(gym.Wrapper):
    """Evaluator-only private physical effects; NEVER changes observations/info."""

    def __init__(self, env):
        super().__init__(env)
        self.effective_key_pickups = 0
        self.effective_door_openings = 0
        self._last_doors = {}

    def _door_states(self):
        world = self.env.unwrapped
        grid = world.grid
        return {
            (x, y): (bool(tile.is_locked), bool(tile.is_open))
            for x in range(grid.width) for y in range(grid.height)
            if isinstance(tile := grid.get(x, y), Door)
        }

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.effective_key_pickups = 0
        self.effective_door_openings = 0
        self._last_doors = self._door_states()
        return observation, info

    def step(self, action):
        previous_key = isinstance(self.env.unwrapped.carrying, Key)
        was_empty = self.env.unwrapped.carrying is None
        previous_doors = self._last_doors
        result = self.env.step(action)
        current_key = isinstance(self.env.unwrapped.carrying, Key)
        self._last_doors = self._door_states()
        if int(action) == 3 and was_empty and current_key:
            self.effective_key_pickups += 1
        if int(action) == 5 and previous_key:
            for cell, (locked, opened) in previous_doors.items():
                now = self._last_doors.get(cell)
                if locked and not opened and now == (False, True):
                    self.effective_door_openings += 1
        return result


def make_task(task: str, *, training_seed: int | None = None,
              audit: bool = False):
    if task not in TASKS:
        raise ValueError("Unknown M5 public-goal task")
    raw = gym.make(task)
    raw = gym.wrappers.TimeLimit(raw, max_episode_steps=EVAL_HORIZON)
    public = PublicGoalActions(raw)
    if training_seed is not None:
        if audit:
            raise ValueError("Privileged physical audit forbidden during training")
        return SeededTrainingNative(public, seed=training_seed, task=task)
    if audit and task != DOORKEY:
        raise ValueError("Physical key audit is only meaningful for DoorKey")
    wrapped = DoorKeyEffectAudit(public) if audit else public
    return CountNativeSteps(wrapped, max_native_steps=EVAL_HORIZON)


def validate_config(config: dict) -> dict:
    expected = {
        "registered_study": "m5",
        "tasks": list(TASKS),
        "arms": list(ARMS),
        "model_seeds": list(SEEDS),
        "steps_per_task": STEPS_PER_TASK,
        "rollout_steps": ROLLOUT,
        "batch_size": BATCH,
        "epochs_per_rollout": EPOCHS,
        "evaluation_horizon": EVAL_HORIZON,
        "memory_monitor_seeds": list(MEMORY_MONITOR),
        "doorkey_monitor_seeds": list(DOORKEY_MONITOR),
        "memory_heldout_seeds": list(MEMORY_TEST),
        "doorkey_heldout_seeds": list(DOORKEY_TEST),
        "mission_encoding_width": MISSION_WIDTH,
        "action_history": ACTION_HISTORY,
        "output_dir": "runs/m5_continual_retention",
    }
    if config != expected:
        raise ValueError("M5 frozen preregistration contract changed")
    return config


def schedule(arm: str) -> tuple[tuple[str, ...], ...]:
    """Exactly two registered phases of whole PPO rollout blocks."""
    n = STEPS_PER_TASK // ROLLOUT
    if arm == "blocked":
        return ((MEMORY,) * n, (DOORKEY,) * n)
    if arm == "interleaved":
        alternating = tuple(TASKS * (n // 2))
        return (alternating, alternating)
    if arm == "memory_only":
        return ((MEMORY,) * (n // 2), (MEMORY,) * (n // 2))
    if arm == "doorkey_only":
        return ((DOORKEY,) * (n // 2), (DOORKEY,) * (n // 2))
    raise ValueError("Unknown training schedule")


def evaluate(agent, task: str, seeds: tuple[int, ...]) -> dict:
    records = []
    for seed in seeds:
        env = make_task(task, audit=task == DOORKEY)
        try:
            env.action_space.seed(seed)
            obs, _ = env.reset(seed=seed)
            terminated = truncated = False
            reward_sum = 0.0
            histogram = [0] * int(env.action_space.n)
            for _ in range(EVAL_HORIZON):
                action, _ = agent.predict(obs, deterministic=True)
                executed = int(np.asarray(action).item())
                if not 0 <= executed < len(histogram):
                    raise ValueError("Policy returned a non-native action")
                obs, reward, terminated, truncated, _ = env.step(executed)
                histogram[executed] += 1
                reward_sum += float(reward)
                if terminated or truncated:
                    break
            if not (terminated or truncated) or env.native_resets != 1:
                raise ValueError("Unfinished/invalid native evaluation episode")
            if sum(histogram) != env.native_steps:
                raise ValueError("Evaluation action provenance inconsistent")
            goal = int(bool(terminated) and reward_sum > 0)
            effects = env.env if task == DOORKEY else None
            pickups = effects.effective_key_pickups if effects is not None else 0
            openings = effects.effective_door_openings if effects is not None else 0
            physical = int(bool(goal and pickups and openings)) if effects else None
            records.append({
                "seed": seed, "reward": reward_sum,
                "terminated": bool(terminated), "truncated": bool(truncated),
                "goal_success": goal,
                "verified_key_pickups": pickups,
                "verified_door_openings": openings,
                "physical_key_door_goal": physical,
                "evaluation_real_actions": env.native_steps,
                "evaluation_real_resets": env.native_resets,
                "executed_action_histogram": histogram,
            })
        finally:
            env.close()
    return {
        "episodes": len(records),
        "rewarded_goals": sum(r["goal_success"] for r in records),
        "verified_physical_goals": (
            sum(r["physical_key_door_goal"] for r in records) if task == DOORKEY else None
        ),
        "evaluation_real_actions": sum(r["evaluation_real_actions"] for r in records),
        "evaluation_real_resets": sum(r["evaluation_real_resets"] for r in records),
        "episode_results": records,
    }


def score(records: list[dict]) -> dict:
    models = {(r["arm"], r["model_seed"]): r for r in records}
    if len(records) != len(ARMS) * len(SEEDS) or set(models) != {
        (arm, seed) for arm in ARMS for seed in SEEDS
    }:
        raise ValueError("Expected all eight independently trained M5 models")
    def goals(arm, task):
        return sum(models[(arm, seed)]["final_test"][task]["rewarded_goals"] for seed in SEEDS)
    blocked_memory, interleaved_memory = goals("blocked", MEMORY), goals("interleaved", MEMORY)
    blocked_door, interleaved_door = goals("blocked", DOORKEY), goals("interleaved", DOORKEY)
    seeded_physical = all(
        models[("interleaved", seed)]["final_test"][MEMORY]["rewarded_goals"] >= 1
        and models[("interleaved", seed)]["final_test"][DOORKEY]["verified_physical_goals"] >= 1
        for seed in SEEDS
    )
    return {
        "blocked_memory_goals_of_24": blocked_memory,
        "interleaved_memory_goals_of_24": interleaved_memory,
        "blocked_doorkey_goals_of_24": blocked_door,
        "interleaved_doorkey_goals_of_24": interleaved_door,
        "memory_retention_difference_goals_of_24": interleaved_memory - blocked_memory,
        "doorkey_difference_goals_of_24": interleaved_door - blocked_door,
        "each_interleaved_seed_has_memory_and_physical_door_goal": seeded_physical,
        "primary_gate_pass": (
            interleaved_memory - blocked_memory >= 4
            and interleaved_door - blocked_door >= -2
            and seeded_physical
        ),
        "note": "Descriptive, two-seed, within-MiniGrid pilot; not cross-domain evidence.",
    }


def run(config_path: Path, output: Path) -> dict:
    if output.exists() or (output.parent / "progress.json").exists():
        raise FileExistsError("Refusing to overwrite previous M5 outputs")
    raw_config = config_path.read_bytes()
    config = validate_config(yaml.safe_load(raw_config))
    if output.parent.as_posix() != config["output_dir"]:
        raise ValueError("M5 output directory changed")
    source_sha = admitted_source_commit()
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    trained, models = [], {}
    stage = "preflight"
    progress = {"training_real_actions": 0, "optimizer_step_calls": 0,
                "evaluation_real_actions": 0, "evaluation_real_resets": 0}
    start_time = time.monotonic()
    active_envs, active_optimizer = None, None
    try:
        for task in TASKS:
            env = make_task(task, training_seed=SEEDS[0])
            try:
                if env.action_space.n != 7 or env.observation_space.shape != (243,):
                    raise ValueError("Incompatible public task observation/action contracts")
            finally:
                env.close()

        for arm in ARMS:
            for seed in SEEDS:
                stage = f"fit-{arm}-{seed}"
                training = {task: make_task(task, training_seed=seed) for task in TASKS}
                active_envs = training
                agent = PPO(
                    "MlpPolicy", training[MEMORY], seed=seed, device="cpu",
                    n_steps=ROLLOUT, batch_size=BATCH, n_epochs=EPOCHS,
                    policy_kwargs={"net_arch": {"pi": [32], "vf": [32]}}, verbose=0,
                )
                counts = count_optimizer_steps(agent)
                active_optimizer = counts
                task_blocks = {task: 0 for task in TASKS}
                monitor = {}
                initial = {}
                for task, seeds in ((MEMORY, MEMORY_MONITOR), (DOORKEY, DOORKEY_MONITOR)):
                    initial[task] = evaluate(agent, task, seeds)
                for phase, blocks in enumerate(schedule(arm), start=1):
                    for task in blocks:
                        # Every 128-action block, even without switching tasks,
                        # follows this identical upstream reset/update contract.
                        agent.set_env(training[task], force_reset=True)
                        agent.learn(total_timesteps=ROLLOUT, reset_num_timesteps=False)
                        task_blocks[task] += 1
                    monitor[f"phase{phase}"] = {
                        "task_blocks": dict(task_blocks),
                        MEMORY: evaluate(agent, MEMORY, MEMORY_MONITOR),
                        DOORKEY: evaluate(agent, DOORKEY, DOORKEY_MONITOR),
                    }
                expected = {
                    MEMORY: STEPS_PER_TASK // ROLLOUT if arm in (
                        "blocked", "interleaved", "memory_only"
                    ) else 0,
                    DOORKEY: STEPS_PER_TASK // ROLLOUT if arm in (
                        "blocked", "interleaved", "doorkey_only"
                    ) else 0,
                }
                if arm in ("memory_only", "doorkey_only"):
                    expected = {
                        MEMORY: STEPS_PER_TASK // ROLLOUT if arm == "memory_only" else 0,
                        DOORKEY: STEPS_PER_TASK // ROLLOUT if arm == "doorkey_only" else 0,
                    }
                if task_blocks != expected:
                    raise ValueError("M5 training task block exposure mismatch")
                real_actions = sum(env.native_steps for env in training.values())
                optimizer_expected = sum(task_blocks.values()) * EPOCHS * (
                    ROLLOUT // BATCH
                )
                if (real_actions != sum(task_blocks.values()) * ROLLOUT
                        or counts["calls"] != optimizer_expected
                        or agent.num_timesteps != real_actions):
                    raise ValueError("M5 training native/optimizer action accounting mismatch")
                checkpoint = numeric_state(
                    agent, output.parent / f"{arm}-seed{seed}.npz"
                )
                row = {
                    "arm": arm, "model_seed": seed,
                    "initial_monitor": initial,
                    "training_phase_monitors": monitor,
                    "parameter_count": sum(p.numel() for p in agent.policy.parameters()),
                    "training_real_actions": real_actions,
                    "optimizer_step_calls": counts["calls"],
                    "task_blocks": task_blocks,
                    "training_native_by_task": {
                        task: env.provenance() for task, env in training.items()
                    },
                    "numeric_policy": checkpoint,
                }
                trained.append(row)
                models[(arm, seed)] = agent
                for task in TASKS:
                    training[task].close()
                active_envs, active_optimizer = None, None
                progress["training_real_actions"] += real_actions
                progress["optimizer_step_calls"] += counts["calls"]
                progress["evaluation_real_actions"] += sum(
                    measurement["evaluation_real_actions"]
                    for measurement in (*initial.values(), *(
                        monitor[phase][task] for phase in ("phase1", "phase2")
                        for task in TASKS
                    ))
                )
                progress["evaluation_real_resets"] += sum(
                    measurement["evaluation_real_resets"]
                    for measurement in (*initial.values(), *(
                        monitor[phase][task] for phase in ("phase1", "phase2")
                        for task in TASKS
                    ))
                )
                save_progress(output.parent / "progress.json", {
                    "status": "partial_not_a_completed_study", "source_commit": source_sha,
                    "completed_models": len(trained), **progress, "records": trained,
                })
        stage = "independent_heldout_after_all_models"
        expected_training = 8 * STEPS_PER_TASK + 4 * STEPS_PER_TASK
        expected_optimizers = (expected_training // ROLLOUT) * (
            ROLLOUT // BATCH
        ) * EPOCHS
        if (progress["training_real_actions"] != expected_training
                or progress["optimizer_step_calls"] != expected_optimizers):
            raise ValueError("M5 study-wide matched training allocation mismatch")

        for row in trained:
            agent = models[(row["arm"], row["model_seed"])]
            row["final_test"] = {
                MEMORY: evaluate(agent, MEMORY, MEMORY_TEST),
                DOORKEY: evaluate(agent, DOORKEY, DOORKEY_TEST),
            }
            progress["evaluation_real_actions"] += sum(
                x["evaluation_real_actions"] for x in row["final_test"].values()
            )
            progress["evaluation_real_resets"] += sum(
                x["evaluation_real_resets"] for x in row["final_test"].values()
            )
            save_progress(output.parent / "progress.json", {
                "status": "partial_not_a_completed_study", "source_commit": source_sha,
                "completed_models": len(trained), "completed_final_tests": sum(
                    "final_test" in r for r in trained
                ), **progress, "records": trained,
            })
    except BaseException as error:
        partial = {
            "status": "interrupted_not_a_completed_study",
            "source_commit": source_sha, "failure_phase": stage,
            "failure_type": type(error).__name__,
            "completed_models": len(trained), **progress, "records": trained,
            "elapsed_seconds": round(time.monotonic() - start_time, 3),
            "in_flight_training": (
                {task: env.provenance() for task, env in active_envs.items()}
                if active_envs is not None else None
            ),
            "in_flight_optimizer_step_calls": (
                active_optimizer["calls"] if active_optimizer is not None else 0
            ),
        }
        save_progress(output.parent / "progress.json", partial)
        raise
    finally:
        if active_envs is not None:
            for env in active_envs.values():
                env.close()

    result = {
        "study": "BAGI-M5 public-goal blocked versus interleaved task retention",
        "registered_study": "m5",
        "status": "completed_independent_continual_retention_pilot",
        "source_commit": source_sha,
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "versions": {name: version(name) for name in
                     ("gymnasium", "minigrid", "stable-baselines3")},
        "task_observation_contract": (
            "public_local_image_plus_hashed_public_mission_plus_four_executed_actions"
        ),
        "new_training_real_actions": progress["training_real_actions"],
        "optimizer_step_calls": progress["optimizer_step_calls"],
        "new_evaluation_real_actions": progress["evaluation_real_actions"],
        "new_evaluation_real_resets": progress["evaluation_real_resets"],
        "teacher_queries": 0,
        "elapsed_seconds": round(time.monotonic() - start_time, 3),
        "pilot_gates": score(trained),
        "records": trained,
        "limitations": [
            "Memory and DoorKey are within one MiniGrid simulator family.",
            "Mission text is public hashed word features, not pretrained language understanding.",
            "DoorKey pickup/opening labels are evaluator-private and absent from policy inputs.",
            "Two model initializations and 12 heldout layouts per task are a small pilot.",
            "On-policy fresh task interleaving is not off-policy CLEAR replay.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m5_continual_retention.yaml"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/m5_continual_retention/summary.json"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("M5 training needs explicit --execute")
    result = run(args.config, args.output)
    print(json.dumps({
        "status": result["status"],
        "new_training_real_actions": result["new_training_real_actions"],
        "optimizer_step_calls": result["optimizer_step_calls"],
        "new_evaluation_real_actions": result["new_evaluation_real_actions"],
        "pilot_gates": result["pilot_gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
