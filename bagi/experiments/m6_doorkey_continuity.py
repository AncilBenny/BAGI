"""M6: matched native episode-continuity ablation for sparse DoorKey PPO.

Both arms use identical on-policy rollout/update calls; only SB3 set_env's
force_reset flag changes. All actual public actions and physical evaluator-only
key/door effects are recorded separately from simulator reward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import torch
import yaml

from bagi.experiments.m1_goal_learning import (
    CountNativeSteps,
    admitted_source_commit,
    count_optimizer_steps,
    numeric_state,
    save_progress,
)
from bagi.experiments.m5_continual_retention import (
    DOORKEY,
    EVAL_HORIZON,
    PublicGoalActions,
)
from bagi.experiments.m5_continual_retention import evaluate as evaluate_door

ARMS = ("reset128", "continue_episode")
SEEDS = (97, 109)
TRAIN_STEPS = 65536
ROLLOUT = 128
BATCH = 64
EPOCHS = 2
TRAIN_BLOCKS = TRAIN_STEPS // ROLLOUT
MONITOR = tuple(range(108001, 108007))
HELDOUT = tuple(range(106001, 106013))


class NativeDoorTraining(CountNativeSteps):
    """Track real task-local episode lengths, natural endings and forced resets."""

    def __init__(self, env, *, model_seed: int):
        if model_seed not in SEEDS:
            raise ValueError("Only preregistered M6 initialization seeds")
        super().__init__(env, max_native_steps=TRAIN_STEPS)
        self.model_seed = model_seed
        self.seed_hash = hashlib.sha256()
        self.first_reset_seed = None
        self.last_reset_seed = None
        self.episode_length = 0
        self.episode_ended = True
        self.max_episode_length = 0
        self.incomplete_episode_resets = 0
        self.native_completed_episodes = 0
        self.native_time_limit_episodes = 0
        self.executed_action_histogram = [0] * 7

    def reset(self, **kwargs):
        if self.episode_length and not self.episode_ended:
            self.incomplete_episode_resets += 1
        seed = 1_000_000 + self.model_seed * 500_000 + self.native_resets
        if self.first_reset_seed is None:
            self.first_reset_seed = seed
        self.last_reset_seed = seed
        self.seed_hash.update(f"{seed}\n".encode())
        self.episode_length = 0
        self.episode_ended = False
        kwargs["seed"] = seed
        return super().reset(**kwargs)

    def step(self, action):
        actual = int(action)
        if not 0 <= actual < len(self.executed_action_histogram):
            raise ValueError("Non-native DoorKey action")
        observation, reward, terminated, truncated, info = super().step(actual)
        self.executed_action_histogram[actual] += 1
        self.episode_length += 1
        self.max_episode_length = max(self.episode_length, self.max_episode_length)
        if terminated or truncated:
            self.episode_ended = True
            self.native_completed_episodes += 1
            self.native_time_limit_episodes += int(bool(truncated))
        return observation, reward, terminated, truncated, info

    def measured(self) -> dict:
        return {
            "native_actions": self.native_steps,
            "native_resets": self.native_resets,
            "rewarded_goal_terminations": self.native_rewarded_goal_terminations,
            "positive_reward_steps": self.native_positive_reward_steps,
            "public_reward_sum": self.native_public_reward_sum,
            "native_completed_episodes": self.native_completed_episodes,
            "native_time_limit_episodes": self.native_time_limit_episodes,
            "max_episode_native_actions": self.max_episode_length,
            "incomplete_episode_resets": self.incomplete_episode_resets,
            "pending_episode_native_actions": self.episode_length,
            "first_reset_seed": self.first_reset_seed,
            "last_reset_seed": self.last_reset_seed,
            "reset_seed_stream_sha256": self.seed_hash.hexdigest(),
            "executed_action_histogram": self.executed_action_histogram,
        }


def make_training_env(model_seed: int) -> NativeDoorTraining:
    if model_seed not in SEEDS:
        raise ValueError("Unknown M6 training seed")
    raw = gym.make(DOORKEY)
    native = gym.wrappers.TimeLimit(raw, max_episode_steps=EVAL_HORIZON)
    public = PublicGoalActions(native)
    return NativeDoorTraining(public, model_seed=model_seed)


def validate_config(config: dict) -> dict:
    expected = {
        "registered_study": "m6",
        "task": DOORKEY,
        "arms": list(ARMS),
        "training_seeds": list(SEEDS),
        "train_steps_per_fit": TRAIN_STEPS,
        "rollout_steps": ROLLOUT,
        "batch_size": BATCH,
        "epochs_per_rollout": EPOCHS,
        "max_episode_steps": EVAL_HORIZON,
        "monitor_seeds": list(MONITOR),
        "heldout_seeds": list(HELDOUT),
        "output_dir": "runs/m6_doorkey_continuity",
    }
    if config != expected:
        raise ValueError("Registered M6 reset-continuity protocol changed")
    return config


def compare_reset_integrity(records: list[dict]) -> dict:
    """Check the hypothesized reset manipulation BEFORE heldout evaluation."""
    models = {(r["arm"], r["model_seed"]): r for r in records}
    if len(records) != len(ARMS) * len(SEEDS) or set(models) != {
        (arm, seed) for arm in ARMS for seed in SEEDS
    }:
        raise ValueError("M6 models or unique seeds missing")
    comparison = {}
    for seed in SEEDS:
        reset = models[("reset128", seed)]["training"]
        carried = models[("continue_episode", seed)]["training"]
        reset_count = reset["incomplete_episode_resets"]
        carry_count = carried["incomplete_episode_resets"]
        passes = (
            reset_count > carry_count
            and carry_count == 0
            and reset["max_episode_native_actions"] <= ROLLOUT
            and carried["max_episode_native_actions"] > ROLLOUT
        )
        comparison[str(seed)] = {
            "reset_incomplete_episode_resets": reset_count,
            "continuous_incomplete_episode_resets": carry_count,
            "reset_max_episode_actions": reset["max_episode_native_actions"],
            "continuous_max_episode_actions": carried["max_episode_native_actions"],
            "valid_continuity_intervention": passes,
        }
    return comparison


def score(records: list[dict], continuity: dict) -> dict:
    models = {(r["arm"], r["model_seed"]): r for r in records}
    reset_goals = sum(
        models[("reset128", seed)]["independent_heldout"]["rewarded_goals"]
        for seed in SEEDS
    )
    continue_goals = sum(
        models[("continue_episode", seed)]["independent_heldout"]["rewarded_goals"]
        for seed in SEEDS
    )
    reset_physical = sum(
        models[("reset128", seed)]["independent_heldout"]["verified_physical_goals"]
        for seed in SEEDS
    )
    continue_physical = sum(
        models[("continue_episode", seed)]["independent_heldout"]["verified_physical_goals"]
        for seed in SEEDS
    )
    each_seed = all(
        models[("continue_episode", seed)]["independent_heldout"][
            "verified_physical_goals"
        ] >= 1 for seed in SEEDS
    )
    return {
        "reset128_rewarded_goals_of_24": reset_goals,
        "continue_episode_rewarded_goals_of_24": continue_goals,
        "reset128_physical_goals_of_24": reset_physical,
        "continue_episode_physical_goals_of_24": continue_physical,
        "physical_goal_difference_of_24": continue_physical - reset_physical,
        "both_continuation_model_seeds_have_physical_goal": each_seed,
        "continuity_manipulation_valid": all(
            row["valid_continuity_intervention"] for row in continuity.values()
        ),
        "primary_gate_pass": (
            continue_physical - reset_physical >= 4 and each_seed
            and all(row["valid_continuity_intervention"] for row in continuity.values())
        ),
        "limitations": (
            "Small two-model-seed pilot; task competence is not cross-task continual learning."
        ),
    }


def run(config_path: Path, output: Path) -> dict:
    if output.exists() or (output.parent / "progress.json").exists():
        raise FileExistsError("Refusing overwrite of M6 experimental results")
    raw_config = config_path.read_bytes()
    config = validate_config(yaml.safe_load(raw_config))
    if output.parent.as_posix() != config["output_dir"]:
        raise ValueError("M6 output location differs from preregistration")
    source_sha = admitted_source_commit()
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    started = time.monotonic()
    records, models = [], {}
    phase = "preflight"
    in_flight_environment = None
    in_flight_updates = None
    in_flight_monitor_before = None
    in_flight_monitor_after = None
    counted = {"training_actions": 0, "optimizer_updates": 0,
               "evaluation_actions": 0, "evaluation_resets": 0}
    try:
        example = make_training_env(SEEDS[0])
        try:
            if example.action_space.n != 7 or example.observation_space.shape != (243,):
                raise ValueError("DoorKey M6 public policy interface changed")
        finally:
            example.close()

        for arm in ARMS:
            for seed in SEEDS:
                phase = f"train-{arm}-{seed}"
                env = make_training_env(seed)
                in_flight_environment = env
                agent = PPO(
                    "MlpPolicy", env, seed=seed, device="cpu",
                    n_steps=ROLLOUT, batch_size=BATCH, n_epochs=EPOCHS,
                    policy_kwargs={"net_arch": {"pi": [32], "vf": [32]}}, verbose=0,
                )
                optimizer = count_optimizer_steps(agent)
                in_flight_updates = optimizer
                initial = evaluate_door(agent, DOORKEY, MONITOR)
                in_flight_monitor_before = initial
                for block in range(TRAIN_BLOCKS):
                    # Identical per-block rebind and learn in both arms.
                    # The only intervention is force_reset's Boolean value.
                    # Rebind the EXISTING VecEnv rather than wrapping a fresh
                    # Monitor every block: the latter is unreset when continuing.
                    agent.set_env(agent.get_env(), force_reset=(arm == "reset128"))
                    agent.learn(total_timesteps=ROLLOUT, reset_num_timesteps=False)
                if (env.native_steps != TRAIN_STEPS
                        or agent.num_timesteps != TRAIN_STEPS
                        or optimizer["calls"] != TRAIN_BLOCKS * EPOCHS * (ROLLOUT // BATCH)):
                    raise ValueError("M6 actual native training/update counts mismatched")
                metrics = env.measured()
                if sum(metrics["executed_action_histogram"]) != env.native_steps:
                    raise ValueError("M6 training action histogram disagrees")
                phase = f"monitor-{arm}-{seed}"
                after = evaluate_door(agent, DOORKEY, MONITOR)
                in_flight_monitor_after = after
                phase = f"export-{arm}-{seed}"
                checkpoint = numeric_state(
                    agent, output.parent / f"{arm}-seed{seed}.npz"
                )
                row = {
                    "arm": arm, "model_seed": seed,
                    "training": metrics,
                    "training_real_actions": env.native_steps,
                    "optimizer_step_calls": optimizer["calls"],
                    "policy_parameters": sum(p.numel() for p in agent.policy.parameters()),
                    "monitor_before": initial, "monitor_after": after,
                    "numeric_policy": checkpoint,
                }
                records.append(row)
                models[(arm, seed)] = agent
                env.close()
                in_flight_environment = None
                in_flight_updates = None
                in_flight_monitor_before = None
                in_flight_monitor_after = None
                counted["training_actions"] += row["training_real_actions"]
                counted["optimizer_updates"] += optimizer["calls"]
                for stage in ("monitor_before", "monitor_after"):
                    counted["evaluation_actions"] += row[stage]["evaluation_real_actions"]
                    counted["evaluation_resets"] += row[stage]["evaluation_real_resets"]
                save_progress(output.parent / "progress.json", {
                    "status": "partial_not_a_completed_study",
                    "phase": phase, "completed_models": len(records),
                    "source_commit": source_sha, **counted, "records": records,
                })

        phase = "preheldout_continuity_validation"
        if counted["training_actions"] != len(ARMS) * len(SEEDS) * TRAIN_STEPS:
            raise ValueError("M6 four-model native action total disagrees")
        if counted["optimizer_updates"] != 8192:
            raise ValueError("M6 four-model optimizer step total disagrees")
        continuity = compare_reset_integrity(records)
        if not all(row["valid_continuity_intervention"] for row in continuity.values()):
            raise ValueError("M6 force_reset vs continue intervention was not observed")

        phase = "independent_heldout"
        for row in records:
            agent = models[(row["arm"], row["model_seed"])]
            row["independent_heldout"] = evaluate_door(agent, DOORKEY, HELDOUT)
            counted["evaluation_actions"] += row["independent_heldout"][
                "evaluation_real_actions"
            ]
            counted["evaluation_resets"] += row["independent_heldout"][
                "evaluation_real_resets"
            ]
            save_progress(output.parent / "progress.json", {
                "status": "partial_not_a_completed_study",
                "phase": phase, "completed_models": len(records),
                "completed_heldout_models": sum(
                    "independent_heldout" in item for item in records
                ), "source_commit": source_sha, **counted, "records": records,
            })
    except BaseException as error:
        save_progress(output.parent / "progress.json", {
            "status": "interrupted_not_a_completed_study",
            "failure_phase": phase,
            "failure_type": type(error).__name__,
            "source_commit": source_sha,
            "completed_models": len(records), **counted,
            "in_flight_native_actions": (
                in_flight_environment.native_steps
                if in_flight_environment is not None else 0
            ),
            "in_flight_optimizer_updates": (
                in_flight_updates["calls"] if in_flight_updates is not None else 0
            ),
            "in_flight_completed_monitor_before_actions": (
                in_flight_monitor_before["evaluation_real_actions"]
                if in_flight_monitor_before is not None else 0
            ),
            "in_flight_completed_monitor_after_actions": (
                in_flight_monitor_after["evaluation_real_actions"]
                if in_flight_monitor_after is not None else 0
            ),
            "records": records,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        raise
    finally:
        if in_flight_environment is not None:
            in_flight_environment.close()

    result = {
        "study": "BAGI-M6 DoorKey PPO 128-action forced-reset versus episode-continuity",
        "registered_study": "m6",
        "status": "completed_independent_episode_continuity_ablation",
        "source_commit": source_sha,
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "versions": {pkg: version(pkg) for pkg in (
            "gymnasium", "minigrid", "stable-baselines3"
        )},
        "training_real_actions": counted["training_actions"],
        "optimizer_step_calls": counted["optimizer_updates"],
        "evaluation_real_actions": counted["evaluation_actions"],
        "evaluation_real_resets": counted["evaluation_resets"],
        "teacher_queries": 0,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "continuity_manipulation_checks": continuity,
        "pilot_gate": score(records, continuity),
        "records": records,
        "limitations": [
            "This tests episodic reset continuity in one sparse MiniGrid task only.",
            "Training monitor and final heldout layouts are distinct and frozen.",
            "Physical DoorKey success requires key pickup + door opening + terminal goal.",
            "No policy access to private evaluator state, tool task or cross-domain transfer.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/m6_doorkey_continuity.yaml"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/m6_doorkey_continuity/summary.json"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("M6 fresh PPO training requires explicit --execute")
    result = run(args.config, args.output)
    print(json.dumps({
        "source_commit": result["source_commit"],
        "status": result["status"],
        "training_real_actions": result["training_real_actions"],
        "optimizer_step_calls": result["optimizer_step_calls"],
        "evaluation_real_actions": result["evaluation_real_actions"],
        "pilot_gate": result["pilot_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
