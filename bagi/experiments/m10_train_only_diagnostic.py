"""M10-D3: train-only causal opportunity diagnosis for public-observation PPO.

Reuses upstream PPO, the established BAGI public wrapper and native auditor.
These monitor layouts are diagnostic/tunable, NEVER independent heldout data.
No privileged environment state enters the actor, model, reward or optimizer.
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
import torch
import yaml

from bagi.experiments.m1_goal_learning import (
    admitted_source_commit,
    count_optimizer_steps,
    numeric_state,
    save_progress,
)
from bagi.experiments.m5_continual_retention import DOORKEY, EVAL_HORIZON, PublicGoalActions
from bagi.experiments.m6_doorkey_continuity import BATCH, EPOCHS, ROLLOUT, NativeDoorTraining
from bagi.experiments.m10_native_audit import NativeDoorKeyAudit

SEEDS = (181, 197)
MONITOR = tuple(range(119001, 119013))
TRAIN_STEPS = 131072
EXPECTED_UPDATES = TRAIN_STEPS // ROLLOUT * EPOCHS * (ROLLOUT // BATCH)
OUTPUT_DIR = "runs/m10_train_only_diagnostic"


def validate_config(config: dict) -> dict:
    """Freeze the diagnostic question, action budget and train-only split."""
    expected = {
        "registered_study": "m10",
        "phase": "train_only_causal_diagnostic",
        "task": DOORKEY,
        "model_seeds": list(SEEDS),
        "monitor_seeds": list(MONITOR),
        "train_steps_per_fit": TRAIN_STEPS,
        "rollout_steps": ROLLOUT,
        "batch_size": BATCH,
        "epochs_per_rollout": EPOCHS,
        "evaluation_horizon": EVAL_HORIZON,
        "policy": "public_raw_codes_and_executed_history",
        "control": "uniform_native_random",
        "output_dir": OUTPUT_DIR,
    }
    if config != expected:
        raise ValueError("M10 train-only causal diagnostic config changed")
    return config


def make_training_env(model_seed: int) -> NativeDoorTraining:
    if model_seed not in SEEDS:
        raise ValueError("M10 unregistered training model seed")
    env = gym.make(DOORKEY)
    timed = gym.wrappers.TimeLimit(env, max_episode_steps=EVAL_HORIZON)
    return NativeDoorTraining(
        PublicGoalActions(timed),
        model_seed=model_seed,
        allowed_seeds=SEEDS,
        max_native_steps=TRAIN_STEPS,
    )


def action_rng_seed(seed: int, layout: int, mode: str) -> int:
    if seed not in SEEDS or layout not in MONITOR:
        raise ValueError("M10 train-only layout/model seed not registered")
    if mode not in ("random", "sampled"):
        raise ValueError("Unregistered M10 action mode")
    return 80_000_000 + SEEDS.index(seed) * 100_000 + MONITOR.index(layout) * 10 + (
        1 if mode == "sampled" else 2
    )


def evaluation_episode(agent, layout: int, model_seed: int, mode: str) -> dict:
    """Native physical audit OUTSIDE the acting public-observation wrapper."""
    if mode not in ("random", "deterministic", "sampled"):
        raise ValueError("M10 evaluation mode is not registered")
    if mode == "random" and agent is not None:
        raise ValueError("Random control must not use a learned policy")
    if mode != "random" and agent is None:
        raise ValueError("PPO evaluation needs its learned public-only policy")
    if layout not in MONITOR or model_seed not in SEEDS:
        raise ValueError("Evaluation must use only new M10 train-monitor layouts")
    rng_seed = (
        action_rng_seed(model_seed, layout, mode) if mode != "deterministic" else None
    )
    random_actor = np.random.default_rng(rng_seed) if mode == "random" else None
    raw = gym.make(DOORKEY)
    env = NativeDoorKeyAudit(
        PublicGoalActions(gym.wrappers.TimeLimit(raw, max_episode_steps=EVAL_HORIZON))
    )
    try:
        observation, _ = env.reset(seed=layout)
        if mode == "sampled":
            torch.manual_seed(rng_seed)
        reward_sum = 0.0
        terminated = truncated = False
        for _ in range(EVAL_HORIZON):
            if random_actor is not None:
                executed = int(random_actor.integers(0, 7))
            else:
                chosen, _ = agent.predict(
                    observation, deterministic=mode == "deterministic"
                )
                executed = int(np.asarray(chosen).item())
            observation, reward, terminated, truncated, _ = env.step(executed)
            reward_sum += float(reward)
            if terminated or truncated:
                break
        summary = env.summary()
        if not (terminated or truncated):
            raise ValueError("M10 native diagnostic episode did not end")
        if summary["native_actions"] > EVAL_HORIZON:
            raise ValueError("M10 exceeded declared native evaluation horizon")
        return {
            "layout_seed": layout,
            "model_seed": model_seed,
            "mode": mode,
            "action_rng_seed": rng_seed,
            "native_reward_sum": reward_sum,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            **summary,
        }
    finally:
        env.env.close()


def aggregate(rows: list[dict]) -> dict:
    """Preserve episode denominators distinct from repeated physical events."""
    if not rows:
        raise ValueError("Cannot aggregate an empty native diagnostic")
    return {
        "episodes": len(rows),
        "native_actions": sum(r["native_actions"] for r in rows),
        "rewarded_goal_episodes": sum(
            int(r["terminated"] and r["native_reward_sum"] > 0) for r in rows
        ),
        "verified_full_physical_goal_episodes": sum(
            int(r["full_physical_goal"]) for r in rows
        ),
        "episodes_with_key_pickup": sum(
            int(r["effective_pickup_events"] > 0) for r in rows
        ),
        "episodes_with_valid_toggle_opportunity": sum(
            int(r["valid_toggle_opportunities"] > 0) for r in rows
        ),
        "episodes_with_effective_door_opening": sum(
            int(r["effective_door_open_events"] > 0) for r in rows
        ),
        "valid_toggle_opportunities": sum(
            r["valid_toggle_opportunities"] for r in rows
        ),
        "valid_toggle_attempts": sum(r["valid_toggle_attempts"] for r in rows),
        "ineffective_valid_toggles": sum(r["ineffective_valid_toggles"] for r in rows),
        "effective_pickup_events": sum(r["effective_pickup_events"] for r in rows),
        "effective_door_open_events": sum(r["effective_door_open_events"] for r in rows),
        "action_concentrated_95pct_episodes": sum(
            int(r["max_action_fraction"] is not None
                and r["max_action_fraction"] >= 0.95) for r in rows
        ),
        "native_action_histogram": [
            sum(r["action_counts"][str(action)] for r in rows)
            for action in range(7)
        ],
    }


def evaluate_suite(agent, model_seed: int, mode: str) -> dict:
    rows = [evaluation_episode(agent, layout, model_seed, mode) for layout in MONITOR]
    totals = aggregate(rows)
    if totals["episodes"] != len(MONITOR):
        raise ValueError("M10 diagnostic split incomplete")
    if sum(totals["native_action_histogram"]) != totals["native_actions"]:
        raise ValueError("M10 native action histogram mismatch")
    return {"totals": totals, "episodes": rows}


def decision_summary(records: list[dict]) -> dict:
    if len(records) != len(SEEDS):
        raise ValueError("M10 requires both independently initialized PPO fits")
    trained = [row["monitor_after_deterministic"]["totals"] for row in records]
    random = [row["random_control"]["totals"] for row in records]
    opportunities = sum(x["valid_toggle_opportunities"] for x in trained)
    attempts = sum(x["valid_toggle_attempts"] for x in trained)
    openings = sum(x["effective_door_open_events"] for x in trained)
    return {
        "trained_valid_toggle_opportunities": opportunities,
        "random_valid_toggle_opportunities": sum(
            x["valid_toggle_opportunities"] for x in random
        ),
        "trained_valid_toggle_attempts": attempts,
        "trained_effective_door_open_events": openings,
        "trained_full_physical_goals": sum(
            x["verified_full_physical_goal_episodes"] for x in trained
        ),
        "random_full_physical_goals": sum(
            x["verified_full_physical_goal_episodes"] for x in random
        ),
        "diagnostic_priority": (
            "navigation_or_exploration_hypothesis" if opportunities == 0
            else "action_selection_or_precondition_hypothesis" if attempts == 0
            else "effectiveness_hypothesis" if openings == 0
            else "post_opening_navigation_hypothesis"
        ),
        "not_causal_identification": True,
        "not_independent_evaluation": True,
    }


def run(config_path: Path, output: Path) -> dict:
    if output.exists() or (output.parent / "progress.json").exists():
        raise FileExistsError("Refusing to overwrite an earlier M10 diagnostic run")
    raw_config = config_path.read_bytes()
    validate_config(yaml.safe_load(raw_config))
    if output.parent.as_posix() != OUTPUT_DIR:
        raise ValueError("M10 diagnostic output path is not preregistered")
    source = admitted_source_commit()
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    start = time.monotonic()
    records = []
    in_flight_env = None
    in_flight_counter = None
    phase = "preflight"
    try:
        for seed in SEEDS:
            phase = f"baseline-seed-{seed}"
            env = make_training_env(seed)
            in_flight_env = env
            if env.action_space.n != 7 or env.observation_space.shape != (243,):
                raise ValueError("M10 public observation contract changed")
            agent = PPO(
                "MlpPolicy", env, seed=seed, device="cpu", n_steps=ROLLOUT,
                batch_size=BATCH, n_epochs=EPOCHS, ent_coef=0.0,
                policy_kwargs={"net_arch": {"pi": [32], "vf": [32]}}, verbose=0,
            )
            counter = count_optimizer_steps(agent)
            in_flight_counter = counter
            random = evaluate_suite(None, seed, "random")
            before = evaluate_suite(agent, seed, "deterministic")
            phase = f"fresh-ppo-training-seed-{seed}"
            agent.learn(total_timesteps=TRAIN_STEPS, reset_num_timesteps=False)
            if env.native_steps != TRAIN_STEPS or agent.num_timesteps != TRAIN_STEPS:
                raise ValueError("M10 expected native training interactions differ")
            if counter["calls"] != EXPECTED_UPDATES:
                raise ValueError("M10 measured PPO optimizer updates differ")
            measured = env.measured()
            if measured["incomplete_episode_resets"] != 0:
                raise ValueError("M10 PPO training prematurely reset native episodes")
            phase = f"posttrain-diagnostic-seed-{seed}"
            after_det = evaluate_suite(agent, seed, "deterministic")
            after_sampled = evaluate_suite(agent, seed, "sampled")
            checkpoint = numeric_state(agent, output.parent / f"ppo-seed{seed}.npz")
            row = {
                "model_seed": seed,
                "training_real_actions": env.native_steps,
                "optimizer_step_calls": counter["calls"],
                "training": measured,
                "policy_parameters": sum(p.numel() for p in agent.policy.parameters()),
                "numeric_policy": checkpoint,
                "random_control": random,
                "monitor_before_deterministic": before,
                "monitor_after_deterministic": after_det,
                "monitor_after_sampled": after_sampled,
            }
            records.append(row)
            env.close()
            in_flight_env = None
            in_flight_counter = None
            save_progress(output.parent / "progress.json", {
                "status": "partial_not_completed", "phase": phase,
                "source_commit": source, "completed_models": len(records),
                "training_real_actions": sum(
                    r["training_real_actions"] for r in records
                ),
                "optimizer_step_calls": sum(
                    r["optimizer_step_calls"] for r in records
                ),
                "records": records,
            })
    except BaseException as exc:
        save_progress(output.parent / "progress.json", {
            "status": "interrupted_not_completed",
            "phase": phase, "failure_type": type(exc).__name__,
            "source_commit": source, "completed_models": len(records),
            "in_flight_training_actions": (
                in_flight_env.native_steps if in_flight_env is not None else 0
            ),
            "in_flight_optimizer_calls": (
                in_flight_counter["calls"] if in_flight_counter is not None else 0
            ),
            "records": records,
            "elapsed_seconds": round(time.monotonic() - start, 3),
        })
        raise
    finally:
        if in_flight_env is not None:
            in_flight_env.close()

    decision = decision_summary(records)
    evaluation_actions = {
        mode: sum(row[key]["totals"]["native_actions"] for row in records)
        for mode, key in (
            ("random_control", "random_control"),
            ("pretrain_deterministic", "monitor_before_deterministic"),
            ("posttrain_deterministic", "monitor_after_deterministic"),
            ("posttrain_sampled", "monitor_after_sampled"),
        )
    }
    result = {
        "study": "BAGI M10-D3 train-only DoorKey physical opportunity diagnosis",
        "registered_study": "m10",
        "status": "completed_train_only_diagnostic_not_independent_evaluation",
        "source_commit": source,
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "versions": {
            package: version(package)
            for package in ("gymnasium", "minigrid", "stable-baselines3")
        },
        "training_real_actions": sum(r["training_real_actions"] for r in records),
        "optimizer_step_calls": sum(r["optimizer_step_calls"] for r in records),
        "train_only_monitor_actions": evaluation_actions,
        "train_only_monitor_resets": len(SEEDS) * len(MONITOR) * 4,
        "model_fits": len(records),
        "teacher_queries": 0,
        "decision": decision,
        "records": records,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "limitations": [
            "All new layouts are TRAIN MONITOR, not independent heldout tests.",
            "Random control does not match PPO optimizer or exploration history.",
            "One sparse DoorKey task cannot establish task-independent learning.",
            "The physical auditor uses simulator-private state ONLY after actions.",
            "Opportunity rates are descriptive and do not prove a causal bottleneck.",
            "M10 public one-hot/direction representation comparison has NOT run.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/m10_train_only_diagnostic.yaml")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/m10_train_only_diagnostic/summary.json"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("M10 fresh native training requires explicit --execute")
    result = run(args.config, args.output)
    print(json.dumps({
        "status": result["status"],
        "source_commit": result["source_commit"],
        "training_real_actions": result["training_real_actions"],
        "optimizer_step_calls": result["optimizer_step_calls"],
        "train_only_monitor_actions": result["train_only_monitor_actions"],
        "decision": result["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
