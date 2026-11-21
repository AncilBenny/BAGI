"""M9: matched native DoorKey reactive PPO versus maintained recurrent PPO.

All four fresh policies train on only public partial observations, the public
mission and four actually executed actions. Recurrent evaluation carries and
resets the upstream LSTM state at true episode boundaries. Private key/door
state is used by the separate evaluation-only physical-effect auditor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
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
from bagi.experiments.m7_frozen_doorkey_sampling import evaluate_episode
from bagi.experiments.m8_doorkey_entropy import _aggregate

ARMS = ("reactive_history", "recurrent_history")
SEEDS = (151, 167)
TRAIN_STEPS = 131072
UPDATES_PER_MODEL = TRAIN_STEPS // ROLLOUT * EPOCHS * (ROLLOUT // BATCH)
MONITOR = tuple(range(117001, 117007))
HELDOUT = tuple(range(118001, 118013))
REPEATS = 3
LSTM_SIZE = 32


def validate_config(config: dict) -> dict:
    expected = {
        "registered_study": "m9",
        "task": DOORKEY,
        "arms": list(ARMS),
        "model_seeds": list(SEEDS),
        "train_steps_per_fit": TRAIN_STEPS,
        "rollout_steps": ROLLOUT,
        "batch_size": BATCH,
        "epochs_per_rollout": EPOCHS,
        "lstm_hidden_size": LSTM_SIZE,
        "lstm_layers": 1,
        "evaluation_horizon": EVAL_HORIZON,
        "monitor_seeds": list(MONITOR),
        "heldout_seeds": list(HELDOUT),
        "sampled_repeats": REPEATS,
        "output_dir": "runs/m9_recurrent_doorkey",
    }
    if config != expected:
        raise ValueError("Registered M9 recurrent DoorKey protocol changed")
    return config


def action_seed(arm: str, model_seed: int, layout: int, repeat: int) -> int:
    if arm not in ARMS or model_seed not in SEEDS or layout not in HELDOUT:
        raise ValueError("Unregistered M9 model/layout/RNG seed combination")
    if type(repeat) is not int or not 1 <= repeat <= REPEATS:
        raise ValueError("Unregistered M9 sampled repetition")
    return (
        70_000_000 + ARMS.index(arm) * 1_000_000
        + SEEDS.index(model_seed) * 100_000
        + HELDOUT.index(layout) * 100 + repeat
    )


def make_training_env(model_seed: int) -> NativeDoorTraining:
    if model_seed not in SEEDS:
        raise ValueError("Unregistered fresh M9 model seed")
    raw = gym.make(DOORKEY)
    timed = gym.wrappers.TimeLimit(raw, max_episode_steps=EVAL_HORIZON)
    public = PublicGoalActions(timed)
    return NativeDoorTraining(
        public, model_seed=model_seed, allowed_seeds=SEEDS,
        max_native_steps=TRAIN_STEPS,
    )


class StatefulPredict:
    """Adapt upstream RecurrentPPO.predict to M7's common physical evaluator.

    A NEW instance is created for every native episode. No privileged effect
    labels, hidden simulator state, reward, or info are passed into the policy.
    """

    def __init__(self, agent):
        self.agent = agent
        self.state = None
        self.episode_start = np.array([True], dtype=bool)

    def predict(self, observation, deterministic: bool):
        action, self.state = self.agent.predict(
            observation, state=self.state, episode_start=self.episode_start,
            deterministic=deterministic,
        )
        self.episode_start[:] = False
        return action, None


def scored_episode(agent, arm: str, layout: int, *, sampled_seed: int | None,
                   meter: dict, reference: bool) -> dict:
    if arm not in ARMS:
        raise ValueError("Unknown M9 policy arm")
    policy = StatefulPredict(agent) if arm == "recurrent_history" else agent
    return evaluate_episode(
        policy, layout, sampled_seed=sampled_seed, meter=meter, reference=reference
    )


def evaluate_monitor(agent, arm: str, meter: dict) -> dict:
    rows = [
        scored_episode(agent, arm, seed, sampled_seed=None, meter=meter, reference=True)
        for seed in MONITOR
    ]
    return {"episode_results": rows, **_aggregate(rows)}


def evaluate_independent(agent, arm: str, model_seed: int, meter: dict) -> dict:
    deterministic, sampled = [], []
    for layout in HELDOUT:
        deterministic.append(scored_episode(
            agent, arm, layout, sampled_seed=None, meter=meter, reference=False
        ))
        for repeat in range(1, REPEATS + 1):
            sampled.append(scored_episode(
                agent, arm, layout,
                sampled_seed=action_seed(arm, model_seed, layout, repeat),
                meter=meter, reference=False,
            ))
    return {
        "deterministic": deterministic,
        "sampled": sampled,
        "deterministic_totals": _aggregate(deterministic),
        "sampled_totals": _aggregate(sampled),
    }


def preheldout_integrity(records: list[dict]) -> dict:
    models = {(r["arm"], r["model_seed"]): r for r in records}
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if len(records) != len(expected) or set(models) != expected:
        raise ValueError("M9 missing or duplicated fresh models")
    checks = {}
    for seed in SEEDS:
        arm_records = []
        for arm in ARMS:
            row = models[(arm, seed)]
            t = row["training"]
            passes = (
                row["training_real_actions"] == TRAIN_STEPS
                and row["optimizer_step_calls"] == UPDATES_PER_MODEL
                and t["native_actions"] == TRAIN_STEPS
                and t["incomplete_episode_resets"] == 0
                and t["max_episode_native_actions"] == EVAL_HORIZON
                and sum(t["executed_action_histogram"]) == TRAIN_STEPS
                and row["policy_family"] == arm
                and row["policy_parameters"] > 0
                and row["numeric_policy"].get("allow_pickle") is False
                and bool(row["numeric_policy"].get("sha256"))
            )
            checks[f"{arm}:{seed}"] = passes
            arm_records.append(t)
        if arm_records[0]["first_reset_seed"] != arm_records[1]["first_reset_seed"]:
            raise ValueError("M9 first training reset layouts differ across policies")
        if arm_records[0]["incomplete_episode_resets"]:
            raise ValueError("M9 reactive control interrupted a native episode")
    return checks


def score(records: list[dict], checks: dict) -> dict:
    models = {(r["arm"], r["model_seed"]): r for r in records}
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if len(records) != len(expected) or set(models) != expected:
        raise ValueError("M9 score requires four independently trained policies")

    def physical(arm: str, seed: int, policy: str) -> int:
        return models[(arm, seed)]["independent"][policy + "_totals"][
            "verified_physical_goals"
        ]

    recurrent = sum(physical("recurrent_history", s, "sampled") for s in SEEDS)
    reactive = sum(physical("reactive_history", s, "sampled") for s in SEEDS)
    recurrent_det = sum(physical("recurrent_history", s, "deterministic") for s in SEEDS)
    reactive_det = sum(physical("reactive_history", s, "deterministic") for s in SEEDS)
    both_recurrent = all(physical("recurrent_history", s, "sampled") >= 1 for s in SEEDS)
    return {
        "recurrent_sampled_physical_goals_of_72": recurrent,
        "reactive_sampled_physical_goals_of_72": reactive,
        "recurrent_deterministic_physical_goals_of_24": recurrent_det,
        "reactive_deterministic_physical_goals_of_24": reactive_det,
        "sampled_physical_difference_of_72": recurrent - reactive,
        "both_recurrent_model_seeds_have_physical_goal": both_recurrent,
        "preheldout_training_checks_pass": all(checks.values()),
        "primary_gate_pass": (
            recurrent - reactive >= 4 and both_recurrent and all(checks.values())
        ),
        "limitations": (
            "Two model seeds and correlated action-sampling repeats per layout; "
            "LSTM and feed-forward model parameter counts differ."
        ),
    }


def run(config_path: Path, output: Path) -> dict:
    if output.exists() or (output.parent / "progress.json").exists():
        raise FileExistsError("Refusing to overwrite existing M9 results")
    raw_config = config_path.read_bytes()
    config = validate_config(yaml.safe_load(raw_config))
    if output.parent.as_posix() != config["output_dir"]:
        raise ValueError("M9 output path differs from preregistration")
    source = admitted_source_commit()
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    start = time.monotonic()
    records, agents = [], {}
    meter = {"reference_actions": 0, "reference_resets": 0,
             "independent_actions": 0, "independent_resets": 0}
    totals = {"training_real_actions": 0, "optimizer_step_calls": 0}
    stage = "preflight"
    in_flight_env = None
    in_flight_optimizer = None
    in_flight_before = None
    in_flight_after = None
    try:
        sanity = make_training_env(SEEDS[0])
        try:
            if sanity.action_space.n != 7 or sanity.observation_space.shape != (243,):
                raise ValueError("M9 public observation/action interface changed")
        finally:
            sanity.close()
        for arm in ARMS:
            for seed in SEEDS:
                stage = f"train-{arm}-{seed}"
                env = make_training_env(seed)
                in_flight_env = env
                recurrent = arm == "recurrent_history"
                kwargs = {"net_arch": {"pi": [32], "vf": [32]}}
                if recurrent:
                    kwargs.update({"lstm_hidden_size": LSTM_SIZE, "n_lstm_layers": 1})
                agent = (RecurrentPPO if recurrent else PPO)(
                    "MlpLstmPolicy" if recurrent else "MlpPolicy",
                    env, seed=seed, device="cpu", n_steps=ROLLOUT,
                    batch_size=BATCH, n_epochs=EPOCHS, ent_coef=0.0,
                    policy_kwargs=kwargs, verbose=0,
                )
                optimizer = count_optimizer_steps(agent)
                in_flight_optimizer = optimizer
                before = evaluate_monitor(agent, arm, meter)
                in_flight_before = before
                # This ONE learn call preserves the native episode across
                # PPO update blocks in both arms.
                agent.learn(total_timesteps=TRAIN_STEPS, reset_num_timesteps=False)
                if env.native_steps != TRAIN_STEPS or agent.num_timesteps != TRAIN_STEPS:
                    raise ValueError("M9 native training interactions mismatched")
                if optimizer["calls"] != UPDATES_PER_MODEL:
                    raise ValueError("M9 actual optimizer-step count mismatched")
                measured = env.measured()
                stage = f"monitor-{arm}-{seed}"
                after = evaluate_monitor(agent, arm, meter)
                in_flight_after = after
                stage = f"export-{arm}-{seed}"
                checkpoint = numeric_state(agent, output.parent / f"{arm}-seed{seed}.npz")
                row = {
                    "arm": arm, "policy_family": arm, "model_seed": seed,
                    "training": measured,
                    "training_real_actions": env.native_steps,
                    "optimizer_step_calls": optimizer["calls"],
                    "policy_parameters": sum(p.numel() for p in agent.policy.parameters()),
                    "monitor_before": before,
                    "monitor_after": after,
                    "numeric_policy": checkpoint,
                }
                records.append(row)
                agents[(arm, seed)] = agent
                env.close()
                in_flight_env = None
                in_flight_optimizer = None
                in_flight_before = None
                in_flight_after = None
                totals["training_real_actions"] += row["training_real_actions"]
                totals["optimizer_step_calls"] += row["optimizer_step_calls"]
                save_progress(output.parent / "progress.json", {
                    "status": "partial_not_a_completed_study",
                    "phase": stage, "source_commit": source,
                    "completed_models": len(records),
                    "evaluation_monitor_actions": meter["reference_actions"],
                    "evaluation_monitor_resets": meter["reference_resets"],
                    **totals, "records": records,
                })

        stage = "preheldout_training_integrity"
        checks = preheldout_integrity(records)
        if not all(checks.values()):
            raise ValueError("M9 native training/continuity/checkpoint integrity failed")
        if totals["training_real_actions"] != len(ARMS) * len(SEEDS) * TRAIN_STEPS:
            raise ValueError("M9 total native training count mismatch")
        if totals["optimizer_step_calls"] != len(ARMS) * len(SEEDS) * UPDATES_PER_MODEL:
            raise ValueError("M9 total optimizer count mismatch")
        stage = "independent_heldout"
        for row in records:
            arm, seed = row["arm"], row["model_seed"]
            row["independent"] = evaluate_independent(agents[(arm, seed)], arm, seed, meter)
            save_progress(output.parent / "progress.json", {
                "status": "partial_not_a_completed_study",
                "phase": stage, "source_commit": source,
                "completed_models": len(records),
                "completed_independent_models": sum(
                    "independent" in item for item in records
                ),
                "evaluation_monitor_actions": meter["reference_actions"],
                "evaluation_monitor_resets": meter["reference_resets"],
                "evaluation_independent_actions": meter["independent_actions"],
                "evaluation_independent_resets": meter["independent_resets"],
                **totals, "records": records,
            })
    except BaseException as error:
        save_progress(output.parent / "progress.json", {
            "status": "interrupted_not_a_completed_study",
            "phase": stage,
            "failure_type": type(error).__name__,
            "source_commit": source,
            "completed_models": len(records),
            "in_flight_native_actions": (
                in_flight_env.native_steps if in_flight_env is not None else 0
            ),
            "in_flight_optimizer_updates": (
                in_flight_optimizer["calls"] if in_flight_optimizer is not None else 0
            ),
            "in_flight_completed_monitor_before_actions": (
                in_flight_before["evaluation_real_actions"]
                if in_flight_before is not None else 0
            ),
            "in_flight_completed_monitor_after_actions": (
                in_flight_after["evaluation_real_actions"]
                if in_flight_after is not None else 0
            ),
            "evaluation_monitor_actions": meter["reference_actions"],
            "evaluation_monitor_resets": meter["reference_resets"],
            "evaluation_independent_actions": meter["independent_actions"],
            "evaluation_independent_resets": meter["independent_resets"],
            **totals, "records": records,
            "elapsed_seconds": round(time.monotonic() - start, 3),
        })
        raise
    finally:
        if in_flight_env is not None:
            in_flight_env.close()

    result = {
        "study": "BAGI-M9 matched public-history PPO vs maintained RecurrentPPO",
        "registered_study": "m9",
        "status": "completed_independent_native_recurrent_ablation",
        "source_commit": source,
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "versions": {pkg: version(pkg) for pkg in (
            "gymnasium", "minigrid", "stable-baselines3", "sb3-contrib"
        )},
        "training_real_actions": totals["training_real_actions"],
        "optimizer_step_calls": totals["optimizer_step_calls"],
        "evaluation_monitor_actions": meter["reference_actions"],
        "evaluation_monitor_resets": meter["reference_resets"],
        "evaluation_independent_actions": meter["independent_actions"],
        "evaluation_independent_resets": meter["independent_resets"],
        "teacher_queries": 0,
        "training_integrity_checks": checks,
        "pilot_gate": score(records, checks),
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "records": records,
        "limitations": [
            "One sparse DoorKey environment only; not task-independent continual learning.",
            "Two model seeds and three correlated sampled repetitions per heldout layout.",
            "PPO and RecurrentPPO differ in parameter count and gradient paths.",
            "Key and door physical effects are evaluator-only, never acting-policy inputs.",
            "Training reward-only goals do not establish independent manipulation.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/m9_recurrent_doorkey.yaml"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/m9_recurrent_doorkey/summary.json"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("M9 new native PPO training requires explicit --execute")
    result = run(args.config, args.output)
    print(json.dumps({
        "status": result["status"],
        "source_commit": result["source_commit"],
        "training_real_actions": result["training_real_actions"],
        "optimizer_step_calls": result["optimizer_step_calls"],
        "evaluation_monitor_actions": result["evaluation_monitor_actions"],
        "evaluation_independent_actions": result["evaluation_independent_actions"],
        "pilot_gate": result["pilot_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
