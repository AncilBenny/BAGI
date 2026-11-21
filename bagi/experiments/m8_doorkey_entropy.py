"""M8: fresh on-policy DoorKey entropy-coefficient ablation.

Both arms reuse native PPO and only change built-in ent_coef. The sole policy
inputs are public MiniGrid observations and actually executed action history.
Heldout evaluation is withheld until ALL four fits pass native continuity,
optimizer-update accounting and numeric checkpoint-export integrity checks.
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
from bagi.experiments.m6_doorkey_continuity import (
    BATCH,
    EPOCHS,
    ROLLOUT,
    NativeDoorTraining,
)
from bagi.experiments.m7_frozen_doorkey_sampling import evaluate_episode

ARMS = ("standard_entropy", "bonus_entropy")
COEFFICIENTS = {"standard_entropy": 0.0, "bonus_entropy": 0.02}
SEEDS = (127, 139)
TRAIN_STEPS = 131072
UPDATES_PER_MODEL = TRAIN_STEPS // ROLLOUT * EPOCHS * (ROLLOUT // BATCH)
MONITOR = tuple(range(114001, 114007))
HELDOUT = tuple(range(112001, 112013))
REPEATS = 3


def validate_config(config: dict) -> dict:
    expected = {
        "registered_study": "m8",
        "task": DOORKEY,
        "arms": list(ARMS),
        "entropy_coefficients": COEFFICIENTS,
        "model_seeds": list(SEEDS),
        "train_steps_per_fit": TRAIN_STEPS,
        "rollout_steps": ROLLOUT,
        "batch_size": BATCH,
        "epochs_per_rollout": EPOCHS,
        "evaluation_horizon": EVAL_HORIZON,
        "monitor_seeds": list(MONITOR),
        "heldout_seeds": list(HELDOUT),
        "sampled_repeats": REPEATS,
        "output_dir": "runs/m8_doorkey_entropy",
    }
    if config != expected:
        raise ValueError("Registered M8 entropy ablation contract changed")
    return config


def action_seed(arm: str, model_seed: int, layout: int, repeat: int) -> int:
    if arm not in ARMS or model_seed not in SEEDS or layout not in HELDOUT:
        raise ValueError("Unregistered M8 action seed combination")
    if type(repeat) is not int or not 1 <= repeat <= REPEATS:
        raise ValueError("Unregistered M8 stochastic repeat")
    return (
        60_000_000 + ARMS.index(arm) * 1_000_000
        + SEEDS.index(model_seed) * 100_000
        + HELDOUT.index(layout) * 100 + repeat
    )


def make_training_env(model_seed: int) -> NativeDoorTraining:
    if model_seed not in SEEDS:
        raise ValueError("Unregistered fresh M8 model seed")
    raw = gym.make(DOORKEY)
    timed = gym.wrappers.TimeLimit(raw, max_episode_steps=EVAL_HORIZON)
    public = PublicGoalActions(timed)
    return NativeDoorTraining(
        public, model_seed=model_seed, allowed_seeds=SEEDS,
        max_native_steps=TRAIN_STEPS,
    )


def _aggregate(episodes: list[dict]) -> dict:
    return {
        "episodes": len(episodes),
        "rewarded_goals": sum(e["goal_success"] for e in episodes),
        "verified_physical_goals": sum(e["physical_key_door_goal"] for e in episodes),
        "verified_key_pickup_events": sum(e["verified_key_pickups"] for e in episodes),
        "verified_door_opening_events": sum(e["verified_door_openings"] for e in episodes),
        "episodes_with_key_pickup": sum(e["verified_key_pickups"] > 0 for e in episodes),
        "episodes_with_door_opening": sum(e["verified_door_openings"] > 0 for e in episodes),
        "distinct_physical_layouts": len({
            e["seed"] for e in episodes if e["physical_key_door_goal"]
        }),
        "evaluation_real_actions": sum(e["evaluation_real_actions"] for e in episodes),
        "evaluation_real_resets": sum(e["evaluation_real_resets"] for e in episodes),
    }


def evaluate_monitor(agent, meter: dict) -> dict:
    episodes = [
        evaluate_episode(
            agent, layout, sampled_seed=None, meter=meter, reference=True
        ) for layout in MONITOR
    ]
    return {"episode_results": episodes, **_aggregate(episodes)}


def evaluate_independent(agent, arm: str, model_seed: int, meter: dict) -> dict:
    deterministic = []
    sampled = []
    for layout in HELDOUT:
        deterministic.append(evaluate_episode(
            agent, layout, sampled_seed=None, meter=meter, reference=False
        ))
        for repeat in range(1, REPEATS + 1):
            sampled.append(evaluate_episode(
                agent, layout,
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
    models = {(row["arm"], row["model_seed"]): row for row in records}
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if len(records) != len(expected) or set(models) != expected:
        raise ValueError("M8 missing or duplicated fresh fits")
    checks = {}
    for seed in SEEDS:
        arm_training = []
        for arm in ARMS:
            row = models[(arm, seed)]
            training = row["training"]
            passes = (
                row["training_real_actions"] == TRAIN_STEPS
                and row["optimizer_step_calls"] == UPDATES_PER_MODEL
                and training["native_actions"] == TRAIN_STEPS
                and training["incomplete_episode_resets"] == 0
                and training["max_episode_native_actions"] == EVAL_HORIZON
                and sum(training["executed_action_histogram"]) == TRAIN_STEPS
                and row["numeric_policy"].get("allow_pickle") is False
                and bool(row["numeric_policy"].get("sha256"))
                and row["entropy_coefficient"] == COEFFICIENTS[arm]
                and row["policy_parameters"] == 15880
            )
            checks[f"{arm}:{seed}"] = passes
            arm_training.append(training)
        if arm_training[0]["first_reset_seed"] != arm_training[1]["first_reset_seed"]:
            raise ValueError("M8 arms used different first training layout seeds")
        if arm_training[0]["incomplete_episode_resets"] != 0:
            raise ValueError("M8 control interrupted native episodes")
    return checks


def score(records: list[dict], checks: dict) -> dict:
    models = {(row["arm"], row["model_seed"]): row for row in records}
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if len(records) != len(expected) or set(models) != expected:
        raise ValueError("M8 score requires all four independently fitted models")
    def physical(arm: str, seed: int, policy: str) -> int:
        return models[(arm, seed)]["independent"][policy + "_totals"][
            "verified_physical_goals"
        ]
    bonus = sum(physical("bonus_entropy", s, "sampled") for s in SEEDS)
    control = sum(physical("standard_entropy", s, "sampled") for s in SEEDS)
    bonus_det = sum(physical("bonus_entropy", s, "deterministic") for s in SEEDS)
    control_det = sum(physical("standard_entropy", s, "deterministic") for s in SEEDS)
    both_bonus_seeds = all(physical("bonus_entropy", s, "sampled") >= 1 for s in SEEDS)
    return {
        "bonus_sampled_physical_goals_of_72": bonus,
        "standard_sampled_physical_goals_of_72": control,
        "bonus_deterministic_physical_goals_of_24": bonus_det,
        "standard_deterministic_physical_goals_of_24": control_det,
        "sampled_physical_difference_of_72": bonus - control,
        "both_bonus_model_seeds_have_physical_goal": both_bonus_seeds,
        "preheldout_training_checks_pass": all(checks.values()),
        "primary_gate_pass": (
            bonus - control >= 4 and both_bonus_seeds and all(checks.values())
        ),
        "limitations": "Two distinct initialization seeds; three sampled repeats per fixed layout.",
    }


def run(config_path: Path, output: Path) -> dict:
    if output.exists() or (output.parent / "progress.json").exists():
        raise FileExistsError("Refusing to overwrite existing M8 results")
    raw_config = config_path.read_bytes()
    config = validate_config(yaml.safe_load(raw_config))
    if output.parent.as_posix() != config["output_dir"]:
        raise ValueError("M8 output path differs from registered configuration")
    source = admitted_source_commit()
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    started = time.monotonic()
    counts = {
        "training_real_actions": 0,
        "optimizer_step_calls": 0,
        "evaluation_reference_actions": 0,
        "evaluation_reference_resets": 0,
        "evaluation_independent_actions": 0,
        "evaluation_independent_resets": 0,
    }
    meter = {"reference_actions": 0, "reference_resets": 0,
             "independent_actions": 0, "independent_resets": 0}
    records, models = [], {}
    phase = "preflight"
    in_flight_env = None
    in_flight_updates = None
    in_flight_monitor_before = None
    in_flight_monitor_after = None
    try:
        sanity = make_training_env(SEEDS[0])
        try:
            if sanity.action_space.n != 7 or sanity.observation_space.shape != (243,):
                raise ValueError("M8 public action/observation contract mismatch")
        finally:
            sanity.close()
        for arm in ARMS:
            for seed in SEEDS:
                phase = f"train-{arm}-{seed}"
                env = make_training_env(seed)
                in_flight_env = env
                agent = PPO(
                    "MlpPolicy", env, seed=seed, device="cpu",
                    n_steps=ROLLOUT, batch_size=BATCH, n_epochs=EPOCHS,
                    ent_coef=COEFFICIENTS[arm],
                    policy_kwargs={"net_arch": {"pi": [32], "vf": [32]}},
                    verbose=0,
                )
                optimizer = count_optimizer_steps(agent)
                in_flight_updates = optimizer
                before = evaluate_monitor(agent, meter)
                in_flight_monitor_before = before
                # One continuous native episode stream; no forced reset at
                # PPO update boundaries and identical learn call for both arms.
                agent.learn(total_timesteps=TRAIN_STEPS, reset_num_timesteps=False)
                if env.native_steps != TRAIN_STEPS or agent.num_timesteps != TRAIN_STEPS:
                    raise ValueError("M8 actual native action count does not match")
                if optimizer["calls"] != UPDATES_PER_MODEL:
                    raise ValueError("M8 measured optimizer calls differ from PPO schedule")
                training = env.measured()
                phase = f"monitor-{arm}-{seed}"
                after = evaluate_monitor(agent, meter)
                in_flight_monitor_after = after
                phase = f"export-{arm}-{seed}"
                checkpoint = numeric_state(agent, output.parent / f"{arm}-seed{seed}.npz")
                row = {
                    "arm": arm, "model_seed": seed,
                    "entropy_coefficient": COEFFICIENTS[arm],
                    "training": training,
                    "training_real_actions": env.native_steps,
                    "optimizer_step_calls": optimizer["calls"],
                    "policy_parameters": sum(p.numel() for p in agent.policy.parameters()),
                    "monitor_before": before,
                    "monitor_after": after,
                    "numeric_policy": checkpoint,
                }
                records.append(row)
                models[(arm, seed)] = agent
                env.close()
                in_flight_env = None
                in_flight_updates = None
                in_flight_monitor_before = None
                in_flight_monitor_after = None
                counts["training_real_actions"] += row["training_real_actions"]
                counts["optimizer_step_calls"] += row["optimizer_step_calls"]
                counts["evaluation_reference_actions"] = meter["reference_actions"]
                counts["evaluation_reference_resets"] = meter["reference_resets"]
                save_progress(output.parent / "progress.json", {
                    "status": "partial_not_a_completed_study",
                    "phase": phase, "source_commit": source,
                    "completed_models": len(records),
                    **counts, "records": records,
                })

        phase = "preheldout_training_integrity"
        checks = preheldout_integrity(records)
        if not all(checks.values()):
            raise ValueError("M8 native training/continuity/entropy checks failed")
        if counts["training_real_actions"] != len(ARMS) * len(SEEDS) * TRAIN_STEPS:
            raise ValueError("M8 total native training interactions differ")
        if counts["optimizer_step_calls"] != len(ARMS) * len(SEEDS) * UPDATES_PER_MODEL:
            raise ValueError("M8 total intercepted optimizer updates differ")
        phase = "independent_heldout"
        for row in records:
            arm, seed = row["arm"], row["model_seed"]
            row["independent"] = evaluate_independent(
                models[(arm, seed)], arm, seed, meter
            )
            counts["evaluation_independent_actions"] = meter["independent_actions"]
            counts["evaluation_independent_resets"] = meter["independent_resets"]
            save_progress(output.parent / "progress.json", {
                "status": "partial_not_a_completed_study",
                "phase": phase, "source_commit": source,
                "completed_models": len(records),
                "completed_independent_models": sum(
                    "independent" in item for item in records
                ),
                **counts, "records": records,
            })
    except BaseException as error:
        save_progress(output.parent / "progress.json", {
            "status": "interrupted_not_a_completed_study",
            "phase": phase, "failure_type": type(error).__name__,
            "source_commit": source,
            "completed_models": len(records),
            "in_flight_training_actions": (
                in_flight_env.native_steps if in_flight_env is not None else 0
            ),
            "in_flight_optimizer_updates": (
                in_flight_updates["calls"] if in_flight_updates is not None else 0
            ),
            "in_flight_monitor_before_actions": (
                in_flight_monitor_before["evaluation_real_actions"]
                if in_flight_monitor_before is not None else 0
            ),
            "in_flight_monitor_after_actions": (
                in_flight_monitor_after["evaluation_real_actions"]
                if in_flight_monitor_after is not None else 0
            ),
            **counts,
            "evaluation_reference_actions": meter["reference_actions"],
            "evaluation_reference_resets": meter["reference_resets"],
            "evaluation_independent_actions": meter["independent_actions"],
            "evaluation_independent_resets": meter["independent_resets"],
            "records": records,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        raise
    finally:
        if in_flight_env is not None:
            in_flight_env.close()

    result = {
        "study": "BAGI-M8 matched PPO entropy-coefficient sparse DoorKey",
        "registered_study": "m8",
        "status": "completed_independent_native_entropy_ablation",
        "source_commit": source,
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "versions": {p: version(p) for p in (
            "gymnasium", "minigrid", "stable-baselines3"
        )},
        "training_real_actions": counts["training_real_actions"],
        "optimizer_step_calls": counts["optimizer_step_calls"],
        "evaluation_monitor_actions": meter["reference_actions"],
        "evaluation_monitor_resets": meter["reference_resets"],
        "evaluation_independent_actions": meter["independent_actions"],
        "evaluation_independent_resets": meter["independent_resets"],
        "teacher_queries": 0,
        "training_integrity_checks": checks,
        "pilot_gate": score(records, checks),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "records": records,
        "limitations": [
            "Only a single sparse MiniGrid task; no task-independent continual learning.",
            "A goal requires native terminal positive reward AND actual pickup and opening.",
            "Private state used only by evaluation effect auditor, never by PPO.",
            "Repeated sampled actions on a fixed layout are not independent layouts.",
            "Only ent_coef changes within M8 arms; M6 is historical, not a fresh M8 control.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/m8_doorkey_entropy.yaml"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/m8_doorkey_entropy/summary.json"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("M8 real native training requires explicit --execute")
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
