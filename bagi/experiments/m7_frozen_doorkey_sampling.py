"""M7: independent frozen-policy DoorKey stochastic-vs-deterministic diagnostic.

No new training; all four M6 numeric-only policies must reproduce the archived
trained-monitor reference episodes before any fresh M7 layout may be evaluated.
The acting policy receives public observations and truly executed action history;
private physical effects remain inside the separate evaluator only.
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
import yaml

from bagi.experiments.m1_goal_learning import admitted_source_commit, save_progress
from bagi.experiments.m4_frozen_policy_eval import restore_numeric_policy
from bagi.experiments.m5_continual_retention import DOORKEY, EVAL_HORIZON, make_task
from bagi.experiments.m6_doorkey_continuity import ARMS, BATCH, EPOCHS, ROLLOUT, SEEDS

M6_SOURCE = "3842c2f7ac4b78a592faac8730e4466e145dbd7b"
M6_CONFIG_SHA256 = "36b302a4a214fa6b547c646e9b8f9ec2733625dce7c0f35e2a57ed69b8140088"
M6_ARTIFACT_ID = 13838577607
M6_ARTIFACT_SHA256 = "10f425ac3fc2420a4ba970180c3947eeffab28414cad9353ebdc2558042a661b"
REFERENCE = tuple(range(108001, 108007))
HELDOUT = tuple(range(109001, 109013))
REPEATS = 5


def validate_config(config: dict) -> dict:
    expected = {
        "registered_study": "m7",
        "task": DOORKEY,
        "arms": list(ARMS),
        "model_seeds": list(SEEDS),
        "m6_source_commit": M6_SOURCE,
        "m6_config_sha256": M6_CONFIG_SHA256,
        "m6_artifact_id": M6_ARTIFACT_ID,
        "m6_artifact_sha256": M6_ARTIFACT_SHA256,
        "reference_seeds": list(REFERENCE),
        "heldout_seeds": list(HELDOUT),
        "sampled_repeats": REPEATS,
        "evaluation_horizon": EVAL_HORIZON,
        "output_dir": "runs/m7_frozen_doorkey_sampling",
    }
    if config != expected:
        raise ValueError("Registered M7 policy-selection protocol changed")
    return config


def action_seed(arm: str, model_seed: int, layout: int, repeat: int) -> int:
    """Independent prespecified episode RNG, never chosen by outcome."""
    if arm not in ARMS or model_seed not in SEEDS or layout not in HELDOUT:
        raise ValueError("Unknown frozen M7 arm, model seed or layout")
    if type(repeat) is not int or not 1 <= repeat <= REPEATS:
        raise ValueError("Unknown M7 stochastic repetition")
    return (
        50_000_000 + ARMS.index(arm) * 1_000_000
        + SEEDS.index(model_seed) * 100_000
        + HELDOUT.index(layout) * 100 + repeat
    )


def read_m6_manifest(root: Path) -> tuple[dict, dict]:
    path = root / "artifacts/research-loop/m6_doorkey_continuity-summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (
        summary.get("source_commit") != M6_SOURCE
        or summary.get("config_sha256") != M6_CONFIG_SHA256
        or summary.get("registered_study") != "m6"
        or summary.get("status") != "completed_independent_episode_continuity_ablation"
        or summary.get("training_real_actions") != 262144
        or summary.get("optimizer_step_calls") != 8192
        or summary.get("teacher_queries") != 0
        or summary.get("pilot_gate", {}).get("continuity_manipulation_valid") is not True
    ):
        raise ValueError("M6 frozen source/config/completion evidence mismatch")
    for pkg in ("gymnasium", "minigrid", "stable-baselines3"):
        if summary["versions"][pkg] != version(pkg):
            raise ValueError("Frozen M6 checkpoint version mismatch: " + pkg)
    records = summary["records"]
    models = {(r["arm"], r["model_seed"]): r for r in records}
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if len(records) != len(expected) or set(models) != expected:
        raise ValueError("Missing or duplicated M6 frozen models")
    for r in records:
        if r["training_real_actions"] != 65536 or r["optimizer_step_calls"] != 2048:
            raise ValueError("M6 individual training provenance mismatch")
        if [x["seed"] for x in r["monitor_after"]["episode_results"]] != list(REFERENCE):
            raise ValueError("M6 reference seed provenance mismatch")
    return summary, models


def evaluate_episode(agent, layout: int, *, sampled_seed: int | None,
                     meter: dict, reference: bool) -> dict:
    """All actions are native and physically executed, no privileged policy input."""
    env = make_task(DOORKEY, audit=True)
    bucket = "reference" if reference else "independent"
    try:
        env.action_space.seed(layout)
        observation, _ = env.reset(seed=layout)
        meter[bucket + "_resets"] += 1
        if sampled_seed is not None:
            torch.manual_seed(sampled_seed)
            np.random.seed(sampled_seed)
        terminated = truncated = False
        reward_sum = 0.0
        histogram = [0] * int(env.action_space.n)
        for _ in range(EVAL_HORIZON):
            action, _ = agent.predict(observation, deterministic=sampled_seed is None)
            executed = int(np.asarray(action).item())
            if not 0 <= executed < len(histogram):
                raise ValueError("Non-native DoorKey action")
            observation, reward, terminated, truncated, _ = env.step(executed)
            histogram[executed] += 1
            reward_sum += float(reward)
            meter[bucket + "_actions"] += 1
            if terminated or truncated:
                break
        steps = sum(histogram)
        if not (terminated or truncated) or env.native_steps != steps or env.native_resets != 1:
            raise ValueError("M7 evaluator episode did not close or native counters differ")
        effects = env.env
        pickups = int(effects.effective_key_pickups)
        openings = int(effects.effective_door_openings)
        goal = int(bool(terminated) and reward_sum > 0)
        return {
            "seed": layout,
            "action_rng_seed": sampled_seed,
            "reward": reward_sum,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "goal_success": goal,
            "verified_key_pickups": pickups,
            "verified_door_openings": openings,
            "physical_key_door_goal": int(bool(goal and pickups and openings)),
            "evaluation_real_actions": steps,
            "evaluation_real_resets": 1,
            "executed_action_histogram": histogram,
        }
    finally:
        env.close()


def verify_reference(agent, row: dict, meter: dict) -> int:
    reference = row["monitor_after"]["episode_results"]
    if len(reference) != len(REFERENCE):
        raise ValueError("M6 trained-monitor reference count mismatch")
    fields = (
        "seed", "terminated", "truncated", "goal_success", "verified_key_pickups",
        "verified_door_openings", "physical_key_door_goal", "evaluation_real_actions",
        "evaluation_real_resets", "executed_action_histogram",
    )
    for original in reference:
        observed = evaluate_episode(
            agent, original["seed"], sampled_seed=None, meter=meter, reference=True
        )
        if any(observed[f] != original[f] for f in fields):
            raise ValueError("M6 reference replay mismatch in physical/native episode fields")
        if abs(observed["reward"] - original["reward"]) > 1e-8:
            raise ValueError("M6 trained-monitor reference reward differs")
    return len(reference)


def _aggregate(episodes: list[dict]) -> dict:
    return {
        "episodes": len(episodes),
        "rewarded_goals": sum(r["goal_success"] for r in episodes),
        "verified_physical_goals": sum(r["physical_key_door_goal"] for r in episodes),
        "verified_key_pickups": sum(r["verified_key_pickups"] for r in episodes),
        "verified_door_openings": sum(r["verified_door_openings"] for r in episodes),
        "distinct_physical_layouts": len({
            r["seed"] for r in episodes if r["physical_key_door_goal"]
        }),
        "evaluation_real_actions": sum(r["evaluation_real_actions"] for r in episodes),
        "evaluation_real_resets": sum(r["evaluation_real_resets"] for r in episodes),
    }


def evaluate_split(agent, arm: str, model_seed: int, meter: dict) -> dict:
    deterministic, sampled = [], []
    for layout in HELDOUT:
        deterministic.append(evaluate_episode(
            agent, layout, sampled_seed=None, meter=meter, reference=False
        ))
        for repeat in range(1, REPEATS + 1):
            sampled.append(evaluate_episode(
                agent, layout, sampled_seed=action_seed(arm, model_seed, layout, repeat),
                meter=meter, reference=False
            ))
    return {
        "deterministic": deterministic, "sampled": sampled,
        "deterministic_totals": _aggregate(deterministic),
        "sampled_totals": _aggregate(sampled),
    }


def score(records: list[dict]) -> dict:
    models = {(r["arm"], r["model_seed"]): r for r in records}
    if len(records) != len(ARMS) * len(SEEDS) or set(models) != {
        (arm, seed) for arm in ARMS for seed in SEEDS
    }:
        raise ValueError("Incomplete M7 scored models")
    cont_sample = sum(
        models[("continue_episode", seed)]["independent"]["sampled_totals"][
            "verified_physical_goals"
        ] for seed in SEEDS
    )
    cont_det = sum(
        models[("continue_episode", seed)]["independent"]["deterministic_totals"][
            "verified_physical_goals"
        ] for seed in SEEDS
    )
    reset_sample = sum(
        models[("reset128", seed)]["independent"]["sampled_totals"][
            "verified_physical_goals"
        ] for seed in SEEDS
    )
    sampled_rate = cont_sample / (len(SEEDS) * len(HELDOUT) * REPEATS)
    deterministic_rate = cont_det / (len(SEEDS) * len(HELDOUT))
    both_seeds = all(
        models[("continue_episode", seed)]["independent"]["sampled_totals"][
            "verified_physical_goals"
        ] >= 1 for seed in SEEDS
    )
    return {
        "continue_sampled_physical_goals_of_120": cont_sample,
        "continue_deterministic_physical_goals_of_24": cont_det,
        "reset128_sampled_physical_goals_of_120": reset_sample,
        "continue_sampled_physical_rate": sampled_rate,
        "continue_deterministic_physical_rate": deterministic_rate,
        "sampled_minus_deterministic_physical_rate": sampled_rate - deterministic_rate,
        "both_continuation_model_seeds_have_sampled_physical_goal": both_seeds,
        "primary_gate_pass": (
            sampled_rate >= 0.10
            and sampled_rate - deterministic_rate >= 0.10
            and both_seeds
        ),
        "limitations": "Two model seeds; five repeats per layout share the same layouts.",
    }


def run(config_path: Path, artifact_root: Path, output: Path) -> dict:
    if output.exists() or (output.parent / "progress.json").exists():
        raise FileExistsError("Refusing to overwrite M7 outputs")
    source = admitted_source_commit()
    raw = config_path.read_bytes()
    config = validate_config(yaml.safe_load(raw))
    if output.parent.as_posix() != config["output_dir"]:
        raise ValueError("M7 output path differs from preregistration")
    m6, models = read_m6_manifest(artifact_root)
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    meter = {"reference_actions": 0, "reference_resets": 0,
             "independent_actions": 0, "independent_resets": 0}
    records = []
    stage = "initialization"
    started = time.monotonic()
    try:
        verified = {}
        # Critical: all four numeric checkpoints must match archived M6
        # deterministic reference episodes before ANY M7 independent episode.
        for arm in ARMS:
            for seed in SEEDS:
                stage = f"restore-{arm}-{seed}"
                env = make_task(DOORKEY)
                try:
                    agent = PPO(
                        "MlpPolicy", env, seed=seed, device="cpu",
                        n_steps=ROLLOUT, batch_size=BATCH, n_epochs=EPOCHS,
                        policy_kwargs={"net_arch": {"pi": [32], "vf": [32]}},
                        verbose=0,
                    )
                    row = models[(arm, seed)]
                    file_name = f"{arm}-seed{seed}.npz"
                    if row["numeric_policy"]["file"] != file_name:
                        raise ValueError("M6 numeric checkpoint filename mismatch")
                    path = artifact_root / "runs/m6_doorkey_continuity" / file_name
                    sha = restore_numeric_policy(agent, path, row["numeric_policy"])
                    stage = f"replay-{arm}-{seed}"
                    n = verify_reference(agent, row, meter)
                    verified[(arm, seed)] = (agent, sha, n)
                finally:
                    env.close()
        if len(verified) != len(ARMS) * len(SEEDS):
            raise ValueError("M7 missing a reference-verified frozen model")
        for arm in ARMS:
            for seed in SEEDS:
                stage = f"independent-{arm}-{seed}"
                agent, sha, n = verified[(arm, seed)]
                independent = evaluate_split(agent, arm, seed, meter)
                records.append({
                    "arm": arm,
                    "model_seed": seed,
                    "m6_numeric_checkpoint_sha256": sha,
                    "m6_reference_episodes_replayed": n,
                    "independent": independent,
                })
                save_progress(output.parent / "progress.json", {
                    "status": "partial_not_a_completed_study",
                    "phase": stage, "source_commit": source,
                    "training_real_actions": 0, "optimizer_step_calls": 0,
                    "completed_models": len(records), **meter, "records": records,
                })
    except BaseException as error:
        save_progress(output.parent / "progress.json", {
            "status": "interrupted_not_a_completed_study",
            "phase": stage, "source_commit": source,
            "failure_type": type(error).__name__,
            "training_real_actions": 0, "optimizer_step_calls": 0,
            "completed_models": len(records), **meter, "records": records,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        raise
    pilot = score(records)
    result = {
        "study": "BAGI-M7 frozen M6 DoorKey sampled-vs-deterministic",
        "registered_study": "m7",
        "status": "completed_independent_frozen_policy_sampling_diagnostic",
        "source_commit": source,
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "m6_source_commit": M6_SOURCE,
        "m6_config_sha256": M6_CONFIG_SHA256,
        "m6_artifact_id": M6_ARTIFACT_ID,
        "m6_artifact_sha256": M6_ARTIFACT_SHA256,
        "m6_training_real_actions_reused_not_new": m6["training_real_actions"],
        "m6_optimizer_step_calls_reused_not_new": m6["optimizer_step_calls"],
        "versions": m6["versions"],
        "training_real_actions": 0,
        "optimizer_step_calls": 0,
        "teacher_queries": 0,
        "reference_replays_excluded_from_m7_denominators": True,
        "evaluation_reference_actions": meter["reference_actions"],
        "evaluation_reference_resets": meter["reference_resets"],
        "evaluation_independent_actions": meter["independent_actions"],
        "evaluation_independent_resets": meter["independent_resets"],
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "pilot_gate": pilot,
        "records": records,
        "limitations": [
            "Frozen M6 PPO policies, no new training fits or optimizer steps.",
            "Reference replay is checkpoint verification, not new independent layouts.",
            "Sampled policy replicates on a layout are correlated.",
            "Rewarded DoorKey goal and evaluator-only physical key/door events remain distinct.",
            "No independent cross-task continual learning or physical robotics claim.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/m7_frozen_doorkey_sampling.yaml"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/m7_frozen_doorkey_sampling/summary.json"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("M7 frozen policy native evaluation requires explicit --execute")
    result = run(args.config, args.artifact_root, args.output)
    print(json.dumps({
        "status": result["status"], "source_commit": result["source_commit"],
        "pilot_gate": result["pilot_gate"],
        "evaluation_reference_actions": result["evaluation_reference_actions"],
        "evaluation_independent_actions": result["evaluation_independent_actions"],
        "training_real_actions": 0, "optimizer_step_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
