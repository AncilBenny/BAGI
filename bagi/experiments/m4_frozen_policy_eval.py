"""M4 evaluation of frozen, numeric-only M3 PPO and RecurrentPPO policies.

No new training is permitted in this diagnostic. All scored layouts are disjoint
from the M1/M2/M3 sets, and M3 replay is only checkpoint verification.
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

from bagi.experiments.m1_goal_learning import (
    ARMS,
    MAX_EVAL_EPISODE_STEPS,
    TRAIN_TASK,
    TRANSFER_TASK,
    admitted_source_commit,
    make_task,
    save_progress,
)

M3_SOURCE = "99f7f194787c1f02014b692aed221957639b77e6"
M3_CONFIG_SHA = "820aab08b951b1447e897225522b382555a12cbbaca0a021169d08d289070824"
M3_ARTIFACT_ID = 13823291153
MODEL_SEEDS = (37, 53)
REFERENCE_SEEDS = tuple(range(96001, 96007))
EVAL_SEEDS = tuple(range(98001, 98013))
TRANSFER_SEEDS = tuple(range(99001, 99013))
REPEATS = 5


def validate_config(config: dict) -> dict:
    expected = {
        "registered_study": "m4",
        "train_task": TRAIN_TASK,
        "transfer_task": TRANSFER_TASK,
        "arms": list(ARMS),
        "model_seeds": list(MODEL_SEEDS),
        "evaluation_seeds": list(EVAL_SEEDS),
        "transfer_seeds": list(TRANSFER_SEEDS),
        "sampled_repeats": REPEATS,
        "max_eval_steps": MAX_EVAL_EPISODE_STEPS,
        "m3_artifact_id": M3_ARTIFACT_ID,
        "m3_source_commit": M3_SOURCE,
        "output_dir": "runs/m4_frozen_policy",
    }
    if config != expected:
        raise ValueError("Registered M4 evaluation contract changed")
    return config


def action_seed(arm: str, model_seed: int, task: str, layout: int, repeat: int) -> int:
    """One public, outcome-independent Torch RNG seed per stochastic episode."""
    if arm not in ARMS or model_seed not in MODEL_SEEDS:
        raise ValueError("Unregistered frozen model")
    if task == TRAIN_TASK:
        task_index, layout_index = 0, EVAL_SEEDS.index(layout)
    elif task == TRANSFER_TASK:
        task_index, layout_index = 1, TRANSFER_SEEDS.index(layout)
    else:
        raise ValueError("Unknown evaluation task")
    if type(repeat) is not int or not 1 <= repeat <= REPEATS:
        raise ValueError("Unknown M4 replicate")
    return (20_000_000 + ARMS.index(arm) * 1_000_000
            + MODEL_SEEDS.index(model_seed) * 100_000
            + task_index * 10_000 + layout_index * 10 + repeat)


def read_m3_manifest(root: Path) -> tuple[dict, dict]:
    path = root / "artifacts/research-loop/m3_goal_learning-summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (summary.get("source_commit") != M3_SOURCE
            or summary.get("registered_study") != "m3"
            or summary.get("config_sha256") != M3_CONFIG_SHA
            or summary.get("training_real_actions") != 524288
            or summary.get("optimizer_step_calls") != 16384):
        raise ValueError("M3 source, configuration or training counters mismatch")
    for package in ("gymnasium", "minigrid", "stable-baselines3", "sb3-contrib"):
        if summary["versions"][package] != version(package):
            raise ValueError("Frozen checkpoint runtime version mismatch: " + package)
    models = {(row["arm"], row["model_seed"]): row for row in summary["records"]}
    expected = {(arm, seed) for arm in ARMS for seed in MODEL_SEEDS}
    if len(summary["records"]) != len(expected) or set(models) != expected:
        raise ValueError("Missing or duplicate M3 model record")
    return summary, models


def restore_numeric_policy(agent, file: Path, manifest: dict) -> str:
    """Restore full exact policy state; refuse pickle, mismatched bytes or tensors."""
    sha = hashlib.sha256(file.read_bytes()).hexdigest()
    if sha != manifest.get("sha256") or manifest.get("allow_pickle") is not False:
        raise ValueError("M3 numeric checkpoint digest/pickle mismatch")
    expected = agent.policy.state_dict()
    members = manifest.get("members", {})
    if not isinstance(members, dict) or not members:
        raise ValueError("Empty checkpoint manifest")
    tensors = {}
    with np.load(file, allow_pickle=False) as archive:
        if set(archive.files) != set(members):
            raise ValueError("M3 numeric checkpoint members mismatch")
        for key, metadata in members.items():
            name = metadata["name"]
            if name not in expected or name in tensors:
                raise ValueError("Unexpected/duplicate checkpoint tensor name")
            array = np.asarray(archive[key])
            if (array.dtype.kind not in "fiub"
                    or list(array.shape) != metadata["shape"]
                    or str(array.dtype) != metadata["dtype"]
                    or not np.isfinite(array).all()):
                raise ValueError("Checkpoint tensor shape, dtype or value mismatch")
            tensor = torch.from_numpy(array.copy())
            if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
                raise ValueError("Checkpoint model architecture does not match")
            tensors[name] = tensor
    if set(tensors) != set(expected):
        raise ValueError("Incomplete frozen policy checkpoint")
    agent.policy.load_state_dict(tensors, strict=True)
    return sha


def evaluate_episode(agent, arm: str, task: str, layout: int, *,
                     sampled_seed: int | None, meter: dict, reference: bool) -> dict:
    """Observe and count only actual native actions; no hidden simulator inputs."""
    env = make_task(task, arm, max_native_steps=MAX_EVAL_EPISODE_STEPS)
    prefix = "reference" if reference else "independent"
    try:
        env.action_space.seed(layout)
        obs, _ = env.reset(seed=layout)
        meter[prefix + "_resets"] += 1
        if sampled_seed is not None:
            torch.manual_seed(sampled_seed)
            np.random.seed(sampled_seed)
        state = None
        episode_start = np.array([True], dtype=bool)
        terminated = truncated = False
        total_reward = 0.0
        histogram = [0] * int(env.action_space.n)
        for _ in range(MAX_EVAL_EPISODE_STEPS):
            action, state = agent.predict(
                obs, state=state, episode_start=episode_start,
                deterministic=sampled_seed is None,
            )
            executed = int(np.asarray(action).item())
            if not 0 <= executed < len(histogram):
                raise ValueError("Non-native action returned by policy")
            obs, reward, terminated, truncated, _ = env.step(executed)
            histogram[executed] += 1
            total_reward += float(reward)
            meter[prefix + "_actions"] += 1
            if terminated or truncated:
                break
            episode_start[:] = False
        steps = sum(histogram)
        if not (terminated or truncated):
            raise ValueError("Evaluation episode exceeded registered horizon")
        if env.native_steps != steps or env.native_resets != 1:
            raise ValueError("Native evaluation counters disagree")
        return {
            "seed": layout, "action_rng_seed": sampled_seed,
            "goal_success": int(bool(terminated) and total_reward > 0),
            "reward": total_reward, "terminated": bool(terminated),
            "truncated": bool(truncated), "evaluation_real_steps": steps,
            "evaluation_real_resets": 1, "executed_action_histogram": histogram,
        }
    finally:
        env.close()


def verify_reference(agent, arm: str, row: dict, meter: dict) -> int:
    """Check restored policy against M3 without adding old seeds to M4 results."""
    episodes = row["after"]["episode_results"]
    if [episode["seed"] for episode in episodes] != list(REFERENCE_SEEDS):
        raise ValueError("M3 replay seed provenance changed")
    for original in episodes:
        observed = evaluate_episode(
            agent, arm, TRAIN_TASK, original["seed"], sampled_seed=None,
            meter=meter, reference=True,
        )
        for field in ("seed", "goal_success", "terminated", "truncated",
                      "evaluation_real_steps"):
            if observed[field] != original[field]:
                raise ValueError("Restored M3 policy deterministic replay differs: " + field)
        if abs(observed["reward"] - original["reward"]) > 1e-8:
            raise ValueError("Restored M3 policy reference reward differs")
    return len(episodes)


def evaluate_split(agent, arm: str, model_seed: int, task: str,
                   layouts: tuple[int, ...], meter: dict) -> dict:
    deterministic, sampled = [], []
    for layout in layouts:
        deterministic.append(evaluate_episode(
            agent, arm, task, layout, sampled_seed=None,
            meter=meter, reference=False,
        ))
        for repeat in range(1, REPEATS + 1):
            sampled.append(evaluate_episode(
                agent, arm, task, layout,
                sampled_seed=action_seed(arm, model_seed, task, layout, repeat),
                meter=meter, reference=False,
            ))
    deterministic_rate = sum(r["goal_success"] for r in deterministic) / len(deterministic)
    sampled_rate = sum(r["goal_success"] for r in sampled) / len(sampled)
    return {
        "task": task, "layout_count": len(layouts),
        "sampled_repeats_per_layout": REPEATS,
        "deterministic": deterministic, "sampled": sampled,
        "deterministic_success_rate": deterministic_rate,
        "sampled_success_rate": sampled_rate,
        "sampled_minus_deterministic": sampled_rate - deterministic_rate,
    }


def run(config_path: Path, artifact_root: Path, output: Path) -> dict:
    if output.exists() or (output.parent / "progress.json").exists():
        raise FileExistsError("Refusing to overwrite an existing M4 study")
    source = admitted_source_commit()
    raw = config_path.read_bytes()
    config = validate_config(yaml.safe_load(raw))
    if output.parent.as_posix() != config["output_dir"]:
        raise ValueError("M4 output directory differs from registered location")
    m3, models = read_m3_manifest(artifact_root)
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    meter = {"reference_actions": 0, "reference_resets": 0,
             "independent_actions": 0, "independent_resets": 0}
    records = []
    stage = "initializing"
    start_time = time.monotonic()
    try:
        # Validate EVERY frozen model against the archived M3 reference split
        # before touching ANY new M4 episode seed.
        verified_models = {}
        for arm in ARMS:
            for model_seed in MODEL_SEEDS:
                stage = f"restore-{arm}-{model_seed}"
                env = make_task(TRAIN_TASK, arm)
                try:
                    recurrent = arm == "recurrent"
                    kwargs = {"net_arch": {"pi": [32], "vf": [32]}}
                    if recurrent:
                        kwargs["lstm_hidden_size"] = 32
                    agent = (RecurrentPPO if recurrent else PPO)(
                        "MlpLstmPolicy" if recurrent else "MlpPolicy",
                        env, seed=model_seed, device="cpu",
                        n_steps=128, batch_size=64, n_epochs=2,
                        policy_kwargs=kwargs, verbose=0,
                    )
                    row = models[(arm, model_seed)]
                    expected_file = f"{arm}-seed{model_seed}.npz"
                    if row["numeric_policy"]["file"] != expected_file:
                        raise ValueError("M3 checkpoint filename mismatch")
                    checkpoint = artifact_root / "runs/m3_goal_learning" / expected_file
                    sha = restore_numeric_policy(agent, checkpoint, row["numeric_policy"])
                    stage = f"replay-{arm}-{model_seed}"
                    verified = verify_reference(agent, arm, row, meter)
                    verified_models[(arm, model_seed)] = (agent, sha, verified)
                finally:
                    env.close()
        if len(verified_models) != len(ARMS) * len(MODEL_SEEDS):
            raise ValueError("Not all frozen M3 models passed reference preflight")

        for arm in ARMS:
            for model_seed in MODEL_SEEDS:
                stage = f"evaluate-{arm}-{model_seed}"
                agent, sha, verified = verified_models[(arm, model_seed)]
                same = evaluate_split(agent, arm, model_seed, TRAIN_TASK,
                                      EVAL_SEEDS, meter)
                transfer = evaluate_split(agent, arm, model_seed, TRANSFER_TASK,
                                          TRANSFER_SEEDS, meter)
                records.append({
                    "arm": arm, "model_seed": model_seed,
                    "m3_checkpoint_sha256": sha,
                    "m3_reference_episodes_replayed": verified,
                    "same_task": same, "transfer": transfer,
                })
                save_progress(output.parent / "progress.json", {
                    "status": "partial_not_a_completed_study",
                    "source_commit": source, "phase": stage,
                    "completed_models": len(records),
                    "optimizer_step_calls": 0, **meter, "records": records,
                })
    except BaseException as error:
        save_progress(output.parent / "progress.json", {
            "status": "interrupted_not_a_completed_study",
            "source_commit": source, "phase": stage,
            "failure_type": type(error).__name__,
            "completed_models": len(records),
            "optimizer_step_calls": 0, **meter, "records": records,
        })
        raise
    if len(records) != len(ARMS) * len(MODEL_SEEDS):
        raise ValueError("Incomplete M4 evaluation model coverage")
    deltas = {str(row["model_seed"]): row["same_task"]["sampled_minus_deterministic"]
              for row in records if row["arm"] == "recurrent"}
    result = {
        "study": "BAGI-M4 frozen-policy repeated action selection evaluation",
        "registered_study": "m4",
        "status": "independent_frozen_policy_evaluation",
        "source_commit": source,
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "m3_source_commit": M3_SOURCE,
        "m3_config_sha256": M3_CONFIG_SHA,
        "m3_artifact_id": M3_ARTIFACT_ID,
        "m3_training_real_actions_reused_not_new": m3["training_real_actions"],
        "m3_optimizer_step_calls_reused_not_new": m3["optimizer_step_calls"],
        "training_real_actions": 0, "optimizer_step_calls": 0, "teacher_queries": 0,
        "reference_replays_excluded_from_m4_denominators": True,
        "evaluation_reference_actions": meter["reference_actions"],
        "evaluation_reference_resets": meter["reference_resets"],
        "evaluation_independent_actions": meter["independent_actions"],
        "evaluation_independent_resets": meter["independent_resets"],
        "elapsed_seconds": round(time.monotonic() - start_time, 3),
        "per_recurrent_seed_same_task_deltas": deltas,
        "primary_hypothesis_pass": all(delta >= 0.25 for delta in deltas.values()),
        "records": records,
        "limitations": [
            "Existing frozen M3 models; no new training or optimizer updates.",
            "Repeated sampled trajectories on each layout are not independent layouts.",
            "Navigation goal requires termination plus native reward; no physical manipulation.",
            "Action sampling is not proof of recurrent memory or continual learning.",
            "M3 reference replay is restoration verification, not independent M4 data.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m4_frozen_policy.yaml"))
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/m4_frozen_policy/summary.json"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("M4 native evaluation requires explicit --execute")
    result = run(args.config, args.artifact_root, args.output)
    print(json.dumps({
        "status": result["status"],
        "source_commit": result["source_commit"],
        "primary_hypothesis_pass": result["primary_hypothesis_pass"],
        "evaluation_independent_actions": result["evaluation_independent_actions"],
        "evaluation_reference_actions": result["evaluation_reference_actions"],
        "training_real_actions": 0, "optimizer_step_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
