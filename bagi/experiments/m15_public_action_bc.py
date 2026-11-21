"""M15: fresh public-action behavior cloning of the audited M14 controller.

Trainer uses only current public observations, a bit derived from OWN executed
pickup effects, and actual reference actions. Native physical outcomes stay in
a separate evaluator; no hidden state enters the learned model or teacher.
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
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from torch import nn

from bagi.experiments.m1_goal_learning import admitted_source_commit, save_progress
from bagi.experiments.m5_continual_retention import mission_vector
from bagi.experiments.m10_native_audit import NativeDoorKeyAudit
from bagi.experiments.m13_public_interaction_coverage import Controller
from bagi.experiments.m14_public_open_door import TASKS, TraversableOpenDoorController

SEEDS = (521, 533)
ARMS = ("ordinary_bc", "balanced_bc")
TRAIN = (tuple(range(211001, 211033)), tuple(range(221001, 221033)))
MONITOR = (tuple(range(212001, 212009)), tuple(range(222001, 222009)))
FINAL = (tuple(range(213001, 213025)), tuple(range(223001, 223025)))
HORIZON = 256
EPOCHS = 10
BATCH = 128
LEARNING_RATE = 0.003
HIDDEN = 64
N_OBJECT = len(OBJECT_TO_IDX)
N_COLOR = len(COLOR_TO_IDX)
N_STATE = len(STATE_TO_IDX)
WIDTH = N_OBJECT + N_COLOR + N_STATE + 64 + 4 + 1
OUT = "runs/m15_public_action_bc"


def validate_config(config: dict) -> dict:
    expected = {
        "registered_study": "m15",
        "tasks": list(TASKS),
        "model_seeds": list(SEEDS),
        "learner_arms": list(ARMS),
        "train_seeds": [list(x) for x in TRAIN],
        "monitor_seeds": [list(x) for x in MONITOR],
        "final_seeds": [list(x) for x in FINAL],
        "episode_horizon": HORIZON,
        "training_epochs": EPOCHS,
        "batch_size": BATCH,
        "learning_rate": LEARNING_RATE,
        "hidden_width": HIDDEN,
        "balanced_weight_cap": 8.0,
        "primary_min_balanced_goals": 10,
        "primary_min_balanced_gain": 4,
        "output_dir": OUT,
    }
    if config != expected:
        raise ValueError("M15 registered public native-action learner changed")
    seeds = sum(TRAIN + MONITOR + FINAL, ())
    if len(seeds) != len(set(seeds)):
        raise ValueError("M15 train/monitor/final layouts overlap")
    return config


def features(observation: dict, public_pickup_memory: bool) -> np.ndarray:
    if not isinstance(observation, dict):
        raise TypeError("Public MiniGrid observation must be a dict")
    if set(observation) != {"image", "direction", "mission"}:
        raise ValueError("M15 policy saw unknown observation field")
    image = np.asarray(observation["image"])
    if image.shape != (7, 7, 3):
        raise ValueError("M15 public MiniGrid input width changed")
    obj, color, state = (int(x) for x in image[3, 5])
    direction = observation["direction"]
    if (not 0 <= obj < N_OBJECT or not 0 <= color < N_COLOR
            or not 0 <= state < N_STATE or not 0 <= direction < 4):
        raise ValueError("M15 observed invalid native categorical fields")
    vec = np.zeros(WIDTH, dtype=np.float32)
    vec[obj] = 1.0
    vec[N_OBJECT + color] = 1.0
    vec[N_OBJECT + N_COLOR + state] = 1.0
    start = N_OBJECT + N_COLOR + N_STATE
    vec[start:start + 64] = mission_vector(observation["mission"])
    vec[start + 64 + int(direction)] = 1.0
    vec[-1] = float(public_pickup_memory)
    return vec


class BCPolicy(nn.Module):
    """Maintained PyTorch MLP trained with ordinary cross-entropy BC."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(WIDTH, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 7)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != WIDTH:
            raise ValueError("M15 model received non-public input width")
        return self.layers(x)


def teacher_episode(task: str, seed: int) -> dict:
    if task not in TASKS or seed not in TRAIN[TASKS.index(task)]:
        raise ValueError("M15 teacher may collect only TRAIN layouts")
    controller = TraversableOpenDoorController()
    rng = np.random.default_rng(14_000_000 + seed)
    audited = NativeDoorKeyAudit(
        gym.wrappers.TimeLimit(gym.make(task), max_episode_steps=HORIZON)
    )
    xs, ys = [], []
    try:
        obs, _ = audited.reset(seed=seed)
        finished = False
        for _ in range(HORIZON):
            xs.append(features(obs, controller.key_observed_after_pickup))
            actual = controller.choose(obs, rng)
            successor, _reward, terminated, truncated, _ = audited.step(actual)
            controller.update(obs, actual, successor)
            ys.append(actual)
            obs = successor
            if terminated or truncated:
                finished = True
                break
        summary = audited.summary()
        if not finished or sum(summary["action_counts"].values()) != len(ys):
            raise ValueError("M15 real teacher executed-action count mismatch")
        if summary["native_actions"] != len(xs):
            raise ValueError("M15 public observation/action pairing mismatch")
        return {
            "task": task, "layout_seed": seed,
            "x": np.stack(xs).astype(np.float32),
            "y": np.asarray(ys, dtype=np.int64),
            "native_actions": summary["native_actions"],
            "seven_action_counts": summary["action_counts"],
            "verified_physical_goal": bool(summary["full_physical_goal"]),
            "executed_actions_sha256": hashlib.sha256(bytes(ys)).hexdigest(),
        }
    finally:
        audited.env.close()


def acquire_teacher() -> tuple[np.ndarray, np.ndarray, dict]:
    data = [
        teacher_episode(task, seed)
        for task, seeds in zip(TASKS, TRAIN) for seed in seeds
    ]
    x = np.concatenate([r["x"] for r in data])
    y = np.concatenate([r["y"] for r in data])
    hist = np.bincount(y, minlength=7)
    if len(x) != sum(row["native_actions"] for row in data):
        raise ValueError("M15 teacher observation labels not conserved")
    return x, y, {
        "training_episodes": len(data),
        "actual_native_training_actions": len(x),
        "actual_action_histogram": hist.tolist(),
        "independent_teacher_goal_episodes_not_claimed": sum(
            int(r["verified_physical_goal"]) for r in data
        ),
        "episode_action_provenance": [
            {key: v for key, v in row.items() if key not in ("x", "y")}
            for row in data
        ],
    }


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=7)
    typical = max(1, max(counts[0], counts[1], counts[2]))
    value = np.sqrt(typical / np.maximum(1, counts))
    value = np.clip(value, 1.0, 8.0)
    return torch.tensor(value, dtype=torch.float32)


def fit(model: BCPolicy, x: np.ndarray, y: np.ndarray, *,
        seed: int, balanced: bool) -> dict:
    if seed not in SEEDS or not len(x) or len(x) != len(y):
        raise ValueError("M15 requires registered seed and actual public labels")
    if set(np.unique(y)) - set(range(7)):
        raise ValueError("M15 illegal native action label")
    model.train()
    weights = class_weights(y) if balanced else None
    source = torch.from_numpy(x)
    labels = torch.from_numpy(y)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(seed + 10_000)
    count = 0
    presentations = 0
    loss_total = 0.0
    for _ in range(EPOCHS):
        for index in torch.randperm(len(y), generator=generator).split(BATCH):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(
                model(source[index]), labels[index], weight=weights
            )
            loss.backward()
            opt.step()
            count += 1
            presentations += len(index)
            loss_total += float(loss.detach())
    if presentations != EPOCHS * len(y):
        raise ValueError("M15 cached label presentations not conserved")
    return {
        "optimizer_step_calls": count,
        "cached_sample_presentations": presentations,
        "mean_loss_per_optimizer": loss_total / count,
        "balanced": balanced,
        "class_weights": weights.tolist() if balanced else None,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def save_numeric(model: BCPolicy, path: Path) -> dict:
    if path.exists():
        raise FileExistsError("M15 numeric policy already saved")
    arrays, manifest = {}, {}
    for n, (name, tensor) in enumerate(model.state_dict().items()):
        value = tensor.detach().cpu().numpy()
        if value.dtype.kind not in "fiub" or not np.isfinite(value).all():
            raise ValueError("Unsafe or invalid M15 numeric checkpoint")
        key = f"tensor_{n:03d}"
        arrays[key] = value
        manifest[key] = {
            "name": name, "dtype": str(value.dtype), "shape": list(value.shape)
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {
        "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "allow_pickle": False, "members": manifest,
    }


def policy_episode(task: str, layout: int, mode: str,
                   model: BCPolicy | None, split: str) -> dict:
    group = MONITOR if split == "monitor" else FINAL if split == "final" else None
    if group is None or task not in TASKS:
        raise ValueError("M15 unregistered task/split")
    if layout not in group[TASKS.index(task)]:
        raise ValueError("M15 attempted unregistered evaluation layout")
    if mode not in ("teacher", "uniform") and mode not in (
        f"{arm}_{seed}" for arm in ARMS for seed in SEEDS
    ):
        raise ValueError("M15 evaluation mode not registered")
    if (model is None) != (mode in ("teacher", "uniform")):
        raise ValueError("M15 mismatched model/controller")
    actor = (
        TraversableOpenDoorController() if mode == "teacher"
        else Controller("uniform") if mode == "uniform"
        else Controller("public_memory")
    )
    mode_id = (
        0 if mode == "teacher" else 1 if mode == "uniform"
        else 2 + [f"{arm}_{seed}" for arm in ARMS for seed in SEEDS].index(mode)
    )
    rng_seed = 19_000_000 + 10 * layout + mode_id
    rng = np.random.default_rng(rng_seed)
    audited = NativeDoorKeyAudit(
        gym.wrappers.TimeLimit(gym.make(task), max_episode_steps=HORIZON)
    )
    try:
        obs, _ = audited.reset(seed=layout)
        finished = False
        reward_total = 0.0
        for _ in range(HORIZON):
            if model is None:
                action = actor.choose(obs, rng)
            else:
                with torch.no_grad():
                    vector = torch.from_numpy(
                        features(obs, actor.key_observed_after_pickup)
                    ).unsqueeze(0)
                    logits = model(vector).squeeze(0).numpy()
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                action = int(rng.choice(7, p=probabilities))
            successor, reward, terminated, truncated, _ = audited.step(action)
            actor.update(obs, action, successor)
            reward_total += float(reward)
            obs = successor
            if terminated or truncated:
                finished = True
                break
        summary = audited.summary()
        if not finished or summary["native_actions"] > HORIZON:
            raise ValueError("M15 native policy evaluation did not terminate")
        if sum(summary["action_counts"].values()) != summary["native_actions"]:
            raise ValueError("M15 native action histogram mismatch")
        return {
            "task": task, "layout_seed": layout, "split": split,
            "mode": mode, "action_rng_seed": rng_seed,
            "actual_native_reward_sum": reward_total,
            **summary,
        }
    finally:
        audited.env.close()


def evaluate_models(split: str, models: dict[str, BCPolicy]) -> dict:
    modes = ("teacher", "uniform", *models)
    result = {}
    for task, seeds in zip(TASKS, MONITOR if split == "monitor" else FINAL):
        for mode in modes:
            model = models.get(mode)
            rows = [
                policy_episode(task, layout, mode, model, split) for layout in seeds
            ]
            histogram = [
                sum(row["action_counts"][str(a)] for row in rows)
                for a in range(7)
            ]
            actions = sum(row["native_actions"] for row in rows)
            if sum(histogram) != actions:
                raise ValueError("M15 evaluation native action conservation failed")
            result[f"{task}:{mode}"] = {
                "totals": {
                    "episodes": len(rows), "native_actions": actions,
                    "native_action_histogram": histogram,
                    "key_pickup_episodes": sum(
                        int(row["effective_pickup_events"] > 0) for row in rows
                    ),
                    "effective_door_opening_episodes": sum(
                        int(row["effective_door_open_events"] > 0) for row in rows
                    ),
                    "physical_full_goal_episodes": sum(
                        int(row["full_physical_goal"]) for row in rows
                    ),
                    "rewarded_terminal_episodes": sum(
                        int(row["rewarded_terminations"] > 0) for row in rows
                    ),
                },
                "episodes": rows,
            }
    return result


def preregistered_gate(result: dict) -> dict:
    expected = {
        f"{task}:{mode}" for task in TASKS
        for mode in ("teacher", "uniform", *(f"{arm}_{s}" for arm in ARMS for s in SEEDS))
    }
    if set(result) != expected:
        raise ValueError("M15 missing independent model/teacher conditions")
    goals = {
        arm: sum(
            result[f"{task}:{arm}_{seed}"]["totals"]["physical_full_goal_episodes"]
            for task in TASKS for seed in SEEDS
        )
        for arm in ARMS
    }
    balanced_all = {
        f"{task}:{seed}": result[
            f"{task}:balanced_bc_{seed}"
        ]["totals"]["physical_full_goal_episodes"]
        for task in TASKS for seed in SEEDS
    }
    return {
        "actual_physical_goals_across_96_episodes_per_arm": goals,
        "balanced_by_task_and_seed": balanced_all,
        "balanced_minus_ordinary_full_goals": (
            goals["balanced_bc"] - goals["ordinary_bc"]
        ),
        "pilot_gate_pass": (
            goals["balanced_bc"] >= 10
            and goals["balanced_bc"] - goals["ordinary_bc"] >= 4
            and all(x >= 1 for x in balanced_all.values())
        ),
        "not_task_independent_learning": True,
        "public_history_feature_is_hand_specified": True,
    }


def run(config: Path, output: Path) -> dict:
    if output.exists() or (output.parent / "progress.json").exists():
        raise FileExistsError("M15 refuses to overwrite earlier evidence")
    config_data = config.read_bytes()
    validate_config(yaml.safe_load(config_data))
    if output.parent.as_posix() != OUT:
        raise ValueError("M15 generated output outside registered directory")
    source_commit = admitted_source_commit()
    torch.set_num_threads(1)
    started = time.monotonic()
    phase = "training_acquisition"
    data, models, trained, monitor, final = None, {}, {}, {}, {}
    total_training_actions = 0
    optimizer_calls = 0
    try:
        x, y, acquisition = acquire_teacher()
        total_training_actions = acquisition["actual_native_training_actions"]
        # Numeric-only training-data artifact. Never submit stored raw data to git.
        data_file = output.parent / "public_teacher_train.npz"
        data_file.parent.mkdir(parents=True, exist_ok=True)
        if data_file.exists():
            raise FileExistsError("M15 training dataset already collected")
        np.savez_compressed(data_file, features=x, executed_actions=y)
        data = {
            "file": data_file.name,
            "sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
            "observations": len(x), "allow_pickle": False,
        }
        phase = "fresh_behavior_cloning_fits"
        for seed in SEEDS:
            torch.manual_seed(seed)
            base = BCPolicy()
            for arm in ARMS:
                model = BCPolicy()
                model.load_state_dict(base.state_dict())
                stats = fit(
                    model, x, y, seed=seed, balanced=arm == "balanced_bc"
                )
                checkpoint = save_numeric(
                    model, output.parent / f"{arm}-seed{seed}.npz"
                )
                models[f"{arm}_{seed}"] = model
                optimizer_calls += stats["optimizer_step_calls"]
                trained[f"{arm}_{seed}"] = {
                    "training": stats, "numeric_checkpoint": checkpoint
                }
                save_progress(output.parent / "progress.json", {
                    "status": "partial_not_completed", "phase": phase,
                    "source_commit": source_commit, "acquisition": acquisition,
                    "model_fits": len(trained),
                    "actual_native_training_actions": total_training_actions,
                    "optimizer_step_calls": optimizer_calls, "models": trained,
                })
        if len(models) != len(SEEDS) * len(ARMS):
            raise ValueError("M15 heldout locked before all four fits")
        phase = "train_only_monitor"
        monitor = evaluate_models("monitor", models)
        phase = "independent_final"
        final = evaluate_models("final", models)
        gate = preregistered_gate(final)
    except BaseException as error:
        save_progress(output.parent / "progress.json", {
            "status": "interrupted_not_completed", "phase": phase,
            "failure_type": type(error).__name__, "source_commit": source_commit,
            "native_training_actions": total_training_actions,
            "optimizer_step_calls": optimizer_calls,
            "model_fits": len(trained), "models": trained,
            "monitor_conditions": len(monitor),
            "independent_final_conditions": len(final),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        raise
    monitor_actions = sum(v["totals"]["native_actions"] for v in monitor.values())
    final_actions = sum(v["totals"]["native_actions"] for v in final.values())
    report = {
        "study": "M15 public physically-verified teacher behavior cloning",
        "registered_study": "m15",
        "status": "completed_new_public_behavior_cloning_and_independent_audit",
        "source_commit": source_commit,
        "config_sha256": hashlib.sha256(config_data).hexdigest(),
        "versions": {
            p: version(p) for p in ("gymnasium", "minigrid", "torch", "numpy")
        },
        "acquisition": acquisition, "train_dataset_numeric": data,
        "fresh_native_actions": total_training_actions + monitor_actions + final_actions,
        "native_training_actions": total_training_actions,
        "monitor_native_actions": monitor_actions,
        "independent_native_actions": final_actions,
        "independent_episodes": sum(v["totals"]["episodes"] for v in final.values()),
        "optimizer_step_calls": optimizer_calls,
        "new_model_fits": len(trained),
        "reference_teacher_action_labels": total_training_actions,
        "privileged_teacher_queries": 0,
        "models": trained, "monitor": monitor, "independent_final": final,
        "pilot": gate,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "limitations": [
            "Teacher is a hand-specified PUBLIC controller, not discovered skill.",
            "Learned state uses a hand-specified public pickup-evidence bit.",
            "Two DoorKey sizes share task rules and action/observation interface.",
            "Rare action balance changes objective, not native training actions.",
            "All learners share independently heldout layout seeds.",
            "No cross-domain transfer or continual task retention demonstrated.",
            "Private physical auditing is never a policy input.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m15_public_bc.yaml"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/m15_public_action_bc/summary.json"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("M15 new teacher actions and fits require --execute")
    result = run(args.config, args.output)
    print(json.dumps({
        "status": result["status"],
        "fresh_native_actions": result["fresh_native_actions"],
        "optimizer_step_calls": result["optimizer_step_calls"],
        "new_model_fits": result["new_model_fits"],
        "pilot": result["pilot"],
    }, indent=2))


if __name__ == "__main__":
    main()
