"""M19: matched exposure control for within-family source replay effects.

Compare previous-task replay to equal-sized extra target-data presentations
under identical training updates, public actions and independently audited goals.
This is NOT cross-domain transfer or task-independent continual learning.
"""
from __future__ import annotations

import argparse
import copy
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
from torch import nn

from bagi.experiments.m1_goal_learning import admitted_source_commit, save_progress
from bagi.experiments.m10_native_audit import NativeDoorKeyAudit
from bagi.experiments.m13_public_interaction_coverage import Controller
from bagi.experiments.m14_public_open_door import TraversableOpenDoorController
from bagi.experiments.m15_public_action_bc import BCPolicy, features, save_numeric
from bagi.experiments.m18_public_door_selection import NativeGoToDoorAudit, PublicGoToDoorTeacher

TASKS = ("MiniGrid-GoToDoor-8x8-v0", "MiniGrid-DoorKey-6x6-v0")
SEEDS = (701, 709, 719, 727)
ARMS = ("source", "scratch", "naive", "extra_target", "source_replay")
TRAIN = (tuple(range(291001, 291049)), tuple(range(301001, 301049)))
MONITOR = (tuple(range(292001, 292013)), tuple(range(302001, 302013)))
FINAL = (tuple(range(293001, 293033)), tuple(range(303001, 303033)))
EPOCHS = 10
BATCH = 128
LR = 0.003
HORIZON = 384
OUT = "runs/m19_reverse_rehearsal"


def validate_config(config: dict) -> dict:
    expected = {
        "registered_study": "m19",
        "tasks": list(TASKS),
        "model_seeds": list(SEEDS),
        "arms": list(ARMS),
        "train_seeds": [list(x) for x in TRAIN],
        "monitor_seeds": [list(x) for x in MONITOR],
        "final_seeds": [list(x) for x in FINAL],
        "training_epochs": EPOCHS,
        "batch_size": BATCH,
        "learning_rate": LR,
        "episode_horizon": HORIZON,
        "additional_cached_label_ratio": 1.0,
        "primary_old_task_min_replay_gain": 12,
        "primary_target_max_replay_deficit": 8,
        "primary_min_positive_model_seeds": 3,
        "output_dir": OUT,
    }
    if config != expected:
        raise ValueError("M19 preregistered sequential learning protocol changed")
    sequences = TRAIN + MONITOR + FINAL
    all_seeds = sum(sequences, ())
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("M19 train/monitor/final layout overlap")
    return config


def make_teacher(task: str):
    return PublicGoToDoorTeacher() if task == TASKS[0] else TraversableOpenDoorController()


def make_audit(task: str):
    env = gym.wrappers.TimeLimit(gym.make(task), max_episode_steps=HORIZON)
    return NativeGoToDoorAudit(env) if task == TASKS[0] else NativeDoorKeyAudit(env)


def collect_episode(task: str, layout: int) -> dict:
    if task not in TASKS or layout not in TRAIN[TASKS.index(task)]:
        raise ValueError("M19 acquisition is TRAIN only")
    actor = make_teacher(task)
    rng = np.random.default_rng(26_000_000 + layout)
    audit = make_audit(task)
    obs_rows, action_rows = [], []
    try:
        obs, _ = audit.reset(seed=layout)
        done = False
        for _ in range(HORIZON):
            obs_rows.append(features(obs, actor.key_observed_after_pickup))
            action = actor.choose(obs, rng)
            next_obs, _, terminated, truncated, _ = audit.step(action)
            action_rows.append(action)
            actor.update(obs, action, next_obs)
            obs = next_obs
            if terminated or truncated:
                done = True
                break
        result = audit.summary()
        if not done or result["native_actions"] != len(action_rows):
            raise ValueError("M19 real teacher action count mismatch")
        if sum(result["action_counts"].values()) != len(action_rows):
            raise ValueError("M19 real teacher action histogram mismatch")
        return {
            "features": np.asarray(obs_rows, dtype=np.float32),
            "actions": np.asarray(action_rows, dtype=np.int64),
            "task": task, "layout_seed": layout,
            "actual_native_actions": result["native_actions"],
            "full_physical_goal": bool(result["full_physical_goal"]),
            "action_sha256": hashlib.sha256(bytes(action_rows)).hexdigest(),
            "action_histogram": result["action_counts"],
        }
    finally:
        audit.env.close()


def acquire(task: str, seeds: tuple[int, ...], out: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    rows = [collect_episode(task, seed) for seed in seeds]
    x = np.concatenate([r["features"] for r in rows])
    y = np.concatenate([r["actions"] for r in rows])
    if len(x) != sum(row["actual_native_actions"] for row in rows):
        raise ValueError("M19 acquired feature/action count disagreement")
    filename = f"train_{TASKS.index(task)}.npz"
    path = out / filename
    if path.exists():
        raise FileExistsError("M19 training traces cannot be overwritten")
    np.savez_compressed(path, public_features=x, executed_actions=y)
    summary = {
        "task": task, "train_episodes": len(rows),
        "actual_native_train_actions": len(y),
        "actual_action_histogram": np.bincount(y, minlength=7).tolist(),
        "train_only_teacher_physical_goals": sum(int(r["full_physical_goal"]) for r in rows),
        "trace": {"file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "allow_pickle": False},
        "episode_provenance": [
            {key: value for key, value in r.items() if key not in ("features", "actions")}
            for r in rows
        ],
    }
    return x, y, summary


def fit_phase(model: BCPolicy, x: np.ndarray, y: np.ndarray, seed: int,
              replay: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """Match target updates; optional extra cached examples from either domain."""
    if seed not in SEEDS or len(x) != len(y) or len(x) == 0:
        raise ValueError("M19 needs a registered seed and genuine native labels")
    if x.ndim != 2 or x.shape[1] != model.layers[0].in_features:
        raise ValueError("M19 malformed public observation features")
    if any(int(label) not in range(7) for label in np.unique(y)):
        raise ValueError("M19 non-native action label")
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    if replay is not None:
        replay_x, replay_y = replay
        if len(replay_x) != len(replay_y) or not len(replay_x):
            raise ValueError("M19 additional cached replay samples invalid")
        source_x = torch.from_numpy(replay_x)
        source_y = torch.from_numpy(replay_y)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    shuffle = torch.Generator().manual_seed(81_000 + seed)
    sample = torch.Generator().manual_seed(93_000 + seed)
    steps = 0
    target_presentations = 0
    extra_presentations = 0
    total_loss = 0.0
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(y), generator=shuffle).split(BATCH):
            optimizer.zero_grad()
            target_loss = nn.functional.cross_entropy(model(x_t[idx]), y_t[idx])
            loss = target_loss
            if replay is not None:
                chosen = torch.randint(len(source_y), (len(idx),), generator=sample)
                source_loss = nn.functional.cross_entropy(
                    model(source_x[chosen]), source_y[chosen]
                )
                loss = 0.5 * (target_loss + source_loss)
                extra_presentations += len(chosen)
            loss.backward()
            optimizer.step()
            steps += 1
            target_presentations += len(idx)
            total_loss += float(loss.detach())
    if target_presentations != EPOCHS * len(y):
        raise ValueError("M19 target action-label presentations not conserved")
    if replay is not None and extra_presentations != target_presentations:
        raise ValueError("M19 additional cached label ratio changed")
    return {
        "optimizer_step_calls": steps,
        "target_cached_label_presentations": target_presentations,
        "additional_cached_label_presentations": extra_presentations,
        "mean_training_loss": total_loss / steps,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def evaluate_episode(task: str, layout: int, mode: str,
                     model: BCPolicy | None, split: str) -> dict:
    group = MONITOR if split == "monitor" else FINAL if split == "final" else None
    if group is None or task not in TASKS or layout not in group[TASKS.index(task)]:
        raise ValueError("M19 unregistered task or heldout layout")
    labels = tuple(f"{arm}_{seed}" for seed in SEEDS for arm in ARMS)
    if mode not in ("teacher", "uniform", *labels):
        raise ValueError("M19 unregistered actor")
    if (model is None) != (mode in ("teacher", "uniform")):
        raise ValueError("M19 model/controller mismatch")
    actor = (
        make_teacher(task) if mode == "teacher"
        else Controller("uniform") if mode == "uniform"
        else Controller("public_memory")
    )
    # Common random numbers across arms for a given layout and model seed.
    # Action histories may diverge; no matched-trajectory causal claim is made.
    rng_key = 0 if mode == "teacher" else 1 if mode == "uniform" else int(
        mode.rsplit("_", 1)[1]
    )
    rng_seed = 39_000_000 + 1000 * layout + rng_key
    rng = np.random.default_rng(rng_seed)
    audit = make_audit(task)
    try:
        obs, _ = audit.reset(seed=layout)
        reward_total = 0.0
        finished = False
        if model is not None:
            model.eval()
        for _ in range(HORIZON):
            if model is None:
                action = actor.choose(obs, rng)
            else:
                with torch.no_grad():
                    logits = model(torch.from_numpy(
                        features(obs, actor.key_observed_after_pickup)
                    ).unsqueeze(0)).squeeze(0).numpy()
                weights = np.exp(logits - logits.max())
                weights /= weights.sum()
                action = int(rng.choice(7, p=weights))
            successor, reward, terminated, truncated, _ = audit.step(action)
            actor.update(obs, action, successor)
            obs = successor
            reward_total += float(reward)
            if terminated or truncated:
                finished = True
                break
        summary = audit.summary()
        if not finished or summary["native_actions"] > HORIZON:
            raise ValueError("M19 native episode did not finish within registered horizon")
        if sum(summary["action_counts"].values()) != summary["native_actions"]:
            raise ValueError("M19 native action audit mismatch")
        return {
            "task": task, "layout": layout, "split": split,
            "mode": mode, "rng_seed": rng_seed,
            "reward_sum": reward_total, **summary,
        }
    finally:
        audit.env.close()


def evaluate(split: str, models: dict[str, BCPolicy]) -> dict:
    if split not in ("monitor", "final") or set(models) != {
        f"{arm}_{seed}" for seed in SEEDS for arm in ARMS
    }:
        raise ValueError("M19 needs all pretrained/fine-tuned models before evaluation")
    group = MONITOR if split == "monitor" else FINAL
    result = {}
    for task, layouts in zip(TASKS, group):
        for mode in ("teacher", "uniform", *models):
            rows = [evaluate_episode(task, layout, mode, models.get(mode), split)
                    for layout in layouts]
            actions = sum(row["native_actions"] for row in rows)
            histogram = [sum(row["action_counts"][str(a)] for row in rows)
                         for a in range(7)]
            if actions != sum(histogram):
                raise ValueError("M19 evaluation action histograms do not reconcile")
            result[f"{task}:{mode}"] = {
                "totals": {
                    "episodes": len(rows), "actual_native_actions": actions,
                    "native_action_histogram": histogram,
                    "effective_key_pickup_episodes": sum(
                        int(row["effective_pickup_events"] > 0) for row in rows
                    ),
                    "effective_door_open_episodes": sum(
                        int(row["effective_door_open_events"] > 0) for row in rows
                    ),
                    "native_rewarded_episodes": sum(
                        int(row["rewarded_terminations"] > 0) for row in rows
                    ),
                    "full_physical_goal_episodes": sum(
                        int(row["full_physical_goal"]) for row in rows
                    ),
                },
                "episodes": rows,
            }
    return result


def registered_gates(final: dict) -> dict:
    """Retain prior native mission skill without sacrificing new physical goals."""
    expected = {f"{task}:{mode}" for task in TASKS for mode in (
        "teacher", "uniform",
        *(f"{arm}_{seed}" for seed in SEEDS for arm in ARMS)
    )}
    if set(final) != expected:
        raise ValueError("M19 requires all independent physical evaluation conditions")

    def score(task_idx: int, arm: str, seed: int) -> int:
        row = final[f"{TASKS[task_idx]}:{arm}_{seed}"]["totals"]
        if row["episodes"] != len(FINAL[task_idx]):
            raise ValueError("M19 missing independently audited final physical episodes")
        value = row["full_physical_goal_episodes"]
        if not 0 <= value <= row["episodes"]:
            raise ValueError("M19 invalid physical goal count")
        return value

    def pooled(task_idx: int, arm: str) -> int:
        return sum(score(task_idx, arm, seed) for seed in SEEDS)

    by_seed = {
        str(seed): {
            "replay_old_goals": score(0, "source_replay", seed),
            "extra_target_old_goals": score(0, "extra_target", seed),
            "replay_minus_extra_target": (
                score(0, "source_replay", seed) - score(0, "extra_target", seed)
            ),
            "replay_new_goals": score(1, "source_replay", seed),
            "naive_new_goals": score(1, "naive", seed),
        }
        for seed in SEEDS
    }
    old_gain = pooled(0, "source_replay") - pooled(0, "extra_target")
    positive = sum(int(row["replay_minus_extra_target"] > 0)
                   for row in by_seed.values())
    target_deficit = pooled(1, "naive") - pooled(1, "source_replay")
    return {
        "source_go_to_door_nonvacuous_goals": {
            arm: pooled(0, arm) for arm in ARMS
        },
        "target_doorkey_full_physical_goals": {
            arm: pooled(1, arm) for arm in ARMS
        },
        "old_replay_minus_exposure_matched_target": old_gain,
        "old_positive_model_seed_contrasts": positive,
        "target_naive_minus_replay": target_deficit,
        "per_model_seed": by_seed,
        "primary_reversal_retention_gate_pass": (
            old_gain >= 12 and positive >= 3 and target_deficit <= 8
        ),
        "shared_minigrid_mechanics_not_task_independent": True,
    }



def run(config: Path, output: Path) -> dict:
    if output.exists() or (output.parent / "progress.json").exists():
        raise FileExistsError("M19 refuses overwrite of prior run evidence")
    content = config.read_bytes()
    validate_config(yaml.safe_load(content))
    if output.parent.as_posix() != OUT:
        raise ValueError("M19 output outside registered ignored directory")
    source_commit = admitted_source_commit()
    torch.set_num_threads(1)
    out = output.parent
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    phase = "native_training_acquisition"
    train_meta: list[dict] = []
    trained: dict[str, dict] = {}
    models: dict[str, BCPolicy] = {}
    optimizer_calls = 0
    native_training_actions = 0
    monitor: dict = {}
    final: dict = {}
    try:
        (xs, ys, source_info) = acquire(TASKS[0], TRAIN[0], out)
        train_meta.append(source_info)
        native_training_actions += len(ys)
        save_progress(out / "progress.json", {
            "status": "partial_not_completed", "phase": phase,
            "source_commit": source_commit, "training": train_meta,
            "actual_native_training_actions": native_training_actions,
            "optimizer_step_calls": optimizer_calls,
        })
        (xt, yt, target_info) = acquire(TASKS[1], TRAIN[1], out)
        train_meta.append(target_info)
        native_training_actions += len(yt)
        phase = "fresh_source_and_target_fits"
        for seed in SEEDS:
            torch.manual_seed(seed)
            common_init = BCPolicy()
            source_model = copy.deepcopy(common_init)
            source_fit = fit_phase(source_model, xs, ys, seed)
            plans = [
                ("source", source_model, source_fit),
                ("scratch", copy.deepcopy(common_init), None),
                ("naive", copy.deepcopy(source_model), None),
                ("extra_target", copy.deepcopy(source_model), None),
                ("source_replay", copy.deepcopy(source_model), None),
            ]
            for arm, model, already_fit in plans:
                extra = (
                    (xs, ys) if arm == "source_replay"
                    else (xt, yt) if arm == "extra_target" else None
                )
                stats = already_fit if already_fit is not None else fit_phase(
                    model, xt, yt, seed, extra
                )
                name = f"{arm}_{seed}"
                checkpoint = save_numeric(model, out / f"{name}.npz")
                models[name] = model
                trained[name] = {
                    "training": stats, "numeric_checkpoint": checkpoint,
                    "stage": "gotodoor_8x8" if arm == "source" else "doorkey_6x6",
                    "additional_cached_label_origin": (
                        "gotodoor_8x8" if arm == "source_replay"
                        else "doorkey_6x6" if arm == "extra_target" else None
                    ),
                }
                optimizer_calls += stats["optimizer_step_calls"]
                save_progress(out / "progress.json", {
                    "status": "partial_not_completed", "phase": phase,
                    "source_commit": source_commit, "training": train_meta,
                    "actual_native_training_actions": native_training_actions,
                    "optimizer_step_calls": optimizer_calls,
                    "completed_model_fits": len(trained), "models": trained,
                })
        if len(trained) != len(SEEDS) * len(ARMS):
            raise ValueError("M19 heldout locked until all model fits completed")
        for seed in SEEDS:
            a = trained[f"source_replay_{seed}"]["training"]
            b = trained[f"extra_target_{seed}"]["training"]
            baseline = trained[f"naive_{seed}"]["training"]
            keys = ("optimizer_step_calls", "target_cached_label_presentations")
            if any(a[key] != b[key] or b[key] != baseline[key] for key in keys):
                raise ValueError("M19 mismatched target optimizer/data exposure")
            if a["additional_cached_label_presentations"] != b[
                "additional_cached_label_presentations"
            ] or a["additional_cached_label_presentations"] != len(yt) * EPOCHS:
                raise ValueError("M19 replay and target-extras not exposure-matched")
        phase = "train_only_monitor"
        monitor = evaluate("monitor", models)
        phase = "independent_final"
        final = evaluate("final", models)
        gate = registered_gates(final)
    except BaseException as exc:
        save_progress(out / "progress.json", {
            "status": "interrupted_not_completed", "phase": phase,
            "error_type": type(exc).__name__,
            "source_commit": source_commit, "training": train_meta,
            "native_training_actions_completed": native_training_actions,
            "optimizer_step_calls": optimizer_calls,
            "completed_model_fits": len(trained),
            "completed_monitor_conditions": len(monitor),
            "completed_final_conditions": len(final),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        raise
    monitor_actions = sum(r["totals"]["actual_native_actions"] for r in monitor.values())
    final_actions = sum(r["totals"]["actual_native_actions"] for r in final.values())
    result = {
        "registered_study": "m19",
        "study": "M19 reverse-order GoToDoor to DoorKey rehearsal and retention",
        "status": "completed_fresh_with_independent_physical_audit",
        "source_commit": source_commit,
        "config_sha256": hashlib.sha256(content).hexdigest(),
        "library_versions": {
            name: version(name) for name in ("gymnasium", "minigrid", "numpy", "torch")
        },
        "training": train_meta,
        "models": trained,
        "fresh_native_actions": native_training_actions + monitor_actions + final_actions,
        "native_training_actions": native_training_actions,
        "monitor_native_actions": monitor_actions,
        "independent_native_actions": final_actions,
        "independent_episodes": sum(r["totals"]["episodes"] for r in final.values()),
        "optimizer_step_calls": optimizer_calls,
        "fresh_model_fits": len(trained),
        "teacher_privileged_queries": 0,
        "monitor": monitor, "independent_final": final,
        "gates": gate,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "limitations": [
            "Reversed GoToDoor to DoorKey sequence, shared MiniGrid physics.",
            "Source GoToDoor requires mission-colored done after nonvacuous movement.",
            "Target DoorKey requires externally audited pickup, open and goal.",
            "Public front-cell features and pickup-evidence bit are engineered.",
            "Scripted reference provides training labels; skills are not discovered.",
            "Both replay arms add exactly matched cached label presentations.",
            "Source replay and target-extras differ only in extra label domain.",
            "Model seeds and stochastic policy trajectories introduce uncertainty.",
            "Private state is only read by external evaluator, not acting policy.",
            "Workshop and XLand provide diagnostics, not generalization proof.",
        ],
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/m19_matched_replay.yaml")
    )
    parser.add_argument("--output", type=Path,
                        default=Path("runs/m19_matched_replay/summary.json"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("M19 requires --execute for new native actions and fresh training")
    report = run(args.config, args.output)
    print(json.dumps({
        "source_commit": report["source_commit"],
        "fresh_native_actions": report["fresh_native_actions"],
        "optimizer_step_calls": report["optimizer_step_calls"],
        "fresh_model_fits": report["fresh_model_fits"],
        "gates": report["gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
