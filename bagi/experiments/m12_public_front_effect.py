"""M12: native public-front-cell effect prediction versus action-only frequency.

The learned output is NEXT PUBLIC IMAGE CHANGE, never physical task success.
All input features come from a current public view and an executed native action.
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
from bagi.experiments.m5_continual_retention import DOORKEY, MEMORY

TASKS = (MEMORY, DOORKEY)
TRAIN = (tuple(range(151101, 151133)), tuple(range(161101, 161133)))
FINAL = (tuple(range(152101, 152117)), tuple(range(162101, 162117)))
SEEDS = (421, 433)
FRONT = (3, 5)
CHANNELS = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)
FEATURES = 7 * CHANNELS
HORIZON = 256
EPOCHS = 12
BATCH = 256
RATE = 0.01
PRIOR = 20.0
OUT = "runs/m12_public_front_effect"


def frozen_config(config: dict) -> dict:
    expected = {
        "registered_study": "m12",
        "tasks": list(TASKS),
        "train_seeds": [list(x) for x in TRAIN],
        "final_seeds": [list(x) for x in FINAL],
        "model_seeds": list(SEEDS),
        "native_episode_horizon": HORIZON,
        "front_cell": list(FRONT),
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "learning_rate": RATE,
        "front_table_prior_strength": PRIOR,
        "primary_action": 2,
        "primary_forward_brier_gain": 0.01,
        "min_forward_positives_and_negatives": 20,
        "output_dir": OUT,
    }
    if config != expected or len(set(sum(TRAIN + FINAL, ()))) != 96:
        raise ValueError("M12 preregistered source/evaluation protocol changed")
    return config


def public_feature(image: np.ndarray, action: int) -> tuple[np.ndarray, int]:
    if image.shape != (7, 7, 3) or type(action) is not int or not 0 <= action < 7:
        raise ValueError("M12 requires public 7x7 image and legal native action")
    obj, color, state = [int(x) for x in image[FRONT]]
    if not (0 <= obj < len(OBJECT_TO_IDX)
            and 0 <= color < len(COLOR_TO_IDX)
            and 0 <= state < len(STATE_TO_IDX)):
        raise ValueError("M12 unrecognized native public tile code")
    values = np.zeros(FEATURES, dtype=np.float32)
    start = action * CHANNELS
    for offset in (
        obj, len(OBJECT_TO_IDX) + color,
        len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state
    ):
        values[start + offset] = 1
    return values, obj


def visible_change(before: np.ndarray, after: np.ndarray) -> int:
    if before.shape != (7, 7, 3) or after.shape != (7, 7, 3):
        raise ValueError("M12 effect target must be from actual public images")
    return int(np.any(before != after))


def collect(task: str, seeds: tuple[int, ...]) -> tuple[list[dict], dict]:
    if task not in TASKS or seeds not in (TRAIN[TASKS.index(task)], FINAL[TASKS.index(task)]):
        raise ValueError("M12 attempted unregistered native episode split")
    episodes = []
    counts = np.zeros(7, dtype=np.int64)
    effects = np.zeros(7, dtype=np.int64)
    for seed in seeds:
        env = gym.wrappers.TimeLimit(gym.make(task), max_episode_steps=HORIZON)
        try:
            obs, _ = env.reset(seed=seed)
            if env.action_space.n != 7 or obs["image"].shape != (7, 7, 3):
                raise ValueError("M12 native public observation contract changed")
            rng = np.random.default_rng(7_000_000 + seed)
            features, labels, actions, objects = [], [], [], []
            reward_goals = 0
            finished = False
            for _ in range(HORIZON):
                action = int(rng.integers(7))
                before = np.asarray(obs["image"]).copy()
                x, front_object = public_feature(before, action)
                following, reward, ended, truncated, _ = env.step(action)
                y = visible_change(before, np.asarray(following["image"]))
                features.append(x)
                labels.append(y)
                actions.append(action)
                objects.append(front_object)
                counts[action] += 1
                effects[action] += y
                reward_goals += int(bool(ended) and float(reward) > 0)
                obs = following
                if ended or truncated:
                    finished = True
                    break
            if not finished or not actions:
                raise ValueError("M12 unfinished native collection episode")
            episodes.append({
                "seed": seed,
                "x": np.stack(features),
                "y": np.asarray(labels, dtype=np.float32),
                "actions": np.asarray(actions, dtype=np.int64),
                "objects": np.asarray(objects, dtype=np.int64),
                "rewarded_terminal": reward_goals,
            })
        finally:
            env.close()
    if np.any(effects > counts) or sum(counts) != sum(
        len(ep["actions"]) for ep in episodes
    ):
        raise ValueError("M12 native executed-action histogram inconsistent")
    return episodes, {
        "task": task, "seeds": list(seeds), "episodes": len(episodes),
        "seed_sha256": hashlib.sha256(
            "".join(f"{seed}\n" for seed in seeds).encode()
        ).hexdigest(),
        "native_actions": int(sum(counts)),
        "action_counts": counts.tolist(),
        "public_change_by_action": effects.tolist(),
        "rewarded_terminals_not_physically_audited": sum(
            ep["rewarded_terminal"] for ep in episodes
        ),
    }


def unpack(episodes: list[dict]) -> tuple[np.ndarray, ...]:
    return tuple(
        np.concatenate([ep[key] for ep in episodes])
        for key in ("x", "y", "actions", "objects")
    )


class FrontLogit(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(FEATURES, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != FEATURES:
            raise ValueError("Expected public action-by-front feature vector")
        return self.linear(x).flatten()


def fit(model: FrontLogit, episodes: list[dict], seed: int,
        calls: dict) -> dict:
    if seed not in SEEDS or not episodes:
        raise ValueError("M12 fresh model seed or dataset missing")
    x, y, _, _ = unpack(episodes)
    features = torch.from_numpy(x)
    targets = torch.from_numpy(y)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=RATE)
    before = calls["optimizer_step_calls"]
    seen = 0
    for _ in range(EPOCHS):
        for selected in torch.randperm(len(x), generator=generator).split(BATCH):
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(features[selected]), targets[selected]
            )
            loss.backward()
            optimizer.step()
            calls["optimizer_step_calls"] += 1
            seen += len(selected)
    return {
        "optimizer_step_calls": calls["optimizer_step_calls"] - before,
        "cached_training_sample_presentations": seen,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def fitted_tables(episodes: list[dict]) -> dict:
    _, y, action, obj = unpack(episodes)
    counts = np.bincount(action, minlength=7)
    positives = np.bincount(action, weights=y, minlength=7)
    action_rates = (positives + 1) / (counts + 2)
    index = action * len(OBJECT_TO_IDX) + obj
    front_counts = np.bincount(
        index, minlength=7 * len(OBJECT_TO_IDX)
    ).reshape(7, len(OBJECT_TO_IDX))
    front_positives = np.bincount(
        index, weights=y, minlength=7 * len(OBJECT_TO_IDX)
    ).reshape(7, len(OBJECT_TO_IDX))
    front_rates = (front_positives + PRIOR * action_rates[:, None]) / (
        front_counts + PRIOR
    )
    return {
        "action_rates": action_rates.tolist(),
        "front_object_rates": front_rates.tolist(),
        "training_actions": len(action),
    }


def split_metrics(y: np.ndarray, a: np.ndarray, pred: np.ndarray) -> dict:
    if len(y) != len(a) or len(y) != len(pred):
        raise ValueError("M12 evaluation step count mismatch")
    if not np.isfinite(pred).all() or np.any((pred < 0) | (pred > 1)):
        raise ValueError("Invalid M12 prediction probabilities")
    result = {}
    for name, selection in (
        ("all", np.ones(len(y), dtype=bool)),
        ("turn", a < 2),
        ("forward", a == 2),
        ("interaction", (a >= 3) & (a <= 5)),
        ("nonturn", (a >= 2) & (a <= 5)),
    ):
        actual = y[selection]
        prediction = pred[selection]
        result[name] = {
            "native_actions": len(actual), "positives": int(actual.sum()),
            "negatives": len(actual) - int(actual.sum()),
            "brier": float(np.square(actual - prediction).mean())
            if len(actual) else None,
            "accuracy": float(((prediction >= 0.5) == (actual >= 0.5)).mean())
            if len(actual) else None,
        }
    return result


def evaluate(episodes: list[dict], table: dict, *,
             model: FrontLogit | None = None, conditional: bool = False) -> dict:
    x, y, action, obj = unpack(episodes)
    if model is not None:
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(x))).numpy()
    elif conditional:
        probs = np.asarray(table["front_object_rates"])[action, obj]
    else:
        probs = np.asarray(table["action_rates"])[action]
    return split_metrics(y, action, probs)


def save_numeric(model: FrontLogit, path: Path) -> dict:
    if path.exists():
        raise FileExistsError("Refusing overwrite of numeric M12 checkpoint")
    arrays_out, manifest = {}, {}
    for i, (name, param) in enumerate(model.state_dict().items()):
        array = param.detach().cpu().numpy()
        if array.dtype.kind not in "fiub" or not np.isfinite(array).all():
            raise ValueError("M12 invalid numeric checkpoint")
        key = f"tensor_{i:03d}"
        arrays_out[key] = array
        manifest[key] = {
            "name": name, "shape": list(array.shape), "dtype": str(array.dtype)
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays_out)
    return {
        "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "members": manifest, "allow_pickle": False,
    }


def pilot(metrics: dict, meta: dict) -> dict:
    expected = {"action_only", "front_table", "logistic_421", "logistic_433"}
    if set(metrics) != expected:
        raise ValueError("M12 must include both new fits and both simple controls")
    coverage, improvements = {}, {}
    for task in TASKS:
        final = meta[f"{task}:final"]
        pos = final["public_change_by_action"][2]
        total = final["action_counts"][2]
        coverage[task] = pos >= 20 and total - pos >= 20
        reference = metrics["action_only"][task]["forward"]["brier"]
        improvements[task] = {
            name: reference - metrics[name][task]["forward"]["brier"]
            for name in ("front_table", "logistic_421", "logistic_433")
        }
    return {
        "forward_coverage": coverage,
        "forward_brier_improvement_over_action_only": improvements,
        "pilot_gate_pass": (
            all(coverage.values()) and all(
                improvements[task][name] >= 0.01 for task in TASKS
                for name in ("logistic_421", "logistic_433")
            )
        ),
        "not_physical_task_success": True,
        "not_continual_skill_learning": True,
    }


def run(config: Path, output: Path) -> dict:
    if output.exists() or (output.parent / "progress.json").exists():
        raise FileExistsError("Preserve all existing M12 run evidence")
    raw = config.read_bytes()
    frozen_config(yaml.safe_load(raw))
    if output.parent.as_posix() != OUT:
        raise ValueError("M12 unregistered output directory")
    source_commit = admitted_source_commit()
    started = time.monotonic()
    torch.set_num_threads(1)
    acquired, trained = {}, []
    calls = {"optimizer_step_calls": 0}
    native_actions = 0
    phase = "training_data"
    try:
        training = []
        for task, seeds in zip(TASKS, TRAIN):
            rows, info = collect(task, seeds)
            training.extend(rows)
            acquired[f"{task}:train"] = info
            native_actions += info["native_actions"]
        table = fitted_tables(training)
        phase = "fresh_public_predictor_fits"
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = FrontLogit()
            stats = fit(model, training, seed, calls)
            if stats["optimizer_step_calls"] <= 0 or stats["parameter_count"] != 141:
                raise ValueError("M12 optimizer/parameter accounting invalid")
            checkpoint = save_numeric(
                model, output.parent / f"front-logit-seed{seed}.npz"
            )
            trained.append((seed, model, stats, checkpoint))
            save_progress(output.parent / "progress.json", {
                "status": "partial_not_completed", "phase": phase,
                "source_commit": source_commit, "acquisition": acquired,
                "new_native_actions": native_actions,
                "optimizer_step_calls": calls["optimizer_step_calls"],
                "completed_models": len(trained),
            })
        if len(trained) != len(SEEDS) or {x[0] for x in trained} != set(SEEDS):
            raise ValueError("Final episode stream is locked until all fits complete")
        phase = "independent_final_collection"
        final_rows = {}
        for task, seeds in zip(TASKS, FINAL):
            rows, info = collect(task, seeds)
            final_rows[task] = rows
            acquired[f"{task}:final"] = info
            native_actions += info["native_actions"]
        metrics = {
            "action_only": {
                task: evaluate(final_rows[task], table) for task in TASKS
            },
            "front_table": {
                task: evaluate(final_rows[task], table, conditional=True)
                for task in TASKS
            },
        }
        for seed, model, _, _ in trained:
            metrics[f"logistic_{seed}"] = {
                task: evaluate(final_rows[task], table, model=model)
                for task in TASKS
            }
        decision = pilot(metrics, acquired)
    except BaseException as exc:
        save_progress(output.parent / "progress.json", {
            "status": "interrupted_not_completed", "phase": phase,
            "failure_type": type(exc).__name__,
            "source_commit": source_commit, "acquisition": acquired,
            "new_native_actions": native_actions,
            "optimizer_step_calls": calls["optimizer_step_calls"],
            "completed_models": len(trained),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        raise
    result = {
        "registered_study": "m12",
        "study": "M12 public front-cell forward-effect prediction versus action-only",
        "status": "completed_public_effect_only_independent_diagnostic",
        "source_commit": source_commit,
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "versions": {
            p: version(p) for p in ("gymnasium", "minigrid", "torch", "numpy")
        },
        "acquisition": acquired,
        "new_native_actions": native_actions,
        "native_train_actions": sum(
            acquired[f"{task}:train"]["native_actions"] for task in TASKS
        ),
        "native_independent_test_actions": sum(
            acquired[f"{task}:final"]["native_actions"] for task in TASKS
        ),
        "optimizer_step_calls": calls["optimizer_step_calls"],
        "new_model_fits": len(trained),
        "teacher_queries": 0,
        "trained_action_rates": table,
        "models": [
            {"seed": seed, "training": stats, "numeric_checkpoint": checkpoint}
            for seed, _, stats, checkpoint in trained
        ],
        "independent_action_conditioned_metrics": metrics,
        "pilot": decision,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "limitations": [
            "Public-view changes are NOT effective pickup/opening or physical goal success.",
            "Both domains share MiniGrid actions/encoding; not broad general transfer.",
            "Uniform random behavior is not the learned predictor acting.",
            "The front-cell feature is MiniGrid-specific and agent-centered.",
            "Final transitions within one episode are correlated.",
            "Two model seeds share final evaluation episodes.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/m12_public_front_effect.yaml")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/m12_public_front_effect/summary.json"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("M12 fresh native acquisition and fitting requires --execute")
    result = run(args.config, args.output)
    print(json.dumps({
        "status": result["status"], "source_commit": result["source_commit"],
        "new_native_actions": result["new_native_actions"],
        "optimizer_step_calls": result["optimizer_step_calls"],
        "pilot": result["pilot"],
    }, indent=2))


if __name__ == "__main__":
    main()
