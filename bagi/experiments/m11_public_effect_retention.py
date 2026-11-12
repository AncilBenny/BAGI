"""M11: self-supervised public-action effect prediction and replay retention.

Supervised diagnostic on real Memory and DoorKey actions; not a control policy.
The binary target is NEXT public image change, never private physical state.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
import yaml
from torch import nn

from bagi.experiments.m1_goal_learning import admitted_source_commit, save_progress
from bagi.experiments.m5_continual_retention import DOORKEY, MEMORY, PublicGoalActions

TASKS = (MEMORY, DOORKEY)
SEEDS = (317, 331)
TRAIN = (tuple(range(131101, 131125)), tuple(range(141101, 141125)))
FINAL = (tuple(range(132101, 132113)), tuple(range(142101, 142113)))
MAX_EPISODE = 256
EPOCHS = 4
HIDDEN = 32
RATE = 0.001
WIDTH = 250
OUT = "runs/m11_public_effect_retention"


@dataclass
class Episode:
    seed: int
    x: torch.Tensor
    y: torch.Tensor
    actions: list[int]
    rewarded_goals: int

    @property
    def steps(self) -> int:
        return len(self.actions)


class EffectGRU(nn.Module):
    """Established PyTorch recurrent predictor, reset at every episode."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(WIDTH, HIDDEN, batch_first=True)
        self.readout = nn.Linear(HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != WIDTH:
            raise ValueError("M11 input requires legal public observations and actions")
        h, _ = self.gru(x.unsqueeze(0))
        return self.readout(h).flatten()


def validate_config(c: dict) -> dict:
    expected = {
        "registered_study": "m11",
        "task_order": list(TASKS),
        "model_seeds": list(SEEDS),
        "train_seeds": [list(a) for a in TRAIN],
        "final_seeds": [list(a) for a in FINAL],
        "max_episode_steps": MAX_EPISODE,
        "source_epochs": EPOCHS,
        "target_epochs": EPOCHS,
        "hidden_size": HIDDEN,
        "learning_rate": RATE,
        "replay_old_loss_weight": 0.5,
        "target": "next_public_image_changed",
        "output_dir": OUT,
    }
    if c != expected:
        raise ValueError("M11 fixed public-effect experiment configuration changed")
    if len(set(sum(TRAIN + FINAL, ()))) != sum(map(len, TRAIN + FINAL)):
        raise ValueError("M11 split contamination")
    return c


def public_input(obs: np.ndarray, action: int) -> np.ndarray:
    if obs.shape != (243,) or not np.isfinite(obs).all():
        raise ValueError("M11 public image/mission/action-history width changed")
    if type(action) is not int or not 0 <= action < 7:
        raise ValueError("Not a valid actually executed native action")
    onehot = np.zeros(7, dtype=np.float32)
    onehot[action] = 1
    return np.concatenate((obs, onehot)).astype(np.float32)


def public_label(before: np.ndarray, after: np.ndarray) -> float:
    if before.shape != (243,) or after.shape != (243,):
        raise ValueError("Missing public 243-dimensional observation")
    return float(np.any(before[:147] != after[:147]))


def collect(task: str, seeds: tuple[int, ...]) -> tuple[list[Episode], dict]:
    if task not in TASKS:
        raise ValueError("Only preregistered native environments")
    ix = TASKS.index(task)
    if seeds not in (TRAIN[ix], FINAL[ix]):
        raise ValueError("Only registered train or independent final layouts")
    rows = []
    for seed in seeds:
        env = PublicGoalActions(gym.wrappers.TimeLimit(
            gym.make(task), max_episode_steps=MAX_EPISODE
        ))
        try:
            obs, _ = env.reset(seed=seed)
            if env.observation_space.shape != (243,) or env.action_space.n != 7:
                raise ValueError("M11 environments must share public action interface")
            rng = np.random.default_rng(5_000_000 + seed)
            x, y, actions = [], [], []
            goals = 0
            finished = False
            for _ in range(MAX_EPISODE):
                actual = int(rng.integers(7))
                features = public_input(obs, actual)
                following, reward, done, truncated, _ = env.step(actual)
                x.append(features)
                y.append(public_label(obs, following))
                actions.append(actual)
                goals += int(bool(done) and reward > 0)
                obs = following
                if done or truncated:
                    finished = True
                    break
            if not finished or not actions:
                raise ValueError("Native M11 trajectory did not finish")
            rows.append(Episode(
                seed=seed,
                x=torch.from_numpy(np.stack(x)),
                y=torch.tensor(y, dtype=torch.float32),
                actions=actions, rewarded_goals=goals,
            ))
        finally:
            env.close()
    counts = [sum(ep.actions.count(a) for ep in rows) for a in range(7)]
    changes = sum(int(ep.y.sum().item()) for ep in rows)
    total = sum(ep.steps for ep in rows)
    if sum(counts) != total:
        raise ValueError("M11 native action tally mismatch")
    return rows, {
        "task": task, "seeds": list(seeds), "episodes": len(rows),
        "seed_sha256": hashlib.sha256(
            "".join(f"{s}\n" for s in seeds).encode()
        ).hexdigest(),
        "native_actions": total, "action_counts": counts,
        "public_image_changes": changes,
        "rewarded_terminations_without_physical_audit": sum(
            ep.rewarded_goals for ep in rows
        ),
    }


def learn(model: EffectGRU, data: list[Episode], *, seed: int,
          replay: list[Episode] | None = None) -> dict:
    if not data or (replay is not None and not replay):
        raise ValueError("M11 training requires nonempty native episodes")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=RATE)
    rng = np.random.default_rng(seed)
    updates = 0
    losses = 0.0
    replay_presentations = 0
    for _ in range(EPOCHS):
        for index in rng.permutation(len(data)):
            row = data[int(index)]
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(row.x), row.y)
            if replay is not None:
                old = replay[int(rng.integers(len(replay)))]
                old_loss = nn.functional.binary_cross_entropy_with_logits(
                    model(old.x), old.y
                )
                loss = (loss + old_loss) / 2
                replay_presentations += old.steps
            loss.backward()
            optimizer.step()
            updates += 1
            losses += float(loss.detach())
    return {
        "optimizer_step_calls": updates,
        "new_task_sample_presentations": EPOCHS * sum(x.steps for x in data),
        "old_cached_replay_sample_presentations": replay_presentations,
        "mean_training_loss": losses / updates,
    }


def evaluate(model: EffectGRU, episodes: list[Episode]) -> dict:
    if not episodes:
        raise ValueError("M11 final effect episodes missing")
    model.eval()
    total, positives, errors, correct = 0, 0, 0.0, 0
    with torch.no_grad():
        for ep in episodes:
            probs = torch.sigmoid(model(ep.x))
            errors += float((probs - ep.y).square().sum())
            correct += int(((probs >= 0.5) == (ep.y >= 0.5)).sum())
            positives += int(ep.y.sum().item())
            total += ep.steps
    return {
        "episodes": len(episodes),
        "native_heldout_actions": total,
        "public_change_positives": positives,
        "prevalence": positives / total,
        "brier": errors / total,
        "accuracy": correct / total,
        "zero_change_brier": positives / total,
    }


def action_baseline(train: list[Episode], heldout: list[Episode]) -> float:
    counts = np.zeros(7)
    positives = np.zeros(7)
    for ep in train:
        for action, label in zip(ep.actions, ep.y.tolist()):
            counts[action] += 1
            positives[action] += label
    rates = (positives + 1) / (counts + 2)
    errors = [
        (rates[action] - label) ** 2
        for ep in heldout for action, label in zip(ep.actions, ep.y.tolist())
    ]
    return float(np.mean(errors))


def numeric_checkpoint(model: EffectGRU, path: Path) -> dict:
    if path.exists():
        raise FileExistsError("M11 checkpoint already exists")
    state, members = {}, {}
    for i, (name, tensor) in enumerate(model.state_dict().items()):
        value = tensor.detach().cpu().numpy()
        if value.dtype.kind not in "fiub" or not np.isfinite(value).all():
            raise ValueError("M11 checkpoint must be finite and numeric-only")
        key = f"tensor_{i:03d}"
        state[key] = value
        members[key] = {"name": name, "shape": list(value.shape),
                        "dtype": str(value.dtype)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **state)
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "allow_pickle": False, "members": members}


def pilot(records: list[dict], sources: dict) -> dict:
    if len(records) != len(SEEDS) or {r["seed"] for r in records} != set(SEEDS):
        raise ValueError("M11 pilot requires both initialization seeds")
    gains, costs = [], []
    for r in records:
        after = r["final"]
        gains.append(after["naive"]["memory"]["brier"]
                     - after["replay"]["memory"]["brier"])
        costs.append(after["replay"]["doorkey"]["brier"]
                     - after["naive"]["doorkey"]["brier"])
    positive = all(x["public_image_changes"] > 0 for x in sources.values())
    return {
        "old_task_brier_gain_per_seed": gains,
        "new_task_brier_cost_per_seed": costs,
        "both_tasks_have_public_changes": positive,
        "pilot_gate_pass": (
            positive and all(x >= 0.015 for x in gains)
            and all(x <= 0.02 for x in costs)
        ),
        "not_goal_learning": True,
        "not_physical_task_success": True,
    }


def run(config: Path, output: Path) -> dict:
    if output.exists() or (output.parent / "progress.json").exists():
        raise FileExistsError("M11 source output is append-only per experiment")
    raw = config.read_bytes()
    validate_config(yaml.safe_load(raw))
    if output.parent.as_posix() != OUT:
        raise ValueError("M11 nonregistered output location")
    source_commit = admitted_source_commit()
    torch.set_num_threads(1)
    begin = time.monotonic()
    phase = "train_acquisition"
    acquired, records, models = {}, [], []
    total_updates = 0
    native_actions = 0
    try:
        old_data, old_meta = collect(TASKS[0], TRAIN[0])
        acquired["memory_train"] = old_meta
        native_actions += old_meta["native_actions"]
        new_data, new_meta = collect(TASKS[1], TRAIN[1])
        acquired["doorkey_train"] = new_meta
        native_actions += new_meta["native_actions"]
        phase = "source_and_target_training"
        for seed in SEEDS:
            torch.manual_seed(seed)
            source = EffectGRU()
            source_fit = learn(source, old_data, seed=seed)
            naive, replay = copy.deepcopy(source), copy.deepcopy(source)
            torch.manual_seed(seed)
            scratch = EffectGRU()
            naive_fit = learn(naive, new_data, seed=seed + 10_000)
            replay_fit = learn(replay, new_data, seed=seed + 10_000, replay=old_data)
            scratch_fit = learn(scratch, new_data, seed=seed + 10_000)
            fits = {
                "source": source_fit, "naive": naive_fit,
                "replay": replay_fit, "scratch": scratch_fit,
            }
            expected_updates = len(TRAIN[0]) * EPOCHS
            if any(fit["optimizer_step_calls"] != expected_updates
                   for fit in fits.values()):
                raise ValueError("M11 actual optimizer step count unexpected")
            total_updates += sum(fit["optimizer_step_calls"] for fit in fits.values())
            group = {
                "source": source, "naive": naive,
                "replay": replay, "scratch": scratch,
            }
            checkpoints = {
                name: numeric_checkpoint(model, output.parent / f"{name}-{seed}.npz")
                for name, model in group.items()
            }
            row = {"seed": seed, "fits": fits, "checkpoints": checkpoints}
            models.append(group)
            records.append(row)
            save_progress(output.parent / "progress.json", {
                "status": "partial_not_completed", "phase": phase,
                "source_commit": source_commit, "acquisition": acquired,
                "native_actions": native_actions, "optimizer_step_calls": total_updates,
                "records": records,
            })
        # Only now access independent effect-evaluation episodes.
        phase = "independent_effect_test"
        old_final, old_final_meta = collect(TASKS[0], FINAL[0])
        acquired["memory_final"] = old_final_meta
        native_actions += old_final_meta["native_actions"]
        new_final, new_final_meta = collect(TASKS[1], FINAL[1])
        acquired["doorkey_final"] = new_final_meta
        native_actions += new_final_meta["native_actions"]
        baseline = {
            "memory_brier": action_baseline(
                old_data + new_data, old_final
            ),
            "doorkey_brier": action_baseline(
                old_data + new_data, new_final
            ),
        }
        for r, group in zip(records, models):
            r["final"] = {
                name: {
                    "memory": evaluate(model, old_final),
                    "doorkey": evaluate(model, new_final),
                }
                for name, model in group.items()
            }
    except BaseException as error:
        save_progress(output.parent / "progress.json", {
            "status": "interrupted_not_completed", "phase": phase,
            "failure_type": type(error).__name__, "source_commit": source_commit,
            "acquisition": acquired, "native_actions": native_actions,
            "optimizer_step_calls": total_updates, "records": records,
            "elapsed_seconds": round(time.monotonic() - begin, 3),
        })
        raise
    result = {
        "study": "M11 continual self-supervised public effect prediction",
        "registered_study": "m11",
        "status": "completed_independent_public_effect_retention_diagnostic",
        "source_commit": source_commit,
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "versions": {p: version(p) for p in ("gymnasium", "minigrid", "torch", "numpy")},
        "acquisition": acquired,
        "native_actions_total": native_actions,
        "native_training_dataset_actions": (
            old_meta["native_actions"] + new_meta["native_actions"]
        ),
        "native_independent_effect_test_actions": (
            old_final_meta["native_actions"] + new_final_meta["native_actions"]
        ),
        "independent_effect_test_episodes": len(old_final) + len(new_final),
        "optimizer_step_calls": total_updates,
        "source_fits": len(SEEDS), "target_continuations": 2 * len(SEEDS),
        "scratch_target_fits": len(SEEDS), "teacher_queries": 0,
        "action_conditioned_baseline": baseline,
        "pilot": pilot(records, {
            "source": old_meta, "target": new_meta,
            "source_final": old_final_meta, "target_final": new_final_meta,
        }),
        "records": records,
        "elapsed_seconds": round(time.monotonic() - begin, 3),
        "limitations": [
            "Public image changes are not physical manipulation or goal success.",
            "No effect predictor was deployed as a control policy.",
            "Both tasks share MiniGrid encoding; no cross-domain generality proven.",
            "Random behavior distribution may not cover actual valid interactions.",
            "Rehearsal uses extra cached-source FLOPs per optimizer update.",
            "Only two model initialization seeds and one task order.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path,
                   default=Path("configs/m11_public_effect_retention.yaml"))
    p.add_argument("--output", type=Path,
                   default=Path("runs/m11_public_effect_retention/summary.json"))
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if not args.execute:
        p.error("M11 fresh native acquisition and fitting need explicit --execute")
    result = run(args.config, args.output)
    print(json.dumps({
        "status": result["status"], "source_commit": result["source_commit"],
        "native_actions_total": result["native_actions_total"],
        "optimizer_step_calls": result["optimizer_step_calls"],
        "pilot": result["pilot"],
    }, indent=2))


if __name__ == "__main__":
    main()
