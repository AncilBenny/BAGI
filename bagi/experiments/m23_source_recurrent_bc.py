"""M23 Stage14 native-source public GRU BC, stateless versus recurrent memory.

Policies receive only native PUBLIC image, mission, direction and own visibly
witnessed pickup bit. Hidden native physics belongs to evaluator-only audits.
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
from torch import nn

from bagi.experiments.m1_goal_learning import admitted_source_commit
from bagi.experiments.m13_public_interaction_coverage import Controller
from bagi.experiments.m15_public_action_bc import class_weights, save_numeric
from bagi.experiments.m23_source_feasibility import HORIZON
from bagi.experiments.m23_source_mission_bc import make_native_audit, make_source_actor
from bagi.experiments.m23_source_public_vision_bc import WIDTH, public_visual_features
from bagi.experiments.m23_study_contract import (
    RECURRENT_MONITOR_LAYOUTS,
    RECURRENT_TRAIN_LAYOUTS,
    validate_split_contract,
)

OUT = "runs/m23_source_recurrent_bc"
TASKS = ("doorkey", "fetch")
ARMS = ("stateless_gru", "recurrent_gru")
MODEL_SEEDS = (941, 947, 953)
EPOCHS = 12
EPISODE_BATCH = 16
HIDDEN = 64
LR = 0.003
BOOTSTRAP_SEED = 231401
MIN_GAIN = 12
MIN_MONITOR_DOOR_GOALS = 16
ACTOR_MARGIN = 24


class PublicGRU(nn.Module):
    """Same GRU tensors, capacity and initialization for both temporal arms."""

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(WIDTH, HIDDEN, batch_first=True)
        self.logits = nn.Linear(HIDDEN, 7)

    def forward(self, x: torch.Tensor, *, reset_each: bool) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != WIDTH:
            raise ValueError("Stage14 GRU requires batched public feature sequences")
        if reset_each:
            # A genuinely stateless policy: zero GRU hidden for EVERY public cell view.
            b, t, _ = x.shape
            out, _hidden = self.gru(x.reshape(b * t, 1, WIDTH))
            out = out.reshape(b, t, HIDDEN)
        else:
            # One recurrent history per episode, zero GRU hidden at episode start.
            out, _hidden = self.gru(x)
        return self.logits(out)

    def native_step(
        self, vector: np.ndarray, state: torch.Tensor | None, *, reset_each: bool
    ) -> tuple[int, torch.Tensor | None]:
        if self.training or vector.shape != (WIDTH,) or not np.isfinite(vector).all():
            raise ValueError("Stage14 public native policy must be eval-only and finite")
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(vector, dtype=np.float32)).reshape(
                1, 1, WIDTH
            )
            if reset_each:
                output, _ = self.gru(x)
                next_state = None
            else:
                if state is not None and state.shape != (1, 1, HIDDEN):
                    raise ValueError("Stage14 recurrent state shape not registered")
                output, next_state = self.gru(x, state)
                next_state = next_state.detach()
            action = int(torch.argmax(self.logits(output)[0, 0]).item())
        if action not in range(7):
            raise ValueError("Stage14 native GRU action outside public interface")
        return action, next_state


def train_episode(task: str, layout: int) -> dict:
    if task not in TASKS or layout not in RECURRENT_TRAIN_LAYOUTS[task]:
        raise ValueError("Stage14 source TRAIN requires fresh preregistered layout")
    actor = make_source_actor(task)
    rng = np.random.default_rng(8_800_000 + layout)
    audit = make_native_audit(task)
    vectors: list[np.ndarray] = []
    actions: list[int] = []
    finished = False
    try:
        obs, _ = audit.reset(seed=layout)
        for _ in range(HORIZON):
            vectors.append(public_visual_features(obs, actor.key_observed_after_pickup))
            action = int(actor.choose(obs, rng))
            if action not in range(7):
                raise ValueError("Stage14 source teacher illegal native action")
            nxt, _reward, done, timeout, _ = audit.step(action)
            actor.update(obs, action, nxt)
            actions.append(action)
            obs = nxt
            if done or timeout:
                finished = True
                break
        truth = audit.summary()
        if (not finished or not actions or len(actions) != len(vectors)
                or truth["native_actions"] != len(actions)
                or sum(truth["action_counts"].values()) != len(actions)):
            raise ValueError("Stage14 native TRAIN temporal labels not truly executed")
        x = np.stack(vectors).astype(np.float32, copy=False)
        y = np.asarray(actions, dtype=np.int64)
        return {
            "task": task,
            "layout_seed": layout,
            "x": x,
            "y": y,
            "actual_native_steps": len(actions),
            "actual_executed_action_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
            "public_episode_x_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            "true_executed_episode_y_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
            "actual_native_action_histogram":
                [truth["action_counts"][str(i)] for i in range(7)],
            "external_physical_full_goal": bool(truth["full_physical_goal"]),
        }
    finally:
        audit.env.close()


def collect_train() -> tuple[list[dict], np.ndarray, dict]:
    validate_split_contract()
    episodes = [
        train_episode(task, layout)
        for task in TASKS for layout in RECURRENT_TRAIN_LAYOUTS[task]
    ]
    if len(episodes) != 192:
        raise ValueError("Stage14 source TRAIN requires 192 actual native episodes")
    ys = np.concatenate([row["y"] for row in episodes])
    native_steps = sum(row["actual_native_steps"] for row in episodes)
    if not len(ys) or len(ys) != native_steps:
        raise ValueError("Stage14 TRAIN source native steps not conserved")
    hist = np.bincount(ys, minlength=7).tolist()
    if sum(hist) != native_steps:
        raise ValueError("Stage14 actual TRAIN action histogram invalid")
    digest = hashlib.sha256()
    for row in episodes:
        digest.update(row["x"].tobytes())
    return episodes, ys, {
        "native_source_train_episode_count": len(episodes),
        "actual_new_native_train_steps": native_steps,
        "native_source_train_executed_action_histogram": hist,
        "native_source_train_public_x_sha256": digest.hexdigest(),
        "actual_source_executed_y_sha256": hashlib.sha256(ys.tobytes()).hexdigest(),
        "per_episode": [
            {k: v for k, v in row.items() if k not in ("x", "y")}
            for row in episodes
        ],
    }


def fit_model(
    episodes: list[dict], model_seed: int, arm: str, weights: torch.Tensor
) -> tuple[PublicGRU, dict]:
    if model_seed not in MODEL_SEEDS or arm not in ARMS or len(episodes) != 192:
        raise ValueError("Stage14 GRU training arm/seed/episodes not preregistered")
    if weights.shape != (7,) or not torch.isfinite(weights).all():
        raise ValueError("Stage14 TRAIN source class-weights invalid")
    total_tokens = 0
    for row in episodes:
        x, y = row["x"], row["y"]
        if (not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)
                or x.ndim != 2 or x.shape[1] != WIDTH
                or y.ndim != 1 or len(x) != len(y)
                or not 1 <= len(y) <= HORIZON or not np.isfinite(x).all()
                or set(np.unique(y)) - set(range(7))):
            raise ValueError("Stage14 public episode/native label alignment invalid")
        total_tokens += len(y)
    torch.manual_seed(model_seed)
    model = PublicGRU()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(10_000 + model_seed)
    steps, presentations, loss_total = 0, 0, 0.0
    started = time.perf_counter()
    for _ in range(EPOCHS):
        for indices in torch.randperm(len(episodes), generator=generator).split(
            EPISODE_BATCH
        ):
            selected = [episodes[int(i)] for i in indices]
            x = nn.utils.rnn.pad_sequence(
                [torch.from_numpy(row["x"]) for row in selected], batch_first=True
            )
            y = nn.utils.rnn.pad_sequence(
                [torch.from_numpy(row["y"]) for row in selected],
                batch_first=True, padding_value=-100,
            )
            valid = y != -100
            if int(valid.sum().item()) != sum(len(row["y"]) for row in selected):
                raise ValueError("Stage14 real public sequence padding not conserved")
            optimizer.zero_grad()
            logits = model(x, reset_each=(arm == "stateless_gru"))
            loss = nn.functional.cross_entropy(
                logits[valid], y[valid], weight=weights
            )
            if not torch.isfinite(loss):
                raise ValueError("Stage14 nonfinite supervised native source CE")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps += 1
            presentations += int(valid.sum().item())
            loss_total += float(loss.detach())
    if steps != EPOCHS * 12 or presentations != EPOCHS * total_tokens:
        raise ValueError("Stage14 equal TRAIN optimizer/label exposures failed")
    model.eval()
    return model, {
        "model_seed": model_seed, "arm": arm,
        "architecture": "GRU(full_public_WIDTH,hidden64,batch_first) Linear(64,7)",
        "public_feature_width": WIDTH,
        "hidden_units": HIDDEN,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "native_source_train_episodes": len(episodes),
        "actual_source_train_label_count": total_tokens,
        "class_balanced": True, "fixed_source_class_weights": weights.tolist(),
        "epochs": EPOCHS, "episode_minibatch": EPISODE_BATCH,
        "adam_learning_rate": LR, "gradient_norm_clip": 1.0,
        "actual_optimizer_step_calls": steps,
        "cached_source_label_presentations_not_native_steps": presentations,
        "mean_training_minibatch_cross_entropy": loss_total / steps,
        "training_seconds": time.perf_counter() - started,
    }


def monitor_episode(
    task: str, layout: int, arm: str, seed: int | None, policy: PublicGRU | None
) -> dict:
    if task not in TASKS or layout not in RECURRENT_MONITOR_LAYOUTS[task]:
        raise ValueError("Stage14 MONITOR source layout outside independent fresh split")
    if arm == "actor_only":
        if seed is not None or policy is not None:
            raise ValueError("Stage14 native actor-only baseline is measured once")
        actor = make_source_actor(task)
    elif arm in ARMS and seed in MODEL_SEEDS and policy is not None:
        if policy.training:
            raise ValueError("Stage14 learned MONITOR policy must be eval-only")
        actor = Controller("public_memory")
    else:
        raise ValueError("Stage14 learned native source evaluation not registered")
    rng = np.random.default_rng(8_800_000 + layout)
    audit = make_native_audit(task)
    action_trace: list[int] = []
    hidden: torch.Tensor | None = None
    reward_sum = 0.0
    finished = False
    try:
        obs, _ = audit.reset(seed=layout)
        for _ in range(HORIZON):
            if policy is None:
                action = int(actor.choose(obs, rng))
            else:
                vector = public_visual_features(obs, actor.key_observed_after_pickup)
                action, hidden = policy.native_step(
                    vector, hidden, reset_each=(arm == "stateless_gru")
                )
            nxt, reward, done, timeout, _ = audit.step(action)
            actor.update(obs, action, nxt)
            action_trace.append(action)
            reward_sum += float(reward)
            obs = nxt
            if done or timeout:
                finished = True
                break
        truth = audit.summary()
        if (not finished or len(action_trace) != truth["native_actions"]
                or sum(truth["action_counts"].values()) != len(action_trace)):
            raise ValueError("Stage14 real native MONITOR physical actions not conserved")
        return {
            "task": task, "layout_seed": layout, "arm": arm, "model_seed": seed,
            "actual_native_steps": len(action_trace),
            "actual_native_executed_action_sha256":
                hashlib.sha256(bytes(action_trace)).hexdigest(),
            "native_executed_action_histogram":
                [truth["action_counts"][str(i)] for i in range(7)],
            "actual_reward_total": reward_sum,
            "external_strict_full_physical_goal": bool(truth["full_physical_goal"]),
            "external_rewarded_terminations": truth["rewarded_terminations"],
            "external_effective_pickup_events": truth["effective_pickup_events"],
            "external_effective_door_open_events":
                truth["effective_door_open_events"] if task == "doorkey" else 0,
            "external_effective_forward_movements":
                truth.get("effective_forward_movements", 0),
            "external_wrong_target_pickup_events":
                truth.get("wrong_target_pickup_events", 0),
        }
    finally:
        audit.env.close()


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Stage14 MONITOR requires actual native source episodes")
    count = sum(row["actual_native_steps"] for row in rows)
    hist = [
        sum(row["native_executed_action_histogram"][i] for row in rows)
        for i in range(7)
    ]
    if sum(hist) != count:
        raise ValueError("Stage14 independent native action histogram invalid")
    return {
        "real_native_episodes": len(rows),
        "real_native_actions": count,
        "native_action_histogram": hist,
        "strict_native_full_physical_goals": sum(
            int(row["external_strict_full_physical_goal"]) for row in rows
        ),
        "reward_positive_episodes": sum(
            int(row["external_rewarded_terminations"] > 0) for row in rows
        ),
        "reward_only_nonphysical_episodes": sum(
            int(row["external_rewarded_terminations"] > 0
                and not row["external_strict_full_physical_goal"]) for row in rows
        ),
        "effective_door_open_episodes": sum(
            int(row["external_effective_door_open_events"] > 0) for row in rows
        ),
        "effective_pickup_events": sum(
            int(row["external_effective_pickup_events"] > 0) for row in rows
        ),
    }


def acquire_monitor(models: dict[tuple[str, int], PublicGRU]) -> dict:
    if set(models) != {(a, s) for a in ARMS for s in MODEL_SEEDS}:
        raise ValueError("Stage14 requires six genuinely trained source GRU fits")
    result = {}
    for task in TASKS:
        layouts = RECURRENT_MONITOR_LAYOUTS[task]
        actor = [
            monitor_episode(task, layout, "actor_only", None, None)
            for layout in layouts
        ]
        result[f"{task}:actor_only"] = {
            "episodes": actor, "summary": summarize(actor)
        }
        for arm in ARMS:
            for seed in MODEL_SEEDS:
                rows = [
                    monitor_episode(task, layout, arm, seed, models[(arm, seed)])
                    for layout in layouts
                ]
                result[f"{task}:{arm}:{seed}"] = {
                    "episodes": rows, "summary": summarize(rows)
                }
    return result


def decide(monitor: dict) -> dict:
    required = {f"{task}:actor_only" for task in TASKS} | {
        f"{task}:{arm}:{seed}"
        for task in TASKS for arm in ARMS for seed in MODEL_SEEDS
    }
    if set(monitor) != required:
        raise ValueError("Stage14 strict native MONITOR missing required controls")
    seed_gains = {s: 0 for s in MODEL_SEEDS}
    task_gains, action_diffs, actor_action_diffs = {}, {}, {}
    paired = {}
    actor_goals = recurrent_goals = recurrent_doorkey_goals = 0
    for task in TASKS:
        layouts = list(RECURRENT_MONITOR_LAYOUTS[task])
        actor = monitor[f"{task}:actor_only"]["episodes"]
        if [r["layout_seed"] for r in actor] != layouts:
            raise ValueError("Stage14 actor MONITOR source layouts unpaired")
        actor_goals += sum(int(r["external_strict_full_physical_goal"]) for r in actor)
        action_diffs[task], actor_action_diffs[task] = {}, {}
        diffs = []
        for seed in MODEL_SEEDS:
            stateless = monitor[f"{task}:stateless_gru:{seed}"]["episodes"]
            recurrent = monitor[f"{task}:recurrent_gru:{seed}"]["episodes"]
            if ([r["layout_seed"] for r in stateless] != layouts
                    or [r["layout_seed"] for r in recurrent] != layouts):
                raise ValueError("Stage14 GRU learned source layout pairing failed")
            values = np.asarray([
                int(r["external_strict_full_physical_goal"])
                - int(s["external_strict_full_physical_goal"])
                for r, s in zip(recurrent, stateless, strict=True)
            ], dtype=np.int64)
            diffs.append(values)
            seed_gains[seed] += int(values.sum())
            recurrent_goals += sum(
                int(r["external_strict_full_physical_goal"]) for r in recurrent
            )
            if task == "doorkey":
                recurrent_doorkey_goals += sum(
                    int(r["external_strict_full_physical_goal"]) for r in recurrent
                )
            action_diffs[task][str(seed)] = sum(
                r["actual_native_executed_action_sha256"]
                != s["actual_native_executed_action_sha256"]
                for r, s in zip(recurrent, stateless, strict=True)
            )
            actor_action_diffs[task][str(seed)] = sum(
                r["actual_native_executed_action_sha256"]
                != a["actual_native_executed_action_sha256"]
                for r, a in zip(recurrent, actor, strict=True)
            )
        paired[task] = np.asarray(diffs, dtype=np.int64)
        task_gains[task] = int(paired[task].sum())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.zeros(20000, dtype=np.int64)
    for task in TASKS:
        values = paired[task]
        seed_ix = rng.integers(0, 3, size=(20000, 3))
        layout_ix = rng.integers(0, values.shape[1], size=(20000, 3, values.shape[1]))
        draws += values[seed_ix[:, :, None], layout_ix].sum(axis=(1, 2))
    interval = [float(x) for x in np.quantile(draws, (0.025, 0.975))]
    distinct = sum(
        all(action_diffs[task][str(seed)] >= 16 for task in TASKS)
        for seed in MODEL_SEEDS
    ) >= 2
    gain = int(sum(task_gains.values()))
    noninferior = recurrent_goals >= 3 * actor_goals - ACTOR_MARGIN
    numeric = bool(
        gain >= MIN_GAIN
        and all(task_gains[task] > 0 for task in TASKS)
        and sum(seed_gains[s] > 0 for s in MODEL_SEEDS) >= 2
        and recurrent_doorkey_goals >= MIN_MONITOR_DOOR_GOALS
        and noninferior
    )
    passed = distinct and numeric and interval[0] > 0
    return {
        "registered_decision": (
            "INSUFFICIENT" if not distinct else "PASS" if passed else "FAIL"
        ),
        "necessary_native_action_distinction": distinct,
        "physical_numeric_gate": numeric,
        "paired_recurrent_minus_stateless_strict_full_goals": gain,
        "per_task_physical_goal_gain": task_gains,
        "per_model_seed_physical_goal_gain": {
            str(s): int(v) for s, v in seed_gains.items()
        },
        "learned_native_path_hash_differences": action_diffs,
        "recurrent_native_hash_difference_vs_actor_once": actor_action_diffs,
        "once_measured_actor_source_goals": actor_goals,
        "actor_repeated_three_times_arithmetic_only": 3 * actor_goals,
        "recurrent_source_strict_full_goals": recurrent_goals,
        "recurrent_doorkey_strict_full_goals": recurrent_doorkey_goals,
        "actor_24_goal_margin_reached": noninferior,
        "paired_seed_task_bootstrap_95pct_gain_interval": interval,
        "bootstrap_95pct_lower_bound_above_zero": interval[0] > 0,
        "original_unlockpickup_final_touched": False,
    }


def execute(output: str) -> dict:
    if output != f"{OUT}/summary.json" or Path(output).exists():
        raise ValueError("Stage14 output must be fresh isolated preregistered path")
    validate_split_contract()
    source_sha = admitted_source_commit()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    episodes, true_y, train = collect_train()
    weights = class_weights(true_y)
    models = {}
    fits = []
    for seed in MODEL_SEEDS:
        for arm in ARMS:
            model, profile = fit_model(episodes, seed, arm, weights)
            profile["numeric_checkpoint"] = save_numeric(
                model, path.parent / f"{arm}-{seed}.npz"
            )
            models[(arm, seed)] = model
            fits.append(profile)
    if (len({f["num_parameters"] for f in fits}) != 1
            or len({f["actual_optimizer_step_calls"] for f in fits}) != 1
            or len({f["cached_source_label_presentations_not_native_steps"]
                    for f in fits}) != 1):
        raise ValueError("Stage14 learned temporal comparison capacity/exposure mismatch")
    monitor = acquire_monitor(models)
    verdict = decide(monitor)
    monitor_steps = sum(
        group["summary"]["real_native_actions"] for group in monitor.values()
    )
    result = {
        "study": "m23_stage14_real_source_public_recurrent_vs_stateless_gru_bc",
        "execution_source_commit": source_sha,
        "train_source_layouts": {
            task: list(ids) for task, ids in RECURRENT_TRAIN_LAYOUTS.items()
        },
        "monitor_source_layouts": {
            task: list(ids) for task, ids in RECURRENT_MONITOR_LAYOUTS.items()
        },
        "model_seeds": list(MODEL_SEEDS), "arms": list(ARMS),
        "public_input_width": WIDTH, "hidden_units": HIDDEN,
        "epochs": EPOCHS, "episode_minibatch": EPISODE_BATCH,
        "adam_learning_rate": LR, "fixed_public_label_class_weights": weights.tolist(),
        "real_source_train": train,
        "actual_native_source_train_steps": train["actual_new_native_train_steps"],
        "real_source_independent_monitor": monitor,
        "actual_native_source_monitor_steps": monitor_steps,
        "actual_native_source_total_steps":
            train["actual_new_native_train_steps"] + monitor_steps,
        "real_model_fits": len(fits),
        "real_optimizer_update_calls": sum(
            fit["actual_optimizer_step_calls"] for fit in fits
        ),
        "cached_label_presentations_not_native_actions": sum(
            fit["cached_source_label_presentations_not_native_steps"] for fit in fits
        ),
        "fit_numeric_checkpoint_manifests": fits,
        "registered_scientific_decision": verdict,
        "new_imagined_actions": 0,
        "original_unlockpickup_target_native_actions": 0,
        "original_target_final_episodes": 0,
        "actual_study_wall_seconds": time.perf_counter() - started,
        "versions": {
            name: version(name) for name in ("numpy", "torch", "gymnasium", "minigrid")
        },
    }
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "execution_sha": source_sha,
        "real_train_steps": result["actual_native_source_train_steps"],
        "real_monitor_steps": monitor_steps,
        "fits": len(fits), "optimizer_updates": result["real_optimizer_update_calls"],
        "physical_gain": verdict["paired_recurrent_minus_stateless_strict_full_goals"],
        "per_task_physical_gain": verdict["per_task_physical_goal_gain"],
        "action_distinct": verdict["necessary_native_action_distinction"],
        "registered_decision": verdict["registered_decision"],
    }), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=f"{OUT}/summary.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Stage14 native public recurrent training needs --execute")
    execute(args.output)


if __name__ == "__main__":
    main()
