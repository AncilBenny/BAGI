"""Matched public observation encoding experiment; exploratory only."""
from __future__ import annotations

import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import OneHotPartialObsWrapper

from bagi.experiments.cumulative_transfer import ACTIONS, TASKS, run_episode
from bagi.experiments.gate_a_competence import BLOCK, PPOControl
from bagi.experiments.gate_a_navigation_policy import FrozenActor, MemorylessActor
from bagi.experiments.gate_a_observation_dependence import tensor_tree_equal
from bagi.workshop_runtime import source_revision

ARMS = ("scalar", "categorical")


class CategoricalView(gym.ObservationWrapper):
    """Apply maintained MiniGrid transform to the exact public image only."""

    def __init__(self, env):
        super().__init__(env)
        self.categorical = OneHotPartialObsWrapper(env.unwrapped)
        size = int(np.prod(self.categorical.observation_space["image"].shape)) + 14
        self.observation_space = gym.spaces.Box(-1, 1, (size,), dtype=np.float32)

    def observation(self, observation):
        image = np.rint(observation[:147] * 10).astype(np.uint8).reshape(7, 7, 3)
        categorical = self.categorical.observation({"image": image})["image"]
        return np.concatenate((categorical.reshape(-1), observation[147:])).astype(np.float32)


class EvaluationActor(FrozenActor):
    def __init__(self, model, transform, *, stale=False, deterministic=False):
        super().__init__(model, deterministic=deterministic)
        self.transform = transform
        self.stale = stale

    def begin_episode(self):
        super().begin_episode()
        self.first = None

    def act(self, observation, legal_actions, *, epsilon):
        obs = observation.copy()
        if self.stale:
            if self.first is None:
                self.first = obs[:147].copy()
            obs[:147] = self.first
        return super().act(self.transform(obs), legal_actions, epsilon=epsilon)


def identity(obs):
    return obs


def evaluate(control, monitor, transform, *, mode, prior=None):
    import copy

    before = (control.training_actions, control.updates, control.presentations)
    weights = copy.deepcopy(control.model.policy.state_dict())
    optimizer = copy.deepcopy(control.model.policy.optimizer.state_dict())
    rng = torch.random.get_rng_state().clone()
    rows = []
    for seed in monitor:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 40_000_000)
            if mode == "prior":
                actor = MemorylessActor(prior, seed + 50_000_000)
            else:
                actor = EvaluationActor(
                    control.model, transform, stale=mode == "stale",
                    deterministic=mode == "greedy",
                )
            rows.append(run_episode(actor, TASKS[0], seed, learn=False))
    if before != (control.training_actions, control.updates, control.presentations):
        raise RuntimeError("Evaluation changed learning counters")
    if not torch.equal(rng, torch.random.get_rng_state()):
        raise RuntimeError("Evaluation consumed training RNG")
    if not tensor_tree_equal(weights, control.model.policy.state_dict()):
        raise RuntimeError("Evaluation changed policy")
    if not tensor_tree_equal(optimizer, control.model.policy.optimizer.state_dict()):
        raise RuntimeError("Evaluation changed Adam")
    return {
        "mode": mode, "success": sum(x["physical_success"] for x in rows) / len(rows),
        "success_by_20": sum(
            x["physical_success"] and x["actions"] <= 20 for x in rows
        ) / len(rows),
        "restricted_mean_actions": sum(
            x["actions"] if x["physical_success"] else 100 for x in rows
        ) / len(rows),
        "actions": sum(x["actions"] for x in rows), "episodes": rows,
    }


def run(output, *, seeds=(101, 103, 107), budget=65536, interval=4096, eval_count=40):
    if (not seeds or len(set(seeds)) != len(seeds) or interval <= 0 or budget <= 0
            or budget % interval or interval % BLOCK or not 1 <= eval_count < 100000
            or budget >= 100000):
        raise ValueError("Invalid diagnostic configuration")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    start = time.monotonic()
    results = []
    with (output / "training_actions.jsonl").open("x") as trace:
        for index, seed in enumerate(seeds):
            base = 41_000_000 + index * 1_000_000
            monitor = range(base + 500_000, base + 500_000 + eval_count)
            for arm in ARMS:
                control = PPOControl(
                    TASKS[0], seed, base,
                    observation_wrapper=CategoricalView if arm == "categorical" else None,
                )
                wrapped = control.model_env
                transform = wrapped.observation if arm == "categorical" else identity
                points = []
                counts = np.zeros(ACTIONS, dtype=np.int64)
                try:
                    for steps in range(0, budget + 1, interval):
                        if steps in (0, budget // 4, budget // 2, budget):
                            points.append({
                                "train_actions": steps,
                                "greedy": evaluate(control, monitor, transform, mode="greedy"),
                                "live": evaluate(control, monitor, transform, mode="live"),
                            })
                        if steps == budget:
                            break
                        control.train(interval)
                        rows = control.env.rows
                        control.env.rows = []
                        counts.fill(0)
                        for row in rows:
                            counts += np.bincount(row["executed_actions"], minlength=ACTIONS)
                            trace.write(json.dumps(dict(row, model_seed=seed, arm=arm)) + "\n")
                        if counts.sum() != interval or control.training_actions != steps + interval:
                            raise RuntimeError("Native action accounting mismatch")
                    controls = [
                        evaluate(control, monitor, transform, mode="stale"),
                        evaluate(control, monitor, transform, mode="prior",
                                 prior=counts / counts.sum()),
                    ]
                    path = output / f"{arm}-{seed}.pt"
                    with path.open("xb") as handle:
                        torch.save({
                            "schema": "gate-a-representation-v1", "arm": arm, "seed": seed,
                            "source": source_revision(),
                            "policy": control.model.policy.state_dict(),
                            "optimizer": control.model.policy.optimizer.state_dict(),
                            "training_actions": control.training_actions,
                            "optimizer_updates": control.updates,
                            "sample_presentations": control.presentations,
                        }, handle)
                    with path.open("rb") as handle:
                        digest = hashlib.file_digest(handle, "sha256").hexdigest()
                    results.append({
                        "arm": arm, "seed": seed, "curve": points, "final_controls": controls,
                        "last_train_action_counts": counts.tolist(),
                        "observation_shape": control.model.observation_space.shape,
                        "parameters": sum(p.numel() for p in control.model.policy.parameters()),
                        "training_actions": control.training_actions,
                        "optimizer_updates": control.updates,
                        "sample_presentations": control.presentations,
                        "evaluation_actions": (
                            sum(p[m]["actions"] for p in points for m in ("greedy", "live"))
                            + sum(p["actions"] for p in controls)
                        ),
                        "policy_sha256": digest,
                    })
                finally:
                    wrapped.close()
    summary = {
        "kind": "gate_a_matched_public_representation", "scientific_pass": False,
        "source": source_revision(),
        "versions": {p: version(p) for p in (
            "gymnasium", "minigrid", "torch", "numpy", "stable-baselines3", "sb3-contrib",
        )},
        "config": {"seeds": seeds, "budget": budget, "interval": interval,
                   "eval_count": eval_count, "arms": ARMS, "domain": TASKS[0]},
        "actual_training_actions": sum(r["training_actions"] for r in results),
        "actual_evaluation_actions": sum(r["evaluation_actions"] for r in results),
        "actual_optimizer_updates": sum(r["optimizer_updates"] for r in results),
        "actual_sample_presentations": sum(r["sample_presentations"] for r in results),
        "seconds": time.monotonic() - start, "results": results,
        "limitations": "Exploratory; input width and parameter count differ; no transfer claim.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
