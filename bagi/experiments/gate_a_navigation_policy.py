"""Exploratory audit of stochastic versus greedy navigation behavior.

Reuse the maintained learner and the existing public native runtime.
No new RL algorithm; no final held-out evaluation.
"""
from __future__ import annotations

import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from bagi.experiments.cumulative_transfer import ACTIONS, TASKS, run_episode
from bagi.experiments.gate_a_competence import BLOCK, PPOControl
from bagi.workshop_runtime import source_revision


class FrozenActor:
    def __init__(self, model, *, deterministic):
        self.model = model
        self.deterministic = deterministic
        self.hidden = None

    def begin_episode(self):
        self.hidden = None

    def act(self, observation, legal_actions, *, epsilon):
        if epsilon:
            raise ValueError("Evaluation must not use external exploration")
        action, self.hidden = self.model.predict(
            observation, state=self.hidden,
            episode_start=np.array([self.hidden is None]),
            deterministic=self.deterministic,
        )
        action = int(np.asarray(action).item())
        if not 0 <= action < legal_actions:
            raise ValueError("Illegal predicted action")
        return action


class MemorylessActor:
    def __init__(self, probabilities, seed):
        self.probabilities = probabilities
        self.rng = np.random.default_rng(seed)

    def begin_episode(self):
        pass

    def act(self, observation, legal_actions, *, epsilon):
        if epsilon or legal_actions != len(self.probabilities):
            raise ValueError("Invalid memoryless control interface")
        return int(self.rng.choice(legal_actions, p=self.probabilities))


def evaluate(control, seeds, *, mode, probabilities=None):
    before = (control.training_actions, control.updates, control.presentations)
    rng_before = torch.random.get_rng_state().clone()
    weights = {k: v.detach().clone() for k, v in control.model.policy.state_dict().items()}
    rows = []
    for seed in seeds:
        # Isolate policy-sampling RNG from training; same reset and sampling seed
        # at each checkpoint. Memoryless controls use a separate local generator.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 40_000_000)
            if mode in ("greedy", "stochastic"):
                actor = FrozenActor(control.model, deterministic=mode == "greedy")
            elif mode in ("uniform", "action_prior"):
                prior = np.asarray(probabilities, dtype=np.float64)
                actor = MemorylessActor(prior, seed + 50_000_000)
            else:
                raise ValueError("Unknown evaluation mode")
            rows.append(run_episode(actor, TASKS[0], seed, learn=False))
    if before != (control.training_actions, control.updates, control.presentations):
        raise RuntimeError("Frozen evaluation modified learning counters")
    if not torch.equal(rng_before, torch.random.get_rng_state()):
        raise RuntimeError("Evaluation consumed training RNG")
    if not all(torch.equal(v, weights[k]) for k, v in control.model.policy.state_dict().items()):
        raise RuntimeError("Evaluation modified model weights")
    return {
        "mode": mode, "success": sum(row["physical_success"] for row in rows) / len(rows),
        "actions": sum(row["actions"] for row in rows), "episodes": rows,
    }


def run(output, *, seeds=(71, 73, 79), budget=65536, interval=4096, eval_count=40):
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
            control = PPOControl(TASKS[0], seed, 11_000_000 + index * 1_000_000)
            monitor = range(31_000_000 + index * 1_000_000,
                            31_000_000 + index * 1_000_000 + eval_count)
            points = []
            counts = np.zeros(ACTIONS, dtype=np.int64)
            try:
                for steps in range(0, budget + 1, interval):
                    if steps in (0, budget // 4, budget // 2, budget):
                        points.append({
                            "train_actions": steps,
                            "greedy": evaluate(control, monitor, mode="greedy"),
                            "stochastic": evaluate(control, monitor, mode="stochastic"),
                        })
                    if steps == budget:
                        break
                    control.train(interval)
                    rows = control.env.rows
                    control.env.rows = []
                    counts.fill(0)
                    for row in rows:
                        counts += np.bincount(row["executed_actions"], minlength=ACTIONS)
                        trace.write(json.dumps(dict(row, model_seed=seed)) + "\n")
                    if counts.sum() != interval or control.training_actions != steps + interval:
                        raise RuntimeError("Native trace accounting mismatch")
                # A fixed distribution, matched to the learner's last TRAIN interval;
                # no monitor data used in its construction.
                prior = counts / counts.sum()
                controls = [
                    evaluate(control, monitor, mode="uniform",
                             probabilities=np.full(ACTIONS, 1 / ACTIONS)),
                    evaluate(control, monitor, mode="action_prior", probabilities=prior),
                ]
                # Preserve policy/optimizer tensors for subsequent frozen diagnostics.
                # Not a full simulator/rollout/RNG-resume checkpoint.
                path = output / f"policy-{seed}.pt"
                with path.open("xb") as handle:
                    torch.save({
                        "schema": "gate-a-ppo-policy-v1", "seed": seed,
                        "source": source_revision(), "policy": control.model.policy.state_dict(),
                        "optimizer": control.model.policy.optimizer.state_dict(),
                        "training_actions": control.training_actions,
                        "optimizer_updates": control.updates,
                        "sample_presentations": control.presentations,
                    }, handle)
                with path.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                results.append({
                    "seed": seed, "curve": points, "final_controls": controls,
                    "last_train_action_counts": counts.tolist(),
                    "training_actions": control.training_actions,
                    "optimizer_updates": control.updates,
                    "sample_presentations": control.presentations,
                    "evaluation_actions": (
                        sum(p[m]["actions"] for p in points for m in ("greedy", "stochastic"))
                        + sum(p["actions"] for p in controls)
                    ),
                    "policy_sha256": digest,
                })
            finally:
                control.env.close()
    summary = {
        "kind": "gate_a_navigation_policy_audit", "scientific_pass": False,
        "source": source_revision(),
        "versions": {p: version(p) for p in (
            "gymnasium", "minigrid", "torch", "numpy", "stable-baselines3", "sb3-contrib",
        )},
        "config": {"seeds": seeds, "budget": budget, "interval": interval,
                   "eval_count": eval_count, "domain": TASKS[0]},
        "actual_training_actions": sum(r["training_actions"] for r in results),
        "actual_evaluation_actions": sum(r["evaluation_actions"] for r in results),
        "actual_optimizer_updates": sum(r["optimizer_updates"] for r in results),
        "actual_sample_presentations": sum(r["sample_presentations"] for r in results),
        "seconds": time.monotonic() - start, "results": results,
        "limitations": "Exploratory monitor; no independent layout split or transfer result.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
