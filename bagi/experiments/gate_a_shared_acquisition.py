"""Fresh native common-interface competence diagnostic, not a transfer study."""
from __future__ import annotations

import copy
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from bagi.experiments.cumulative_transfer import TASKS, run_episode
from bagi.experiments.gate_a_competence import BLOCK
from bagi.experiments.gate_a_navigation_policy import MemorylessActor
from bagi.experiments.gate_a_observation_dependence import tensor_tree_equal
from bagi.experiments.gate_a_shared_interface import (
    INTERFACE,
    PACKAGES,
    SharedPPO,
    load_boundary,
    native_action,
    save_boundary,
)
from bagi.workshop_runtime import source_revision


class SharedActor:
    def __init__(self, control, mode):
        self.control = control
        self.mode = mode

    def begin_episode(self):
        self.hidden = None
        self.first = None

    def act(self, observation, legal_actions, *, epsilon):
        if epsilon:
            raise ValueError("Frozen policy only")
        obs = observation.copy()
        if self.mode == "stale":
            if self.first is None:
                self.first = obs[:147].copy()
            obs[:147] = self.first
        action, self.hidden = self.control.model.predict(
            self.control.model_env.encode(obs), state=self.hidden,
            episode_start=np.array([self.hidden is None]), deterministic=self.mode == "greedy",
        )
        native = native_action(self.control.env.domain, action)
        if not 0 <= native < legal_actions:
            raise RuntimeError("Illegal native policy action")
        return native


def evaluate(control, seeds, mode, prior=None):
    before = (control.training_actions, control.updates, control.presentations)
    weights = copy.deepcopy(control.model.policy.state_dict())
    optimizer = copy.deepcopy(control.model.policy.optimizer.state_dict())
    rng = torch.random.get_rng_state().clone()
    rows = []
    for seed in seeds:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 40_000_000)
            actor = (MemorylessActor(prior, seed + 50_000_000) if mode == "prior"
                     else SharedActor(control, mode))
            rows.append(run_episode(actor, control.env.domain, seed, learn=False))
    if (before != (control.training_actions, control.updates, control.presentations)
            or not tensor_tree_equal(weights, control.model.policy.state_dict())
            or not tensor_tree_equal(optimizer, control.model.policy.optimizer.state_dict())
            or not torch.equal(rng, torch.random.get_rng_state())):
        raise RuntimeError("Frozen evaluation modified learning state")
    return {
        "mode": mode, "success": sum(x["physical_success"] for x in rows) / len(rows),
        "success_by_20": sum(x["physical_success"] and x["actions"] <= 20
                             for x in rows) / len(rows),
        "actions": sum(x["actions"] for x in rows), "episodes": rows,
    }


def run(output, *, seeds=(131, 137, 139), budgets=(65536, 32768), interval=4096, eval_count=40):
    if (not seeds or len(set(seeds)) != len(seeds) or len(budgets) != len(TASKS)
            or interval <= 0 or interval % BLOCK or eval_count < 1 or eval_count >= 100000
            or any(b <= 0 or b >= 100000 or b % interval for b in budgets)):
        raise ValueError("Invalid diagnostic configuration")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    results = []
    with (output / "training_actions.jsonl").open("x") as trace:
        for di, domain in enumerate(TASKS):
            budget = budgets[di]
            for index, seed in enumerate(seeds):
                base = 51_000_000 + di * 10_000_000 + index * 1_000_000
                monitors = range(base + 500_000, base + 500_000 + eval_count)
                control = SharedPPO(domain, seed, base)
                points = []
                counts = np.zeros(7 if di == 0 else 2, dtype=np.int64)
                midpoint = None
                try:
                    for steps in range(0, budget + 1, interval):
                        if steps in (0, budget // 4, budget // 2, budget):
                            points.append({"train_actions": steps,
                                           "greedy": evaluate(control, monitors, "greedy"),
                                           "live": evaluate(control, monitors, "live")})
                        if steps == budget:
                            break
                        if steps == budget // 2:
                            path = output / f"mid-{di}-{seed}.pt"
                            midpoint = save_boundary(control, path)
                            control.model_env.close()
                            control = load_boundary(path, midpoint)
                        control.train(interval)
                        rows = control.env.rows
                        control.env.rows = []
                        counts.fill(0)
                        for row in rows:
                            counts += np.bincount(row["executed_actions"], minlength=len(counts))
                            trace.write(json.dumps(dict(row, model_seed=seed)) + "\n")
                        if counts.sum() != interval or control.training_actions != steps + interval:
                            raise RuntimeError("Native action budget mismatch")
                    extra = [evaluate(control, monitors, "prior", counts / counts.sum())]
                    if di == 0:
                        extra.append(evaluate(control, monitors, "stale"))
                    digest = save_boundary(control, output / f"final-{di}-{seed}.pt")
                    results.append({
                        "domain": domain, "seed": seed, "curve": points, "final_controls": extra,
                        "training_actions": control.training_actions,
                        "optimizer_updates": control.updates,
                        "sample_presentations": control.presentations,
                        "evaluation_actions": (sum(p[m]["actions"] for p in points
                                                   for m in ("greedy", "live"))
                                               + sum(p["actions"] for p in extra)),
                        "midpoint_sha256": midpoint, "final_sha256": digest,
                    })
                finally:
                    control.model_env.close()
    summary = {
        "kind": "gate_a_shared_interface_acquisition", "scientific_pass": False,
        "source": source_revision(), "interface": INTERFACE,
        "versions": {p: version(p) for p in PACKAGES},
        "config": {"seeds": seeds, "budgets": budgets, "interval": interval,
                   "eval_count": eval_count, "tasks": TASKS},
        "actual_training_actions": sum(r["training_actions"] for r in results),
        "actual_evaluation_actions": sum(r["evaluation_actions"] for r in results),
        "actual_optimizer_updates": sum(r["optimizer_updates"] for r in results),
        "actual_sample_presentations": sum(r["sample_presentations"] for r in results),
        "seconds": time.monotonic() - started, "results": results,
        "limitations": "Fresh competence only; inactive-channel logprob; no replay or transfer.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
