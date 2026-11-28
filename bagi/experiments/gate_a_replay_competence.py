"""Exploratory matched native competence for recurrent replay; not transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import deque
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from bagi.agents.general_brain import GeneralBrain, Skill
from bagi.agents.recurrent_checkpoint import load_checkpoint, save_checkpoint
from bagi.agents.recurrent_replay import Episode, RecurrentReplay
from bagi.agents.torchrl_recurrent import (
    ACTIONS,
    HORIZON,
    INTERFACE,
    LEGAL,
    WIDTH,
    PersistentFeedForwardDQN,
    PersistentRecurrentDQN,
)
from bagi.experiments.cumulative_transfer import GOAL, TASKS, EndSensor
from bagi.experiments.gate_a_competence import NativePPOEnv
from bagi.experiments.gate_a_shared_interface import SharedActuators
from bagi.runtime import AgentRuntime, StepFeedback
from bagi.workshop_runtime import source_revision

ARMS = ("torchrl", "internal_gru")
SUPPORTED_ARMS = (*ARMS, "feedforward")
SEEDS = (151, 157, 163)
CAPACITY = 256


def counters(learner):
    return (learner.training_actions, learner.updates, learner.presentations)


class NativeReplayRuntime:
    """A single reusable public-action runtime; physical success stays evaluator-only."""

    def __init__(self, arm, seed):
        if arm not in SUPPORTED_ARMS:
            raise ValueError("Unknown control")
        self.arm = arm
        self.learner = (
            PersistentRecurrentDQN(seed, capacity=CAPACITY)
            if arm == "torchrl" else PersistentFeedForwardDQN(seed, capacity=CAPACITY)
            if arm == "feedforward" else RecurrentReplay(WIDTH, ACTIONS, seed=seed)
        )
        self.domain, self.epsilon = 0, 0.0
        skill = Skill("control", frozenset(), GOAL.required, self.choose, max_actions=HORIZON)
        self.runtime = AgentRuntime(
            GeneralBrain((skill,), EndSensor(), max_episode_actions=HORIZON), INTERFACE,
        )

    def choose(self, observation, facts):
        arg = self.domain if self.arm != "internal_gru" else LEGAL[self.domain]
        return self.learner.act(observation, arg, epsilon=self.epsilon)

    def episode(self, domain, seed, *, learn, limit=HORIZON):
        if domain not in (0, 1) or not 1 <= limit <= HORIZON:
            raise ValueError("Invalid native episode")
        self.domain, self.epsilon = domain, 0.2 if learn else 0.0
        env = SharedActuators(NativePPOEnv(TASKS[domain], seed))
        self.learner.begin_episode()
        actions, rewards, terms, truncs = [], [], [], []
        try:
            initial, _ = env.reset()
            observations = [initial]
            native_end = False
            physical = False

            def step(action):
                nonlocal native_end, physical
                if not 0 <= action < LEGAL[domain]:
                    raise RuntimeError("Illegal action")
                proposal = [action, 0] if domain == 0 else [0, action]
                obs, reward, term, trunc, _ = env.step(proposal)
                actual = env.env.episode_actions[-1]
                if actual != action:
                    raise RuntimeError("Incorrect native action projection")
                actions.append(actual)
                rewards.append(float(reward))
                terms.append(bool(term))
                native_end = bool(term or trunc)
                cut = len(actions) == limit and not native_end
                truncs.append(bool(trunc or cut))
                observations.append(obs)
                if native_end:
                    physical = bool(env.env.rows[-1]["physical_success"])
                return StepFeedback(actual, obs, float(reward), bool(term), bool(trunc or cut))

            result = self.runtime.run_episode(GOAL, initial, step, learn=learn)
            if tuple(actions) != result.actions or not actions or len(actions) > limit:
                raise RuntimeError("Native trace mismatch")
            self.learner.begin_episode()
            if learn:
                if self.arm != "internal_gru":
                    self.learner.remember(
                        domain, np.stack(observations), actions, rewards, terms, truncs,
                    )
                else:
                    key = TASKS[domain]
                    self.learner.replay.setdefault(key, deque(maxlen=CAPACITY))
                    self.learner.remember(Episode(
                        np.stack(observations), np.asarray(actions, dtype=np.int64),
                        np.asarray(rewards, dtype=np.float32), np.asarray(terms, dtype=np.bool_),
                        key, LEGAL[domain],
                    ))
            return {
                "seed": seed, "domain": TASKS[domain], "actions": len(actions),
                "executed_actions": actions, "reward": sum(rewards),
                "action_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
                "physical_success": physical, "native_end": native_end,
                "terminated": terms[-1], "truncated": truncs[-1], "learn": learn,
            }
        finally:
            env.close()
            self.learner.begin_episode()

    def evaluate(self, domain, seeds):
        before = counters(self.learner), self.learner.rng.getstate(), self.runtime.snapshot()
        rows = [self.episode(domain, seed, learn=False) for seed in seeds]
        after = counters(self.learner), self.learner.rng.getstate(), self.runtime.snapshot()
        if before != after:
            raise RuntimeError("Frozen evaluation mutated learning state")
        return {
            "success": mean(row["physical_success"] for row in rows),
            "mean_reward": mean(row["reward"] for row in rows),
            "actions": sum(row["actions"] for row in rows), "episodes": rows,
        }

    def block(self, domain, steps, seed, updates):
        before = counters(self.learner)
        rows = []
        remaining = steps
        while remaining:
            row = self.episode(domain, seed + len(rows), learn=True,
                               limit=min(remaining, HORIZON))
            rows.append(row)
            remaining -= row["actions"]
        presentations = 0
        for _ in range(updates):
            start = self.learner.presentations
            loss = self.learner.update(4)
            if not np.isfinite(loss):
                raise RuntimeError("Nonfinite update")
            presentations += self.learner.presentations - start
        after = counters(self.learner)
        if tuple(a - b for a, b in zip(after, before, strict=True)) != (
            steps, updates, presentations,
        ):
            raise RuntimeError("Native budget mismatch")
        return rows

    def checkpoint(self, path, *, restore):
        path.mkdir(exist_ok=False)
        self.runtime.save(path / "runtime.json")
        if self.arm != "internal_gru":
            self.learner.save(path / "learner")
            if restore:
                self.learner = type(self.learner).load(path / "learner")
        else:
            save_checkpoint(self.learner, path / "learner.pt", interface=INTERFACE)
            if restore:
                self.learner = load_checkpoint(
                    path / "learner.pt", interface=INTERFACE, width=WIDTH, actions=ACTIONS,
                )
                self.learner.replay = {
                    k: deque(v, maxlen=CAPACITY) for k, v in self.learner.replay.items()
                }
        if restore:
            self.runtime.load(path / "runtime.json")
        digests = {}
        for p in sorted(path.rglob("*")):
            if p.is_file():
                with p.open("rb") as handle:
                    digests[str(p.relative_to(path))] = hashlib.file_digest(
                        handle, "sha256",
                    ).hexdigest()
        return digests


def run(output, *, seeds=SEEDS, budget=32768, interval=8192, block=256,
        updates_per_block=8, eval_count=40, arms=ARMS, domains=(0, 1)):
    if (not seeds or len(set(seeds)) != len(seeds) or not arms
            or len(set(arms)) != len(arms) or not set(arms) <= set(SUPPORTED_ARMS)
            or not domains or len(set(domains)) != len(domains) or not set(domains) <= {0, 1}
            or min(budget, interval, block, updates_per_block, eval_count) < 1
            or budget % interval or interval % block or budget % (2 * interval)
            or budget >= 100000 or eval_count >= 100000):
        raise ValueError("Invalid exploratory competence registration")
    source = source_revision()
    if source["tracked_worktree_dirty"] is not False:
        raise ValueError("Require clean source")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    manifest = {
        "source": source,
        "versions": {p: version(p) for p in
                     ("torch", "torchrl", "tensordict", "gymnasium", "minigrid", "numpy")},
        "config": {"seeds": seeds, "budget": budget, "interval": interval, "block": block,
                   "updates_per_block": updates_per_block, "eval_count": eval_count,
                   "arms": arms, "domains": domains, "tasks": TASKS, "capacity_per_domain": CAPACITY,
                   "epsilon": 0.2, "episodes_per_update": 4, "horizon": HORIZON},
        "kind": "gate_a_replay_competence", "scientific_pass": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    results = []
    with ((output / "training.jsonl").open("x") as trace,
          (output / "evaluation.jsonl").open("x") as probes):
        for domain in domains:
            for index, seed in enumerate(seeds):
                base = 85_000_000 + domain * 10_000_000 + index * 1_000_000
                monitor = range(base + 500_000, base + 500_000 + eval_count)
                for arm in arms:
                    agent = NativeReplayRuntime(arm, seed)
                    curve, cursor = [], base
                    midpoint = None
                    for steps in range(0, budget + 1, interval):
                        point = {"train_actions": steps, **agent.evaluate(domain, monitor)}
                        curve.append(point)
                        probes.write(json.dumps(dict(point, domain=domain, seed=seed,
                                                     arm=arm)) + "\n")
                        probes.flush()
                        if steps == budget:
                            break
                        if steps == budget // 2:
                            path = output / f"mid-{domain}-{seed}-{arm}"
                            midpoint = agent.checkpoint(path, restore=True)
                            shutil.rmtree(path)
                        for _ in range(interval // block):
                            rows = agent.block(domain, block, cursor, updates_per_block)
                            cursor += len(rows)
                            for row in rows:
                                trace.write(json.dumps(dict(row, model_seed=seed, arm=arm)) + "\n")
                            trace.flush()
                    actual = counters(agent.learner)
                    if actual[:2] != (budget, budget // block * updates_per_block):
                        raise RuntimeError("Incorrect lifetime counters")
                    values = [point["success"] for point in curve]
                    auc = sum((a + b) / 2 for a, b in pairwise(values)) / (
                        len(values) - 1
                    )
                    final = agent.checkpoint(output / f"final-{domain}-{seed}-{arm}", restore=False)
                    result = {
                        "domain": domain, "seed": seed, "arm": arm, "curve": curve,
                        "success_auc": auc, "final_success": values[-1],
                        "qualifies": values[-1] >= 0.9,
                        "training_actions": actual[0], "optimizer_updates": actual[1],
                        "replay_presentations": actual[2], "next_reset_seed": cursor,
                        "evaluation_actions": sum(p["actions"] for p in curve),
                        "evaluation_episodes": sum(len(p["episodes"]) for p in curve),
                        "midpoint_digests": midpoint, "final_digests": final,
                    }
                    results.append(result)
                    print(json.dumps({k: result[k] for k in (
                        "domain", "seed", "arm", "success_auc", "final_success",
                        "training_actions", "optimizer_updates", "replay_presentations",
                    )}), flush=True)
    comparison = []
    for domain in domains:
        for arm in arms:
            group = [r for r in results if r["domain"] == domain and r["arm"] == arm]
            comparison.append({
                "domain": domain, "arm": arm,
                "auc_mean": mean(r["success_auc"] for r in group),
                "auc_seed_sd": stdev(r["success_auc"] for r in group) if len(group) > 1 else None,
                "all_seeds_competent": all(r["qualifies"] for r in group),
            })
    summary = {
        **manifest, "results": results, "comparison": comparison,
        "actual_training_actions": sum(r["training_actions"] for r in results),
        "actual_optimizer_updates": sum(r["optimizer_updates"] for r in results),
        "actual_replay_presentations": sum(r["replay_presentations"] for r in results),
        "actual_evaluation_actions": sum(r["evaluation_actions"] for r in results),
        "actual_evaluation_episodes": sum(r["evaluation_episodes"] for r in results),
        "seconds": time.monotonic() - started,
        "limitations": "Single-family fits, repeated exploratory monitors, no transfer claim.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.output)
