"""Source-bound exploratory CartPole acquisition mechanism probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path

import torch

from bagi.experiments.gate_a_replay_competence import (
    SEEDS,
    NativeReplayRuntime,
    counters,
)
from bagi.workshop_runtime import source_revision

ARMS = ("exploration", "density", "both")
BASELINE = {
    151: "728f2de9d243478e4a445db2a61f23f8419ff1395ea2017e01d4028fac3ed777",
    157: "cd37d877f2a5f0667195ad00c8bf9dd393eb124c2c6d9e6536478344eeaa706b",
    163: "72566691f4c077c55522904d93fa28abd1365c302f720d4dac1c2f6034f02fe7",
}
VERSIONS = {
    "torch": "2.15.0", "torchrl": "0.15.0", "tensordict": "0.15.6",
    "gymnasium": "1.4.0", "minigrid": "3.2.0", "numpy": "2.4.6",
}


def settings(arm, steps, budget):
    if arm not in ARMS or not 0 <= steps <= budget or budget < 1:
        raise ValueError("Invalid acquisition setting")
    epsilon = (
        0.9 - 0.8 * min(1.0, 2 * steps / budget)
        if arm in ("exploration", "both") else 0.2
    )
    return epsilon, 64 if arm in ("density", "both") else 8


class AcquisitionRuntime(NativeReplayRuntime):
    def __init__(self, seed):
        super().__init__("torchrl", seed)
        self.training_epsilon = 0.2

    def choose(self, observation, facts):
        epsilon = self.training_epsilon if self.epsilon else 0.0
        return self.learner.act(observation, self.domain, epsilon=epsilon)


def trace_digest(point):
    hashes = [r["action_sha256"] for r in point["episodes"]]
    return hashlib.sha256(json.dumps(hashes, separators=(",", ":")).encode()).hexdigest()


def run(output, *, seeds=SEEDS, arms=ARMS, budget=32768, interval=8192,
        block=256, eval_count=40, verify_baseline=True):
    if (not seeds or not arms or len(set(seeds)) != len(seeds)
            or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS)
            or min(budget, interval, block, eval_count) < 1
            or budget % (2 * interval) or interval % block):
        raise ValueError("Invalid exploratory budget")
    source = source_revision()
    versions = {p: version(p) for p in VERSIONS}
    if source["tracked_worktree_dirty"] is not False:
        raise ValueError("Require clean source")
    if verify_baseline and (versions != VERSIONS or tuple(seeds) != SEEDS
                            or budget != 32768 or interval != 8192
                            or block != 256 or eval_count != 40):
        raise ValueError("Frozen baseline requires exact settings and dependencies")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    start = time.monotonic()
    manifest = {
        "source": source, "versions": versions, "scientific_pass": False,
        "kind": "gate_a_cartpole_acquisition",
        "historical_baseline": {"run": 46141697260, "artifact": 15262685472,
                                "source": "7e884e7fc6e8fb02678cfd6d2fd3b76129d7be61",
                                "greedy_auc": 0, "final_success": 0},
        "config": {"seeds": seeds, "arms": arms, "budget": budget, "block": block,
                   "interval": interval, "eval_count": eval_count,
                   "verify_baseline": verify_baseline, "capacity": 256},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    with ((output / "training.jsonl").open("x") as trace,
          (output / "evaluation.jsonl").open("x") as probes,
          (output / "updates.jsonl").open("x") as updates):
        for index, seed in enumerate(seeds):
            base = 95_000_000 + index * 1_000_000
            monitor = range(base + 500_000, base + 500_000 + eval_count)
            for arm in arms:
                agent = AcquisitionRuntime(seed)
                cursor, curve, midpoint = base, [], None
                expected_updates = 0
                for steps in range(0, budget + 1, interval):
                    point = {"train_actions": steps, **agent.evaluate(1, monitor)}
                    if steps == 0 and verify_baseline:
                        if trace_digest(point) != BASELINE[seed]:
                            raise RuntimeError("Historical initial frozen trace mismatch")
                    curve.append(point)
                    probes.write(json.dumps(dict(point, seed=seed, arm=arm)) + "\n")
                    probes.flush()
                    if steps == budget:
                        break
                    if steps == budget // 2:
                        path = output / f"mid-{seed}-{arm}"
                        midpoint = agent.checkpoint(path, restore=True)
                        shutil.rmtree(path)
                    for pos in range(steps, steps + interval, block):
                        epsilon, count = settings(arm, pos, budget)
                        agent.training_epsilon = epsilon
                        before = counters(agent.learner)
                        rows = agent.block(1, block, cursor, count)
                        after = counters(agent.learner)
                        cursor += len(rows)
                        expected_updates += count
                        for row in rows:
                            trace.write(json.dumps(dict(row, model_seed=seed, arm=arm,
                                                        epsilon=epsilon)) + "\n")
                        updates.write(json.dumps({
                            "seed": seed, "arm": arm, "train_actions": pos + block,
                            "epsilon": epsilon, "updates": after[1] - before[1],
                            "presentations": after[2] - before[2],
                        }) + "\n")
                        trace.flush()
                        updates.flush()
                actual = counters(agent.learner)
                if actual[:2] != (budget, expected_updates):
                    raise RuntimeError("Incorrect acquisition counters")
                values = [p["success"] for p in curve]
                result = {
                    "seed": seed, "arm": arm, "curve": curve,
                    "success_auc": sum((a + b) / 2 for a, b in pairwise(values))
                    / (len(values) - 1),
                    "final_success": values[-1], "training_actions": actual[0],
                    "optimizer_updates": actual[1], "replay_presentations": actual[2],
                    "evaluation_actions": sum(p["actions"] for p in curve),
                    "evaluation_episodes": sum(len(p["episodes"]) for p in curve),
                    "midpoint_digests": midpoint, "next_reset_seed": cursor,
                    "final_digests": agent.checkpoint(
                        output / f"final-{seed}-{arm}", restore=False,
                    ),
                }
                results.append(result)
                print(json.dumps({k: result[k] for k in (
                    "seed", "arm", "success_auc", "final_success", "optimizer_updates",
                    "replay_presentations",
                )}), flush=True)
    summary = {
        **manifest, "results": results,
        "actual_training_actions": sum(r["training_actions"] for r in results),
        "actual_optimizer_updates": sum(r["optimizer_updates"] for r in results),
        "actual_replay_presentations": sum(r["replay_presentations"] for r in results),
        "actual_evaluation_actions": sum(r["evaluation_actions"] for r in results),
        "actual_evaluation_episodes": sum(r["evaluation_episodes"] for r in results),
        "qualifying_arms": [
            arm for arm in arms
            if all(r["final_success"] >= 0.9 for r in results if r["arm"] == arm)
        ],
        "seconds": time.monotonic() - start,
        "limitations": "Historical baseline, repeated exploratory monitors, no transfer claim.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.output)
