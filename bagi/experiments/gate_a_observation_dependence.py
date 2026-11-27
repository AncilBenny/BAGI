"""Frozen observation-dependence diagnostic; no new optimization or final data."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from bagi.agents.ppo_policy_restore import restore_policy_snapshot
from bagi.experiments.cumulative_transfer import ACTIONS, TASKS, run_episode
from bagi.experiments.gate_a_competence import PPOControl
from bagi.experiments.gate_a_navigation_policy import FrozenActor, MemorylessActor
from bagi.workshop_runtime import source_revision

SOURCE = "798bbd9e642db9d02444e5f9e9a04808ecd3aae4"
SUMMARY_SHA = "a2f3b9c870f91548b8f561b00998d628eeddd66ff4b2eb14a396544960bcc104"
DIGESTS = {
    71: "59a433665846ad5b19e7b21da14e484d4b34ea9cac958b3555d53bb5885e5411",
    73: "823b8547bd94b813d4c12522c01f32373e336550d66ccaad8e8cd4f859336c99",
    79: "42cda9526ecde94890fb3f951f00fe603ed0b7e526b393e89d98ff1333c116bd",
}
MODES = ("live_image", "stale_first_image", "action_prior")


class StaleImageActor(FrozenActor):
    """Hold the first public image; keep actual-action history live."""

    def begin_episode(self):
        super().begin_episode()
        self.initial_image = None

    def act(self, observation, legal_actions, *, epsilon):
        if self.initial_image is None:
            self.initial_image = observation[:147].copy()
        stale = observation.copy()
        stale[:147] = self.initial_image
        return super().act(stale, legal_actions, epsilon=epsilon)


def tensor_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(tensor_tree_equal(v, right[k]) for k, v in left.items()))
    if isinstance(left, (tuple, list)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(tensor_tree_equal(a, b) for a, b in zip(left, right, strict=True)))
    return left == right


def run(inputs, output, *, eval_count=60):
    if not 1 <= eval_count <= 1000:
        raise ValueError("Invalid exploratory monitor count")
    inputs, output = Path(inputs), Path(output)
    raw = (inputs / "summary.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != SUMMARY_SHA:
        raise ValueError("Source summary digest mismatch")
    original = json.loads(raw)
    if original["source"] != {"commit": SOURCE, "tracked_worktree_dirty": False}:
        raise ValueError("Source summary provenance mismatch")
    versions = {p: version(p) for p in original["versions"]}
    if versions != original["versions"]:
        raise ValueError("Snapshot dependency versions differ")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows = []
    for index, row in enumerate(original["results"]):
        seed = row["seed"]
        control = PPOControl(TASKS[0], seed, 11_000_000 + index * 1_000_000)
        try:
            if row["policy_sha256"] != DIGESTS[seed]:
                raise ValueError("Unexpected policy digest")
            restored = restore_policy_snapshot(
                control.model.policy, inputs / f"policy-{seed}.pt",
                sha256=DIGESTS[seed], source=SOURCE, seed=seed,
            )
            if restored != {key: row[key] for key in restored}:
                raise ValueError("Historical counters mismatch")
            prior = np.asarray(row["last_train_action_counts"], dtype=np.float64)
            if prior.shape != (ACTIONS,) or prior.sum() != 4096:
                raise ValueError("Invalid frozen action prior")
            prior /= prior.sum()
            policy_before = copy.deepcopy(control.model.policy.state_dict())
            adam_before = copy.deepcopy(control.model.policy.optimizer.state_dict())
            rng_before = torch.random.get_rng_state().clone()
            monitors = range(34_000_000 + index * 1_000_000,
                             34_000_000 + index * 1_000_000 + eval_count)
            for mode in MODES:
                episodes = []
                for monitor in monitors:
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(monitor + 40_000_000)
                        if mode == "live_image":
                            actor = FrozenActor(control.model, deterministic=False)
                        elif mode == "stale_first_image":
                            actor = StaleImageActor(control.model, deterministic=False)
                        else:
                            actor = MemorylessActor(prior, monitor + 50_000_000)
                        episodes.append(run_episode(actor, TASKS[0], monitor, learn=False))
                rows.append({
                    "seed": seed, "mode": mode, "restored_counters": restored,
                    "success": sum(x["physical_success"] for x in episodes) / eval_count,
                    "success_by_20": sum(
                        x["physical_success"] and x["actions"] <= 20 for x in episodes
                    ) / eval_count,
                    "restricted_mean_actions": sum(
                        x["actions"] if x["physical_success"] else 100 for x in episodes
                    ) / eval_count,
                    "actual_evaluation_actions": sum(x["actions"] for x in episodes),
                    "episodes": episodes,
                })
            if not tensor_tree_equal(policy_before, control.model.policy.state_dict()):
                raise RuntimeError("Frozen policy changed")
            if not tensor_tree_equal(adam_before, control.model.policy.optimizer.state_dict()):
                raise RuntimeError("Frozen optimizer changed")
            if not torch.equal(rng_before, torch.random.get_rng_state()):
                raise RuntimeError("Frozen evaluation consumed training RNG")
            if (control.training_actions, control.updates, control.presentations) != (0, 0, 0):
                raise RuntimeError("Unexpected new training")
        finally:
            control.env.close()
    if {row["seed"] for row in rows} != set(DIGESTS) or len(rows) != 9:
        raise ValueError("Missing source seed or evaluation arm")
    summary = {
        "kind": "gate_a_frozen_observation_dependence", "scientific_pass": False,
        "source": source_revision(), "restored_source": SOURCE, "restored_artifact": 15172477979,
        "versions": versions, "config": {"eval_count": eval_count, "modes": MODES},
        "actual_training_actions": 0, "actual_optimizer_updates": 0,
        "actual_sample_presentations": 0,
        "actual_evaluation_actions": sum(x["actual_evaluation_actions"] for x in rows),
        "results": rows,
        "limitations": "Exploratory, stale-view intervention is out of training distribution.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.inputs, args.output)
