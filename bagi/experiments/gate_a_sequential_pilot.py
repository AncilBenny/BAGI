"""Exploratory sequential transfer from verified navigation checkpoints."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from bagi.experiments.cumulative_transfer import TASKS, run_episode
from bagi.experiments.gate_a_competence import BLOCK, NativePPOEnv
from bagi.experiments.gate_a_observation_dependence import tensor_tree_equal
from bagi.experiments.gate_a_shared_interface import (
    INTERFACE,
    PACKAGES,
    SharedActuators,
    SharedPPO,
    capture_rng,
    load_boundary,
    native_action,
    restore_rng,
    save_boundary,
)
from bagi.workshop_runtime import source_revision

SOURCE = "90322715afab01d068e16c64c00969563c7da5bf"
SUMMARY_SHA = "f1a8d2a410bdb917a75fddc0da92b7cc04617945815518808051e2602da350e8"
DIGESTS = {
    131: "458cf9fc120e3966cf603db5369e544ec8eda6510ba17ec75c7151b40587a18f",
    137: "eb97bf93de5aadd928d6cd5e8747055c80b3068a88ab053e9d8f6625733165a3",
    139: "97ea864314bafd171bc502b656ece7869c75064a26d57adbdfbe6a54f9a388d0",
}
ARMS = ("carry", "fresh", "frozen")


class TargetActor:
    def __init__(self, model, encoder, domain, *, deterministic):
        self.model, self.encoder, self.domain = model, encoder, domain
        self.deterministic = deterministic
        self.begin_episode()

    def begin_episode(self):
        self.hidden = None

    def act(self, observation, legal_actions, *, epsilon):
        if epsilon:
            raise ValueError("Frozen policy only")
        p, self.hidden = self.model.predict(
            self.encoder.encode(observation), state=self.hidden,
            episode_start=np.array([self.hidden is None]), deterministic=self.deterministic,
        )
        a = native_action(self.domain, p)
        if not 0 <= a < legal_actions:
            raise RuntimeError("Illegal native action")
        return a


def same_tree(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(
            same_tree(v, b[k]) for k, v in a.items()
        )
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(
            same_tree(x, y) for x, y in zip(a, b, strict=True)
        )
    return tensor_tree_equal(a, b)


@contextmanager
def frozen_state(control):
    m, p, e = control.model, control.model.policy, control.env
    o, v, b, wrapper = p.optimizer, m.get_env(), m.rollout_buffer, control.model_env

    def snapshot():
        return copy.deepcopy((
            p.state_dict(), o.state_dict(), m._last_obs, m._last_episode_starts,
            m._last_lstm_states, control.training_actions, control.updates,
            control.presentations, m.num_timesteps, e.episodes, e.rows,
            getattr(e, "current_seed", None), e.episode_actions,
        ))

    before, rng, mode = snapshot(), capture_rng(), p.training
    try:
        yield
    finally:
        restore_rng(rng)
        p.train(mode)
        if (control.model is not m or m.policy is not p or p.optimizer is not o
                or m.get_env() is not v or m.rollout_buffer is not b
                or control.env is not e or control.model_env is not wrapper
                or not same_tree(before, snapshot())):
            raise RuntimeError("Frozen target evaluation mutated training state")


def evaluate(control, domain, seeds, *, deterministic):
    rows = []
    encoder = SharedActuators(NativePPOEnv(domain, 0))
    try:
        with frozen_state(control):
            for seed in seeds:
                torch.manual_seed(seed + 40_000_000)
                actor = TargetActor(control.model, encoder, domain, deterministic=deterministic)
                rows.append(run_episode(actor, domain, seed, learn=False))
    finally:
        encoder.close()
    return {"success": mean(r["physical_success"] for r in rows),
            "actions": sum(r["actions"] for r in rows), "episodes": rows}


def frozen_exposure(control, base, cursor, budget):
    """Match native B exposure and rollout cuts, with no parameter update."""
    if budget % BLOCK:
        raise ValueError("Exposure must contain complete blocks")
    rows = []
    domain = TASKS[1]
    encoder = SharedActuators(NativePPOEnv(domain, 0))
    try:
        with frozen_state(control):
            for _ in range(budget // BLOCK):
                remaining = BLOCK
                while remaining:
                    seed = base + cursor
                    torch.manual_seed(seed + 60_000_000)
                    actor = TargetActor(control.model, encoder, domain, deterministic=False)
                    row = run_episode(actor, domain, seed, learn=False, limit=min(100, remaining))
                    if not 0 < row["actions"] <= remaining:
                        raise RuntimeError("Invalid exposure accounting")
                    rows.append(row)
                    remaining -= row["actions"]
                    cursor += 1
    finally:
        encoder.close()
    return rows, cursor


def auc(points, key):
    steps = [p["B_actions"] for p in points]
    values = [p[key]["success"] for p in points]
    if steps[0] != 0 or any(a >= b for a, b in zip(steps, steps[1:])):
        raise ValueError("Invalid curve checkpoints")
    return sum((b - a) * (u + v) / 2 for a, b, u, v in zip(
        steps[:-1], steps[1:], values[:-1], values[1:], strict=True
    )) / steps[-1]


def metrics(results):
    estimates = []
    for seed in sorted({r["seed"] for r in results}):
        arms = {r["arm"]: r["curve"] for r in results if r["seed"] == seed}
        carry, fresh, frozen = (arms[a] for a in ARMS)
        if not [p["B_actions"] for p in carry] == [p["B_actions"] for p in fresh] == [
            p["B_actions"] for p in frozen
        ]:
            raise ValueError("Unmatched B budgets")
        a = [p["A_greedy"]["success"] for p in carry]
        estimates.append({
            "seed": seed, "forward_auc": auc(carry, "B_greedy") - auc(fresh, "B_greedy"),
            "forward_stochastic_auc": auc(carry, "B_live") - auc(fresh, "B_live"),
            "backward_transfer": a[-1] - a[0],
            "peak_forgetting": max(a) - a[-1],
            "worst_forgetting": max(0.0, a[0] - min(a)),
            "carry_B_auc": auc(carry, "B_greedy"), "fresh_B_auc": auc(fresh, "B_greedy"),
            "frozen_B_auc": auc(frozen, "B_greedy"),
            "zero_shot_difference": carry[0]["B_greedy"]["success"] -
            fresh[0]["B_greedy"]["success"],
            "final_B_difference": carry[-1]["B_greedy"]["success"] -
            fresh[-1]["B_greedy"]["success"],
        })
    return {"per_seed": estimates, "spread": {
        k: {"mean": mean(r[k] for r in estimates),
            "seed_sd": stdev(r[k] for r in estimates) if len(estimates) > 1 else None,
            "min": min(r[k] for r in estimates), "max": max(r[k] for r in estimates)}
        for k in estimates[0] if k != "seed"
    }}


def run(inputs, output, *, budget=32768, interval=8192, eval_count=40):
    if budget < interval or budget % interval or interval % BLOCK or eval_count < 1:
        raise ValueError("Invalid matched budget")
    inputs, output = Path(inputs), Path(output)
    raw = (inputs / "summary.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != SUMMARY_SHA:
        raise ValueError("Historical summary digest mismatch")
    prior = json.loads(raw)
    source = {"commit": SOURCE, "tracked_worktree_dirty": False}
    if prior["source"] != source or prior["interface"] != INTERFACE:
        raise ValueError("Historical source mismatch")
    versions = {p: version(p) for p in PACKAGES}
    if versions != prior["versions"]:
        raise ValueError("Dependency versions changed")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    results = []
    with (output / "B_actions.jsonl").open("x") as trace:
        for index, (seed, digest) in enumerate(DIGESTS.items()):
            path = inputs / f"final-0-{seed}.pt"
            record = next(r for r in prior["results"] if r["domain"] == TASKS[0]
                          and r["seed"] == seed)
            if record["final_sha256"] != digest:
                raise ValueError("Historical checkpoint mismatch")
            base = 71_000_000 + index * 1_000_000
            a_seeds = range(base + 300_000, base + 300_000 + eval_count)
            b_seeds = range(base + 500_000, base + 500_000 + eval_count)
            for arm in ARMS:
                c = (SharedPPO(TASKS[1], seed, base) if arm == "fresh"
                     else load_boundary(path, digest, expected_source=source))
                initial = (c.training_actions, c.updates, c.presentations)
                curve, cursor = [], 0
                try:
                    for steps in range(0, budget + 1, interval):
                        point = {"B_actions": steps}
                        for label, domain, seeds in (
                            ("A", TASKS[0], a_seeds), ("B", TASKS[1], b_seeds),
                        ):
                            point[label + "_greedy"] = evaluate(c, domain, seeds,
                                                                deterministic=True)
                            point[label + "_live"] = evaluate(c, domain, seeds,
                                                              deterministic=False)
                        curve.append(point)
                        if steps == budget:
                            break
                        if arm == "frozen":
                            rows, cursor = frozen_exposure(c, base, cursor, interval)
                        else:
                            if arm == "carry" and steps == 0:
                                c.switch_domain(TASKS[1], base)
                            c.train(interval)
                            rows, c.env.rows = c.env.rows, []
                        if sum(row["actions"] for row in rows) != interval:
                            raise RuntimeError("Mismatched B native exposure")
                        for row in rows:
                            trace.write(json.dumps(dict(row, arm=arm, model_seed=seed)) + "\n")
                    delta = tuple(a - b for a, b in zip(
                        (c.training_actions, c.updates, c.presentations), initial, strict=True
                    ))
                    if delta != ((0, 0, 0) if arm == "frozen"
                                 else (budget, budget // 16, 4 * budget)):
                        raise RuntimeError("Incorrect phase B training counters")
                    final_digest = (None if arm == "frozen" else save_boundary(
                        c, output / f"{arm}-{seed}.pt"
                    ))
                    results.append({
                        "seed": seed, "arm": arm, "curve": curve,
                        "historical_A_actions": initial[0], "B_exposure_actions": budget,
                        "B_training_actions": delta[0], "B_optimizer_updates": delta[1],
                        "B_sample_presentations": delta[2], "final_sha256": final_digest,
                        "evaluation_actions": sum(p[k]["actions"] for p in curve for k in (
                            "A_greedy", "A_live", "B_greedy", "B_live",
                        )),
                    })
                finally:
                    c.model_env.close()
    summary = {
        "kind": "gate_a_exploratory_sequential_pilot", "scientific_pass": False,
        "source": source_revision(), "historical_source": source,
        "historical_artifact": 15175703476, "interface": INTERFACE, "versions": versions,
        "config": {"budget": budget, "interval": interval, "eval_count": eval_count},
        "actual_B_exposure_actions": sum(r["B_exposure_actions"] for r in results),
        "actual_training_actions": sum(r["B_training_actions"] for r in results),
        "actual_optimizer_updates": sum(r["B_optimizer_updates"] for r in results),
        "actual_sample_presentations": sum(r["B_sample_presentations"] for r in results),
        "actual_evaluation_actions": sum(r["evaluation_actions"] for r in results),
        "metrics": metrics(results), "seconds": time.monotonic() - started, "results": results,
        "limitations": "Cached A, exploratory resets, no replay/maintained continual baseline.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
