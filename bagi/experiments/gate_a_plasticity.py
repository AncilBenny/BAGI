"""Matched optimizer and actuator-head interventions on cached A policies."""
from __future__ import annotations

import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path
from statistics import mean, stdev

import torch

from bagi.agents.plasticity import reset_adam_history, reset_control_head
from bagi.experiments.cumulative_transfer import TASKS
from bagi.experiments.gate_a_sequential_pilot import (
    DIGESTS,
    SOURCE,
    SUMMARY_SHA,
    auc,
    evaluate,
)
from bagi.experiments.gate_a_shared_interface import (
    INTERFACE,
    PACKAGES,
    SharedPPO,
    capture_rng,
    load_boundary,
    restore_rng,
    save_boundary,
)
from bagi.workshop_runtime import source_revision

BASELINE_SOURCE = "31a149afe9643e0bb540988a7f486a2eac324a86"
BASELINE_SUMMARY = "0a3cb7a36d059c987327b7ec533ddba8dea64dd681563db987c19dc5c3677140"
ARMS = ("adam_reset", "adam_and_B_head_reset")


def read_summary(path, digest, source):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Historical summary digest mismatch")
    s = json.loads(raw)
    if (s["source"] != {"commit": source, "tracked_worktree_dirty": False}
            or s["interface"] != INTERFACE
            or s["versions"] != {p: version(p) for p in PACKAGES}):
        raise ValueError("Historical source or versions mismatch")
    return s


def compare(curve, carry, fresh):
    if not [p["B_actions"] for p in curve] == [p["B_actions"] for p in carry] == [
        p["B_actions"] for p in fresh
    ]:
        raise ValueError("Mismatched checkpoints")
    a = [p["A_greedy"]["success"] for p in curve]
    return {
        "B_auc": auc(curve, "B_greedy"),
        "B_auc_delta_carry": auc(curve, "B_greedy") - auc(carry, "B_greedy"),
        "B_auc_delta_fresh": auc(curve, "B_greedy") - auc(fresh, "B_greedy"),
        "B_live_auc_delta_carry": auc(curve, "B_live") - auc(carry, "B_live"),
        "B_live_auc_delta_fresh": auc(curve, "B_live") - auc(fresh, "B_live"),
        "B_final": curve[-1]["B_greedy"]["success"],
        "A_final": a[-1], "A_backward_transfer": a[-1] - a[0],
        "A_worst_forgetting": max(0.0, a[0] - min(a)),
        "A_live_final": curve[-1]["A_live"]["success"],
    }


def run(inputs, baseline, output):
    inputs, baseline, output = map(Path, (inputs, baseline, output))
    prior = read_summary(inputs / "summary.json", SUMMARY_SHA, SOURCE)
    old = read_summary(baseline / "summary.json", BASELINE_SUMMARY, BASELINE_SOURCE)
    if old["historical_source"] != prior["source"]:
        raise ValueError("Cached baseline used another A source")
    config = old["config"]
    budget, interval, count = (config[k] for k in ("budget", "interval", "eval_count"))
    if (budget, interval, count) != (32768, 8192, 40):
        raise ValueError("Unexpected matched experiment settings")
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    results = []
    with (output / "B_actions.jsonl").open("x") as trace:
        for index, (seed, digest) in enumerate(DIGESTS.items()):
            record = next(r for r in prior["results"] if r["domain"] == TASKS[0]
                          and r["seed"] == seed)
            if record["final_sha256"] != digest:
                raise ValueError("Unexpected historical policy digest")
            cached = {r["arm"]: r for r in old["results"] if r["seed"] == seed}
            for name in ("carry", "fresh"):
                if cached[name]["B_training_actions"] != budget:
                    raise ValueError("Cached control budget mismatch")
            base = 71_000_000 + index * 1_000_000
            monitors = (range(base + 300_000, base + 300_000 + count),
                        range(base + 500_000, base + 500_000 + count))
            for arm in ARMS:
                c = load_boundary(inputs / f"final-0-{seed}.pt", digest,
                                  expected_source=prior["source"])
                initial = (c.training_actions, c.updates, c.presentations)
                curve = []
                try:
                    cleared = reset_adam_history(c.model.policy)
                    if not cleared:
                        raise RuntimeError("Expected historical Adam state")
                    if arm == "adam_and_B_head_reset":
                        rng = capture_rng()
                        fresh = SharedPPO(TASKS[1], seed, base)
                        try:
                            reset_control_head(c.model.policy, fresh.model.policy)
                        finally:
                            fresh.model_env.close()
                            restore_rng(rng)
                    for steps in range(0, budget + 1, interval):
                        point = {"B_actions": steps}
                        for label, domain, seeds in zip(
                            ("A", "B"), TASKS, monitors, strict=True
                        ):
                            point[label + "_greedy"] = evaluate(c, domain, seeds,
                                                                deterministic=True)
                            point[label + "_live"] = evaluate(c, domain, seeds,
                                                              deterministic=False)
                        curve.append(point)
                        if steps == 0:
                            # A head/weights are unchanged; greedy native traces must match.
                            previous = cached["carry"]["curve"][0]["A_greedy"]
                            if point["A_greedy"] != previous:
                                raise RuntimeError("Intervention changed A greedy behavior")
                        if steps == budget:
                            break
                        if steps == 0:
                            c.switch_domain(TASKS[1], base)
                        c.train(interval)
                        rows, c.env.rows = c.env.rows, []
                        if sum(row["actions"] for row in rows) != interval:
                            raise RuntimeError("Native action budget mismatch")
                        for row in rows:
                            trace.write(json.dumps(dict(row, arm=arm, model_seed=seed)) + "\n")
                    delta = tuple(a - b for a, b in zip(
                        (c.training_actions, c.updates, c.presentations), initial, strict=True
                    ))
                    if delta != (budget, budget // 16, 4 * budget):
                        raise RuntimeError("Phase B accounting mismatch")
                    results.append({
                        "seed": seed, "arm": arm, "curve": curve, "adam_states_cleared": cleared,
                        "B_training_actions": delta[0], "B_optimizer_updates": delta[1],
                        "B_sample_presentations": delta[2],
                        "evaluation_actions": sum(p[k]["actions"] for p in curve for k in (
                            "A_greedy", "A_live", "B_greedy", "B_live",
                        )),
                        "final_sha256": save_boundary(c, output / f"{arm}-{seed}.pt"),
                        "metrics": compare(curve, cached["carry"]["curve"],
                                           cached["fresh"]["curve"]),
                    })
                finally:
                    c.model_env.close()
    spread = {}
    for arm in ARMS:
        rows = [r["metrics"] for r in results if r["arm"] == arm]
        spread[arm] = {key: {"mean": mean(r[key] for r in rows),
                             "seed_sd": stdev(r[key] for r in rows)}
                       for key in rows[0]}
    summary = {
        "kind": "gate_a_plasticity_interventions", "scientific_pass": False,
        "source": source_revision(), "versions": prior["versions"], "interface": INTERFACE,
        "cached_A_source": prior["source"], "cached_baseline_source": old["source"],
        "cached_A_artifact": 15175703476, "cached_baseline_artifact": 15178421886,
        "config": config, "actual_training_actions": sum(r["B_training_actions"] for r in results),
        "actual_optimizer_updates": sum(r["B_optimizer_updates"] for r in results),
        "actual_sample_presentations": sum(r["B_sample_presentations"] for r in results),
        "actual_evaluation_actions": sum(r["evaluation_actions"] for r in results),
        "metrics": spread, "results": results, "seconds": time.monotonic() - started,
        "limitations": "Exploratory cached controls, no replay, no independent physical groups.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
