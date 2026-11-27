"""Native alternating rehearsal with matched total interaction controls."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch

from bagi.agents.plasticity import reset_adam_history
from bagi.agents.rehearsal_schedule import next_reset_seed
from bagi.experiments.cumulative_transfer import TASKS
from bagi.experiments.gate_a_plasticity import (
    BASELINE_SOURCE,
    BASELINE_SUMMARY,
    compare,
    read_summary,
)
from bagi.experiments.gate_a_sequential_pilot import (
    DIGESTS,
    SOURCE,
    SUMMARY_SHA,
    evaluate,
    same_tree,
)
from bagi.experiments.gate_a_shared_interface import (
    INTERFACE,
    check_boundary,
    load_boundary,
    save_boundary,
)
from bagi.workshop_runtime import source_revision

ADAM_SOURCE = "03771b72dc87a41ce0c8162e0a24a0bb680ca3c3"
ADAM_SUMMARY = "5d5fc712771d684b3fd0eed4426e3389da591a009461f41a281ee47a8908efa7"
B_BLOCK, A_BLOCK, CYCLES = 8192, 2048, 4
KEYS = ("A_greedy", "A_live", "B_greedy", "B_live")


def train_block(c, domain, first, budget):
    check_boundary(c)
    m, p = c.model, c.model.policy
    o, buffer = p.optimizer, m.rollout_buffer
    initial = (c.training_actions, c.updates, c.presentations)
    if c.env.domain != domain:
        weights, adam = copy.deepcopy(p.state_dict()), copy.deepcopy(o.state_dict())
        c.switch_domain(domain, first)
        if (not same_tree(weights, p.state_dict()) or not same_tree(adam, o.state_dict())
                or c.model is not m or m.policy is not p or p.optimizer is not o
                or m.rollout_buffer is not buffer):
            raise RuntimeError("Task switch changed persistent learner")
    elif c.env.current_seed != first:
        raise ValueError("Pending reset does not match cursor")
    c.train(budget)
    rows, c.env.rows = c.env.rows, []
    pending = next_reset_seed(rows, first, budget, domain, 7 if domain == TASKS[0] else 2)
    check_boundary(c)
    delta = tuple(a - b for a, b in zip(
        (c.training_actions, c.updates, c.presentations), initial, strict=True
    ))
    if delta != (budget, budget // 16, 4 * budget) or c.env.current_seed != pending:
        raise RuntimeError("Training or reset cursor accounting mismatch")
    return rows, pending


def measure(c, monitors, b_actions, a_actions):
    point = {"B_actions": b_actions, "A_actions": a_actions}
    for label, domain, seeds in zip(("A", "B"), TASKS, monitors, strict=True):
        for suffix, greedy in (("greedy", True), ("live", False)):
            point[f"{label}_{suffix}"] = evaluate(c, domain, seeds, deterministic=greedy)
    return point


def require_match(point, cached):
    if any(point[key] != cached[key] for key in KEYS):
        raise RuntimeError("Cached trace positive control mismatch")


def unique_record(summary, seed, arm):
    records = [r for r in summary["results"] if r["seed"] == seed and r["arm"] == arm]
    if len(records) != 1 or records[0]["B_training_actions"] != 32768:
        raise ValueError("Expected unique matched B record")
    return records[0]


def totals(c, initial):
    a, u, p = (c.training_actions, c.updates, c.presentations)
    return {"training_actions": a - initial[0], "optimizer_updates": u - initial[1],
            "sample_presentations": p - initial[2]}


def run(inputs, baseline, adam_dir, output):
    inputs, baseline, adam_dir, output = map(Path, (inputs, baseline, adam_dir, output))
    a = read_summary(inputs / "summary.json", SUMMARY_SHA, SOURCE)
    old = read_summary(baseline / "summary.json", BASELINE_SUMMARY, BASELINE_SOURCE)
    adam = read_summary(adam_dir / "summary.json", ADAM_SUMMARY, ADAM_SOURCE)
    if (adam["cached_A_source"] != a["source"]
            or adam["cached_baseline_source"] != old["source"]
            or adam["config"] != old["config"]
            or adam["config"] != {"budget": 32768, "interval": 8192, "eval_count": 40}):
        raise ValueError("Historical controls are not matched")
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    results = []
    with (output / "training_actions.jsonl").open("x") as trace:
        for index, (seed, digest) in enumerate(DIGESTS.items()):
            prior = next(r for r in a["results"] if r["seed"] == seed
                         and r["domain"] == TASKS[0])
            if prior["final_sha256"] != digest:
                raise ValueError("Historical A checkpoint mismatch")
            cached = unique_record(adam, seed, "adam_reset")
            fresh = unique_record(old, seed, "fresh")
            carry = unique_record(old, seed, "carry")
            base = 71_000_000 + index * 1_000_000
            monitors = (range(base + 300_000, base + 300_040),
                        range(base + 500_000, base + 500_040))
            c = load_boundary(inputs / f"final-0-{seed}.pt", digest,
                              expected_source=a["source"])
            try:
                initial = (c.training_actions, c.updates, c.presentations)
                if not reset_adam_history(c.model.policy):
                    raise RuntimeError("Missing historical Adam state")
                cursors = {TASKS[0]: 81_000_000 + index * 1_000_000, TASKS[1]: base}
                first = measure(c, monitors, 0, 0)
                require_match(first, cached["curve"][0])
                curve, after_rehearsal = [first], []
                for cycle in range(1, CYCLES + 1):
                    for domain, budget in ((TASKS[1], B_BLOCK), (TASKS[0], A_BLOCK)):
                        rows, pending = train_block(c, domain, cursors[domain], budget)
                        cursors[domain] = pending
                        for row in rows:
                            trace.write(json.dumps(dict(row, arm="rehearsal", cycle=cycle,
                                                       model_seed=seed)) + "\n")
                        if domain == TASKS[1]:
                            point = measure(c, monitors, cycle * B_BLOCK, (cycle - 1) * A_BLOCK)
                            if cycle == 1:
                                require_match(point, cached["curve"][1])
                            curve.append(point)
                        else:
                            after_rehearsal.append(measure(c, monitors, cycle * B_BLOCK,
                                                           cycle * A_BLOCK))
                counts = totals(c, initial)
                if counts["training_actions"] != 40960:
                    raise RuntimeError("Wrong rehearsal total")
                metrics = compare(curve, carry["curve"], fresh["curve"])
                adam_metrics = compare(curve, cached["curve"], fresh["curve"])
                final = after_rehearsal[-1]
                results.append({
                    "seed": seed, "arm": "rehearsal", "curve": curve,
                    "after_rehearsal": after_rehearsal, "cursors": cursors,
                    "metrics": metrics, "metrics_vs_adam": adam_metrics,
                    "final": final, "A_training_actions": 8192, "B_training_actions": 32768,
                    **counts, "final_sha256": save_boundary(c, output / f"rehearsal-{seed}.pt"),
                })
            finally:
                c.model_env.close()
            control = load_boundary(adam_dir / f"adam_reset-{seed}.pt",
                                    cached["final_sha256"], expected_source=adam["source"])
            try:
                initial = (control.training_actions, control.updates, control.presentations)
                before = measure(control, monitors, 32768, 0)
                require_match(before, cached["curve"][-1])
                rows, pending = train_block(control, TASKS[1], control.env.current_seed, 8192)
                for row in rows:
                    trace.write(json.dumps(dict(row, arm="B_only_extension",
                                               model_seed=seed)) + "\n")
                final = measure(control, monitors, 40960, 0)
                results.append({
                    "seed": seed, "arm": "B_only_extension", "curve": [before, final],
                    "after_rehearsal": [], "final": final, "pending_seed": pending,
                    "A_training_actions": 0, "B_training_actions": 8192,
                    "cached_B_training_actions": 32768, **totals(control, initial),
                    "final_sha256": save_boundary(control, output / f"B-only-{seed}.pt"),
                })
            finally:
                control.model_env.close()
    eval_actions = sum(p[k]["actions"] for r in results
                       for p in r["curve"] + r["after_rehearsal"] for k in KEYS)
    contrasts = []
    for seed in DIGESTS:
        rehearsal = next(r for r in results if r["seed"] == seed and r["arm"] == "rehearsal")
        extension = next(r for r in results if r["seed"] == seed
                         and r["arm"] == "B_only_extension")
        points = rehearsal["curve"] + rehearsal["after_rehearsal"]
        contrasts.append({
            "seed": seed,
            "matched_total_final": {k: rehearsal["final"][k]["success"]
                                    - extension["final"][k]["success"] for k in KEYS},
            "worst_A_greedy_loss": 1 - min(p["A_greedy"]["success"] for p in points),
            "matched_B_AUC_delta_adam": rehearsal["metrics_vs_adam"]["B_auc_delta_carry"],
        })
    summary = {
        "kind": "gate_a_native_rehearsal", "scientific_pass": False,
        "source": source_revision(), "interface": INTERFACE, "versions": a["versions"],
        "historical_sources": [a["source"], old["source"], adam["source"]],
        "historical_artifacts": [15175703476, 15178421886, 15184379503],
        "actual_training_actions": sum(r["training_actions"] for r in results),
        "actual_optimizer_updates": sum(r["optimizer_updates"] for r in results),
        "actual_sample_presentations": sum(r["sample_presentations"] for r in results),
        "actual_A_training_actions": sum(r["A_training_actions"] for r in results),
        "actual_B_training_actions": sum(r["B_training_actions"] for r in results),
        "actual_evaluation_actions": eval_actions, "contrasts": contrasts,
        "seconds": time.monotonic() - started, "results": results,
        "limitations": "Additional live A practice, not historical replay. Exploratory cached "
                       "controls; no held-out groups or maintained continual-learning baseline.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
