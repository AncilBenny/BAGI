"""Native alternating rehearsal with matched total interaction controls."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch

from bagi.agents.plasticity import reset_adam_history
from bagi.agents.rehearsal_comparison import KEYS, check_fork_counts, compare_traces
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
    results, contrasts, forks = [], [], []
    evaluation_actions, evaluation_episodes = 0, 0
    with ((output / "training_actions.jsonl").open("x") as trace,
          (output / "evaluation_points.jsonl").open("x") as probes):

        def probe(c, monitors, b, a_count, seed, stage):
            nonlocal evaluation_actions, evaluation_episodes
            point = measure(c, monitors, b, a_count)
            evaluation_actions += sum(point[k]["actions"] for k in KEYS)
            evaluation_episodes += sum(len(point[k]["episodes"]) for k in KEYS)
            probes.write(json.dumps(dict(point, model_seed=seed, stage=stage)) + "\n")
            probes.flush()
            return point

        def advance(c, domain, first, budget, seed, arm, cycle):
            rows, pending = train_block(c, domain, first, budget)
            for row in rows:
                trace.write(json.dumps(dict(row, arm=arm, cycle=cycle, model_seed=seed)) + "\n")
            trace.flush()
            return pending

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
            fork_path = output / f"common-{seed}.pt"
            try:
                initial = (c.training_actions, c.updates, c.presentations)
                if not reset_adam_history(c.model.policy):
                    raise RuntimeError("Missing historical Adam state")
                first = probe(c, monitors, 0, 0, seed, "common_initial")
                require_match(first, cached["curve"][0])
                pending = advance(c, TASKS[1], base, B_BLOCK, seed, "common", 1)
                shared = probe(c, monitors, B_BLOCK, 0, seed, "common_first_B")
                divergence = compare_traces(shared, cached["curve"][1])
                common_counts = totals(c, initial)
                fork_digest = save_boundary(c, fork_path)
                forks.append({
                    "seed": seed, "sha256": fork_digest, "pending_B_seed": pending,
                    "cached_first_B_comparison": divergence,
                    "A_training_actions": 0, "B_training_actions": B_BLOCK,
                    **common_counts,
                })
                print(json.dumps({"seed": seed, "common_first_B": divergence}), flush=True)
            finally:
                c.model_env.close()

            for arm in ("rehearsal", "B_only"):
                c = load_boundary(fork_path, fork_digest)
                try:
                    initial = (c.training_actions, c.updates, c.presentations)
                    cursors = {TASKS[0]: 81_000_000 + index * 1_000_000, TASKS[1]: pending}
                    restored = probe(c, monitors, B_BLOCK, 0, seed, arm + "_restore")
                    require_match(restored, shared)
                    curve, after_rehearsal = [first, shared], []
                    if arm == "rehearsal":
                        for cycle in range(1, CYCLES + 1):
                            if cycle > 1:
                                cursors[TASKS[1]] = advance(
                                    c, TASKS[1], cursors[TASKS[1]], B_BLOCK, seed, arm, cycle,
                                )
                                curve.append(probe(c, monitors, cycle * B_BLOCK,
                                                   (cycle - 1) * A_BLOCK, seed, arm + "_B"))
                            cursors[TASKS[0]] = advance(
                                c, TASKS[0], cursors[TASKS[0]], A_BLOCK, seed, arm, cycle,
                            )
                            after_rehearsal.append(probe(
                                c, monitors, cycle * B_BLOCK, cycle * A_BLOCK, seed, arm + "_A",
                            ))
                        a_count, b_count = 8192, 24576
                        final = after_rehearsal[-1]
                    else:
                        for cycle in range(2, 6):
                            cursors[TASKS[1]] = advance(
                                c, TASKS[1], cursors[TASKS[1]], B_BLOCK, seed, arm, cycle,
                            )
                            curve.append(probe(c, monitors, cycle * B_BLOCK, 0, seed, arm))
                        a_count, b_count = 0, 32768
                        final = curve[-1]
                    counts = totals(c, initial)
                    if counts["training_actions"] != a_count + b_count:
                        raise RuntimeError("Wrong branch budget")
                    matched = [p for p in curve if p["B_actions"] <= 32768]
                    metrics = compare(matched, carry["curve"], fresh["curve"])
                    results.append({
                        "seed": seed, "arm": arm, "curve": curve,
                        "after_rehearsal": after_rehearsal, "cursors": cursors,
                        "metrics_vs_cached_carry_fresh": metrics,
                        "final": final, "A_training_actions": a_count,
                        "B_training_actions": b_count, "shared_B_training_actions": B_BLOCK,
                        "fork_sha256": fork_digest, **counts,
                        "final_sha256": save_boundary(c, output / f"{arm}-{seed}.pt"),
                    })
                finally:
                    c.model_env.close()

            rehearsal, control = results[-2:]
            check_fork_counts(common_counts["training_actions"],
                              rehearsal["training_actions"], control["training_actions"])
            points = rehearsal["curve"] + rehearsal["after_rehearsal"]
            matched_B = compare(rehearsal["curve"], control["curve"][:-1], fresh["curve"])
            contrasts.append({
                "seed": seed, "matched_total_final": {
                    k: rehearsal["final"][k]["success"] - control["final"][k]["success"]
                    for k in KEYS
                },
                "worst_A_greedy_loss": 1 - min(p["A_greedy"]["success"] for p in points),
                "matched_B_AUC_delta_control": matched_B["B_auc_delta_carry"],
                "matched_B_live_AUC_delta_control": matched_B["B_live_auc_delta_carry"],
            })
    unique = forks + results
    summary = {
        "kind": "gate_a_native_rehearsal_same_run_fork", "scientific_pass": False,
        "source": source_revision(), "interface": INTERFACE, "versions": a["versions"],
        "historical_sources": [a["source"], old["source"], adam["source"]],
        "historical_artifacts": [15175703476, 15178421886, 15184379503],
        "actual_training_actions": sum(r["training_actions"] for r in unique),
        "actual_optimizer_updates": sum(r["optimizer_updates"] for r in unique),
        "actual_sample_presentations": sum(r["sample_presentations"] for r in unique),
        "actual_A_training_actions": sum(r["A_training_actions"] for r in unique),
        "actual_B_training_actions": sum(r["B_training_actions"] for r in unique),
        "actual_evaluation_actions": evaluation_actions,
        "actual_evaluation_episodes": evaluation_episodes,
        "contrasts": contrasts, "forks": forks,
        "seconds": time.monotonic() - started, "results": results,
        "limitations": "Same-run matched forks; additional live practice, not historical replay. "
                       "Cached fresh comparison descriptive; no independent physical groups.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
