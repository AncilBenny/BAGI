"""Registered four-arm Gate A objective/Adam pilot; no scientific PASS."""
from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean, stdev

import torch

from bagi.agents.plasticity import reset_adam_history
from bagi.agents.rehearsal_schedule import next_reset_seed
from bagi.experiments.cumulative_transfer import TASKS
from bagi.experiments.gate_a_plasticity import BASELINE_SOURCE, BASELINE_SUMMARY, read_summary
from bagi.experiments.gate_a_sequential_pilot import DIGESTS, SOURCE, SUMMARY_SHA, auc, evaluate
from bagi.experiments.gate_a_shared_interface import (
    INTERFACE,
    check_boundary,
    load_boundary,
    save_boundary,
)
from bagi.workshop_runtime import source_revision

ARMS = ("joint_carry", "joint_reset", "active_carry", "active_reset")
KEYS = ("A_greedy", "A_live", "B_greedy", "B_live")
BUDGET, INTERVAL, COUNT = 32768, 8192, 40


def train_interval(control, arm, first, steps):
    """Validate native trace and actual counters at every drained boundary."""
    if arm not in ARMS:
        raise ValueError("Unknown objective/Adam arm")
    before = (control.training_actions, control.updates, control.presentations)
    if control.env.domain != TASKS[1]:
        control.switch_domain(TASKS[1], first)
    elif control.env.current_seed != first:
        raise ValueError("Incorrect pending B reset")
    if arm.startswith("active_"):
        from bagi.agents.active_ppo import train_active_block

        train_active_block(control, steps)
    else:
        control.train(steps)
    rows, control.env.rows = control.env.rows, []
    pending = next_reset_seed(rows, first, steps, TASKS[1], 2)
    check_boundary(control)
    delta = tuple(a - b for a, b in zip(
        (control.training_actions, control.updates, control.presentations), before, strict=True
    ))
    if delta != (steps, steps // 16, steps * 4) or control.env.current_seed != pending:
        raise RuntimeError("Incorrect native B counters or reset cursor")
    return rows, pending


def measure(control, monitors, steps):
    point = {"B_actions": steps}
    for label, domain, seeds in zip(("A", "B"), TASKS, monitors, strict=True):
        for suffix, deterministic in (("greedy", True), ("live", False)):
            point[f"{label}_{suffix}"] = evaluate(
                control, domain, seeds, deterministic=deterministic
            )
    return point


def arm_metrics(curve):
    if [p["B_actions"] for p in curve] != list(range(0, BUDGET + 1, INTERVAL)):
        raise ValueError("Unmatched registered measurement grid")
    result = {}
    for mode in ("greedy", "live"):
        a = [p[f"A_{mode}"]["success"] for p in curve]
        result.update({
            f"B_{mode}_auc": auc(curve, f"B_{mode}"),
            f"B_{mode}_final": curve[-1][f"B_{mode}"]["success"],
            f"A_{mode}_final": a[-1],
            f"A_{mode}_final_loss": a[0] - a[-1],
            f"A_{mode}_worst_loss": max(0.0, a[0] - min(a)),
        })
    return result


def summarize(results):
    by_key = {(r["seed"], r["arm"]): arm_metrics(r["curve"]) for r in results}
    expected = {(seed, arm) for seed in DIGESTS for arm in ARMS}
    if set(by_key) != expected or len(results) != len(expected):
        raise ValueError("Incomplete or duplicated four-arm comparison")
    rows = []
    for seed in DIGESTS:
        contrasts = {}
        for adam in ("carry", "reset"):
            joint = by_key[seed, "joint_" + adam]
            active = by_key[seed, "active_" + adam]
            delta = {key: active[key] - joint[key] for key in active}
            contrasts[adam] = {
                "joint": joint, "active": active, "active_minus_joint": delta,
                "qualifies": (
                    delta["B_greedy_auc"] >= 0.10 - 1e-12
                    and active["A_greedy_final"] >= 0.90
                    and active["B_greedy_final"] >= 0.90
                    and active["A_live_final_loss"] <= 0.10 + 1e-12
                    and delta["B_live_auc"] >= -1e-12
                ),
            }
        interaction = {
            k: contrasts["reset"]["active_minus_joint"][k]
            - contrasts["carry"]["active_minus_joint"][k]
            for k in contrasts["carry"]["active_minus_joint"]
        }
        rows.append({"seed": seed, **contrasts, "interaction_reset_minus_carry": interaction})
    spread = {}
    for label in ("carry", "reset", "interaction_reset_minus_carry"):
        values = [r[label] if label.startswith("interaction") else r[label]["active_minus_joint"]
                  for r in rows]
        spread[label] = {
            k: {"mean": mean(v[k] for v in values), "seed_sd": stdev(v[k] for v in values)}
            for k in values[0]
        }
    qualified = [adam for adam in ("carry", "reset") if all(r[adam]["qualifies"] for r in rows)]
    return {"per_seed": rows, "spread": spread, "qualifying_adam_conditions": qualified,
            "pilot_threshold_met": bool(qualified),
            "next_decision": ("register independent confirmation" if qualified
                              else "stop objective tuning; prioritize maintained recurrent replay")}


def run(inputs, baseline, output):
    inputs, baseline, output = map(Path, (inputs, baseline, output))
    prior = read_summary(inputs / "summary.json", SUMMARY_SHA, SOURCE)
    old = read_summary(baseline / "summary.json", BASELINE_SUMMARY, BASELINE_SOURCE)
    config = {"budget": BUDGET, "interval": INTERVAL, "eval_count": COUNT}
    if old["historical_source"] != prior["source"] or old["config"] != config:
        raise ValueError("Historical positive control is not source/budget matched")
    source = source_revision()
    if source["tracked_worktree_dirty"] is not False:
        raise ValueError("Require clean source")
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"source": source, "historical_source": prior["source"],
                "historical_artifacts": [15175703476, 15178421886],
                "historical_checkpoint_digests": DIGESTS, "config": config, "arms": ARMS,
                "versions": prior["versions"], "interface": INTERFACE,
                "resume_note": "Reselect the recorded objective explicitly; no automatic resume."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    results = []
    with ((output / "B_actions.jsonl").open("x") as trace,
          (output / "evaluation_points.jsonl").open("x") as probes):
        for index, (seed, digest) in enumerate(DIGESTS.items()):
            record = next(r for r in prior["results"] if r["seed"] == seed
                          and r["domain"] == TASKS[0])
            if record["final_sha256"] != digest:
                raise ValueError("Historical checkpoint digest mismatch")
            cached = next(r for r in old["results"] if r["seed"] == seed and r["arm"] == "carry")
            if cached["B_training_actions"] != BUDGET:
                raise ValueError("Historical carry budget mismatch")
            base = 71_000_000 + index * 1_000_000
            monitors = (range(base + 300_000, base + 300_000 + COUNT),
                        range(base + 500_000, base + 500_000 + COUNT))
            for arm in ARMS:
                c = load_boundary(inputs / f"final-0-{seed}.pt", digest,
                                  expected_source=prior["source"])
                try:
                    initial = (c.training_actions, c.updates, c.presentations)
                    if not c.model.policy.optimizer.state:
                        raise RuntimeError("Expected historical Adam state")
                    cleared = reset_adam_history(c.model.policy) if arm.endswith("_reset") else 0
                    curve, pending = [], base
                    for steps in range(0, BUDGET + 1, INTERVAL):
                        point = measure(c, monitors, steps)
                        if steps == 0 and any(point[k] != cached["curve"][0][k] for k in KEYS):
                            raise RuntimeError("Initial frozen traces differ from exact A source")
                        if any(len(point[k]["episodes"]) != COUNT for k in KEYS):
                            raise RuntimeError("Incorrect evaluation episode count")
                        curve.append(point)
                        probes.write(json.dumps(dict(point, seed=seed, arm=arm)) + "\n")
                        probes.flush()
                        if steps == BUDGET:
                            break
                        rows, pending = train_interval(c, arm, pending, INTERVAL)
                        for row in rows:
                            trace.write(json.dumps(dict(row, model_seed=seed, arm=arm)) + "\n")
                        trace.flush()
                    delta = tuple(a - b for a, b in zip(
                        (c.training_actions, c.updates, c.presentations), initial, strict=True
                    ))
                    if delta != (BUDGET, BUDGET // 16, BUDGET * 4):
                        raise RuntimeError("Incorrect total registered training budget")
                    result = {
                        "seed": seed, "arm": arm, "curve": curve, "adam_states_cleared": cleared,
                        "B_training_actions": delta[0], "B_optimizer_updates": delta[1],
                        "B_sample_presentations": delta[2], "pending_B_seed": pending,
                        "evaluation_actions": sum(p[k]["actions"] for p in curve for k in KEYS),
                        "evaluation_episodes": sum(len(p[k]["episodes"]) for p in curve for k in KEYS),
                        "final_sha256": save_boundary(c, output / f"{arm}-{seed}.pt"),
                        "metrics": arm_metrics(curve),
                    }
                    results.append(result)
                    print(json.dumps({"seed": seed, "arm": arm, "metrics": result["metrics"]}),
                          flush=True)
                finally:
                    c.model_env.close()
    summary = {
        **manifest, "kind": "gate_a_active_objective_four_arm", "scientific_pass": False,
        "actual_training_actions": sum(r["B_training_actions"] for r in results),
        "actual_optimizer_updates": sum(r["B_optimizer_updates"] for r in results),
        "actual_sample_presentations": sum(r["B_sample_presentations"] for r in results),
        "actual_evaluation_actions": sum(r["evaluation_actions"] for r in results),
        "actual_evaluation_episodes": sum(r["evaluation_episodes"] for r in results),
        "comparison": summarize(results), "results": results,
        "seconds": time.monotonic() - started,
        "limitations": "Exploratory paired monitors; no held-out or contemporaneous fresh/frozen "
                       "maintained continual controls; not evidence of general transfer.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
