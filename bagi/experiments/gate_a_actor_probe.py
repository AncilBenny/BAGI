"""Frozen actor component counterfactuals from verified A and B snapshots."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from bagi.agents.actor_graft import MODES, temporary_actor
from bagi.experiments.cumulative_transfer import TASKS
from bagi.experiments.gate_a_plasticity import read_summary
from bagi.experiments.gate_a_sequential_pilot import (
    DIGESTS,
    SOURCE,
    SUMMARY_SHA,
    evaluate,
    frozen_state,
)
from bagi.experiments.gate_a_shared_interface import INTERFACE, load_boundary
from bagi.workshop_runtime import source_revision

ADAM_SOURCE = "03771b72dc87a41ce0c8162e0a24a0bb680ca3c3"
ADAM_SUMMARY = "5d5fc712771d684b3fd0eed4426e3389da591a009461f41a281ee47a8908efa7"
CRITIC_SOURCE = "0c9ad91c597466eac3446ac813da48f4402e02ca"
CRITIC_SUMMARY = "83d679084779795ef6f58f915506ae9d6f7e5f4125cbde231adc574b9a95a3e9"
ARMS = ("adam_reset", "adam_and_critic_reset")
KEYS = ("A_greedy", "A_live", "B_greedy", "B_live")


def final_record(summary, seed, arm):
    rows = [r for r in summary["results"] if r["seed"] == seed and r["arm"] == arm]
    if len(rows) != 1 or rows[0]["B_training_actions"] != 32768:
        raise ValueError("Expected unique matched B record")
    return rows[0]


def drift(recipient, donor):
    result = {}
    for k, v in recipient.items():
        if not k.startswith(("lstm_actor.", "mlp_extractor.policy_net.", "action_net.")):
            continue
        delta = v.double() - donor[k].double()
        result[k] = {"changed": int(torch.count_nonzero(delta)),
                     "total": delta.numel(), "l2": float(torch.linalg.vector_norm(delta))}
    return result


def run(inputs, adam_dir, critic_dir, output):
    inputs, adam_dir, critic_dir, output = map(Path, (inputs, adam_dir, critic_dir, output))
    a = read_summary(inputs / "summary.json", SUMMARY_SHA, SOURCE)
    adam = read_summary(adam_dir / "summary.json", ADAM_SUMMARY, ADAM_SOURCE)
    critic = read_summary(critic_dir / "summary.json", CRITIC_SUMMARY, CRITIC_SOURCE)
    for summary in (adam, critic):
        if (summary["cached_A_source"] != a["source"]
                or summary["config"] != {"budget": 32768, "interval": 8192, "eval_count": 40}):
            raise ValueError("Historical sources or monitor settings mismatch")
    if critic["cached_adam_source"] != adam["source"]:
        raise ValueError("Critic used another Adam control")
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    results = []
    for index, (seed, digest) in enumerate(DIGESTS.items()):
        record = next(r for r in a["results"] if r["seed"] == seed
                      and r["domain"] == TASKS[0])
        if record["final_sha256"] != digest:
            raise ValueError("A checkpoint digest mismatch")
        donor_control = load_boundary(inputs / f"final-0-{seed}.pt", digest,
                                      expected_source=a["source"])
        try:
            donor = {k: v.detach().clone()
                     for k, v in donor_control.model.policy.state_dict().items()}
        finally:
            donor_control.model_env.close()
        base = 71_000_000 + index * 1_000_000
        monitors = (range(base + 300_000, base + 300_040),
                    range(base + 500_000, base + 500_040))
        initial = final_record(adam, seed, ARMS[0])["curve"][0]
        for arm, folder, summary in zip(ARMS, (adam_dir, critic_dir), (adam, critic),
                                        strict=True):
            record = final_record(summary, seed, arm)
            control = load_boundary(folder / f"{arm}-{seed}.pt", record["final_sha256"],
                                    expected_source=summary["source"])
            try:
                p = control.model.policy
                parameter_drift = drift(p.state_dict(), donor)
                for mode in MODES:
                    evaluated = {}
                    with frozen_state(control), temporary_actor(p, donor, mode):
                        for label, domain, seeds in zip(("A", "B"), TASKS, monitors, strict=True):
                            for suffix, greedy in (("greedy", True), ("live", False)):
                                evaluated[f"{label}_{suffix}"] = evaluate(
                                    control, domain, seeds, deterministic=greedy
                                )
                    if mode in ("original", "A_actor"):
                        expected = record["curve"][-1] if mode == "original" else initial
                        if any(evaluated[k] != expected[k] for k in KEYS):
                            raise RuntimeError("Frozen positive control trace mismatch")
                    if mode == "A_path" and evaluated["A_greedy"] != initial["A_greedy"]:
                        raise RuntimeError("Restored navigation path mismatch")
                    results.append({
                        "seed": seed, "recipient": arm, "mode": mode,
                        "recipient_sha256": record["final_sha256"], "donor_sha256": digest,
                        "actor_parameter_drift": parameter_drift, "evaluations": evaluated,
                        "evaluation_actions": sum(v["actions"] for v in evaluated.values()),
                    })
            finally:
                control.model_env.close()
    summary = {
        "kind": "gate_a_frozen_actor_component_probe", "scientific_pass": False,
        "source": source_revision(), "interface": INTERFACE, "versions": a["versions"],
        "historical_sources": [a["source"], adam["source"], critic["source"]],
        "historical_artifacts": [15175703476, 15184379503, 15188037454],
        "actual_training_actions": 0, "actual_optimizer_updates": 0,
        "actual_sample_presentations": 0,
        "actual_evaluation_actions": sum(r["evaluation_actions"] for r in results),
        "actual_evaluation_episodes": sum(len(v["episodes"]) for r in results
                                          for v in r["evaluations"].values()),
        "seconds": time.monotonic() - started, "results": results,
        "limitations": "Frozen component substitutions; interactions and coadaptation confound "
                       "localization. No new learning, replay, or independent physical groups.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
