"""REUSE-36: fixed diversity of *complete physical TRAIN* demonstrations.

Same imitation.BC, 291 original source +248 original skill rehearsal +248
complete physical putdown examples per arm. The expanded corpus contains the
original 22 valid TRAIN episodes PLUS 72 fixed new TRAIN layout attempts.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bagi.experiments import xland_complete_demo_bc as original
from bagi.experiments.xland_complete_demo_confirmation import load_frozen_teacher
from bagi.experiments.xland_gate_b_native_collection import (
    MODEL_SEEDS,
    SOURCE_RUN,
    source_record_checks,
)

DIVERSE_TRAIN_LAYOUTS = tuple(range(10900, 10972))
DIVERSE_EVAL_LAYOUTS = tuple(range(11000, 11020))
MIN_COMPLETE_SOURCE_EPISODES = 50
EXTRA_REHEARSAL_ROWS = 248
EXTRA_PHYSICAL_ROWS = 248
ARMS = ("replay_22_teacher_layouts", "expanded_22_plus_72_train_layouts")


def validate_register() -> None:
    original.verify_registration()
    if (DIVERSE_TRAIN_LAYOUTS != tuple(range(10900, 10972))
            or DIVERSE_EVAL_LAYOUTS != tuple(range(11000, 11020))
            or MIN_COMPLETE_SOURCE_EPISODES != 50
            or EXTRA_REHEARSAL_ROWS != 248 or EXTRA_PHYSICAL_ROWS != 248
            or MODEL_SEEDS != (17, 29)
            or original.SOURCE_BASE_ROWS != 291
            or original.TRAIN_EPOCHS != 12 or original.TRAIN_BATCH != 32
            or set(DIVERSE_TRAIN_LAYOUTS) & (
                set(original.TRAIN_LAYOUTS) | set(original.EVAL_LAYOUTS)
                | set(range(10700, 10710)) | set(range(10800, 10810))
                | set(DIVERSE_EVAL_LAYOUTS)
            ) or set(DIVERSE_EVAL_LAYOUTS) & (
                set(original.TRAIN_LAYOUTS) | set(original.EVAL_LAYOUTS)
                | set(range(10700, 10710)) | set(range(10800, 10810))
            )):
        raise ValueError("REUSE-36 train/test layout or published BC budget changed")


def collect_diverse_complete() -> tuple[list[dict], dict]:
    """Attempt ALL fixed TRAIN layouts, returning provenance even if gate fails."""
    validate_register()
    with patch.object(original, "TRAIN_LAYOUTS", DIVERSE_TRAIN_LAYOUTS):
        rows, audit = original.collect_train_demonstrations()
    return rows, audit


def verify_diverse_source(rows: list[dict], audit: dict) -> dict:
    """Reject incomplete or nonphysical TRAIN labels after preserving attempt counts."""
    with patch.object(original, "TRAIN_LAYOUTS", DIVERSE_TRAIN_LAYOUTS):
        original.validate_demonstrations(rows, audit)
    if (audit["predeclared_source_layouts"] != list(DIVERSE_TRAIN_LAYOUTS)
            or [c["layout"] for c in audit["cases"]] != list(DIVERSE_TRAIN_LAYOUTS)
            or audit["eligible_physical_episode_count"] < MIN_COMPLETE_SOURCE_EPISODES
            or len(rows) != audit["label_rows"]
            or audit["native_teacher_actions"] > 72 * 40
            or audit["native_teacher_actions"] > audit["attempted_teacher_queries"]):
        raise ValueError("Not enough physically valid TRAIN source episodes")
    extras = {
        "obs": np.stack([r["public_obs"] for r in rows]).astype(np.float32),
        "expert_label": np.asarray([r["expert_label"] for r in rows], dtype=np.int64),
        "done": np.asarray([r["done"] for r in rows], dtype=bool),
    }
    if (extras["obs"].shape != (len(rows), 115)
            or int(np.sum(extras["expert_label"] == 3)) < MIN_COMPLETE_SOURCE_EPISODES
            or int(np.sum(extras["expert_label"] == 4)) < MIN_COMPLETE_SOURCE_EPISODES):
        raise ValueError("No diverse real pickup/drop teacher stage coverage")
    return audit, extras


def sample_fixed_extra(base: dict, physical: dict, seed: int) -> dict:
    """Exactly 291 +248 original rehearsal +248 phase-complete data rows."""
    validate_register()
    bx = np.asarray(base["obs"], dtype=np.float32)
    by = np.asarray(base["expert_label"], dtype=np.int64)
    bd = np.asarray(base["done"], dtype=bool)
    px = np.asarray(physical["obs"], dtype=np.float32)
    py = np.asarray(physical["expert_label"], dtype=np.int64)
    pd = np.asarray(physical["done"], dtype=bool)
    if (bx.shape != (291, 115) or by.shape != (291,) or bd.shape != (291,)
            or px.ndim != 2 or px.shape[1] != 115
            or len(px) < EXTRA_PHYSICAL_ROWS or py.shape != (len(px),)
            or pd.shape != (len(px),) or seed not in MODEL_SEEDS):
        raise ValueError("Invalid equal exposure source/physical dataset")
    base_ix = original.sampled_indices(by, 291, seed)
    rehearse_ix = original.sampled_indices(by, EXTRA_REHEARSAL_ROWS, seed + 4100)
    physical_ix = original.sampled_indices(py, EXTRA_PHYSICAL_ROWS, seed + 5300)
    return {
        "obs": np.concatenate((bx[base_ix], bx[rehearse_ix], px[physical_ix])),
        "expert_label": np.concatenate((by[base_ix], by[rehearse_ix], py[physical_ix])),
        "done": np.concatenate((bd[base_ix], bd[rehearse_ix], pd[physical_ix])),
    }


def fit_original_bc(base: dict, physical: dict, seed: int):
    import gymnasium as gym
    import torch
    from imitation.algorithms.bc import BC
    from imitation.data.types import Transitions

    data = sample_fixed_extra(base, physical, seed)
    x, y, d = (data[k] for k in ("obs", "expert_label", "done"))
    if x.shape != (787, 115) or len(y) != 787:
        raise ValueError("Changes to fixed source training exposures")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    bc = BC(
        observation_space=gym.spaces.Box(low=-np.inf, high=np.inf,
                                          shape=(115,), dtype=np.float32),
        action_space=gym.spaces.Discrete(6),
        demonstrations=Transitions(
            obs=x, acts=y, next_obs=x.copy(), dones=d,
            infos=np.array([None] * len(y), dtype=object),
        ),
        rng=np.random.default_rng(seed), batch_size=original.TRAIN_BATCH,
        device="cpu",
    )
    bc.train(n_epochs=original.TRAIN_EPOCHS, progress_bar=False)
    return bc.policy, {
        "sampled_original_source": 291,
        "sampled_original_rehearsal": EXTRA_REHEARSAL_ROWS,
        "sampled_complete_physical": EXTRA_PHYSICAL_ROWS,
        "total_rows": len(y), "epochs": original.TRAIN_EPOCHS,
        "nominal_optimization_exposures": len(y) * original.TRAIN_EPOCHS,
        "sampling_method": "torch.WeightedRandomSampler",
        "label_action_counts": dict(Counter(map(int, y))),
        "public_observation_width": 115,
    }


def evaluate_original(policy) -> dict:
    """Only fixed NEW source-disjoint test IDs; original physical evaluator."""
    validate_register()
    with patch.object(original, "EVAL_LAYOUTS", DIVERSE_EVAL_LAYOUTS):
        report = original.evaluate_policy(policy)
    if (set(report) != {"hold", "near", "putdown"}
            or any([c["layout"] for c in x["cases"]] != list(DIVERSE_EVAL_LAYOUTS)
                   for x in report.values())):
        raise ValueError("Changed source task/evaluation policy")
    return report


def run(source_dir: Path, original_teacher_dir: Path, output_dir: Path) -> dict:
    from bagi.experiments.xland_existing_bc import TRAIN_TASKS, get_rules
    from bagi.experiments.xland_instruction_action_history import assemble, check_true_history
    from bagi.experiments.xland_upstream_instruction_bc import (
        load_verified_source,
        task_instructions,
    )

    validate_register()
    _old, old_demo = load_frozen_teacher(original_teacher_dir)
    source_meta = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    source_record_checks(source_meta)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "summary.json"
    result = {
        "experiment": "REUSE-36 equal-exposure 22 vs 22-plus-72 source demonstration diversity",
        "status": "collecting", "source_run": SOURCE_RUN,
        "source_artifact": 12362840074,
        "archived_small_complete_teacher_run": 40101923830,
        "archived_small_teacher_artifact": 12563231248,
        "predeclared_diverse_train_layouts": list(DIVERSE_TRAIN_LAYOUTS),
        "predeclared_fresh_eval_layouts": list(DIVERSE_EVAL_LAYOUTS),
        "minimum_diverse_complete_train_episodes": MIN_COMPLETE_SOURCE_EPISODES,
        "model_seeds": list(MODEL_SEEDS), "arms": list(ARMS),
        "new_teacher_queries": 0, "new_native_teacher_actions": 0,
        "new_native_eval_actions": 0,
        "heldout_composition_teacher_queries": 0,
        "results": [],
    }
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows, audit = collect_diverse_complete()
    result["source_teacher_provenance"] = audit
    result["new_teacher_queries"] = audit["attempted_teacher_queries"]
    result["new_native_teacher_actions"] = audit["native_teacher_actions"]
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    diverse_demo = verify_diverse_source(rows, audit)
    np.savez_compressed(output_dir / "diverse_train_physical.npz", **diverse_demo)
    # The only intervention is EXPANDING the exact frozen teacher corpus with
    # all physically complete episodes from the 72 additional TRAIN attempts.
    expanded = {
        field: np.concatenate((old_demo[field], diverse_demo[field]))
        for field in ("obs", "expert_label", "done")
    }
    result["original_teacher_rows"] = len(old_demo["expert_label"])
    result["new_teacher_rows"] = len(diverse_demo["expert_label"])
    result["expanded_teacher_rows"] = len(expanded["expert_label"])
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    instructions = {
        name: task_instructions(get_rules(name).goal, get_rules(name).rules, enabled=True)
        for name in TRAIN_TASKS
    }
    for seed in MODEL_SEEDS:
        verified = [r for r in source_meta["model_results"] if r["model_seed"] == seed]
        if len(verified) != 1:
            raise ValueError("Missing independently verified original data seed")
        data, tasks, checksum = load_verified_source(
            source_dir / f"action_provenance_seed{seed}.npz", verified[0], seed
        )
        check_true_history(data)
        base = {
            "obs": assemble(data["obs"],
                            np.stack([instructions[t] for t in tasks]),
                            data["prior_executed_actions"],
                            "real_executed_action_history"),
            "expert_label": data["expert_label"], "done": data["done"],
        }
        for arm, demos in zip(ARMS, (old_demo, expanded)):
            policy, tr = fit_original_bc(base, demos, seed)
            evaluation = evaluate_original(policy)
            native = sum(task["native_evaluation_steps"] for task in evaluation.values())
            result["new_native_eval_actions"] += native
            result["results"].append({
                "model_seed": seed, "arm": arm, "original_source_sha256": checksum,
                "training": tr, "evaluation": evaluation,
                "native_evaluation_actions": native, "teacher_queries_during_eval": 0,
            })
            destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"REUSE36 seed={seed} {arm} physical="
                  f"{evaluation['putdown']['verified_physical_placements']}/"
                  f"{evaluation['putdown']['eligible']} "
                  f"hold={evaluation['hold']['native_rewarded']} "
                  f"near={evaluation['near']['native_rewarded']} "
                  f"actions={native}", flush=True)
    if len(result["results"]) != 4:
        raise RuntimeError("Incomplete diversity control, no advantage claim")
    result["status"] = "complete"
    result["limitations"] = [
        "One fixed 72-new-layout TRAIN demonstration expansion; no sweep or replacements.",
        "Equal 787 sampled labels, 12 epochs and identical original skill rehearsal.",
        "Different teacher-source diversity requires real extra TRAIN teacher queries.",
        "All initial ineligible source and eval layouts are recorded, never replaced.",
        "Only evaluator uses true native grid/pocket; model receives public 115 features.",
        "Native reward-only putdown without effective BLUE pickup/drop is not physical skill.",
        "No heldout compositional expert labels, unsafe pickle or original architecture changes.",
    ]
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--old-teacher-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/xland_diverse_physical_demos"))
    args = parser.parse_args()
    run(args.source_dir, args.old_teacher_dir, args.output_dir)
