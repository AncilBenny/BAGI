"""REUSE-35: one preregistered complete-demonstration + original-skill rehearsal mix.

Reuses published XLand / imitation.BC, same 291 original verified rows,
248 additional balanced samples, 12 epochs, 115 public features, native
physical evaluator. No new teacher examples or novel optimizer.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bagi.experiments import xland_complete_demo_bc as original
from bagi.experiments.xland_complete_demo_confirmation import (
    load_frozen_teacher,
    validate_original_recipe,
)
from bagi.experiments.xland_gate_b_native_collection import (
    MODEL_SEEDS,
    SOURCE_RUN,
    source_record_checks,
)

REHEARSAL_LAYOUTS = tuple(range(10800, 10810))
REHEARSAL_ORIGINAL = 124
REHEARSAL_COMPLETE = 124
MIXED_ARM = "half_original_half_complete_teacher"
CONDITIONS = (*original.ARM_NAMES, MIXED_ARM)


def check_registration() -> None:
    original.verify_registration()
    if (REHEARSAL_LAYOUTS != tuple(range(10800, 10810))
            or REHEARSAL_ORIGINAL != 124 or REHEARSAL_COMPLETE != 124
            or original.SOURCE_BASE_ROWS != 291
            or original.TRAIN_EPOCHS != 12
            or original.TRAIN_BATCH != 32 or MODEL_SEEDS != (17, 29)
            or set(REHEARSAL_LAYOUTS) & (
                set(original.TRAIN_LAYOUTS) | set(original.EVAL_LAYOUTS)
                | set(range(10700, 10710))
            )):
        raise ValueError("Unpreregistered REUSE-35 model or evaluation controls")


def sample_mixed_training_rows(base: dict, demo: dict, seed: int) -> dict:
    """Same sampled original 291 rows plus exactly 124 original+124 teacher."""
    check_registration()
    x = np.asarray(base["obs"], dtype=np.float32)
    y = np.asarray(base["expert_label"], dtype=np.int64)
    done = np.asarray(base["done"], dtype=bool)
    tx = np.asarray(demo["obs"], dtype=np.float32)
    ty = np.asarray(demo["expert_label"], dtype=np.int64)
    td = np.asarray(demo["done"], dtype=bool)
    if (x.shape != (291, 115) or y.shape != (291,) or done.shape != (291,)
            or tx.shape != (248, 115) or ty.shape != (248,)
            or td.shape != (248,) or seed not in MODEL_SEEDS):
        raise ValueError("Changed original/teacher data shapes or model seed")
    base_ids = original.sampled_indices(y, 291, seed)
    old_ids = original.sampled_indices(y, REHEARSAL_ORIGINAL, seed + 4100)
    demo_ids = original.sampled_indices(ty, REHEARSAL_COMPLETE, seed + 4100)
    answer = {
        "obs": np.concatenate((x[base_ids], x[old_ids], tx[demo_ids])),
        "expert_label": np.concatenate((y[base_ids], y[old_ids], ty[demo_ids])),
        "done": np.concatenate((done[base_ids], done[old_ids], td[demo_ids])),
    }
    if answer["obs"].shape != (539, 115):
        raise ValueError("Unpreregistered additional sampling ratio")
    return answer


def train_mixed(base: dict, teacher: dict, seed: int):
    import gymnasium as gym
    import torch
    from imitation.algorithms.bc import BC
    from imitation.data.types import Transitions

    data = sample_mixed_training_rows(base, teacher, seed)
    x, y, done = (data[k] for k in ("obs", "expert_label", "done"))
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    trainer = BC(
        observation_space=gym.spaces.Box(low=-np.inf, high=np.inf,
                                          shape=(115,), dtype=np.float32),
        action_space=gym.spaces.Discrete(6),
        demonstrations=Transitions(
            obs=x, acts=y, next_obs=x.copy(), dones=done,
            infos=np.array([None] * len(y), dtype=object),
        ),
        rng=np.random.default_rng(seed), batch_size=original.TRAIN_BATCH,
        device="cpu",
    )
    trainer.train(n_epochs=original.TRAIN_EPOCHS, progress_bar=False)
    summary = {
        "original_source_rows": 291, "added_sampled_rows": 248,
        "added_original_rehearsal_rows": REHEARSAL_ORIGINAL,
        "added_complete_teacher_rows": REHEARSAL_COMPLETE,
        "total_rows": 539, "epochs": original.TRAIN_EPOCHS,
        "nominal_optimization_exposures": 539 * original.TRAIN_EPOCHS,
        "sampling_method": "torch.WeightedRandomSampler",
        "label_action_counts": dict(Counter(map(int, y))),
        "same_original_base_rows_as_other_arms": True,
    }
    return trainer.policy, summary


def evaluate_original(policy) -> dict:
    """Run the EXACT established native skill/physical evaluator on new seeds."""
    check_registration()
    with patch.object(original, "EVAL_LAYOUTS", REHEARSAL_LAYOUTS):
        data = original.evaluate_policy(policy)
    if (set(data) != {"hold", "near", "putdown"}
            or any([row["layout"] for row in item["cases"]] != list(REHEARSAL_LAYOUTS)
                   for item in data.values())):
        raise ValueError("Unexpected source task/evaluation layout substitution")
    return data


def run(source_dir: Path, teacher_dir: Path, output_dir: Path) -> dict:
    from bagi.experiments.xland_existing_bc import TRAIN_TASKS, get_rules
    from bagi.experiments.xland_instruction_action_history import assemble, check_true_history
    from bagi.experiments.xland_upstream_instruction_bc import (
        load_verified_source,
        task_instructions,
    )

    check_registration()
    older, complete_demos = load_frozen_teacher(teacher_dir)
    comparison = {(row["model_seed"], row["arm"]): row for row in older["results"]}
    if (len(older["results"]) != 4
            or set(comparison) != {(seed, arm) for seed in MODEL_SEEDS
                                  for arm in original.ARM_NAMES}):
        raise ValueError("Original REUSE-33 arm/archive provenance changed")
    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    source_record_checks(summary)
    instructions = {
        task: task_instructions(get_rules(task).goal, get_rules(task).rules, enabled=True)
        for task in TRAIN_TASKS
    }
    bases = {}
    for seed in MODEL_SEEDS:
        old_rows = [m for m in summary["model_results"] if m["model_seed"] == seed]
        if len(old_rows) != 1:
            raise ValueError("Original verified learner seed absent or duplicated")
        data, tasks, checksum = load_verified_source(
            source_dir / f"action_provenance_seed{seed}.npz", old_rows[0], seed
        )
        check_true_history(data)
        visible = assemble(data["obs"],
                           np.stack([instructions[task] for task in tasks]),
                           data["prior_executed_actions"],
                           "real_executed_action_history")
        bases[seed] = ({
            "obs": visible, "expert_label": data["expert_label"],
            "done": data["done"],
        }, checksum)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "summary.json"
    result = {
        "experiment": "REUSE-35 one fixed 50/50 original-skill rehearsal comparison",
        "status": "running", "source_run": SOURCE_RUN,
        "original_source_artifact": 12362840074,
        "archived_teacher_run": 40101923830,
        "archived_teacher_artifact": 12563231248,
        "old_train_layouts": list(original.TRAIN_LAYOUTS),
        "old_eval_layouts_excluded": [*original.EVAL_LAYOUTS, *range(10700, 10710)],
        "new_frozen_eval_layouts": list(REHEARSAL_LAYOUTS),
        "model_seeds": list(MODEL_SEEDS), "conditions": list(CONDITIONS),
        "rehearsal_added_original_samples": 124,
        "rehearsal_added_complete_teacher_samples": 124,
        "all_arm_training_rows": 539, "all_arm_epochs": 12,
        "new_teacher_queries": 0, "new_native_teacher_actions": 0,
        "heldout_composition_teacher_queries": 0,
        "new_native_evaluation_actions": 0, "results": [],
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for seed in MODEL_SEEDS:
        base, sha = bases[seed]
        original_arms = original.train_two_arms(base, complete_demos, seed)
        mixed_policy, mixed_training = train_mixed(base, complete_demos, seed)
        for arm in CONDITIONS:
            if arm == MIXED_ARM:
                policy, training = mixed_policy, mixed_training
            else:
                policy = original_arms[arm]["policy"]
                training = original_arms[arm]["training"]
                validate_original_recipe(training, comparison[(seed, arm)], sha)
            evaluated = evaluate_original(policy)
            n = sum(item["native_evaluation_steps"] for item in evaluated.values())
            result["new_native_evaluation_actions"] += n
            result["results"].append({
                "model_seed": seed, "condition": arm, "source_sha256": sha,
                "training": training, "evaluation": evaluated,
                "native_evaluation_actions": n, "teacher_queries_during_eval": 0,
            })
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"REUSE35 model={seed} arm={arm} BLUE="
                  f"{evaluated['putdown']['verified_physical_placements']}/"
                  f"{evaluated['putdown']['eligible']}, "
                  f"hold={evaluated['hold']['native_rewarded']}, "
                  f"near={evaluated['near']['native_rewarded']}, "
                  f"real_eval_actions={n}", flush=True)
    if len(result["results"]) != 6:
        raise RuntimeError("Incomplete REUSE-35 three-arm comparison")
    result["status"] = "complete"
    result["limitations"] = [
        "Only one preregistered rehearsal ratio, no mixture hyperparameter sweep.",
        "No new teacher labels; original 248 complete teacher examples reused.",
        "Three arms share original 291 base samples/seed, 539 sampled rows and 12 epochs.",
        "The third arm mixes original skill and physical-stage TRAIN supervision only.",
        "Hold/near skill retention and physical BLUE pickup/drop must BOTH be reported.",
        "Old evaluated layouts are excluded and unchanged native evaluator reused.",
        "A positive result in two model seeds is not established cross-seed competence.",
    ]
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/xland_complete_demo_rehearsal"))
    args = parser.parse_args()
    run(args.source_dir, args.teacher_dir, args.output_dir)
