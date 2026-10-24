"""REUSE-34: independent frozen-REUSE-33 complete-demonstration replication.

Keep original imitation.BC recipe/evaluator. Reuse SHA-verified numerical TRAIN
data and test only untouched XLand layouts 10700..10709. No new teacher queries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bagi.experiments import xland_complete_demo_bc as original
from bagi.experiments.xland_gate_b_native_collection import (
    MODEL_SEEDS,
    SOURCE_RUN,
    source_record_checks,
)

CONFIRM_LAYOUTS = tuple(range(10700, 10710))
SOURCE_ARTIFACT = 12362840074
TEACHER_RUN = 40101923830
TEACHER_ARTIFACT = 12563231248
SUMMARY_SHA256 = "eaf2f99a2dae59f3e4fe78a1f296edf72afbe09bc107cbeb382fcb69f3eaf57d"
DEMO_SHA256 = "b405e1ba816f424179dbf65a93a31e5a5db5b471bd8079671ecfd99215688efd"
TEACHER_ROWS = 248
PHYSICAL_TRAIN_EPISODES = 22


def validate_layouts() -> None:
    original.verify_registration()
    if (CONFIRM_LAYOUTS != tuple(range(10700, 10710))
            or len(CONFIRM_LAYOUTS) != len(original.EVAL_LAYOUTS)
            or set(CONFIRM_LAYOUTS) & (set(original.TRAIN_LAYOUTS) | set(original.EVAL_LAYOUTS))
            or MODEL_SEEDS != (17, 29)):
        raise ValueError("Unregistered independent confirmation seeds")


def load_frozen_teacher(directory: Path) -> tuple[dict, dict]:
    """Verify original published source data, including every physical phase."""
    summary = directory / "summary.json"
    numeric = directory / "train_source_complete_demos.npz"
    if (hashlib.sha256(summary.read_bytes()).hexdigest() != SUMMARY_SHA256
            or hashlib.sha256(numeric.read_bytes()).hexdigest() != DEMO_SHA256):
        raise ValueError("REUSE-33 source SHA256 mismatch")
    old = json.loads(summary.read_text(encoding="utf-8"))
    provenance = old["source_teacher_provenance"]
    if (old["status"] != "complete"
            or old["source_run"] != SOURCE_RUN
            or old["source_artifact"] != SOURCE_ARTIFACT
            or old["predeclared_train_layouts"] != list(original.TRAIN_LAYOUTS)
            or old["predeclared_fresh_eval_layouts"] != list(original.EVAL_LAYOUTS)
            or old["heldout_composition_teacher_queries"] != 0
            or old["native_training_teacher_actions"] != TEACHER_ROWS
            or old["source_teacher_query_attempts"] != TEACHER_ROWS
            or provenance["eligible_physical_episode_count"] != PHYSICAL_TRAIN_EPISODES
            or provenance["label_rows"] != TEACHER_ROWS
            or [x["layout"] for x in provenance["cases"]] != list(original.TRAIN_LAYOUTS)):
        raise ValueError("Changed REUSE-33 teacher evidence")
    with np.load(numeric, allow_pickle=False) as archive:
        if set(archive.files) != {"obs", "expert_label", "done"}:
            raise ValueError("Unapproved source NPZ fields")
        extra = {name: np.array(archive[name], copy=True) for name in archive.files}
    labels = extra["expert_label"]
    if (extra["obs"].shape != (TEACHER_ROWS, 115)
            or extra["obs"].dtype != np.float32
            or labels.shape != (TEACHER_ROWS,)
            or labels.dtype != np.int64
            or extra["done"].shape != (TEACHER_ROWS,)
            or extra["done"].dtype != np.bool_
            or not np.isfinite(extra["obs"]).all()
            or np.any((labels < 0) | (labels > 5))
            or int(np.sum(labels == 3)) != PHYSICAL_TRAIN_EPISODES
            or int(np.sum(labels == 4)) != PHYSICAL_TRAIN_EPISODES
            or int(np.sum(extra["done"])) != PHYSICAL_TRAIN_EPISODES
            or Counter(map(str, labels)) != provenance["complete_teacher_action_counts"]):
        raise ValueError("Original complete BLUE pickup/drop labels not preserved")
    return old, extra


def evaluate_original_on_confirm(policy) -> dict:
    """Call the exact earlier XLand evaluator, changing ONLY frozen layout IDs."""
    validate_layouts()
    with patch.object(original, "EVAL_LAYOUTS", CONFIRM_LAYOUTS):
        out = original.evaluate_policy(policy)
    if (set(out) != {"hold", "near", "putdown"}
            or any([x["layout"] for x in result["cases"]] != list(CONFIRM_LAYOUTS)
                   for result in out.values())):
        raise ValueError("Confirmation evaluator layout/skill mismatch")
    return out


def validate_original_recipe(actual: dict, former: dict, checksum: str) -> None:
    if (former["original_source_sha256"] != checksum
            or actual != former["training"]
            or actual["original_source_rows"] != 291
            or actual["added_sampled_rows"] != TEACHER_ROWS
            or actual["total_rows"] != 539
            or actual["epochs"] != 12
            or actual["nominal_optimization_exposures"] != 6468):
        raise ValueError("Original published BC sampling and training changed")


def run(source_dir: Path, teacher_dir: Path, output_dir: Path) -> dict:
    from bagi.experiments.xland_existing_bc import TRAIN_TASKS, get_rules
    from bagi.experiments.xland_instruction_action_history import assemble, check_true_history
    from bagi.experiments.xland_upstream_instruction_bc import (
        load_verified_source,
        task_instructions,
    )

    validate_layouts()
    old, extra = load_frozen_teacher(teacher_dir)
    previous = {(row["model_seed"], row["arm"]): row for row in old["results"]}
    expected = {(seed, name) for seed in MODEL_SEEDS for name in original.ARM_NAMES}
    if len(old["results"]) != len(expected) or set(previous) != expected:
        raise ValueError("Archived REUSE-33 control arms incomplete")
    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    source_record_checks(summary)
    instructions = {
        task: task_instructions(get_rules(task).goal, get_rules(task).rules, enabled=True)
        for task in TRAIN_TASKS
    }
    bases = {}
    for seed in MODEL_SEEDS:
        rows = [r for r in summary["model_results"] if r["model_seed"] == seed]
        if len(rows) != 1:
            raise ValueError("Missing original REUSE-19 model seed")
        data, tasks, checksum = load_verified_source(
            source_dir / f"action_provenance_seed{seed}.npz", rows[0], seed
        )
        check_true_history(data)
        features = assemble(
            data["obs"], np.stack([instructions[task] for task in tasks]),
            data["prior_executed_actions"], "real_executed_action_history",
        )
        bases[seed] = ({
            "obs": features, "expert_label": data["expert_label"], "done": data["done"]
        }, checksum)

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "summary.json"
    report = {
        "experiment": "REUSE-34 original-BC independent physical confirmation",
        "status": "running",
        "source_run": SOURCE_RUN,
        "source_artifact": SOURCE_ARTIFACT,
        "previous_teacher_run": TEACHER_RUN,
        "previous_teacher_artifact": TEACHER_ARTIFACT,
        "verified_old_summary_sha256": SUMMARY_SHA256,
        "verified_old_teacher_data_sha256": DEMO_SHA256,
        "previous_train_layouts": list(original.TRAIN_LAYOUTS),
        "previous_test_layouts_not_reused": list(original.EVAL_LAYOUTS),
        "new_preregistered_test_layouts": list(CONFIRM_LAYOUTS),
        "model_seeds": list(MODEL_SEEDS),
        "arms": list(original.ARM_NAMES),
        "fresh_teacher_queries": 0,
        "fresh_teacher_controlled_native_actions": 0,
        "heldout_composition_teacher_queries": 0,
        "archived_teacher_actions_reused_not_reexecuted": TEACHER_ROWS,
        "total_new_learner_evaluation_native_actions": 0,
        "results": [],
    }
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for seed in MODEL_SEEDS:
        base, checksum = bases[seed]
        learners = original.train_two_arms(base, extra, seed)
        for arm in original.ARM_NAMES:
            tr = learners[arm]["training"]
            validate_original_recipe(tr, previous[(seed, arm)], checksum)
            evaluation = evaluate_original_on_confirm(learners[arm]["policy"])
            steps = sum(x["native_evaluation_steps"] for x in evaluation.values())
            report["total_new_learner_evaluation_native_actions"] += steps
            report["results"].append({
                "model_seed": seed, "arm": arm,
                "original_source_sha256": checksum,
                "training": tr, "evaluation": evaluation,
                "native_evaluation_actions": steps,
                "teacher_queries_during_confirmation": 0,
            })
            target.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"REUSE34 seed={seed} arm={arm} "
                  f"physically_verified_BLUE="
                  f"{evaluation['putdown']['verified_physical_placements']}/"
                  f"{evaluation['putdown']['eligible']} "
                  f"eval_steps={steps}", flush=True)
    if len(report["results"]) != len(expected):
        raise RuntimeError("Unfinished replication; no confirmation success claim")
    report["status"] = "complete"
    report["limitations"] = [
        "Same published upstream imitation.BC recipe is retrained, not a saved-weight checkpoint.",
        "Model seed17 was positive only on two prior layouts; check independent replication.",
        "Original 248 teacher labels are archived TRAIN-only data, no fresh teacher queries.",
        "Hold/near native reward retention and BLUE physical pickup/drop are distinct metrics.",
        "No teacher queries on new layouts, heldout causal chains or unsafe pickle archives.",
        "Two model seeds and ten new layouts are not broad or robust AGI.",
    ]
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--source-dir", type=Path, required=True)
    cli.add_argument("--teacher-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path,
                     default=Path("runs/xland_complete_demo_confirmation"))
    args = cli.parse_args()
    run(args.source_dir, args.teacher_dir, args.output_dir)
