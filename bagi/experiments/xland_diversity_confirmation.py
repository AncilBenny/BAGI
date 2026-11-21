"""REUSE-37: independent unchanged-BC confirmation of REUSE-36 source diversity.

Repeat original REUSE-36 published BC sampler, policy and optimization budgets.
Replay the exact SHA256-locked numeric TRAIN demonstrations without new teacher
queries; evaluate on a new preregistered source-task XLand layout set only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bagi.experiments import xland_complete_demo_bc as original
from bagi.experiments import xland_diverse_physical_demos as diversity
from bagi.experiments.xland_complete_demo_confirmation import load_frozen_teacher
from bagi.experiments.xland_gate_b_native_collection import (
    MODEL_SEEDS,
    SOURCE_RUN,
    source_record_checks,
)

CONFIRM_EVAL_LAYOUTS = tuple(range(11100, 11120))
SOURCE_DIVERSITY_RUN = 40110945017
SOURCE_DIVERSITY_ARTIFACT = 12565959908
SOURCE_DIVERSITY_SUMMARY_SHA256 = "a178131f01724a9c14cd1ea16e45b8bd62eb17e447b6e751e2297264402277af"
SOURCE_DIVERSITY_NPZ_SHA256 = "e6c605bc9a109e2e366eb50635361bae9cd4c93563c282d787640694280d75da"


def check_confirmation_registration() -> None:
    diversity.validate_register()
    if (CONFIRM_EVAL_LAYOUTS != tuple(range(11100, 11120))
            or SOURCE_DIVERSITY_RUN != 40110945017
            or SOURCE_DIVERSITY_ARTIFACT != 12565959908
            or original.TRAIN_EPOCHS != 12 or original.TRAIN_BATCH != 32
            or MODEL_SEEDS != (17, 29)
            or set(CONFIRM_EVAL_LAYOUTS) & (
                set(diversity.DIVERSE_TRAIN_LAYOUTS)
                | set(diversity.DIVERSE_EVAL_LAYOUTS)
                | set(original.TRAIN_LAYOUTS)
                | set(original.EVAL_LAYOUTS)
                | set(range(10700, 10710))
                | set(range(10800, 10810))
            )):
        raise ValueError("REUSE-37 locked source, model, or eval layouts changed")


def require_previous_training(prior: dict, seed: int, arm: str,
                              current: dict, source_sha256: str) -> None:
    """Reject changes in sampled labels/seed/config before confirmation."""
    original_record = [r for r in prior.get("results", [])
                       if r.get("model_seed") == seed and r.get("arm") == arm]
    if (len(original_record) != 1
            or original_record[0]["original_source_sha256"] != source_sha256
            or json.loads(json.dumps(current)) != original_record[0]["training"]
            or current["total_rows"] != 787
            or current["epochs"] != 12
            or current["nominal_optimization_exposures"] != 9444):
        raise ValueError("Original published REUSE-36 sampling recipe changed")


def load_frozen_diversity(directory: Path) -> tuple[dict, dict]:
    """Safe numeric-only rows and exact original artifact hashes; no pickle."""
    check_confirmation_registration()
    summary = directory / "summary.json"
    numeric = directory / "diverse_train_physical.npz"
    if (hashlib.sha256(summary.read_bytes()).hexdigest()
            != SOURCE_DIVERSITY_SUMMARY_SHA256
            or hashlib.sha256(numeric.read_bytes()).hexdigest()
            != SOURCE_DIVERSITY_NPZ_SHA256):
        raise ValueError("Unreviewed REUSE-36 source archive SHA256")
    data = json.loads(summary.read_text(encoding="utf-8"))
    counts = data["source_teacher_provenance"]
    if (data.get("status") != "complete"
            or data.get("source_run") != SOURCE_RUN
            or data.get("source_artifact") != 12362840074
            or data.get("predeclared_diverse_train_layouts")
            != list(diversity.DIVERSE_TRAIN_LAYOUTS)
            or data.get("predeclared_fresh_eval_layouts")
            != list(diversity.DIVERSE_EVAL_LAYOUTS)
            or data.get("model_seeds") != list(MODEL_SEEDS)
            or data.get("arms") != list(diversity.ARMS)
            or len(data.get("results", [])) != 4
            or data.get("new_teacher_queries") != 707
            or data.get("new_native_teacher_actions") != 707
            or data.get("new_native_eval_actions") != 7457
            or counts.get("eligible_physical_episode_count") != 66
            or counts.get("label_rows") != 707
            or data.get("expanded_teacher_rows") != 955
            or data.get("new_heldout_composition_teacher_queries", 0) != 0):
        raise ValueError("Source XLand diversity evidence changed")
    with np.load(numeric, allow_pickle=False) as arr:
        if set(arr.files) != {"obs", "expert_label", "done"}:
            raise ValueError("Unknown REUSE-36 archive field")
        teacher = {key: np.array(arr[key], copy=True) for key in arr.files}
    if (teacher["obs"].shape != (707, 115)
            or teacher["obs"].dtype != np.float32
            or teacher["expert_label"].shape != (707,)
            or teacher["expert_label"].dtype != np.int64
            or teacher["done"].shape != (707,)
            or teacher["done"].dtype != np.bool_
            or not np.isfinite(teacher["obs"]).all()
            or int(np.count_nonzero(teacher["expert_label"] == 3)) != 66
            or int(np.count_nonzero(teacher["expert_label"] == 4)) != 66
            or int(np.count_nonzero(teacher["done"])) != 66):
        raise ValueError("Physical BLUE TRAIN numeric label coverage altered")
    return data, teacher


def evaluate_fresh(policy) -> dict:
    """Use the ORIGINAL 40-native-step physical evaluator, new layouts only."""
    check_confirmation_registration()
    with patch.object(original, "EVAL_LAYOUTS", CONFIRM_EVAL_LAYOUTS):
        outcomes = original.evaluate_policy(policy)
    if (set(outcomes) != {"hold", "near", "putdown"}
            or any([c["layout"] for c in row["cases"]]
                   != list(CONFIRM_EVAL_LAYOUTS) for row in outcomes.values())):
        raise ValueError("Confirmation evaluator layout/skill was substituted")
    return outcomes


def run(source_dir: Path, old_teacher_dir: Path,
        diverse_dir: Path, output_dir: Path) -> dict:
    from bagi.experiments.xland_existing_bc import TRAIN_TASKS, get_rules
    from bagi.experiments.xland_instruction_action_history import assemble, check_true_history
    from bagi.experiments.xland_upstream_instruction_bc import (
        load_verified_source,
        task_instructions,
    )

    check_confirmation_registration()
    _, old_teacher = load_frozen_teacher(old_teacher_dir)
    previous, new_teacher = load_frozen_diversity(diverse_dir)
    expanded = {name: np.concatenate((old_teacher[name], new_teacher[name]))
                for name in ("obs", "expert_label", "done")}
    if len(expanded["expert_label"]) != 955:
        raise ValueError("Original+new completed teacher source rows changed")
    source_meta = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    source_record_checks(source_meta)
    instructions = {
        name: task_instructions(get_rules(name).goal, get_rules(name).rules, enabled=True)
        for name in TRAIN_TASKS
    }
    base_by_seed = {}
    for seed in MODEL_SEEDS:
        verified = [x for x in source_meta["model_results"] if x["model_seed"] == seed]
        if len(verified) != 1:
            raise ValueError("Original verified source model seed not unique")
        data, tasks, sha = load_verified_source(
            source_dir / f"action_provenance_seed{seed}.npz", verified[0], seed
        )
        check_true_history(data)
        features = assemble(data["obs"],
                            np.stack([instructions[task] for task in tasks]),
                            data["prior_executed_actions"],
                            "real_executed_action_history")
        base_by_seed[seed] = ({
            "obs": features, "expert_label": data["expert_label"],
            "done": data["done"],
        }, sha)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "summary.json"
    report = {
        "experiment": "REUSE-37 independent original-versus-expanded physical-demo replication",
        "status": "running", "source_run": SOURCE_RUN,
        "source_artifact": 12362840074,
        "frozen_old_teacher_run": 40101923830,
        "frozen_diversity_run": SOURCE_DIVERSITY_RUN,
        "frozen_diversity_artifact": SOURCE_DIVERSITY_ARTIFACT,
        "frozen_diversity_sha256": SOURCE_DIVERSITY_NPZ_SHA256,
        "predeclared_fresh_eval_layouts": list(CONFIRM_EVAL_LAYOUTS),
        "never_reuse_old_eval_layouts": list(diversity.DIVERSE_EVAL_LAYOUTS),
        "original_training_layouts": list(original.TRAIN_LAYOUTS),
        "additional_training_layouts": list(diversity.DIVERSE_TRAIN_LAYOUTS),
        "model_seeds": list(MODEL_SEEDS), "arms": list(diversity.ARMS),
        "new_teacher_queries": 0, "new_teacher_executed_actions": 0,
        "heldout_composition_teacher_queries": 0,
        "new_native_eval_actions": 0, "results": [],
    }
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for seed in MODEL_SEEDS:
        base, sha = base_by_seed[seed]
        for arm, teachers in zip(diversity.ARMS, (old_teacher, expanded)):
            policy, tr = diversity.fit_original_bc(base, teachers, seed)
            require_previous_training(previous, seed, arm, tr, sha)
            e = evaluate_fresh(policy)
            actions = sum(t["native_evaluation_steps"] for t in e.values())
            report["new_native_eval_actions"] += actions
            report["results"].append({
                "model_seed": seed, "arm": arm, "original_source_sha256": sha,
                "training": tr, "evaluation": e,
                "native_evaluation_actions": actions,
                "teacher_queries_during_eval": 0,
            })
            target.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"REUSE37 model={seed} arm={arm} BLUE="
                  f"{e['putdown']['verified_physical_placements']}/"
                  f"{e['putdown']['eligible']} "
                  f"hold={e['hold']['native_rewarded']} "
                  f"near={e['near']['native_rewarded']} "
                  f"real_native_eval={actions}", flush=True)
    if len(report["results"]) != 4:
        raise ValueError("Incomplete REUSE-37 independent paired confirmation")
    report["status"] = "complete"
    report["limitations"] = [
        "Exactly previous imitation.BC and weights/sampler/exposures; new source-task layouts only.",
        "No new teacher queries or teacher-controlled native actions.",
        "Physical nonvacuous BLUE sequence is required, reward alone is insufficient.",
        "20 frozen layout seeds and two policy seeds; neither success nor null proves AGI.",
        "A single independent confirmation does not make cross-seed behavior reliable.",
    ]
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--old-teacher-dir", type=Path, required=True)
    p.add_argument("--diverse-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/reuse37_confirmation"))
    args = p.parse_args()
    run(args.source_dir, args.old_teacher_dir, args.diverse_dir, args.output_dir)
