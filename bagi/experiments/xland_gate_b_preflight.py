"""REUSE-31: audit archived correction-label provenance before Gate B training.

Pure JSON analysis of immutable REUSE-29/30 native-action summaries.
No simulation, model reconstruction, oracle queries, or training.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SOURCE_RUN = 39735997245
SOURCE_ARTIFACT = 12407282471
SOURCE_COMMIT = "18586bb19a81795bfe6a8a642fbadfd1257eeffb"
REGISTERED = [[9701, 4], [9702, 10]]
REUSE29_TRACE = "3b620a8c6090f756a0dcf4e9af276a1526c59f040c5708dcad77f8ba8a630d36"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(reuse30, control):
    """Count learner-reached checkpoints separately from expert-induced states."""
    require(reuse30["registered_probes"] == REGISTERED, "changed probe seeds/checkpoints")
    require(reuse30["policy_model_seed"] == 29, "changed frozen model seed")
    require(reuse30["fixed_model_training_labels_source"] == 39637067506,
            "changed source data")
    require(reuse30["valid_distinct_checkpoints"] is True, "duplicate checkpoints")
    require(reuse30["new_policy_training_steps"] == 0, "unexpected retraining")
    require(reuse30["heldout_composition_teacher_queries"] == 0, "heldout labels")
    require(reuse30["total_train_only_teacher_label_queries"] == 20,
            "changed REUSE-30 teacher budget")
    require(reuse30["total_native_simulator_step_calls"] == 114,
            "changed REUSE-30 real step count")
    recovery = control["recovery_probe"]
    require(control["policy_seed"] == 29, "changed REUSE-29 model seed")
    require(recovery["native_actions_sha256_le_int64"] == REUSE29_TRACE,
            "REUSE-29 physical trace not verified")
    require(recovery["teacher_wrong_red_release_queries"] == 1
            and recovery["subsequent_train_only_correction_queries"] == 19,
            "changed REUSE-29 correction budget")
    require(control["actual_train_only_expert_label_queries"] == 20
            and control["heldout_composition_teacher_queries"] == 0,
            "changed REUSE-29 label split")

    cases = reuse30["cases"]
    require(len(cases) == 2, "expected two fixed cases")
    stage_counts = Counter()
    action_counts = Counter()
    checkpoint_labels = []
    continuation_queries = 0
    archived_observation_rows = 0
    for (seed, checkpoint), case in zip(REGISTERED, cases, strict=True):
        require((case["seed"], case["checkpoint"]) == (seed, checkpoint),
                "case order or checkpoint changed")
        require(case["excluded_initial_adjacency"] is False
                and case["initial_distance"] > 1, "vacuous layout")
        require(case["teacher_assisted_physical_placement"] is True
                and case["learned_success"] is False, "changed competence outcome")
        require(case["effective_blue_pickups_on_full_trace"] >= 1
                and case["effective_blue_drops_on_full_trace"] >= 1
                and case["ordered_blue_pickup_then_drop"] is True
                and case["final_blue_adjacent_to_red"] is True,
                "physical placement not established")
        require(not case["teacher_errors"], "unsupported teacher query")
        learner = case["learner_only"]
        prefix = case["learner_prefix_actions"]
        actions = case["teacher_actions"]
        require(learner["reward_success"] == 0
                and learner["real_steps"] == 40
                and len(learner["executed_actions"]) == 40,
                "learner-only control changed")
        require(len(prefix) == checkpoint
                and prefix == learner["executed_actions"][:checkpoint],
                "teacher handoff was not at a real learner checkpoint")
        require(len(actions) == case["teacher_label_queries"] and actions,
                "teacher actions/queries disagree")
        require(all(isinstance(a, int) and 0 <= a < 6 for a in actions),
                "non-native teacher action")
        require(sum(case["teacher_correction_stage_counts"].values())
                == len(actions), "causal-stage counts disagree")
        require(case["native_simulator_step_calls"]
                == learner["real_steps"] + len(prefix) + len(actions),
                "real interaction accounting mismatch")
        # Archived actions are NOT a supervised dataset: no PUBLIC observations.
        require(not any(k in case for k in ("observations", "obs", "public_obs")),
                "unexpected archived observations; revisit preflight conclusion")
        checkpoint_labels.append({"layout_seed": seed,
                                  "checkpoint_step": checkpoint,
                                  "first_teacher_action": actions[0]})
        continuation_queries += len(actions) - 1
        stage_counts.update(case["teacher_correction_stage_counts"])
        action_counts.update(actions)

    require(sum(stage_counts.values()) == 20, "unexpected correction query count")
    require(continuation_queries == 18, "unexpected expert-controlled continuation")
    return {
        "study": "REUSE-31 / Gate B correction-label provenance preflight",
        "contribution_type": "archived-evidence audit, not policy learning",
        "source_commit": SOURCE_COMMIT,
        "source_run_id": SOURCE_RUN,
        "source_artifact_id": SOURCE_ARTIFACT,
        "frozen_policy_training_seed": 29,
        "planner_random_seed": None,
        "new_model_training_steps": 0,
        "new_real_simulator_steps": 0,
        "new_teacher_queries": 0,
        "new_heldout_teacher_queries": 0,
        "historical_reuse30_real_simulator_steps": 114,
        "historical_reuse30_teacher_queries": 20,
        "historical_reuse29_teacher_queries": 20,
        "reuse30_learner_visited_checkpoint_labels": len(checkpoint_labels),
        "reuse30_expert_controlled_continuation_labels": continuation_queries,
        "reuse30_learner_checkpoint_fraction": len(checkpoint_labels) / 20,
        "combined_reuse29_30_learner_checkpoint_labels": 3,
        "combined_reuse29_30_total_teacher_labels": 40,
        "archived_public_observation_rows": archived_observation_rows,
        "reset_arm_new_labels": 0,
        "checkpoint_labels": checkpoint_labels,
        "correction_stage_label_counts": dict(sorted(stage_counts.items())),
        "correction_action_label_counts": {
            str(a): action_counts[a] for a in range(6)
        },
        "teacher_assisted_physical_successes": 2,
        "learner_only_physical_successes": 0,
        "interpretation": (
            "Only the first query at each frozen learner checkpoint labels a "
            "learner-visited state; 18 subsequent queries label teacher-controlled "
            "states. Archived action traces lack legal public observation features "
            "and cannot alone train a matched BC/DAgger arm."
        ),
        "next_measurement": (
            "Preregister a matched 20-query train-only correction-vs-reset "
            "collection with PUBLIC observation/true executed-action-history "
            "rows, matched simulator steps and model seeds [17,29]; then test "
            "nonadjacent fresh-layout BLUE pickup/drop and hold/near retention."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse30", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(
        json.loads(Path(args.reuse30).read_text(encoding="utf-8")),
        json.loads(Path(args.control).read_text(encoding="utf-8")),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("Gate B preflight: learner checkpoints",
          result["reuse30_learner_visited_checkpoint_labels"],
          "/20; teacher-continuation",
          result["reuse30_expert_controlled_continuation_labels"], "/20")


if __name__ == "__main__":
    main()
