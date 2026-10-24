"""REUSE-31 Gate B: matched upstream imitation.BC training and native evaluation.

Run ONLY after exact 20+20 XLand label collection passed provenance checks.
The established BC model, observation contract and source label pool are reused.
Privileged XLand state is used only for physical *evaluation*, never the policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bagi.experiments.xland_gate_b_exact_collection import (
    ARMS,
    CAP,
    FRESH_EVAL_LAYOUTS,
    INPUT_WIDTH,
    QUERY_BUDGET,
    SOURCE_LAYOUTS,
    load_verified_collection_pair,
    native_action,
    public_features,
)
from bagi.experiments.xland_gate_b_native_collection import (
    ENVIRONMENT,
    MODEL_SEEDS,
    SOURCE_RUN,
    screen_source_layout,
    source_record_checks,
)

BASE_ROWS = 291
EPOCHS = 12
BATCH_SIZE = 32
RETAINED_TASKS = ("hold", "near")
EVALUATED_TASKS = (*RETAINED_TASKS, "putdown")


def validate_collection_header(
    directory: Path, source_summary: dict
) -> tuple[dict, list[dict]]:
    """Check the full two-seed experiment before accessing data or training.

    The existing verified numeric-pair loader independently checks each saved
    teacher-stage code, action/history bit, learner origin, and source SHA.
    """
    source_record_checks(source_summary)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (report.get("source_run") != SOURCE_RUN
            or report.get("source_artifact") != 12362840074
            or report.get("model_seeds") != list(MODEL_SEEDS)
            or report.get("source_layouts") != list(SOURCE_LAYOUTS)
            or report.get("future_evaluation_layouts_not_queried")
            != list(FRESH_EVAL_LAYOUTS)
            or report.get("new_heldout_teacher_queries") != 0
            or report.get("new_postintervention_policy_training_steps") != 0
            or report.get("real_native_collection_steps") != 4 * QUERY_BUDGET
            or report.get("real_teacher_query_attempts") != 4 * QUERY_BUDGET
            or report.get("comparison_complete") is not True):
        raise ValueError("REUSE-31 native collection gate not passed")
    models = report.get("results", [])
    if (len(models) != len(MODEL_SEEDS)
            or [m.get("model_seed") for m in models] != list(MODEL_SEEDS)):
        raise ValueError("Missing or reordered model seed collection")
    if any(not m.get("matched") for m in models):
        raise ValueError("Incomplete model seed comparison")
    return report, models


def train_matched_bc(base_x: np.ndarray, base_y: np.ndarray,
                     base_done: np.ndarray, new: dict, seed: int):
    """Same base rows, original class-balanced base sampler, BC/optimizer budget."""
    if (base_x.shape != (BASE_ROWS, INPUT_WIDTH)
            or base_y.shape != (BASE_ROWS,) or base_done.shape != (BASE_ROWS,)
            or new["obs"].shape != (QUERY_BUDGET, INPUT_WIDTH)
            or new["expert_label"].shape != (QUERY_BUDGET,)
            or new["done"].shape != (QUERY_BUDGET,)):
        raise ValueError("Unexpected original or added BC feature contract")
    import gymnasium as gym
    import torch
    from imitation.algorithms.bc import BC
    from imitation.data.types import Transitions

    from bagi.experiments.xland_bc_class_balance import balanced_indices

    chosen = balanced_indices(base_y, seed)
    x = np.concatenate((base_x[chosen], new["obs"]))
    y = np.concatenate((base_y[chosen], new["expert_label"]))
    d = np.concatenate((base_done[chosen], new["done"]))
    assert x.shape == (BASE_ROWS + QUERY_BUDGET, INPUT_WIDTH)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    bc = BC(
        observation_space=gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(INPUT_WIDTH,), dtype=np.float32
        ),
        action_space=gym.spaces.Discrete(6),
        demonstrations=Transitions(
            obs=x, acts=y, next_obs=x.copy(), dones=d.astype(bool),
            infos=np.array([None] * len(y), dtype=object),
        ),
        rng=np.random.default_rng(seed), batch_size=BATCH_SIZE, device="cpu",
    )
    bc.train(n_epochs=EPOCHS, progress_bar=False)
    return bc.policy, {"training_rows": len(y), "base_rows": BASE_ROWS,
                       "added_labels": QUERY_BUDGET, "epochs": EPOCHS,
                       "batch_size": BATCH_SIZE,
                       "optimization_sample_exposures": EPOCHS * len(y),
                       "sampled_original_base_rows_identical_between_arms": True}


def evaluate_native(policy, env, defaults) -> dict:
    """Closed-loop 40-step evaluation: true physical milestones, no oracle."""
    import jax
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles

    from bagi.experiments.xland_existing_bc import get_rules
    from bagi.experiments.xland_putdown_reward_validity import (
        blue_square_transition,
        physically_nonvacuous,
        tiles_adjacent,
    )

    blue = np.asarray(TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE])
    red = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.RED])
    empty = int(Tiles.EMPTY)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    by_task = {}
    for task in EVALUATED_TASKS:
        params = defaults.replace(ruleset=get_rules(task), max_steps=CAP)
        cases = []
        for layout in FRESH_EVAL_LAYOUTS:
            ts = reset(params, jax.random.key(layout))
            is_putdown = task == "putdown"
            eligible = not is_putdown or screen_source_layout(ts, blue, red)
            if not eligible:
                cases.append({"layout": layout, "eligible": False,
                              "native_actions": 0, "native_reward": False,
                              "physical_placement": False})
                continue  # No posthoc layout substitution.
            history = [6, 6, 6, 6]
            reward_seen = False
            blue_pickups = blue_drops = 0
            saw_blue = ordered_drop = False
            initial_adjacent = bool(is_putdown and tiles_adjacent(
                np.asarray(ts.state.grid), blue, red
            ))
            count = 0
            while count < CAP and not bool(ts.last()):
                # The learner gets only PUBLIC data, never the pocket/grid below.
                feature = public_features(ts, tuple(history))
                action, _ = policy.predict(feature, deterministic=True)
                executed = native_action(np.asarray(action))
                if is_putdown:
                    before_pocket = np.asarray(ts.state.agent.pocket).copy()
                    before_grid = np.asarray(ts.state.grid).copy()
                nxt = step(params, ts, executed)
                if is_putdown:
                    picked, dropped = blue_square_transition(
                        executed, before_pocket,
                        np.asarray(nxt.state.agent.pocket), before_grid,
                        np.asarray(nxt.state.grid), blue, empty,
                    )
                    blue_pickups += int(picked)
                    blue_drops += int(dropped)
                    ordered_drop |= bool(dropped and saw_blue)
                    saw_blue |= bool(picked)
                ts = nxt
                reward_seen |= float(ts.reward) > 0
                history = [*history[1:], executed]
                count += 1
            final_adjacent = bool(is_putdown and tiles_adjacent(
                np.asarray(ts.state.grid), blue, red
            ))
            actual_skill = bool(is_putdown and physically_nonvacuous(
                reward_seen, initial_adjacent, blue_pickups, blue_drops,
                final_adjacent, ordered_drop,
            ))
            cases.append({"layout": layout, "eligible": True,
                          "native_actions": count, "native_reward": bool(reward_seen),
                          "blue_pickups": blue_pickups, "blue_drops": blue_drops,
                          "final_adjacent": final_adjacent,
                          "physical_placement": actual_skill})
        by_task[task] = {"cases": cases,
                         "eligible_layouts": sum(int(x["eligible"]) for x in cases),
                         "native_rewarded": sum(int(x["native_reward"]) for x in cases),
                         "verified_physical_placements": sum(
                             int(x["physical_placement"]) for x in cases),
                         "native_actions": sum(x["native_actions"] for x in cases)}
    return by_task


def run(source_dir: Path, collection_dir: Path, output_dir: Path) -> dict:
    """No new teacher labels; report actual native evaluation steps only."""
    from bagi.experiments.xland_existing_bc import TRAIN_TASKS, get_rules
    from bagi.experiments.xland_instruction_action_history import (
        assemble,
        check_true_history,
    )
    from bagi.experiments.xland_upstream_instruction_bc import (
        load_verified_source,
        task_instructions,
    )

    source_summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    collection_report, records = validate_collection_header(collection_dir, source_summary)

    # Validate BOTH source policies and BOTH matched arms before ANY retraining.
    # The authoritative REUSE-31 loader checks NPZ history, stages, source
    # provenance and paired accounting, so we do not duplicate its rules.
    verified = {}
    for model in records:
        seed = model["model_seed"]
        meta = [x for x in source_summary["model_results"] if x["model_seed"] == seed]
        if len(meta) != 1:
            raise ValueError("Missing unique REUSE-19 source learner seed")
        source, tasks, checksum = load_verified_source(
            source_dir / f"action_provenance_seed{seed}.npz", meta[0], seed
        )
        check_true_history(source)
        paired = load_verified_collection_pair(
            collection_dir, model_seed=seed, source_checksum=checksum,
            model_record=model,
        )
        verified[seed] = (source, tasks, checksum, paired)

    import xminigrid
    from xminigrid.wrappers import (
        DirectionObservationWrapper,
        RulesAndGoalsObservationWrapper,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    room, defaults = xminigrid.make(ENVIRONMENT)
    env = RulesAndGoalsObservationWrapper(DirectionObservationWrapper(room))
    models = []
    for model in records:
        seed = model["model_seed"]
        source, tasks, checksum, paired = verified[seed]
        instructions = {
            task: task_instructions(
                get_rules(task).goal, get_rules(task).rules, enabled=True
            ) for task in TRAIN_TASKS
        }
        codes = np.stack([instructions[task] for task in tasks])
        base_x = assemble(source["obs"], codes,
                          source["prior_executed_actions"],
                          "real_executed_action_history")
        arms = []
        for arm in ARMS:
            policy, counts = train_matched_bc(
                base_x, source["expert_label"], source["done"], paired[arm], seed
            )
            evaluated = evaluate_native(policy, env, defaults)
            arms.append({"arm": arm, "training": counts,
                         "evaluation": evaluated, "new_teacher_queries": 0,
                         "actual_evaluation_native_actions": sum(
                             task["native_actions"] for task in evaluated.values())})
        models.append({"model_seed": seed, "source_sha256": checksum, "arms": arms})
    report = {"experiment": "REUSE-31 Gate B matched upstream BC closed-loop",
              "source_run": SOURCE_RUN, "source_artifact": 12362840074,
              "predeclared_eval_layouts": list(FRESH_EVAL_LAYOUTS),
              "evaluation_episode_cap": CAP, "model_seeds": list(MODEL_SEEDS),
              "training_method": "unchanged imitation.BC and base WeightedRandomSampler",
              "per_arm_training_labels": BASE_ROWS + QUERY_BUDGET,
              "new_teacher_queries": 0, "heldout_teacher_queries": 0,
              "actual_evaluation_native_actions": sum(
                  arm["actual_evaluation_native_actions"]
                  for model in models for arm in model["arms"]),
              "learner_only_results": models,
              "limitations": [
                  "Tiny 20-label augmentation is diagnostic, not proof of broad AGI.",
                  "Evaluation is 5 frozen source-task layouts and two model seeds.",
                  "Original base teacher demonstrations are retained identically.",
                  "Initial adjacent putdown layouts are ineligible, not replaced.",
                  "Rewards without actual BLUE pickup/drop are not physical success.",
                  "Teacher controls the reset-arm DATA, never its evaluated policy.",
                  "No heldout composition teacher labels or unsafe pickle data.",
              ]}
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/xland_gate_b_matched_learning"))
    args = parser.parse_args()
    run(args.source_dir, args.collection_dir, args.output_dir)
