"""REUSE-31: run the bounded Gate B collector on actual upstream XLand.

This runner only collects labels; it does NOT train post-intervention policies
or claim physical competence. Source policy reconstruction uses the unchanged
REUSE-23 imitation.BC recipe and SHA256-validated REUSE-19 numeric traces.
It never exposes native state/grid/pocket information to the learner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bagi.experiments.xland_gate_b_exact_collection import (
    ARMS,
    BALANCED_LAYOUT_QUOTAS,
    CAP,
    FRESH_EVAL_LAYOUTS,
    QUERY_BUDGET,
    SOURCE_LAYOUTS,
    collect_arm,
    public_features,
    save_numeric,
    validate_pair,
)

SOURCE_RUN = 39637067506
MODEL_SEEDS = (17, 29)
ENVIRONMENT = "XLand-MiniGrid-R1-9x9"


def screen_source_layout(timestep: object, blue: np.ndarray,
                         red: np.ndarray) -> bool:
    """Diagnostic-only nonadjacent screening; never passes geometry to policy."""
    grid = np.asarray(timestep.state.grid)
    locations = [np.argwhere(np.all(grid == obj, axis=-1))
                 for obj in (blue, red)]
    if any(len(positions) != 1 for positions in locations):
        return False
    distance = sum(abs(int(a) - int(b))
                   for a, b in zip(locations[0][0], locations[1][0]))
    return distance > 1


def source_record_checks(summary: dict) -> None:
    if (summary.get("total_expert_queries") != 582
            or summary.get("heldout_teacher_labels") != 0
            or {x.get("model_seed") for x in summary.get("model_results", [])}
            != set(MODEL_SEEDS)):
        raise ValueError("Unrecognized REUSE-19 verified source metadata")


def rebuild_frozen_policy(source_dir: Path, source_summary: dict,
                          seed: int):
    """Rebuild original REUSE-23 public-instruction+real-history BC, unchanged."""
    import gymnasium as gym
    import torch
    from imitation.algorithms.bc import BC
    from imitation.data.types import Transitions

    from bagi.experiments.xland_bc_class_balance import balanced_indices
    from bagi.experiments.xland_existing_bc import TRAIN_TASKS, get_rules
    from bagi.experiments.xland_instruction_action_history import (
        WIDTH,
        assemble,
        check_true_history,
    )
    from bagi.experiments.xland_upstream_instruction_bc import (
        load_verified_source,
        task_instructions,
    )

    if seed not in MODEL_SEEDS or WIDTH != 115:
        raise ValueError("Unapproved learner seed or feature width")
    meta = [x for x in source_summary["model_results"]
            if x["model_seed"] == seed]
    if len(meta) != 1:
        raise ValueError("Missing unique verified policy source")
    data, tasks, checksum = load_verified_source(
        source_dir / f"action_provenance_seed{seed}.npz", meta[0], seed
    )
    check_true_history(data)
    codes = {
        task: task_instructions(
            get_rules(task).goal, get_rules(task).rules, enabled=True
        ) for task in TRAIN_TASKS
    }
    token_rows = np.stack([codes[task] for task in tasks])
    features = assemble(
        data["obs"], token_rows, data["prior_executed_actions"],
        "real_executed_action_history",
    )
    sampled = balanced_indices(data["expert_label"], seed)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    learner = BC(
        observation_space=gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(WIDTH,), dtype=np.float32
        ),
        action_space=gym.spaces.Discrete(6),
        demonstrations=Transitions(
            obs=features[sampled], acts=data["expert_label"][sampled],
            next_obs=features[sampled].copy(),
            dones=np.asarray(data["done"][sampled], dtype=bool),
            infos=np.array([None] * len(sampled), dtype=object),
        ),
        rng=np.random.default_rng(seed), batch_size=32, device="cpu",
    )
    learner.train(n_epochs=12, progress_bar=False)
    return learner.policy, checksum


def collect_native(policy, room, defaults, blue: np.ndarray,
                   red: np.ndarray):
    """Use identical native params/source seeds and actual dynamics for arms."""
    import jax
    from xminigrid.wrappers import (
        DirectionObservationWrapper,
        RulesAndGoalsObservationWrapper,
    )

    from bagi.experiments.xland_dagger_teacher_feasibility import correction_plan
    from bagi.experiments.xland_existing_bc import get_rules
    from bagi.experiments.xland_gate_b_exact_collection import native_action

    env = RulesAndGoalsObservationWrapper(DirectionObservationWrapper(room))
    params = defaults.replace(ruleset=get_rules("putdown"), max_steps=CAP)
    reset_jax = jax.jit(env.reset)
    step_jax = jax.jit(env.step)

    def reset(seed):
        return reset_jax(params, jax.random.key(int(seed)))

    def step(ts, action):
        return step_jax(params, ts, native_action(np.asarray(action)))

    def teacher(ts):
        plan, stage = correction_plan(ts, "putdown")
        if not plan:
            raise ValueError("Empty privileged teacher plan")
        return native_action(np.asarray(plan[0])), stage

    def learner(obs):
        command, _ = policy.predict(obs, deterministic=True)
        return native_action(np.asarray(command))

    kwargs = {
        "reset": reset,
        "step": step,
        "screen_nonadjacent": lambda ts: screen_source_layout(ts, blue, red),
        "feature_fn": public_features,
        "teacher": teacher,
        "learner": learner,
        "seeds": tuple(seed for seed, _ in BALANCED_LAYOUT_QUOTAS),
        "per_layout_quota": tuple(count for _, count in BALANCED_LAYOUT_QUOTAS),
        "budget": QUERY_BUDGET,
        "cap": CAP,
    }
    return {
        arm: collect_arm(arm=arm, **kwargs)
        for arm in ARMS
    }


def run(source_dir: Path, output_dir: Path) -> dict:
    """Create numeric artifact+accounting even if a preregistered gate fails."""
    import xminigrid
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles

    if not source_dir.is_dir():
        raise FileNotFoundError("Supply extracted REUSE-19 artifact 12362840074")
    summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    source_record_checks(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    blue = np.asarray(TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE])
    red = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.RED])
    if set(FRESH_EVAL_LAYOUTS) & set(SOURCE_LAYOUTS):
        raise ValueError("Source and fresh evaluation seeds overlapped")

    results = []
    for seed in MODEL_SEEDS:
        frozen, checksum = rebuild_frozen_policy(source_dir, summary, seed)
        room, defaults = xminigrid.make(ENVIRONMENT)
        pairs = collect_native(frozen, room, defaults, blue, red)
        row = {
            "model_seed": seed,
            "numeric_source_sha256": checksum,
            "arms": {k: pairs[k].summary() for k in ARMS},
            "matched": False,
        }
        for arm, collection in pairs.items():
            if collection.rows:
                save_numeric(str(output_dir / f"seed{seed}_{arm}.npz"), collection)
        try:
            row["pair_accounting"] = validate_pair(
                pairs["reset_expert"], pairs["learner_correction"],
                per_layout_quota=tuple(count for _, count in BALANCED_LAYOUT_QUOTAS),
            )
            row["matched"] = True
        except ValueError as exc:
            row["failure_reason"] = str(exc)
        results.append(row)
        print(f"seed={seed} matched={row['matched']} "
              f"reset={pairs['reset_expert'].summary()['label_rows']} "
              f"correction={pairs['learner_correction'].summary()['label_rows']}")

    answer = {
        "experiment": "REUSE-31 actual native XLand bounded label collection",
        "source_run": SOURCE_RUN,
        "source_artifact": 12362840074,
        "model_seeds": list(MODEL_SEEDS),
        "source_layouts": list(SOURCE_LAYOUTS),
        "fixed_source_layout_quotas": {
            str(seed): count for seed, count in BALANCED_LAYOUT_QUOTAS
        },
        "future_evaluation_layouts_not_queried": list(FRESH_EVAL_LAYOUTS),
        "teacher_only_train_putdown": True,
        "new_heldout_teacher_queries": 0,
        "new_postintervention_policy_training_steps": 0,
        "new_independent_physical_skill_successes": None,
        "real_native_collection_steps": sum(
            a["native_steps"] for x in results for a in x["arms"].values()
        ),
        "real_teacher_query_attempts": sum(
            a["teacher_queries"] for x in results for a in x["arms"].values()
        ),
        "baseline_bc_training_source_rows_per_seed": 291,
        "baseline_bc_epochs": 12,
        "baseline_bc_nominal_sample_exposures_per_seed": 291 * 12,
        "intervention_training_sample_exposures": 0,
        "comparison_complete": all(x["matched"] for x in results),
        "results": results,
        "limitations": [
            "This is label COLLECTION, not matched postintervention retraining.",
            "Original base learner is reconstructed by the prior 12-epoch BC recipe.",
            "Teacher can see simulator state, but model features are public only.",
            "Unsupported/adjacent predeclared layouts fail closed; never replaced.",
            "Teacher-controlled reset rollout rewards are not learned competence.",
            "Fresh evaluation layouts are reserved and have no teacher labels.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(answer, indent=2), encoding="utf-8"
    )
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/xland_gate_b_native_collection"))
    args = parser.parse_args()
    run(args.source_dir, args.output_dir)
