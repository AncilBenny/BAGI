"""REUSE-20: audit XLand's existing rule/goal instruction observation boundary.

The 54-feature prior learner sees local tiles + heading, not the task's goal
and rules. Upstream RulesAndGoalsObservationWrapper exists for exactly this.
This TRAINING-INTERFACE audit NEVER fits a policy or queries a heldout oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from bagi.experiments.xland_existing_bc import (
    ALL_TASKS,
    TRAIN_TASKS,
    get_rules,
    visible_features,
)

SHARED_INITIAL_WORLD_PAIR = ("hold", "hold_then_near")
DIAGNOSTIC_SEEDS = (9300, 9301, 9302)


def encoding_digest(value):
    array = np.ascontiguousarray(np.asarray(value))
    sha = hashlib.sha256()
    sha.update(array.dtype.str.encode())
    sha.update(str(array.shape).encode())
    sha.update(array.tobytes())
    return sha.hexdigest()


def task_observation_record(plain, instructed, state, ruleset):
    """Report only shapes/hashes; learner obtains rule tokens via upstream wrapper."""
    baseline_keys = tuple(sorted(plain))
    instructed_keys = tuple(sorted(instructed))
    if baseline_keys != ("direction", "img"):
        raise ValueError("Prior feature boundary unexpectedly changed")
    if instructed_keys != (
        "direction", "goal_encoding", "img", "rule_encoding"
    ):
        raise ValueError("Upstream XLand instruction wrapper missing fields")
    prior = visible_features(plain)
    same_local = np.array_equal(prior, visible_features(instructed))
    same_goal = np.array_equal(
        np.asarray(instructed["goal_encoding"]),
        np.asarray(state.goal_encoding)
    )
    same_rules = np.array_equal(
        np.asarray(instructed["rule_encoding"]),
        np.asarray(state.rule_encoding)
    )
    matches_params = (
        np.array_equal(
            np.asarray(instructed["goal_encoding"]),
            np.asarray(ruleset.goal)
        ) and np.array_equal(
            np.asarray(instructed["rule_encoding"]),
            np.asarray(ruleset.rules)
        )
    )
    if not all((same_local, same_goal, same_rules, matches_params)):
        raise AssertionError("Task instruction wrapper was not identity preserving")
    return {
        "baseline_observation_keys": list(baseline_keys),
        "instructed_observation_keys": list(instructed_keys),
        "prior_local_image_heading_feature_count": int(prior.size),
        "visible_local_features_unchanged_by_instruction_wrapper": same_local,
        "wrapped_goal_is_upstream_state_and_params_goal": same_goal and matches_params,
        "wrapped_rules_are_upstream_state_and_params_rules": same_rules and matches_params,
        "goal_shape": list(np.asarray(instructed["goal_encoding"]).shape),
        "rules_shape": list(np.asarray(instructed["rule_encoding"]).shape),
        "goal_sha256": encoding_digest(instructed["goal_encoding"]),
        "rules_sha256": encoding_digest(instructed["rule_encoding"]),
        "initial_image_heading_sha256": encoding_digest(prior),
    }


def same_physical_initial_state(left, right):
    """Compare full state ONLY for benchmark IDENTIFIABILITY diagnostic."""
    for member in ("grid",):
        if not np.array_equal(
            np.asarray(getattr(left, member)),
            np.asarray(getattr(right, member))
        ):
            return False
    for member in ("position", "direction", "pocket"):
        if not np.array_equal(
            np.asarray(getattr(left.agent, member)),
            np.asarray(getattr(right.agent, member))
        ):
            return False
    return True


def run(config_path):
    import jax
    import xminigrid
    from xminigrid.wrappers import (
        DirectionObservationWrapper,
        RulesAndGoalsObservationWrapper,
    )

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if (cfg["environment"] != "XLand-MiniGrid-R1-9x9"
            or cfg["train_tasks"] != list(TRAIN_TASKS)
            or cfg["comparison_task"] != "hold_then_near"
            or tuple(cfg["audit_seeds"]) != DIAGNOSTIC_SEEDS
            or cfg["step_budget"] != 0):
        raise ValueError("Frozen no-learning task-interface audit changed")
    room, defaults = xminigrid.make(cfg["environment"])
    plain_env = DirectionObservationWrapper(room)
    task_env = RulesAndGoalsObservationWrapper(plain_env)
    plain_reset = jax.jit(plain_env.reset)
    instructed_reset = jax.jit(task_env.reset)
    data = []
    state_cache = {}
    reset_calls = 0
    for task in ALL_TASKS:
        params = defaults.replace(ruleset=get_rules(task), max_steps=40)
        for seed in DIAGNOSTIC_SEEDS:
            key = jax.random.key(seed)
            plain = plain_reset(params, key)
            instructed = instructed_reset(params, key)
            reset_calls += 2
            if not same_physical_initial_state(
                plain.state, instructed.state
            ):
                raise AssertionError("Adding task obs changed physical world")
            observation = task_observation_record(
                plain.observation, instructed.observation,
                instructed.state, params.ruleset
            )
            state_cache[(task, seed)] = {
                "state": plain.state, "record": observation,
            }
            data.append({
                "task": task,
                "split": "train" if task in TRAIN_TASKS else "heldout_specs_only",
                "seed": seed,
                **observation,
            })
    same_views = []
    for seed in DIAGNOSTIC_SEEDS:
        first = state_cache[(SHARED_INITIAL_WORLD_PAIR[0], seed)]
        other = state_cache[(SHARED_INITIAL_WORLD_PAIR[1], seed)]
        same_state = same_physical_initial_state(first["state"], other["state"])
        same_local_features = (
            first["record"]["initial_image_heading_sha256"]
            == other["record"]["initial_image_heading_sha256"]
        )
        different_goal = (
            first["record"]["goal_sha256"]
            != other["record"]["goal_sha256"]
        )
        different_rules = (
            first["record"]["rules_sha256"]
            != other["record"]["rules_sha256"]
        )
        if not all((same_state, same_local_features,
                    different_goal, different_rules)):
            raise AssertionError(
                "Hold and composition instruction identifiability assumptions changed"
            )
        same_views.append({
            "seed": seed,
            "identical_initial_physical_state": same_state,
            "identical_old_local_learner_features": same_local_features,
            "different_upstream_goal_encoding": different_goal,
            "different_upstream_rule_encoding": different_rules,
        })
    summary = {
        "question": "Was earlier BAGI causal transfer tested without the benchmark's own task-instruction observation?",
        "upstream_component": "xminigrid.wrappers.RulesAndGoalsObservationWrapper",
        "prior_observation": "DirectionObservationWrapper -> img+direction only",
        "task_observation": "DirectionObservationWrapper + RulesAndGoalsObservationWrapper",
        "training_tasks": list(TRAIN_TASKS),
        "heldout_spec_inspection_only": ["hold_then_near"],
        "policy_training_steps": 0,
        "real_simulator_steps": 0,
        "environment_reset_calls": reset_calls,
        "teacher_label_queries": 0,
        "heldout_expert_labels": 0,
        "no_learned_policy_in_this_audit": True,
        "tasks": data,
        "hold_vs_heldout_identifiability": same_views,
        "limitations": [
            "Instruction tokens are upstream learner-available only when the wrapper is explicitly applied; existing 54-feature policies were not conditioned on them.",
            "Identical initial inputs for hold and the chain do not prove that every possible policy must ultimately fail; subsequent observations and termination can differ.",
            "Inspecting heldout goal/rule SHAPES/HASHES only is not using heldout trajectories, labels or reward to train or tune.",
            "Rule-encoding length can vary across tasks; an eventual trained task-conditioned baseline needs explicit fixed-shape padding/masking and matched source-task interactions.",
            "Task-conditioned policy would change the observation information budget and is NOT directly comparable to old partially observed learner as a same-input control.",
            "No learning, evaluation reward or claim of compositional transfer arises from this interface audit.",
        ],
    }
    dst = Path(cfg["output_dir"]) / "summary.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Upstream instruction observation verified for {len(data)} cases; "
          f"shared initial layout matches={len(same_views)}; resets={reset_calls}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/xland_rule_goal_observation.yaml")
    run(p.parse_args().config)
