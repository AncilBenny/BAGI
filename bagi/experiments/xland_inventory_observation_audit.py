"""REUSE-22: test XLand's hidden inventory feedback at rewarded putdown states.

The standard transparent_field_of_view code does NOT include agent.pocket.
Replay an existing train-only successful putdown oracle to an actual, reachable
pre-drop state. Construct a declared COUNTERFACTUAL version changing ONLY the
hidden pocket. Compare real upstream local and instruction-wrapped observations
and one-step native-action outcomes. No learned model, new oracle rules, or
held-out composition trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from bagi.experiments.xland_existing_bc import expert_actions, get_rules, visible_features

SEEDS = (7300, 7301, 7302)
ACTION_PUTDOWN = 4


def hash_array(value):
    array = np.ascontiguousarray(np.asarray(value))
    dig = hashlib.sha256()
    dig.update(str(array.shape).encode())
    dig.update(str(array.dtype).encode())
    dig.update(array.tobytes())
    return dig.hexdigest()


def assert_identical_learner_views(full, altered):
    """All upstream local, direction, goals and rules must be identical."""
    if tuple(sorted(full)) != ("direction", "goal_encoding", "img", "rule_encoding"):
        raise ValueError("Expected canonical upstream instructed observation")
    if set(full) != set(altered):
        raise AssertionError("Changing pocket changed observable field names")
    equal = {
        name: bool(np.array_equal(np.asarray(full[name]), np.asarray(altered[name])))
        for name in full
    }
    if not all(equal.values()):
        raise AssertionError("Pocket-only counterfactual was visible in upstream view")
    local = visible_features(full)
    if local.shape != (54,) or not np.isfinite(local).all():
        raise AssertionError("Changed learner input geometry or values")
    return equal


def one_case(room, params, seed):
    import jax
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
    from xminigrid.core.observation import transparent_field_of_view
    from xminigrid.wrappers import (
        DirectionObservationWrapper,
        RulesAndGoalsObservationWrapper,
    )

    instructed = RulesAndGoalsObservationWrapper(DirectionObservationWrapper(room))
    # This is a TRAINING-SKILL oracle; only its action list is used for replay.
    actions = expert_actions(room, params, "putdown", seed, 40)
    if len(actions) >= 40 or actions[-1] != ACTION_PUTDOWN:
        raise AssertionError("Expected bounded, rewarded final putdown action")
    reset, step = jax.jit(instructed.reset), jax.jit(instructed.step)
    ts = reset(params, jax.random.key(seed))
    for act in actions[:-1]:
        if bool(ts.last()):
            raise AssertionError("Oracle terminated before rewarded drop")
        ts = step(params, ts, int(act))
    pre = ts.state
    square = np.asarray(TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE])
    empty = TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY]
    if not np.array_equal(np.asarray(pre.agent.pocket), square):
        raise AssertionError("Actual oracle pre-drop state must carry blue square")
    # Deliberately a COUNTERFACTUAL intervention, not claimed reachable:
    # all physical state elements except pocket are literally unchanged.
    empty_state = pre.replace(agent=pre.agent.replace(pocket=empty))
    alt = ts.replace(state=empty_state)
    # Recompute the exact upstream core image from the two states.
    original_img = transparent_field_of_view(
        pre.grid, pre.agent, params.view_size, params.view_size
    )
    empty_img = transparent_field_of_view(
        empty_state.grid, empty_state.agent, params.view_size, params.view_size
    )
    if not np.array_equal(
        np.asarray(original_img), np.asarray(ts.observation["img"])
    ):
        raise AssertionError("Core observation reconstruction disagreed with actual")
    if not np.array_equal(np.asarray(original_img), np.asarray(empty_img)):
        raise AssertionError("Pocket unexpectedly changed upstream image")
    # Both timesteps have the same learner-visible img/direction/instructions;
    # construction is checked against the pristine simulator result above.
    fields = assert_identical_learner_views(ts.observation, alt.observation)
    full_after = step(params, ts, ACTION_PUTDOWN)
    empty_after = step(params, alt, ACTION_PUTDOWN)
    real_reward = float(full_after.reward)
    cf_reward = float(empty_after.reward)
    real_empty = int(np.asarray(full_after.state.agent.pocket)[0]) == int(Tiles.EMPTY)
    cf_empty = int(np.asarray(empty_after.state.agent.pocket)[0]) == int(Tiles.EMPTY)
    if real_reward <= 0 or cf_reward != 0:
        raise AssertionError("Counterfactual did not distinguish native putdown reward")
    if not real_empty or not cf_empty:
        raise AssertionError("Pocket did not match post-action putdown contract")
    real_grid = np.asarray(full_after.state.grid)
    cf_grid = np.asarray(empty_after.state.grid)
    if np.array_equal(real_grid, cf_grid):
        raise AssertionError("Native action did not differ when only pocket changed")
    return {
        "training_only_putdown_seed": seed,
        "oracle_actions_generated": len(actions),
        "oracle_prefix_replay_simulator_steps": len(actions) - 1,
        "native_counterfactual_action_calls": 2,
        "pre_action_teacher_goal_reward": float(ts.reward),
        "real_pre_action_carried_blue_square": True,
        "alternative_pre_action_pocket_empty": True,
        "all_upstream_observable_fields_equal": fields,
        "local_learner_features_sha256": hash_array(visible_features(ts.observation)),
        "full_instruction_fields_sha256": {
            name: hash_array(ts.observation[name])
            for name in ("goal_encoding", "rule_encoding")
        },
        "original_pocket_putdown_reward": real_reward,
        "empty_pocket_counterfactual_reward": cf_reward,
        "original_pocket_putdown_changes_grid": True,
        "same_action_produces_different_next_grid": True,
        "counterfactual_is_simulator_reachable": False,
    }


def run(config_path):
    import xminigrid

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if (cfg["environment"] != "XLand-MiniGrid-R1-9x9"
            or cfg["task"] != "putdown"
            or cfg["seeds"] != list(SEEDS)
            or cfg["native_action"] != ACTION_PUTDOWN
            or cfg["episode_cap"] != 40
            or cfg["heldout_oracle_queries"] != 0):
        raise ValueError("Preregistered inventory aliasing audit changed")
    room, defaults = xminigrid.make(cfg["environment"])
    params = defaults.replace(ruleset=get_rules("putdown"), max_steps=40)
    cases = [one_case(room, params, seed) for seed in SEEDS]
    report = {
        "question": "Is carried-object pocket state omitted from upstream task-conditioned learner observation although it controls putdown reward?",
        "established_components": [
            "xminigrid.core.observation.transparent_field_of_view",
            "xminigrid.wrappers.DirectionObservationWrapper",
            "xminigrid.wrappers.RulesAndGoalsObservationWrapper",
            "BAGI prior train-only putdown expert_actions",
        ],
        "method": "3 real expert-reached pre-drop states + one hidden-pocket-only counterfactual per state",
        "source_train_skill": "putdown",
        "tested_native_action": ACTION_PUTDOWN,
        "seeds": list(SEEDS),
        "evaluated_cases": cases,
        "full_view_aliases": len(cases),
        "actual_teacher_generator_simulator_steps": sum(
            c["oracle_actions_generated"] for c in cases
        ),
        "actual_replay_simulator_steps": sum(
            c["oracle_prefix_replay_simulator_steps"] for c in cases
        ),
        "actual_counterfactual_step_calls": sum(
            c["native_counterfactual_action_calls"] for c in cases
        ),
        "total_simulator_step_calls": sum(
            c["oracle_actions_generated"]
            + c["oracle_prefix_replay_simulator_steps"]
            + c["native_counterfactual_action_calls"]
            for c in cases
        ),
        "new_policy_training_steps": 0,
        "new_learner_model": False,
        "heldout_teacher_labels": 0,
        "hidden_pocket_exposed_to_policy": False,
        "limitations": [
            "The EMPTY-pocket variant is an explicitly constructed nonphysical counterfactual: there is no claim it could be reached through valid XLand actions.",
            "Proves current upstream local+instruction observation does not report pocket, not that a history-based recurrent model cannot infer it from reachable actions.",
            "Full-grid simulator state is used only to construct/measure a diagnostic pair, not policy input or learner training data.",
            "Teacher solves only TRAIN putdown, never the heldout compositional task.",
            "Only three source layouts; no reward-learning or generalization claims.",
            "Actual oracle generator and replay simulator calls plus counterfactual one-step calls are all counted.",
        ],
    }
    target = Path(cfg["output_dir"]) / "summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"REUSE-22: {len(cases)} identical visible pairs with different native-action rewards")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/xland_inventory_observation_audit.yaml")
    run(p.parse_args().config)
