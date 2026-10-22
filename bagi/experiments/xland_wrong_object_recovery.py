"""REUSE-25: recovery feasibility at a REAL wrong-object-held BC putdown state.

Rebuild the already-verified seed29 REUSE-23 upstream imitation.BC policy
from safe REUSE-19 numeric data. Replay actual failed putdown layout9708.
Probe the existing TRAIN-only correction_plan at its true RED-ball pickup,
then test whether releasing the wrong tile with the native action4 and
resuming original corrective BFS is viable within the original 40-step cap.
No new learning algorithm, checkpoint, training or heldout demonstration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from bagi.experiments.xland_action_coverage import execute_putdown_oracle, locate_exact
from bagi.experiments.xland_bc_class_balance import balanced_indices
from bagi.experiments.xland_dagger_teacher_feasibility import correction_plan
from bagi.experiments.xland_existing_bc import TRAIN_TASKS, get_rules, visible_features
from bagi.experiments.xland_instruction_action_history import (
    HISTORY_LENGTH,
    PAD_ACTION,
    WIDTH,
    assemble,
    check_true_history,
)
from bagi.experiments.xland_putdown_reward_validity import (
    blue_square_transition,
    physically_nonvacuous,
    tiles_adjacent,
)
from bagi.experiments.xland_upstream_instruction_bc import (
    load_verified_source,
    task_instructions,
)

MODEL_SEED = 29
FAILED_LAYOUT = 9708
REFERENCE_LAYOUT = 9604
CONDITION = "real_executed_action_history"
CAP = 40

# REUSE-30: two already-tested TRAIN layouts, never replacement seeds.
# Checkpoints are predetermined actual learner roll-ins, not selected post-hoc.
REUSE30_PROBES = ((9701, 4), (9702, 10))


def action_from_policy(policy, timestep, history):
    """Legal learner observation only: tiles, orientation, goal/rules, own history."""
    view = visible_features(timestep.observation)[None, :]
    obs = timestep.observation
    tokens = task_instructions(
        obs["goal_encoding"], obs["rule_encoding"], enabled=True
    )[None, :]
    selected, _ = policy.predict(
        assemble(view, tokens, np.asarray([history], dtype=np.int64), CONDITION)[0],
        deterministic=True,
    )
    action = int(np.asarray(selected).item())
    if action not in range(6):
        raise AssertionError("Non-native learned action")
    return action


def replay_learner(policy, env, params, seed, *, limit):
    import jax
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles

    red = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.RED])
    square = np.asarray(TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE])
    ts = jax.jit(env.reset)(params, jax.random.key(seed))
    step = jax.jit(env.step)
    history = [PAD_ACTION] * HISTORY_LENGTH
    actions, milestones = [], {}
    successes = 0
    while len(actions) < limit and not bool(ts.last()):
        chosen = action_from_policy(policy, ts, history)
        before = np.asarray(ts.state.agent.pocket)
        ts = step(params, ts, chosen)
        after = np.asarray(ts.state.agent.pocket)
        actions.append(chosen)
        history = [*history[1:], chosen]
        successes |= int(float(ts.reward) > 0)
        if np.array_equal(after, red) and not np.array_equal(before, red):
            milestones.setdefault("red_ball_picked_up_step", len(actions))
        if np.array_equal(after, square) and not np.array_equal(before, square):
            milestones.setdefault("blue_square_picked_up_step", len(actions))
        if (chosen == 4 and np.array_equal(before, red)
                and not np.array_equal(after, red)):
            milestones.setdefault("red_ball_put_down_step", len(actions))
    return {
        "layout_seed": seed, "reward_success": successes,
        "real_steps": len(actions), "executed_actions": actions,
        "milestones": milestones,
    }


def native_trace_sha256(prefix_actions, release_action, oracle_actions):
    """Hash exactly the native actions executed; ignore future/planned actions."""
    actions = [*prefix_actions, int(release_action), *oracle_actions]
    return hashlib.sha256(np.asarray(actions, dtype="<i8").tobytes()).hexdigest()


def recover_from_actual_failure(policy, env, params):
    import jax
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles

    red = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.RED])
    square = np.asarray(TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE])
    empty = int(Tiles.EMPTY)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    ts = reset(params, jax.random.key(FAILED_LAYOUT))
    history = [PAD_ACTION] * HISTORY_LENGTH
    prefix_actions = []
    for _ in range(4):
        act = action_from_policy(policy, ts, history)
        ts = step(params, ts, act)
        history = [*history[1:], act]
        prefix_actions.append(act)
    if not np.array_equal(np.asarray(ts.state.agent.pocket), red):
        raise AssertionError("Missing REAL learner wrong-red-ball pickup checkpoint")
    if np.array_equal(np.asarray(ts.state.agent.pocket), square):
        raise AssertionError("Wrong-object checkpoint confused with blue-square carry")

    # The train-only expert, not a manual intervention or the learned agent,
    # must choose the valid native RED-ball release at this REAL failure.
    proposal, release_stage = correction_plan(ts, "putdown")
    if proposal != [4] or release_stage != "release_wrong_red":
        raise AssertionError("Teacher did not choose native wrong-RED release")
    release_action = int(proposal[0])
    before_grid = np.asarray(ts.state.grid).copy()
    ts = step(params, ts, release_action)
    after_grid = np.asarray(ts.state.grid)
    pocket_is_empty = int(np.asarray(ts.state.agent.pocket)[0]) == empty
    red_on_map = np.any(np.all(after_grid == red, axis=-1))
    if not pocket_is_empty or not red_on_map:
        raise AssertionError("Native wrong-held-ball release did not restore source tile")
    if np.array_equal(before_grid, after_grid):
        raise AssertionError("Native wrong-object release did not change object grid")

    stages, oracle_actions, errors = Counter({release_stage: 1}), [], []
    while not bool(ts.last()) and 5 + len(oracle_actions) < CAP:
        try:
            proposal, stage = correction_plan(ts, "putdown")
            if not proposal or int(proposal[0]) not in range(6):
                raise ValueError("No valid train-only corrective expert action")
        except (ValueError, IndexError) as exc:
            errors.append(str(exc))
            break
        chosen = int(proposal[0])
        oracle_actions.append(chosen)
        stages[stage] += 1
        ts = step(params, ts, chosen)
    recovered = int(float(ts.reward) > 0)
    digest = native_trace_sha256(prefix_actions, release_action, oracle_actions)
    archived_digest = "3b620a8c6090f756a0dcf4e9af276a1526c59f040c5708dcad77f8ba8a630d36"
    if digest != archived_digest:
        raise AssertionError("Teacher trace diverged from REUSE-28 physically verified replay")
    return {
        "actual_learner_failure_seed": FAILED_LAYOUT,
        "checkpoint_learner_prefix_steps": len(prefix_actions),
        "learner_executed_prefix": prefix_actions,
        "checkpoint_wrong_red_held": True,
        "historical_REUSE25_teacher_error": "Expected exactly one required initial object, got 0",
        "teacher_wrong_red_release_queries": 1,
        "intervention_kind": "privileged teacher-selected native action4; not learner action",
        "teacher_wrong_red_release_steps": 1,
        "manual_wrong_red_release_steps": 0,
        "native_actions_sha256_le_int64": digest,
        "matches_REUSE28_physically_verified_trace": True,
        "wrong_red_release_restored_empty_pocket": pocket_is_empty,
        "wrong_red_release_restored_red_tile": bool(red_on_map),
        "subsequent_train_only_correction_queries": len(oracle_actions),
        "correction_stage_counts": dict(stages),
        "correction_actions": oracle_actions,
        "successful_goal_recovery": recovered,
        "recovered_in_40_step_cap": bool(recovered and
                                         5 + len(oracle_actions) <= CAP),
        "total_sequence_steps": 5 + len(oracle_actions),
        "post_intervention_teacher_errors": errors,
        "blue_square_carried_at_finish": bool(np.array_equal(
            np.asarray(ts.state.agent.pocket), square
        )),
    }



def state_fingerprint(state):
    """Privileged state signature for deduplicating evaluation checkpoints only."""
    parts = [
        np.asarray(state.grid),
        np.asarray(state.agent.position, dtype=np.int32),
        np.asarray(state.agent.direction, dtype=np.int32),
        np.asarray(state.agent.pocket),
    ]
    return hashlib.sha256(b"".join(value.tobytes() for value in parts)).hexdigest()


def probe_one_additional_state(policy, environment, params, seed, checkpoint):
    """Test one precommitted learner-visited TRAIN putdown state, no policy retraining."""
    import jax
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles

    if (seed, checkpoint) not in REUSE30_PROBES:
        raise ValueError("Unregistered REUSE-30 diagnostic case")
    blue = np.asarray(TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE])
    red = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.RED])
    empty = int(Tiles.EMPTY)
    reset, step = jax.jit(environment.reset), jax.jit(environment.step)
    ts = reset(params, jax.random.key(seed))
    grid = np.asarray(ts.state.grid)
    blue_pos, red_pos = locate_exact(grid, blue), locate_exact(grid, red)
    initial_distance = sum(abs(int(a) - int(b)) for a, b in zip(blue_pos, red_pos))
    # No silent post-hoc replacement: report an invalid precommitted layout.
    if initial_distance <= 1:
        return {
            "seed": seed, "checkpoint": checkpoint,
            "excluded_initial_adjacency": True,
            "initial_distance": initial_distance,
            "native_simulator_step_calls": 0,
            "teacher_label_queries": 0,
            "learned_success": None,
            "teacher_assisted_physical_placement": None,
        }

    learner = replay_learner(policy, environment, params, seed, limit=CAP)
    if learner["reward_success"] or learner["real_steps"] < checkpoint:
        raise AssertionError("Expected actual frozen REUSE-24 failed learner roll-in")
    # Audit legal learner actions separately from privileged teacher actions.
    history = [PAD_ACTION] * HISTORY_LENGTH
    prefix, teacher_actions, stages, errors = [], [], Counter(), []
    teacher_queries = 0
    blue_pickups = blue_drops = 0
    seen_blue = ordered_blue_drop = False
    native_reward = 0.0

    def advance(action):
        nonlocal ts, blue_pickups, blue_drops, seen_blue, ordered_blue_drop, native_reward
        pocket_before = np.asarray(ts.state.agent.pocket).copy()
        grid_before = np.asarray(ts.state.grid).copy()
        ts = step(params, ts, int(action))
        pocket_after = np.asarray(ts.state.agent.pocket)
        grid_after = np.asarray(ts.state.grid)
        pickup, drop = blue_square_transition(
            int(action), pocket_before, pocket_after, grid_before, grid_after,
            blue, empty,
        )
        blue_pickups += int(pickup)
        blue_drops += int(drop)
        ordered_blue_drop |= bool(drop and seen_blue)
        seen_blue |= pickup
        native_reward = max(native_reward, float(ts.reward))

    for _ in range(checkpoint):
        if bool(ts.last()):
            raise AssertionError("Learner roll-in terminated before registered checkpoint")
        action = action_from_policy(policy, ts, history)
        prefix.append(action)
        advance(action)
        history = [*history[1:], action]
    if prefix != learner["executed_actions"][:checkpoint]:
        raise AssertionError("Frozen policy diverged in observed prefix")

    checkpoint_hash = state_fingerprint(ts.state)
    checkpoint_pocket = np.asarray(ts.state.agent.pocket).tolist()
    checkpoint_position = np.asarray(ts.state.agent.position).tolist()
    while not bool(ts.last()) and len(prefix) + len(teacher_actions) < CAP:
        teacher_queries += 1
        try:
            proposal, stage = correction_plan(ts, "putdown")
            if not proposal or int(proposal[0]) not in range(6):
                raise ValueError("No legal train-only corrective label")
        except (ValueError, IndexError) as exc:
            errors.append(str(exc))
            break
        action = int(proposal[0])
        teacher_actions.append(action)
        stages[stage] += 1
        advance(action)

    final_adjacent = tiles_adjacent(np.asarray(ts.state.grid), blue, red)
    physical = physically_nonvacuous(
        native_reward > 0, initial_distance == 1,
        blue_pickups, blue_drops, final_adjacent, ordered_blue_drop,
    )
    return {
        "seed": seed,
        "checkpoint": checkpoint,
        "excluded_initial_adjacency": False,
        "initial_distance": initial_distance,
        "checkpoint_privileged_state_sha256_diagnostic_only": checkpoint_hash,
        "checkpoint_position_diagnostic_only": checkpoint_position,
        "checkpoint_pocket_diagnostic_only": checkpoint_pocket,
        "learner_only": learner,
        "learner_prefix_actions": prefix,
        "teacher_actions": teacher_actions,
        "teacher_correction_stage_counts": dict(stages),
        "teacher_errors": errors,
        "teacher_label_queries": teacher_queries,
        "native_simulator_step_calls": (
            learner["real_steps"] + len(prefix) + len(teacher_actions)
        ),
        "native_game_reward_from_teacher_trace": native_reward,
        "effective_blue_pickups_on_full_trace": blue_pickups,
        "effective_blue_drops_on_full_trace": blue_drops,
        "ordered_blue_pickup_then_drop": ordered_blue_drop,
        "final_blue_adjacent_to_red": final_adjacent,
        "teacher_assisted_physical_placement": physical,
        "learned_success": bool(learner["reward_success"]),
        "independent_learned_recovery": False,
    }


def probe_two_additional_states(policy, environment, params):
    """Report exactly two predeclared cases, including negative/excluded outcomes."""
    cases = [
        probe_one_additional_state(policy, environment, params, seed, checkpoint)
        for seed, checkpoint in REUSE30_PROBES
    ]
    hashes = [
        item["checkpoint_privileged_state_sha256_diagnostic_only"]
        for item in cases if not item["excluded_initial_adjacency"]
    ]
    return {
        "study": "REUSE-30 / two additional real learner-visited TRAIN putdown states",
        "registered_probes": [list(x) for x in REUSE30_PROBES],
        "policy_model_seed": MODEL_SEED,
        "fixed_model_training_labels_source": 39637067506,
        "previous_recovery_control_run": 39733971331,
        "native_action_cap": CAP,
        "new_policy_training_steps": 0,
        "heldout_composition_teacher_queries": 0,
        "privileged_learner_policy_input": False,
        "valid_distinct_checkpoints": len(hashes) == 2 and len(set(hashes)) == 2,
        "cases": cases,
        "total_native_simulator_step_calls": sum(
            item["native_simulator_step_calls"] for item in cases
        ),
        "total_train_only_teacher_label_queries": sum(
            item["teacher_label_queries"] for item in cases
        ),
        "limitations": [
            "These layouts were already evaluated in REUSE-24: not fresh confirmation seeds.",
            "Full-state labels, grid/pocket, and state hashes are teacher/evaluator only.",
            "No learner was updated: teacher-controlled reward is not learned skill.",
            "A failed or vacuous case is retained; no substitute seeds or extra attempts.",
            "At most two new actual learner checkpoints, no heldout chain labels.",
        ],
    }


def run(config_path):
    import gymnasium as gym
    import torch
    import xminigrid
    from imitation.algorithms.bc import BC
    from imitation.data.types import Transitions
    from xminigrid.wrappers import (
        DirectionObservationWrapper,
        RulesAndGoalsObservationWrapper,
    )

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if (cfg["environment"] != "XLand-MiniGrid-R1-9x9"
            or cfg["task"] != "putdown" or cfg["model_seed"] != MODEL_SEED
            or cfg["condition"] != CONDITION
            or cfg["reference_seed"] != REFERENCE_LAYOUT
            or cfg["failure_seed"] != FAILED_LAYOUT
            or cfg["episode_cap"] != CAP or cfg["epochs"] != 12
            or cfg["batch_size"] != 32
            or cfg["source_run_id"] != 39637067506
            or cfg["policy_run_id"] != 39644180240):
        raise ValueError("REUSE-25 preregistered recovery checkpoint changed")
    source_dir = Path(cfg["source_numeric_dir"])
    src = json.loads(
        (source_dir / "summary.json").read_text(encoding="utf-8")
    )
    original = json.loads(
        (Path(cfg["source_policy_dir"]) / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    if (src["total_expert_queries"] != 582 or src["heldout_teacher_labels"] != 0
            or original["source_run"] != 39637067506
            or original["new_privileged_teacher_queries"] != 0):
        raise ValueError("Unexpected original labeling or frozen-policy provenance")
    selected = [m for m in src["model_results"] if m["model_seed"] == MODEL_SEED]
    if len(selected) != 1:
        raise ValueError("Missing unique trusted policy source")
    demo, tasks, checksum = load_verified_source(
        source_dir / f"action_provenance_seed{MODEL_SEED}.npz",
        selected[0], MODEL_SEED
    )
    check_true_history(demo)
    public_codes = {
        name: task_instructions(
            get_rules(name).goal, get_rules(name).rules, enabled=True
        ) for name in TRAIN_TASKS
    }
    token_rows = np.stack([public_codes[task] for task in tasks])
    features = assemble(
        demo["obs"], token_rows, demo["prior_executed_actions"], CONDITION
    )
    idx = balanced_indices(demo["expert_label"], MODEL_SEED)
    torch.set_num_threads(1)
    torch.manual_seed(MODEL_SEED)
    bc = BC(
        observation_space=gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(WIDTH,), dtype=np.float32
        ),
        action_space=gym.spaces.Discrete(6),
        demonstrations=Transitions(
            obs=features[idx], acts=demo["expert_label"][idx],
            next_obs=features[idx].copy(),
            dones=np.asarray(demo["done"][idx], dtype=bool),
            infos=np.array([None] * len(idx), dtype=object)
        ),
        rng=np.random.default_rng(MODEL_SEED),
        batch_size=32, device="cpu",
    )
    bc.train(n_epochs=12, progress_bar=False)

    room, defaults = xminigrid.make(cfg["environment"])
    instructed = RulesAndGoalsObservationWrapper(DirectionObservationWrapper(room))
    params = defaults.replace(ruleset=get_rules("putdown"), max_steps=CAP)
    good = replay_learner(
        bc.policy, instructed, params, REFERENCE_LAYOUT, limit=CAP
    )
    prior = [m for m in original["models"] if m["model_seed"] == MODEL_SEED]
    condition = [c for c in prior[0]["conditions"] if c["condition"] == CONDITION]
    old_success = [
        x["seed"] for x in condition[0]["independent_evaluation"]["putdown"]["cases"]
        if x["goal_success"]
    ]
    if good["reward_success"] != 1 or REFERENCE_LAYOUT not in old_success:
        raise AssertionError("Frozen learner failed its original rewarded putdown anchor")

    failed = replay_learner(
        bc.policy, instructed, params, FAILED_LAYOUT, limit=CAP
    )
    if (failed["reward_success"] != 0
            or failed["milestones"].get("red_ball_picked_up_step") != 4
            or failed["milestones"].get("red_ball_put_down_step") != 16
            or "blue_square_picked_up_step" in failed["milestones"]):
        raise AssertionError("REUSE-24 real learner failure was not reproduced")
    correction = recover_from_actual_failure(
        bc.policy, instructed, params
    )
    baseline = execute_putdown_oracle(
        room, params, FAILED_LAYOUT, CAP
    )
    if not baseline["goal_success"]:
        raise AssertionError("Train putdown layout not solvable from standard reset")
    report = {
        "question": "Can existing train-only expert select its own legal RED release at a real learner failure?",
        "source_numeric_run_id": 39637067506,
        "source_original_policy_run_id": 39644180240,
        "source_phase_replication_run_id": 39646704229,
        "validated_source_sha256": checksum,
        "policy_seed": MODEL_SEED,
        "independent_reference_layout": good,
        "original_failed_layout": failed,
        "recovery_probe": correction,
        "reset_expert_control": {
            "goal_success": baseline["goal_success"],
            "real_oracle_generator_actions": baseline["real_actions"],
        },
        "actual_native_simulator_step_calls": (
            good["real_steps"] + failed["real_steps"]
            + correction["checkpoint_learner_prefix_steps"]
            + correction["teacher_wrong_red_release_steps"]
            + len(correction["correction_actions"])
            + baseline["real_actions"]
        ),
        "actual_train_only_expert_label_queries": (
            correction["teacher_wrong_red_release_queries"]
            + correction["subsequent_train_only_correction_queries"]
        ),
        "new_online_policy_training_steps": 0,
        "heldout_composition_teacher_queries": 0,
        "privileged_learner_policy_input": False,
        "limitations": [
            "Privileged corrective teacher selected native RED release; learner did not.",
            "The teacher releases RED only into an unobstructed FLOOR.",
            "REUSE-28 archived identical action trace physically verifies BLUE pickup/drop.",
            "A successful native teacher recovery does not mean BC learned it or that generic arbitrary state correction exists.",
            "Single preregistered real learner failure layout and one frozen model seed only.",
            "Original expert reset baseline is separate and uses full state only as oracle.",
            "Zero new DAgger rounds or teacher labels on heldout chain; all simulator and label queries counted.",
        ],
    }
    out = Path(cfg["output_dir"]) / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    reuse30 = probe_two_additional_states(bc.policy, instructed, params)
    second = out.parent / "reuse30_summary.json"
    second.write_text(json.dumps(reuse30, indent=2), encoding="utf-8")
    print("REUSE30 outcomes=", [
        c["teacher_assisted_physical_placement"] for c in reuse30["cases"]
    ], "train_only_labels=", reuse30["total_train_only_teacher_label_queries"])
    print("REUSE-25 recovery success=",correction["successful_goal_recovery"],
          "native steps=",report["actual_native_simulator_step_calls"])
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/xland_wrong_object_recovery.yaml")
    run(p.parse_args().config)
