"""REUSE-26: audit whether the first learned putdown REWARD was vacuous.

Replay the EXACT native actions archived by successful REUSE-25 at model seed
29 / layout9604. Check initial BLUE-square and RED-ball adjacency, actual
pocket/grid transitions, and upstream reward. No new learner training or
oracle labels. Distinguish native TileNearGoal reward from genuine placement
competence; the goal's upstream predicate does NOT demand inventory changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from bagi.experiments.xland_action_coverage import locate_exact
from bagi.experiments.xland_existing_bc import get_rules


def physically_nonvacuous(reward_success, initial_adjacent, effective_pickups,
                         effective_putdowns):
    """Strict research skill gate, separate from genuine upstream reward."""
    return bool(
        reward_success and not initial_adjacent
        and effective_pickups > 0 and effective_putdowns > 0
    )


def parse_saved_reference(document):
    """A result must originate from the prior successful frozen BC rollout."""
    row = document["independent_reference_layout"]
    if (document["policy_seed"] != 29
            or row["layout_seed"] != 9604
            or row["reward_success"] != 1
            or row["real_steps"] != len(row["executed_actions"])
            or not 1 <= len(row["executed_actions"]) <= 40
            or row["executed_actions"][-1] != 4
            or any(int(a) not in range(6) for a in row["executed_actions"])
            or document["new_online_policy_training_steps"] != 0
            or document["heldout_composition_teacher_queries"] != 0):
        raise ValueError("Unrecognized source learner rewarded putdown trace")
    return [int(a) for a in row["executed_actions"]]


def run(config_path):
    import jax
    import xminigrid
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if (cfg["environment"] != "XLand-MiniGrid-R1-9x9"
            or cfg["task"] != "putdown"
            or cfg["policy_model_seed"] != 29
            or cfg["learner_success_layout_seed"] != 9604
            or cfg["source_recovery_run_id"] != 39649325255
            or cfg["episode_cap"] != 40
            or cfg["heldout_teacher_queries"] != 0):
        raise ValueError("Frozen REUSE-26 reward-validity protocol changed")
    parent = json.loads(
        (Path(cfg["source_dir"]) / "summary.json").read_text(encoding="utf-8")
    )
    actions = parse_saved_reference(parent)
    env, defaults = xminigrid.make(cfg["environment"])
    params = defaults.replace(ruleset=get_rules("putdown"), max_steps=40)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    ts = reset(params, jax.random.key(9604))
    square = np.asarray(TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE])
    red = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.RED])
    empty_type = int(Tiles.EMPTY)
    grid = np.asarray(ts.state.grid)
    initial_square_pos = locate_exact(grid, square)
    initial_red_pos = locate_exact(grid, red)
    initial_distance = sum(
        abs(int(a) - int(b))
        for a, b in zip(initial_square_pos, initial_red_pos)
    )
    initial_adjacent = initial_distance == 1
    if int(np.asarray(ts.state.agent.pocket)[0]) != empty_type:
        raise AssertionError("Unexpected initially nonempty pocket")
    events = []
    effective_pickups = effective_putdowns = grid_modifications = 0
    rewarded = 0
    for i, action in enumerate(actions, start=1):
        if bool(ts.last()):
            raise AssertionError("Original learner trace continues after terminal")
        pocket_before = np.asarray(ts.state.agent.pocket).copy()
        grid_before = np.asarray(ts.state.grid).copy()
        ts = step(params, ts, action)
        pocket_after = np.asarray(ts.state.agent.pocket).copy()
        grid_after = np.asarray(ts.state.grid).copy()
        changed_pocket = not np.array_equal(pocket_before, pocket_after)
        changed_grid = not np.array_equal(grid_before, grid_after)
        pickup = (action == 3 and int(pocket_before[0]) == empty_type
                  and int(pocket_after[0]) != empty_type)
        drop = (action == 4 and int(pocket_before[0]) != empty_type
                and int(pocket_after[0]) == empty_type)
        effective_pickups += int(pickup)
        effective_putdowns += int(drop)
        grid_modifications += int(changed_grid)
        rewarded |= int(float(ts.reward) > 0)
        events.append({
            "step": i,
            "native_action": action,
            "actual_pocket_changed": changed_pocket,
            "actual_grid_changed": changed_grid,
            "effective_native_pickup": bool(pickup),
            "effective_native_putdown": bool(drop),
            "upstream_goal_reward": float(ts.reward),
        })
    if not rewarded or float(ts.reward) <= 0 or not bool(ts.last()):
        raise AssertionError("Frozen source BC positive reward failed to reproduce")
    result = {
        "question": "Was REUSE-23 first learner rewarded TileNearGoal episode nonvacuous physical manipulation?",
        "source_recovery_run_id": 39649325255,
        "source_model_seed": 29,
        "source_layout_seed": 9604,
        "native_XLand_goal": "TileNearGoal",
        "original_reward_valid_under_upstream": bool(rewarded),
        "physical_placement_skill_criterion": "nonadjacent initial tiles AND effective pickup AND effective putdown AND reward",
        "initial_square_position": list(initial_square_pos),
        "initial_red_position": list(initial_red_pos),
        "initial_manhattan_distance": initial_distance,
        "initial_tiles_adjacent": initial_adjacent,
        "learner_native_actions_replayed": actions,
        "total_native_simulator_step_calls": len(actions),
        "actual_effective_pickups": effective_pickups,
        "actual_effective_putdowns": effective_putdowns,
        "actual_grid_modifications": grid_modifications,
        "final_pocket_empty": int(np.asarray(ts.state.agent.pocket)[0]) == empty_type,
        "physically_nonvacuous_placement_success": physically_nonvacuous(
            rewarded, initial_adjacent, effective_pickups, effective_putdowns
        ),
        "per_step_native_events": events,
        "fingerprint_of_archived_actions_sha256": hashlib.sha256(
            np.asarray(actions, dtype=np.int64).tobytes()
        ).hexdigest(),
        "new_expert_oracle_queries": 0,
        "new_learned_policy_training_actions": 0,
        "heldout_composition_actions_or_labels": 0,
        "learner_training_in_this_audit": False,
        "limitations": [
            "Exact earlier learned action trace replayed as fixed native controls, not a newly trained model.",
            "Original XLand reward is authoritative as GAME SCORE but may not certify manipulation.",
            "One known positive layout from posthoc investigation; screen all future evaluation layouts for initial vacuity before scoring physical skill.",
            "A negative physical-competence conclusion for this case does not imply no future learner can learn real placement.",
            "No heldout composition examples, full grid learner features or teacher actions.",
        ],
    }
    path = Path(cfg["output_dir"]) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "REUSE26 original_reward=",result["original_reward_valid_under_upstream"],
        "initial_adjacent=",initial_adjacent,
        "pickup=",effective_pickups,"drop=",effective_putdowns,
        "nonvacuous=",result["physically_nonvacuous_placement_success"]
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/xland_putdown_reward_validity.yaml")
    run(p.parse_args().config)
