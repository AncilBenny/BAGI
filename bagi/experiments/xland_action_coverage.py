"""REUSE-8: validate existing XLand TileNearGoal for a rewarded put-down skill.

The held-out causal chain requires put_down action 4, absent from our
previously demonstrated train primitive action sequences. This diagnostic
adds an upstream-goal train primitive with actual successful action 4.
Privileged BFS is used only for oracle benchmark construction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from bagi.experiments.xland_composition_smoke import fingerprint
from bagi.experiments.xland_solvability_oracle import (
    DIRECTIONS,
    shortest_route_to_face_object,
)
from bagi.experiments.xland_structural_composition import (
    KINDS,
    ORACLE_SUFFIXES,
    construct_ruleset,
    episode,
)

ACTIONS = {
    "forward": 0, "right": 1, "left": 2,
    "pick_up": 3, "put_down": 4,
}
WALKABLE = frozenset((1, 6, 10))


def putdown_ruleset():
    """Use upstream TileNearGoal, not a BAGI-specific goal implementation."""
    import jax.numpy as jnp
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
    from xminigrid.core.goals import TileNearGoal
    from xminigrid.core.rules import EmptyRule
    from xminigrid.types import RuleSet

    square = TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE]
    ball = TILES_REGISTRY[Tiles.BALL, Colors.RED]
    return RuleSet(
        goal=TileNearGoal(tile_a=square, tile_b=ball).encode(),
        rules=EmptyRule().encode()[None, :],
        init_tiles=jnp.stack([square, ball]),
    )


def locate_exact(grid, tile):
    found = np.argwhere(np.all(grid == tile, axis=-1))
    if len(found) != 1:
        raise ValueError(f"Expected exactly one required initial object, got {len(found)}")
    return tuple(int(x) for x in found[0])


def plan_putdown_after_pickup(grid, position, direction, ball_position):
    """BFS to face a free cell immediately adjacent to the other object."""
    candidate_routes = []
    for dy, dx in DIRECTIONS:
        dest = (ball_position[0] + dy, ball_position[1] + dx)
        if (dest[0] <= 0 or dest[1] <= 0
                or dest[0] >= grid.shape[0] - 1
                or dest[1] >= grid.shape[1] - 1):
            continue
        if int(grid[dest[0], dest[1], 0]) not in WALKABLE:
            continue
        try:
            route, _ = shortest_route_to_face_object(
                grid, position, direction, dest, WALKABLE
            )
        except ValueError:
            continue
        candidate_routes.append((len(route), route, dest))
    if not candidate_routes:
        raise ValueError("No valid drop cell adjacent to the other object")
    _, actions, cell = min(candidate_routes, key=lambda item: (item[0], item[2]))
    return actions, cell


def execute_putdown_oracle(env, params, seed: int, limit: int) -> dict:
    import jax
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles

    timestep = jax.jit(env.reset)(params, jax.random.key(seed))
    start = timestep.state
    grid = np.asarray(start.grid)
    square = np.asarray(TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE])
    ball = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.RED])
    square_pos, ball_pos = locate_exact(grid, square), locate_exact(grid, ball)
    if abs(square_pos[0] - ball_pos[0]) + abs(square_pos[1] - ball_pos[1]) <= 1:
        raise ValueError("Initial objects adjacent: vacuous TileNearGoal benchmark")
    position = tuple(int(x) for x in np.asarray(start.agent.position))
    direction = int(start.agent.direction)
    route, _ = shortest_route_to_face_object(
        grid, position, direction, square_pos, WALKABLE
    )
    step = jax.jit(env.step)
    actions = []
    def perform(action):
        nonlocal timestep
        if len(actions) >= limit or bool(timestep.last()):
            raise ValueError("Oracle ran out of steps before completing the task")
        timestep = step(params, timestep, int(action))
        actions.append(int(action))

    for action in [*route, ACTIONS["pick_up"]]:
        perform(action)
    if not np.array_equal(np.asarray(timestep.state.agent.pocket), square):
        raise AssertionError("Pickup did not place source object in the pocket")
    after_pick = timestep.state
    updated = np.asarray(after_pick.grid)
    position = tuple(int(x) for x in np.asarray(after_pick.agent.position))
    direction = int(after_pick.agent.direction)
    route2, drop_cell = plan_putdown_after_pickup(
        updated, position, direction, ball_pos
    )
    for action in [*route2, ACTIONS["put_down"]]:
        perform(action)
    final = timestep.state
    held = np.asarray(final.agent.pocket)
    returned = {
        "seed": seed,
        "goal_success": int(float(timestep.reward) > 0),
        "real_actions": len(actions),
        "actions": actions,
        "put_down_count": actions.count(ACTIONS["put_down"]),
        "put_down_was_last_action": actions[-1] == ACTIONS["put_down"],
        "source_square_placed_next_to_red_ball": (
            np.array_equal(np.asarray(final.grid)[drop_cell], square)
            and abs(drop_cell[0] - ball_pos[0]) +
                abs(drop_cell[1] - ball_pos[1]) == 1
        ),
        "pocket_empty_after_put_down": int(held[0]) == Tiles.EMPTY,
    }
    return returned


def run(config_path: str) -> dict:
    import xminigrid

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if cfg["environment"] != "XLand-MiniGrid-R1-9x9":
        raise ValueError("Only reviewed upstream 9x9 room")
    seeds = [int(x) for x in cfg["evaluation"]["seeds"]]
    limit = int(cfg["evaluation"]["max_steps"])
    if not seeds or len(set(seeds)) != len(seeds) or limit < 1:
        raise ValueError("Positive cap and unique evaluation seeds required")
    env, defaults = xminigrid.make(cfg["environment"])
    original = {}
    for kind in KINDS:
        params = defaults.replace(
            ruleset=construct_ruleset(kind), max_steps=limit
        )
        original[kind] = [
            episode(env, params, kind, seed=seed,
                    suffix=ORACLE_SUFFIXES[kind], max_steps=limit)
            for seed in seeds
        ]
    putdown = putdown_ruleset()
    params = defaults.replace(ruleset=putdown, max_steps=limit)
    new_task = [execute_putdown_oracle(env, params, seed, limit) for seed in seeds]
    train_actions = [
        action for kind in ("hold", "near") for item in original[kind]
        for action in item["actions"]
    ]
    heldout_actions = [
        action for item in original["hold_then_near"] for action in item["actions"]
    ]
    all_train_actions = train_actions + [
        action for row in new_task for action in row["actions"]
    ]
    new_covered = sorted(set(all_train_actions) - set(train_actions))
    report = {
        "question": "Can an established rewarded put-down task close the missing action-support gap?",
        "upstream_rules": "Unmodified XLand EmptyRule, TileNearGoal, AgentHoldGoal, AgentHoldRule, AgentNearRule",
        "original_train_action_support": sorted(set(train_actions)),
        "heldout_oracle_action_support": sorted(set(heldout_actions)),
        "initial_action_support_missing_from_training": sorted(
            set(heldout_actions) - set(train_actions)
        ),
        "added_train_task": "place_blue_square_adjacent_to_red_ball",
        "putdown_rule_fingerprint": fingerprint(putdown),
        "new_train_actions": new_covered,
        "new_combined_train_action_support": sorted(set(all_train_actions)),
        "added_task_goal_successes": sum(x["goal_success"] for x in new_task),
        "added_task_evaluation_episodes": len(new_task),
        "added_task_rewarded_putdown_episodes": sum(
            x["goal_success"] * int(x["put_down_was_last_action"])
            for x in new_task
        ),
        "added_task_details": new_task,
        "simulator_steps_all_diagnostics": sum(
            x["real_simulator_steps"] for rows in original.values() for x in rows
        ) + sum(x["real_actions"] for x in new_task),
        "limits": [
            "Full grid and object locations supplied only to diagnostic BFS, not an agent.",
            "The new skill is part of an existing goal primitive, not a new environment or algorithm.",
            "Coverage of action 4 in successful demonstrations does not prove the skill has been learned.",
            "Adding a third training task changes the distribution and evaluation comparability; use matched-budget controls.",
            "XLand task goals and rules remain hidden from learned agent; zero-shot rule inference is not implied.",
            "If seeds create initially adjacent objects the test rejects the vacuous instance.",
        ],
    }
    dest = Path(cfg["evaluation"]["output_dir"]) / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Old training actions: {report['original_train_action_support']}; "
          f"new: {report['new_combined_train_action_support']}")
    print(f"Put-down task success: {report['added_task_goal_successes']}/{len(new_task)}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xland_action_coverage.yaml")
    run(parser.parse_args().config)
