"""Diagnostic solvability ceiling for *existing* XLand causal rule/goal tasks.

Uses classical breadth-first navigation over privileged map geometry, NOT a
partially observable agent and not a newly proposed learning algorithm.
Executes the resulting actions with the upstream JAX simulator.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from bagi.experiments.xland_composition_smoke import disjoint_tasks

DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def shortest_route_to_face_object(
    grid: np.ndarray, start: tuple[int, int], direction: int,
    target: tuple[int, int], walkable: frozenset[int],
) -> tuple[list[int], int]:
    """Standard BFS over grid position+heading; 0 forward, 1 right, 2 left."""
    first = (start[0], start[1], direction)
    queue = deque([(first, ())])
    visited = {first}
    expansions = 0
    while queue:
        state, path = queue.popleft()
        expansions += 1
        row, col, heading = state
        dy, dx = DIRECTIONS[heading]
        if (row + dy, col + dx) == target:
            return list(path), expansions
        successors = [
            ((row, col, (heading + 1) % 4), 1),
            ((row, col, (heading - 1) % 4), 2),
        ]
        nr, nc = row + dy, col + dx
        if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]
                and int(grid[nr, nc, 0]) in walkable):
            successors.append(((nr, nc, heading), 0))
        for nxt, action in successors:
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, (*path, action)))
    raise ValueError("No reachable adjacent cell facing the source tile")


def diagnostic_episode(
    env, params, spec: dict, *, seed: int, max_steps: int,
) -> dict:
    """Privileged geometry for construction, visible-only learning not involved."""
    import jax
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles

    source = np.asarray(
        TILES_REGISTRY[
            getattr(Tiles, spec["source_tile"]), getattr(Colors, spec["source_color"])
        ]
    )
    result = np.asarray(
        TILES_REGISTRY[
            getattr(Tiles, spec["result_tile"]), getattr(Colors, spec["result_color"])
        ]
    )
    timestep = jax.jit(env.reset)(params, jax.random.key(seed))
    grid = np.asarray(timestep.state.grid)
    occurrences = np.argwhere(np.all(grid == source, axis=-1))
    if len(occurrences) != 1:
        raise ValueError(f"Expected exactly one source object, found {len(occurrences)}")
    target = tuple(int(v) for v in occurrences[0])
    position = tuple(int(v) for v in np.asarray(timestep.state.agent.position))
    heading = int(np.asarray(timestep.state.agent.direction))
    walkable = frozenset(int(v) for v in np.asarray(
        [Tiles.FLOOR, Tiles.GOAL, Tiles.DOOR_OPEN]
    ))
    route, expansions = shortest_route_to_face_object(
        grid, position, heading, target, walkable
    )
    # The existing AgentNearRule fires on forward; AgentHoldGoal checks on pickup.
    actions = [*route, 0, 3]
    transformed = False
    steps = 0
    reward = 0.0
    final_pocket_matches = False
    step = jax.jit(env.step)
    for action in actions[:max_steps]:
        if bool(timestep.last()):
            break
        timestep = step(params, timestep, int(action))
        steps += 1
        reward += float(timestep.reward)
        state_grid = np.asarray(timestep.state.grid)
        transformed |= bool(np.array_equal(state_grid[target], result))
        final_pocket_matches = bool(np.array_equal(
            np.asarray(timestep.state.agent.pocket), result
        ))
    return {
        "seed": seed,
        "planned_actions": len(actions),
        "real_environment_steps": steps,
        "search_states_expanded": expansions,
        "within_step_budget": len(actions) <= max_steps,
        "observed_object_transformation": transformed,
        "held_result_at_end": final_pocket_matches,
        "goal_success": int(reward > 0),
        "return": reward,
        "diagnostic_action_trace": actions[:max_steps],
    }


def run(config_path: str) -> dict:
    import xminigrid

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    train, held = disjoint_tasks(config)
    env_id = config["environment"]
    if env_id != "XLand-MiniGrid-R1-9x9":
        raise ValueError("Only reviewed XLand 9x9 room is supported")
    env, default_params = xminigrid.make(env_id)
    evaluation = config["evaluation"]
    seeds = [int(evaluation["evaluation_seed_base"]) + i
             for i in range(int(evaluation["episodes_per_task"]))]
    limit = int(evaluation["max_steps_per_episode"])
    if not seeds or limit < 1:
        raise ValueError("Invalid evaluation seeds or episode cap")
    results = []
    for split, tasks in (("train", train), ("held_out", held)):
        for task in tasks:
            params = default_params.replace(
                ruleset=task["ruleset"], max_steps=limit
            )
            episodes = [
                diagnostic_episode(
                    env, params, task["specification"], seed=seed, max_steps=limit
                )
                for seed in seeds
            ]
            result = {
                "split": split,
                "task": task["name"],
                "ruleset_sha256": task["fingerprint"],
                "episodes": episodes,
                "goal_successes": sum(x["goal_success"] for x in episodes),
                "real_environment_steps": sum(x["real_environment_steps"] for x in episodes),
                "search_states_expanded": sum(x["search_states_expanded"] for x in episodes),
            }
            results.append(result)
            print(f"{split}/{task['name']}: "
                  f"{result['goal_successes']}/{len(episodes)} solvable, "
                  f"{result['real_environment_steps']} executed steps")
    report = {
        "question": "Are the existing XLand goal+rule compositions executable within the preset episode cap?",
        "algorithm_reused": "classical breadth-first graph search (diagnostic navigation oracle)",
        "environment": env_id,
        "ruleset_hashes_disjoint": True,
        "seeds": seeds,
        "max_steps_per_episode": limit,
        "successes": sum(r["goal_successes"] for r in results),
        "episodes": len(seeds) * len(results),
        "real_environment_steps": sum(r["real_environment_steps"] for r in results),
        "search_states_expanded": sum(r["search_states_expanded"] for r in results),
        "results": results,
        "previous_random_baseline": {
            "github_run_id": 39576647873,
            "successes": 0,
            "episodes": 12,
            "note": "Previous separate run, identical seeds and rule-goal tasks; no training",
        },
        "limits": [
            "Full simulator grid and object identity are used solely by the oracle; this is NOT an information-matched policy comparison.",
            "BFS is classical deterministic graph search over navigable cells/headings, not learning.",
            "The rule and goal are explicitly supplied; no rule discovery, adaptation or transfer is evaluated.",
            "No pretrained checkpoints, external benchmark pickle files or agent training are used.",
            "Only three fixed evaluation seeds per distinct task and one room size; not a general solvability theorem.",
        ],
    }
    path = Path(evaluation["oracle_output_dir"]) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved diagnostic solvability ceiling: {path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xland_composition_oracle.yaml")
    run(parser.parse_args().config)
