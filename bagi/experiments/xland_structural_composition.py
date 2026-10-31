"""REUSE-6: rule-composition contract using only upstream XLand primitives.

Established RuleSet/AgentHoldRule/AgentNearRule/AgentHoldGoal implementations
are reused. Privileged classical BFS is diagnostic-only, never a learner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from bagi.experiments.xland_composition_smoke import fingerprint
from bagi.experiments.xland_solvability_oracle import shortest_route_to_face_object

# Existing rule types: pick up transforms held object (3); near triggers on move (0).
# The held-out task requires composition of BOTH previously available mechanisms.
KINDS = ("hold", "near", "hold_then_near")
PRIMITIVE_SUFFIXES = {"hold": (3,), "near": (0, 3)}
ORACLE_SUFFIXES = {**PRIMITIVE_SUFFIXES, "hold_then_near": (3, 4, 0, 3)}


def construct_ruleset(kind: str):
    """No homegrown environment or learning rule; just upstream encodings."""
    import jax.numpy as jnp
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
    from xminigrid.core.goals import AgentHoldGoal
    from xminigrid.core.rules import AgentHoldRule, AgentNearRule
    from xminigrid.types import RuleSet

    if kind not in KINDS:
        raise ValueError(f"Unapproved structural rule composition: {kind}")
    blue_square = TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE]
    red_ball = TILES_REGISTRY[Tiles.BALL, Colors.RED]
    green_ball = TILES_REGISTRY[Tiles.BALL, Colors.GREEN]
    if kind == "hold":
        source, goal = blue_square, red_ball
        rules = [AgentHoldRule(tile=blue_square, prod_tile=red_ball).encode()]
    elif kind == "near":
        source, goal = red_ball, green_ball
        rules = [AgentNearRule(tile=red_ball, prod_tile=green_ball).encode()]
    else:
        source, goal = blue_square, green_ball
        rules = [
            AgentHoldRule(tile=blue_square, prod_tile=red_ball).encode(),
            AgentNearRule(tile=red_ball, prod_tile=green_ball).encode(),
        ]
    return RuleSet(
        goal=AgentHoldGoal(tile=goal).encode(),
        rules=jnp.stack(rules),
        init_tiles=jnp.stack([source]),
    )


def episode(env, params, kind: str, *, seed: int, suffix: tuple[int, ...],
            max_steps: int) -> dict:
    """Execute an explicitly declared reference action recipe in the real simulator."""
    import jax
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles

    timestep = jax.jit(env.reset)(params, jax.random.key(seed))
    state = timestep.state
    grid = np.asarray(state.grid)
    source = np.asarray(TILES_REGISTRY[
        Tiles.BALL if kind == "near" else Tiles.SQUARE,
        Colors.RED if kind == "near" else Colors.BLUE,
    ])
    matching = np.argwhere(np.all(grid == source, axis=-1))
    if len(matching) != 1:
        raise ValueError("Expected exactly one initial source object")
    target = tuple(int(v) for v in matching[0])
    start = tuple(int(v) for v in np.asarray(state.agent.position))
    direction = int(np.asarray(state.agent.direction))
    walkable = frozenset((Tiles.FLOOR, Tiles.GOAL, Tiles.DOOR_OPEN))
    nav, expansions = shortest_route_to_face_object(
        grid, start, direction, target, walkable
    )
    planned = [*nav, *suffix]
    step = jax.jit(env.step)
    actions_executed = []
    pocket_red = False
    grid_green = False
    reward = 0.0
    for action in planned[:max_steps]:
        if bool(timestep.last()):
            break
        timestep = step(params, timestep, int(action))
        actions_executed.append(int(action))
        reward += float(timestep.reward)
        pocket = np.asarray(timestep.state.agent.pocket)
        pocket_red |= bool(np.array_equal(
            pocket, np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.RED])
        ))
        grid_green |= bool(np.any(np.all(
            np.asarray(timestep.state.grid) ==
            np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.GREEN]),
            axis=-1,
        )))
    return {
        "seed": seed,
        "planned_action_suffix": list(suffix),
        "actions": actions_executed,
        "planned_length": len(planned),
        "within_budget": len(planned) <= max_steps,
        "real_simulator_steps": len(actions_executed),
        "privileged_bfs_states_expanded": expansions,
        "intermediate_red_ball_carried": pocket_red,
        "green_ball_transformation_observed": grid_green,
        "goal_success": int(reward > 0),
    }


def run(config_path: str) -> dict:
    import xminigrid

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if cfg["environment"] != "XLand-MiniGrid-R1-9x9":
        raise ValueError("Only reviewed XLand room is allowed")
    eval_cfg = cfg["evaluation"]
    seeds = [int(x) for x in eval_cfg["seeds"]]
    max_steps = int(eval_cfg["max_steps"])
    if not seeds or len(seeds) != len(set(seeds)) or max_steps < 1:
        raise ValueError("Require unique episode seeds and positive cap")
    env, defaults = xminigrid.make(cfg["environment"])
    tasks = {}
    for kind in KINDS:
        rule_set = construct_ruleset(kind)
        tasks[kind] = {
            "fingerprint": fingerprint(rule_set),
            "rule_ids": np.asarray(rule_set.rules)[:, 0].tolist(),
            "ruleset": rule_set,
        }
    if len({x["fingerprint"] for x in tasks.values()}) != len(tasks):
        raise AssertionError("Rule/goal holdout has overlapping encoded task IDs")
    results = []
    for kind in KINDS:
        params = defaults.replace(ruleset=tasks[kind]["ruleset"], max_steps=max_steps)
        for seed in seeds:
            oracle = episode(
                env, params, kind, seed=seed,
                suffix=ORACLE_SUFFIXES[kind], max_steps=max_steps,
            )
            record = {"task": kind, "split": (
                "held_out" if kind == "hold_then_near" else "train"
            ), "seed": seed, "rule_ids": tasks[kind]["rule_ids"],
                      "ruleset_sha256": tasks[kind]["fingerprint"],
                      "oracle": oracle}
            if kind == "hold_then_near":
                record["primitive_replay_controls"] = {
                    training_kind: episode(
                        env, params, kind, seed=seed,
                        suffix=PRIMITIVE_SUFFIXES[training_kind],
                        max_steps=max_steps,
                    )
                    for training_kind in ("hold", "near")
                }
            results.append(record)
            print(f"{kind} seed={seed} oracle={oracle['goal_success']}, "
                  f"steps={oracle['real_simulator_steps']}")
    held_out = [x for x in results if x["split"] == "held_out"]
    report = {
        "purpose": "Structural causal composition: training primitives vs unseen multi-rule chain",
        "rules_and_goals": "Unmodified XLand 0.9.3 RuleSet/AgentHoldRule/AgentNearRule/AgentHoldGoal",
        "training_tasks": ["hold", "near"],
        "held_out_tasks": ["hold_then_near"],
        "train_heldout_hashes_disjoint": True,
        "evaluation_seeds": seeds,
        "max_steps_per_episode": max_steps,
        "all_oracle_successes": sum(x["oracle"]["goal_success"] for x in results),
        "oracle_episodes": len(results),
        "heldout_oracle_successes": sum(x["oracle"]["goal_success"] for x in held_out),
        "primitive_replay_successes_on_heldout": sum(
            y["goal_success"] for x in held_out
            for y in x["primitive_replay_controls"].values()
        ),
        "primitive_replay_episodes_on_heldout": len(held_out) * 2,
        "total_real_simulator_steps": sum(
            x["oracle"]["real_simulator_steps"] for x in results
        ) + sum(
            y["real_simulator_steps"] for x in held_out
            for y in x["primitive_replay_controls"].values()
        ),
        "results": results,
        "limitations": [
            "Privileged object position and full grid are used for this oracle and controls; learned agents must not receive them.",
            "Suffix recipes are hand-specified from public upstream rule semantics, not inferred or learned.",
            "Replay of a training recipe is a necessary baseline, not proof of behavioral generalization.",
            "One held-out structural chain and limited seeds do not constitute broad compositional transfer.",
            "No learned policy is trained or evaluated; no new algorithm or pretrained model is claimed.",
            "Environment is the same 9x9 room, with variation from seeded initial objects and positions.",
            "Safety: upstream external benchmark pickle loaders are never used.",
        ],
    }
    dest = Path(eval_cfg["output_dir"]) / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved structural held-out diagnostic: {dest}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xland_structural_composition.yaml")
    run(parser.parse_args().config)
