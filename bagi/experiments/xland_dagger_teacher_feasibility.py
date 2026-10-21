"""REUSE-14: validate whether a privileged TRAIN-ONLY teacher can recover off-path.

This is the prerequisite for established imitation DAgger, not a new imitation
algorithm. Full XLand state is used for teacher LABELS and diagnostic replay
only. No learner policy or held-out chain oracle is queried.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from bagi.experiments.xland_action_coverage import (
    WALKABLE,
    locate_exact,
    plan_putdown_after_pickup,
)
from bagi.experiments.xland_existing_bc import TRAIN_TASKS, expert_actions, get_rules
from bagi.experiments.xland_solvability_oracle import shortest_route_to_face_object

WEAK_PREFIXES = {
    "weak_turn": (0, 1, 0, 1, 0, 1),
    "weak_putdown": (4, 0, 4, 1, 4, 2),
}
POST_EVENT_TASKS = ("near", "putdown")


def require_training_task(task):
    if task not in TRAIN_TASKS:
        raise ValueError("Never query the held-out chain or unapproved task")


def initial_teacher_prefix(task, actions):
    """Reach a nonterminal causal stage before taking one wrong turn."""
    require_training_task(task)
    if task == "near":
        if len(actions) < 2 or actions[-2:] != [0, 3]:
            raise ValueError("Near teacher needs transform then pickup")
        return [*actions[:-1], 1]
    if task == "putdown":
        if 3 not in actions or actions[-1] != 4:
            raise ValueError("Putdown teacher needs pickup then final drop")
        return [*actions[:actions.index(3) + 1], 1]
    raise ValueError("Hold terminates on pickup; no nonterminal post-pickup stage")


def correction_plan(timestep, task):
    """Privileged BFS + existing XLand causal action recipes from CURRENT state."""
    require_training_task(task)
    from xminigrid.core.constants import Colors, TILES_REGISTRY, Tiles

    state = timestep.state
    grid = np.asarray(state.grid)
    position = tuple(map(int, np.asarray(state.agent.position)))
    heading = int(np.asarray(state.agent.direction))
    pocket = np.asarray(state.agent.pocket)
    empty = int(pocket[0]) == int(Tiles.EMPTY)
    square = np.asarray(TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE])
    red = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.RED])
    green = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.GREEN])

    if task == "putdown":
        ball_position = locate_exact(grid, red)
        if np.array_equal(pocket, square):
            navigation, _ = plan_putdown_after_pickup(
                grid, position, heading, ball_position
            )
            return [*navigation, 4], "carrying_square"
        if not empty:
            raise ValueError("Unsupported object held in putdown task")
        location = locate_exact(grid, square)
        navigation, _ = shortest_route_to_face_object(
            grid, position, heading, location, WALKABLE
        )
        return [*navigation, 3], "seek_square"

    if not empty:
        raise ValueError("Unexpected held tile for train-only hold/near task")
    if task == "hold":
        location = locate_exact(grid, square)
        navigation, _ = shortest_route_to_face_object(
            grid, position, heading, location, WALKABLE
        )
        return [*navigation, 3], "seek_hold_square"

    # The upstream near rule can change a RED ball into GREEN at a previous
    # step. A reset-only teacher incorrectly searches for RED at this stage.
    if np.any(np.all(grid == green, axis=-1)):
        location = locate_exact(grid, green)
        navigation, _ = shortest_route_to_face_object(
            grid, position, heading, location, WALKABLE
        )
        return [*navigation, 3], "green_ready"
    location = locate_exact(grid, red)
    navigation, _ = shortest_route_to_face_object(
        grid, position, heading, location, WALKABLE
    )
    return [*navigation, 0], "seek_near_red"


def audit_case(env, params, *, task, seed, mode, cap, reset, step):
    """Validate labels from a visited off-expert state in simulator."""
    require_training_task(task)
    import jax

    timestep = reset(params, jax.random.key(seed))
    generator_steps = 0
    if mode in WEAK_PREFIXES:
        actions = list(WEAK_PREFIXES[mode])
    elif mode == "post_event" and task in POST_EVENT_TASKS:
        expert = expert_actions(env, params, task, seed, cap)
        generator_steps = len(expert)
        actions = initial_teacher_prefix(task, expert)
    else:
        raise ValueError("Unsupported off-path intervention")

    rollin_steps = 0
    for action in actions:
        if bool(timestep.last()):
            break
        timestep = step(params, timestep, int(action))
        rollin_steps += 1

    first_label = None
    stages = Counter()
    correction_steps = 0
    errors = []
    if bool(timestep.last()):
        errors.append("rollin_terminal_before_oracle_query")
    while not bool(timestep.last()) and rollin_steps + correction_steps < cap:
        try:
            plan, stage = correction_plan(timestep, task)
            if not plan or int(plan[0]) not in range(6):
                raise ValueError("Oracle returned no valid next action")
        except (ValueError, IndexError) as exc:
            errors.append(str(exc))
            break
        action = int(plan[0])
        if first_label is None:
            first_label = action
        stages[stage] += 1
        timestep = step(params, timestep, action)
        correction_steps += 1

    success = int(float(timestep.reward) > 0)
    return {
        "task": task,
        "seed": seed,
        "rollin_mode": mode,
        "rollin_steps": rollin_steps,
        "expert_generation_steps": generator_steps,
        "label_queries": sum(stages.values()),
        "first_corrective_label": first_label,
        "oracle_stage_counts": dict(stages),
        "correction_simulator_steps": correction_steps,
        "goal_success": success,
        "failure_reasons": errors if not success else [],
        "terminal": bool(timestep.last()),
        "total_episode_steps": rollin_steps + correction_steps,
    }


def run(config_path):
    import jax
    import xminigrid

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if cfg["environment"] != "XLand-MiniGrid-R1-9x9":
        raise ValueError("Unreviewed environment")
    if tuple(cfg["training_tasks"]) != TRAIN_TASKS:
        raise ValueError("Held-out chain forbidden in corrective oracle audit")
    seeds = [int(s) for s in cfg["seeds"]]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Exactly three independent diagnostic seeds")
    cap = int(cfg["episode_cap"])
    if cap != 40:
        raise ValueError("Preserve prior XLand 40-action episode cap")
    if cfg["modes"] != ["weak_turn", "weak_putdown", "post_event"]:
        raise ValueError("Off-expert modes are preregistered")
    env, defaults = xminigrid.make(cfg["environment"])
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    episodes = []
    for task in TRAIN_TASKS:
        params = defaults.replace(ruleset=get_rules(task), max_steps=cap)
        for seed in seeds:
            for mode in cfg["modes"]:
                if mode == "post_event" and task not in POST_EVENT_TASKS:
                    continue
                row = audit_case(
                    env, params, task=task, seed=seed, mode=mode,
                    cap=cap, reset=reset, step=step
                )
                episodes.append(row)
                print(f"{task} {mode} seed={seed}: "
                      f"teacher_recovered={row['goal_success']} "
                      f"corrected_steps={row['correction_simulator_steps']}")
    summary = {}
    for task in TRAIN_TASKS:
        for mode in cfg["modes"]:
            subset = [r for r in episodes
                      if r["task"] == task and r["rollin_mode"] == mode]
            if subset:
                summary[f"{task}:{mode}"] = {
                    "successes": sum(r["goal_success"] for r in subset),
                    "episodes": len(subset),
                    "queried_states": sum(r["label_queries"] for r in subset),
                }
    result = {
        "question": "Can existing privileged training-skill expert recover from off-path states?",
        "type": "Correction-label validity prerequisite, NOT DAgger training",
        "existing_components": [
            "XLand 0.9.3 upstream transition/reward semantics",
            "BAGI existing privileged BFS pathfinder and train-skill recipes",
            "imitation.DAgger reserved for subsequent verified integration",
        ],
        "upstream_version": version("xminigrid"),
        "training_tasks_only": list(TRAIN_TASKS),
        "heldout_chain_label_queries": 0,
        "learner_policy_inputs_from_privileged_state": 0,
        "weak_rollins_are_trained_agents": False,
        "seeds": seeds,
        "episode_cap": cap,
        "actual_rollin_simulator_steps": sum(r["rollin_steps"] for r in episodes),
        "actual_oracle_recovery_simulator_steps": sum(
            r["correction_simulator_steps"] for r in episodes
        ),
        "actual_expert_generation_simulator_steps": sum(
            r["expert_generation_steps"] for r in episodes
        ),
        "total_real_simulator_steps": sum(
            r["rollin_steps"] + r["correction_simulator_steps"]
            + r["expert_generation_steps"] for r in episodes
        ),
        "teacher_label_queries": sum(r["label_queries"] for r in episodes),
        "successful_recoveries": sum(r["goal_success"] for r in episodes),
        "attempted_recoveries": len(episodes),
        "by_task_and_mode": summary,
        "episodes": episodes,
        "limitations": [
            "Weak scripted behavior and perturbed expert routes are not actual trained-policy rollouts.",
            "Privileged teacher state/geometry is used to choose labels only; not an information-matched learner.",
            "Post-event interventions use extra teacher-generated simulator steps, counted separately.",
            "A recovering expert does not prove an imitation learner can acquire the skills.",
            "Each recovery action recomputes a BFS label, and every query and simulator step is counted.",
            "Only the three original training tasks and independent diagnostic seeds are audited.",
        ],
    }
    output = Path(cfg["output_dir"]) / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Teacher recovery {result['successful_recoveries']}/{len(episodes)}; "
          f"queries={result['teacher_label_queries']}; "
          f"real_steps={result['total_real_simulator_steps']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xland_dagger_teacher_feasibility.yaml")
    run(parser.parse_args().config)
