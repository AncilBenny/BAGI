"""G2-D2: evaluator-only geometry of real frozen XLand learner navigation.

The exact source-trained upstream BC policies see only their established 115
public input features and the last four REAL executed actions. Hidden grid,
agent position, pocket, shortest-path route and diagnostic pickup branches
are NEVER passed to the learner or allowed to determine its actions.

A transparent stage funnel distinguishes public sight, approach, orientation,
verified pickup opportunity, actual acquisition and ordered placement. It is
descriptive, not proof that one intervention would fix behavior.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

MODEL_SEEDS = (17, 29)
NEW_LAYOUTS = tuple(range(11600, 11610))
CAP = 40
INPUT_WIDTH = 115
PICKUP_ACTION = 3
PAD = 6


def verify_registration() -> None:
    """A single prospective 20-episode, native-step-bounded diagnostic."""
    if (MODEL_SEEDS != (17, 29) or NEW_LAYOUTS != tuple(range(11600, 11610))
            or CAP != 40 or INPUT_WIDTH != 115 or PICKUP_ACTION != 3):
        raise ValueError("Unregistered D2 experiment seeds or native budget")
    forbidden = (
        set(range(9700, 9900)) | set(range(10400, 10424))
        | set(range(10600, 10810)) | set(range(10900, 10972))
        | set(range(11000, 11600))
    )
    if set(NEW_LAYOUTS) & forbidden:
        raise ValueError("Train/previous-test layout leakage")


def next_history(history: tuple[int, ...], action: int) -> tuple[int, ...]:
    if (len(history) != 4 or any(v not in range(7) for v in history)
            or type(action) is not int or action not in range(6)):
        raise ValueError("Invalid actual executed action or history")
    return (*history[1:], action)


def observation_for_policy(public_features: object, history: tuple[int, ...]) -> np.ndarray:
    """Check policy input provenance, not merely numerical shape."""
    x = np.asarray(public_features, dtype=np.float32)
    if x.shape != (INPUT_WIDTH,) or not np.isfinite(x).all():
        raise ValueError("Invalid 115-feature public observation")
    expected = np.eye(7, dtype=np.float32)[list(history)].reshape(-1)
    if not np.array_equal(x[-28:], expected):
        raise ValueError("Public input history is not truly executed actions")
    return x


def evaluator_geometry(position: object, heading: int, target: object,
                       directions: tuple[tuple[int, int], ...]) -> dict:
    """Distance and facing. EVALUATOR ONLY; never a policy observation."""
    if len(directions) != 4 or sorted(directions) != sorted(
        ((-1, 0), (0, 1), (1, 0), (0, -1))
    ):
        raise ValueError("Unrecognized reviewed upstream heading conventions")
    if int(heading) not in range(4):
        raise ValueError("Bad simulator heading")
    p = tuple(int(v) for v in np.asarray(position).reshape(-1))
    t = tuple(int(v) for v in np.asarray(target).reshape(-1))
    if len(p) != 2 or len(t) != 2:
        raise ValueError("Expected evaluator-only 2D positions")
    distance = abs(p[0] - t[0]) + abs(p[1] - t[1])
    dy, dx = directions[int(heading)]
    facing = distance == 1 and (p[0] + dy, p[1] + dx) == t
    return {"position": p, "heading": int(heading), "distance": distance,
            "adjacent": distance == 1, "facing": bool(facing)}


def failure_stage(case: dict) -> str:
    if not case["eligible"]:
        return "ineligible_nonadjacent_gate"
    if case["physical_placement"]:
        return "completed_physical_placement"
    if not case["target_visible_steps"]:
        return "never_publicly_saw_target"
    if not case["adjacent_steps"]:
        return "saw_target_never_approached_adjacent"
    if not case["facing_steps"]:
        return "approached_adjacent_never_faced"
    if not case["pickup_ready_steps"]:
        return "faced_but_pickup_precondition_not_met"
    if not case["effective_blue_pickups"]:
        return "pickup_ready_but_not_acquired"
    if not case["effective_blue_drops"]:
        return "acquired_but_never_effectively_dropped"
    return "dropped_but_no_valid_final_goal"


def diagnostic_episode(
    policy, timestep, *, layout: int, eligible: bool,
    observe, native_step, inspect_state, visible_target,
    effective_transition, final_adjacent,
    shortest_pickup_actions: int | None,
) -> dict:
    """Learner never receives inspect_state, geometry, or discarded branch."""
    if layout not in NEW_LAYOUTS:
        raise ValueError("Not a preregistered diagnostic evaluation layout")
    row = {
        "layout": layout, "eligible": bool(eligible), "native_actions": 0,
        "discarded_pickup_probes": 0, "native_reward": False,
        "physical_placement": False, "target_visible_steps": 0,
        "adjacent_steps": 0, "facing_steps": 0, "pickup_ready_steps": 0,
        "missed_pickup_ready_steps": 0, "effective_blue_pickups": 0,
        "effective_blue_drops": 0, "ordered_blue_drop": False,
        "wrong_object_pickups": 0, "pickup_action_attempts": 0,
        "pickup_probe_geometry_disagreements": 0,
        "min_target_distance": None, "first_adjacent_action": None,
        "first_facing_action": None, "first_pickup_action": None,
        "public_seen_but_nonimproving_steps": 0,
        "revisited_pose_steps": 0, "revisited_position_steps": 0,
        "unique_positions": 0, "unique_poses": 0,
        "shortest_pickup_actions_at_reset": shortest_pickup_actions,
        "actions": [],
    }
    if not eligible:
        row["failure_stage"] = failure_stage(row)
        return row
    if bool(timestep.last()):
        raise ValueError("Eligible native putdown episode terminal at reset")
    history = (PAD, PAD, PAD, PAD)
    saw_pickup = False
    visited_positions: set[tuple[int, int]] = set()
    visited_poses: set[tuple[int, int, int]] = set()
    while row["native_actions"] < CAP and not bool(timestep.last()):
        x = observation_for_policy(observe(timestep, history), history)
        predicted, _ = policy.predict(x, deterministic=True)
        action_np = np.asarray(predicted)
        if action_np.shape not in ((), (1,)) or action_np.dtype.kind not in "iu":
            raise ValueError("Policy proposed a nonnative action")
        action = int(action_np.reshape(-1)[0])
        if action not in range(6):
            raise ValueError("Policy action outside native 0..5")
        # The policy has already selected its action; from here everything is
        # *passive* evaluator-only telemetry. None of the values below are used
        # to change action, public input, future trajectory or policy updates.
        before = timestep
        s = inspect_state(before)
        target = s["target"]
        g = s["geometry"]
        visible = bool(visible_target(before))
        if visible:
            row["target_visible_steps"] += 1
        prior_min = row["min_target_distance"]
        if target is not None:
            d = int(g["distance"])
            row["min_target_distance"] = (
                d if prior_min is None else min(prior_min, d)
            )
            if visible and prior_min is not None and d >= prior_min:
                row["public_seen_but_nonimproving_steps"] += 1
            if g["adjacent"]:
                row["adjacent_steps"] += 1
                if row["first_adjacent_action"] is None:
                    row["first_adjacent_action"] = row["native_actions"]
            if g["facing"]:
                row["facing_steps"] += 1
                if row["first_facing_action"] is None:
                    row["first_facing_action"] = row["native_actions"]
        p = tuple(g["position"])
        pose = (*p, int(g["heading"]))
        row["revisited_position_steps"] += int(p in visited_positions)
        row["revisited_pose_steps"] += int(pose in visited_poses)
        visited_positions.add(p)
        visited_poses.add(pose)
        # This additional action3 branch is an evaluator-only discarded
        # counterfactual on a geometrically face-aligned, empty-pocket state.
        # No probe executes on or changes the learner's real trajectory.
        candidate = bool(target is not None and g["facing"] and s["pocket_empty"])
        if action == PICKUP_ACTION:
            row["pickup_action_attempts"] += 1
        after = native_step(before, action)
        row["native_actions"] += 1
        row["actions"].append(action)
        row["native_reward"] |= float(after.reward) > 0
        picked, dropped, wrong = effective_transition(before, after, action)
        if candidate:
            verified_candidate = bool(picked)
            if action != PICKUP_ACTION:
                branch = native_step(before, PICKUP_ACTION)
                row["discarded_pickup_probes"] += 1
                verified_candidate = bool(
                    effective_transition(before, branch, PICKUP_ACTION)[0]
                )
            row["pickup_probe_geometry_disagreements"] += int(
                not verified_candidate
            )
            if verified_candidate:
                row["pickup_ready_steps"] += 1
                if action != PICKUP_ACTION:
                    row["missed_pickup_ready_steps"] += 1
        if picked:
            row["effective_blue_pickups"] += 1
            if row["first_pickup_action"] is None:
                row["first_pickup_action"] = row["native_actions"] - 1
        if dropped:
            row["ordered_blue_drop"] |= bool(saw_pickup)
            row["effective_blue_drops"] += 1
        saw_pickup |= bool(picked)
        row["wrong_object_pickups"] += int(bool(wrong))
        history = next_history(history, action)
        timestep = after
    row["unique_positions"] = len(visited_positions)
    row["unique_poses"] = len(visited_poses)
    row["physical_placement"] = bool(
        row["native_reward"] and row["effective_blue_pickups"] > 0
        and row["effective_blue_drops"] > 0 and row["ordered_blue_drop"]
        and final_adjacent(timestep)
    )
    row["failure_stage"] = failure_stage(row)
    return row


def aggregate_cases(cases: list[dict]) -> dict:
    if [c["layout"] for c in cases] != list(NEW_LAYOUTS):
        raise ValueError("A registered evaluation layout was replaced or skipped")
    elig = [c for c in cases if c["eligible"]]
    return {
        "eligible": len(elig), "native_actions": sum(c["native_actions"] for c in cases),
        "discarded_pickup_probes": sum(c["discarded_pickup_probes"] for c in cases),
        "native_rewarded": sum(bool(c["native_reward"]) for c in elig),
        "physical_placements": sum(bool(c["physical_placement"]) for c in elig),
        "target_seen_episodes": sum(c["target_visible_steps"] > 0 for c in elig),
        "adjacent_episodes": sum(c["adjacent_steps"] > 0 for c in elig),
        "facing_episodes": sum(c["facing_steps"] > 0 for c in elig),
        "pickup_ready_episodes": sum(c["pickup_ready_steps"] > 0 for c in elig),
        "effective_pickup_episodes": sum(c["effective_blue_pickups"] > 0 for c in elig),
        "effective_drop_episodes": sum(c["effective_blue_drops"] > 0 for c in elig),
        "revisited_pose_steps": sum(c["revisited_pose_steps"] for c in elig),
        "revisited_position_steps": sum(c["revisited_position_steps"] for c in elig),
        "public_seen_but_nonimproving_steps": sum(
            c["public_seen_but_nonimproving_steps"] for c in elig
        ),
        "missed_pickup_ready_steps": sum(c["missed_pickup_ready_steps"] for c in elig),
        "pickup_probe_geometry_disagreements": sum(
            c["pickup_probe_geometry_disagreements"] for c in elig
        ),
        "stage_counts": dict(Counter(c["failure_stage"] for c in cases)),
        "cases": cases,
    }


def run(source_dir: Path, old_dir: Path, diverse_dir: Path,
        output_dir: Path) -> dict:
    """Recreate verified frozen BC; NEVER use evaluator-only state as input."""
    import jax
    import xminigrid
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
    from xminigrid.wrappers import (
        DirectionObservationWrapper,
        RulesAndGoalsObservationWrapper,
    )

    from bagi.experiments import xland_diverse_physical_demos as diversity
    from bagi.experiments.general_brain_native_g1 import source_policies
    from bagi.experiments.xland_action_coverage import WALKABLE, locate_exact
    from bagi.experiments.xland_existing_bc import get_rules
    from bagi.experiments.xland_gate_b_exact_collection import public_features
    from bagi.experiments.xland_gate_b_native_collection import screen_source_layout
    from bagi.experiments.xland_putdown_reward_validity import (
        blue_square_transition,
        tiles_adjacent,
    )
    from bagi.experiments.xland_solvability_oracle import (
        DIRECTIONS,
        shortest_route_to_face_object,
    )

    verify_registration()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "summary.json"
    report = {
        "experiment": "G2-D2 learner navigation versus facing geometry diagnostic",
        "status": "running", "model_seeds": list(MODEL_SEEDS),
        "fresh_layouts": list(NEW_LAYOUTS), "episode_cap": CAP,
        "policy_input": "original public115 img+direction+goal/rules+four executed actions",
        "evaluator_only_geometry_and_discarded_native_branches": True,
        "new_teacher_queries": 0, "new_policy_training_steps": 0,
        "real_learner_native_actions": 0, "discarded_native_probes": 0,
        "results": [],
    }
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    models, training = source_policies(source_dir, old_dir, diverse_dir)
    room, defaults = xminigrid.make("XLand-MiniGrid-R1-9x9")
    wrapped = RulesAndGoalsObservationWrapper(DirectionObservationWrapper(room))
    reset, step = jax.jit(wrapped.reset), jax.jit(wrapped.step)
    blue = np.asarray(TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE])
    red = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.RED])
    empty = int(Tiles.EMPTY)
    params = defaults.replace(ruleset=get_rules("putdown"), max_steps=CAP)
    for model_seed in MODEL_SEEDS:
        policy = models[model_seed][diversity.ARMS[1]]
        cases = []
        for layout in NEW_LAYOUTS:
            ts = reset(params, jax.random.key(layout))
            eligible = screen_source_layout(ts, blue, red)
            shortest = None
            if eligible:
                grid = np.asarray(ts.state.grid)
                source = locate_exact(grid, blue)
                agent = ts.state.agent
                try:
                    path, _ = shortest_route_to_face_object(
                        grid, tuple(map(int, np.asarray(agent.position))),
                        int(np.asarray(agent.direction)), source, WALKABLE,
                    )
                except ValueError:
                    # A preregistered layout is NOT replaced for being hard or
                    # unreachable. Preserve the real rollout and a null ceiling.
                    shortest = None
                else:
                    shortest = len(path) + 1  # include final pickup action

            def inspect_state(before):
                state = before.state
                grid = np.asarray(state.grid)
                found = np.argwhere(np.all(grid == blue, axis=-1))
                if len(found) > 1:
                    raise ValueError("Unreviewed multiple BLUE targets")
                target_pos = found[0] if len(found) == 1 else None
                g = evaluator_geometry(
                    state.agent.position, int(np.asarray(state.agent.direction)),
                    target_pos if target_pos is not None else (-99, -99),
                    DIRECTIONS,
                )
                return {
                    "target": target_pos, "geometry": g,
                    "pocket_empty": int(np.asarray(state.agent.pocket)[0]) == empty,
                }

            def native_step(before, action):
                return step(params, before, action)

            def observe(before, history):
                return public_features(before, history)

            def see_target(before):
                img = np.asarray(before.observation["img"])
                return bool(np.any(np.all(img == blue, axis=-1)))

            def transition(before, after, action):
                a = np.asarray(before.state.agent.pocket)
                b = np.asarray(after.state.agent.pocket)
                grid_a = np.asarray(before.state.grid)
                grid_b = np.asarray(after.state.grid)
                pickup, drop = blue_square_transition(
                    action, a, b, grid_a, grid_b, blue, empty,
                )
                wrong = bool(
                    action == 3 and not pickup
                    and int(a[0]) == empty and int(b[0]) != empty
                )
                return pickup, drop, wrong

            def adjacent(before):
                return tiles_adjacent(np.asarray(before.state.grid), blue, red)

            case = diagnostic_episode(
                policy, ts, layout=layout, eligible=eligible,
                observe=observe, native_step=native_step,
                inspect_state=inspect_state, visible_target=see_target,
                effective_transition=transition, final_adjacent=adjacent,
                shortest_pickup_actions=shortest,
            )
            cases.append(case)
            report["real_learner_native_actions"] += case["native_actions"]
            report["discarded_native_probes"] += case["discarded_pickup_probes"]
            report["results"] = [r for r in report["results"]
                                 if r["model_seed"] != model_seed]
            report["results"].append({
                "model_seed": model_seed, "training": training[model_seed][diversity.ARMS[1]],
                "metrics": {"cases": cases},
            })
            target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["results"][-1]["metrics"] = aggregate_cases(cases)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"G2-D2 seed={model_seed} BLUE physical="
              f"{report['results'][-1]['metrics']['physical_placements']}/"
              f"{report['results'][-1]['metrics']['eligible']} "
              f"adjacent={report['results'][-1]['metrics']['adjacent_episodes']} "
              f"facing={report['results'][-1]['metrics']['facing_episodes']} "
              f"learner_steps={report['real_learner_native_actions']}", flush=True)
    if len(report["results"]) != 2 or report["real_learner_native_actions"] > 800:
        raise ValueError("Incomplete or over-budget native root-cause diagnosis")
    if report["discarded_native_probes"] > 800:
        raise ValueError("Unbounded native discarded branch diagnostics")
    report["status"] = "complete"
    report["limitations"] = [
        "The frozen imitation.BC policy is NOT improved or retrained in D2.",
        "Hidden grid, heading, pocket and BFS are evaluator-only; no policy leakage.",
        "A native discarded pickup branch is counted but never controls the agent.",
        "Manhattan distance and revisits do not isolate an optimal causal navigation change.",
        "This is one XLand task and two policy seeds, not a cross-domain root cause proof.",
        "Native reward without ordered BLUE pickup/drop and adjacency never counts as physical success.",
    ]
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--old-teacher-dir", type=Path, required=True)
    parser.add_argument("--diverse-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/g2_d2_navigation"))
    args = parser.parse_args()
    run(args.source_dir, args.old_teacher_dir, args.diverse_dir, args.output_dir)
