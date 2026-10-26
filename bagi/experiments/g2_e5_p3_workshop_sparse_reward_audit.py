"""Exact random-policy sparse-reward diagnostic on Workshop's evaluator graph.

Evaluator ONLY: never a learner, expert, action selector, reward-shaping input,
or native simulator rollout. Reuses the G2-E5-P2 source-pinned graph.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

from bagi.experiments import g2_e5_p2_workshop_reachability_audit as graph

ACTIONS = tuple(range(4))
CAP = graph.CAP


def _states(world) -> set[graph.State]:
    """All reachable evaluator states; stop expansion at native goal."""
    starts = graph._initial_states(world)
    goal = tuple(map(int, world._goal))
    visited = set(starts)
    queue = deque(starts)
    while queue:
        state = queue.popleft()
        if state[:2] == goal:
            continue
        for action in ACTIONS:
            after = graph._next_state(world, state, action)
            if after not in visited:
                visited.add(after)
                queue.append(after)
    return visited


def audit_random_policy(world) -> dict:
    """Uniform legal starts and uniformly random native commands, cap 40."""
    if CAP != 40 or len(ACTIONS) != 4 or world.max_steps != CAP:
        raise ValueError("Changed native horizon or action contract")
    starts = graph._initial_states(world)
    if len(starts) != 13 or len(set(starts)) != 13:
        raise ValueError("Changed registered Workshop start distribution")
    goal = tuple(map(int, world._goal))
    states = _states(world)
    if len(states) != 79:
        raise ValueError("Changed source-pinned evaluator graph")

    # Backward dynamic program: exact chance of goal by t from each start.
    by_state = {state: float(state[:2] == goal) for state in states}
    for _ in range(CAP):
        by_state = {
            state: (
                1.0 if state[:2] == goal
                else sum(by_state[graph._next_state(world, state, a)] for a in ACTIONS)
                / len(ACTIONS)
            )
            for state in states
        }
    per_start = [by_state[state] for state in starts]

    # Independent forward occupancy: goal is absorbing; stage flags monotonic.
    occupancy = {state: 1.0 / len(starts) for state in starts}
    checkpoints: dict[int, dict[str, float]] = {}
    for step in range(1, CAP + 1):
        updated: dict[graph.State, float] = defaultdict(float)
        for state, probability in occupancy.items():
            if state[:2] == goal:
                updated[state] += probability
                continue
            for action in ACTIONS:
                after = graph._next_state(world, state, action)
                updated[after] += probability / len(ACTIONS)
        occupancy = dict(updated)
        if abs(sum(occupancy.values()) - 1.0) > 1e-12:
            raise ValueError("Random-policy probability mass was not conserved")
        if step in (15, 20, 25, 30, 35, 40):
            checkpoints[step] = {
                "switch_activated": sum(p for s, p in occupancy.items() if s[2]),
                "key_acquired": sum(p for s, p in occupancy.items() if s[3]),
                "native_goal": sum(p for s, p in occupancy.items() if s[:2] == goal),
            }
    probability = checkpoints[CAP]["native_goal"]
    if abs(probability - sum(per_start) / len(per_start)) > 1e-12:
        raise ValueError("Independent forward/backward goal probabilities disagree")
    if not (0 <= probability <= checkpoints[CAP]["key_acquired"]
            <= checkpoints[CAP]["switch_activated"] <= 1):
        raise ValueError("Workshop causal-stage probability invariant failed")

    return {
        "registered_starts": len(starts),
        "reachable_evaluator_states": len(states),
        "uniform_random_goal_probability_40": probability,
        "uniform_random_goal_probability_by_start": per_start,
        "minimum_start_probability": min(per_start),
        "maximum_start_probability": max(per_start),
        "checkpoint_probabilities": {str(t): row for t, row in checkpoints.items()},
        "hypothetical_100_episode_no_goal_probability": (1 - probability) ** 100,
        "hypothetical_410_episode_no_goal_probability": (1 - probability) ** 410,
    }


def run() -> dict:
    """Pin native source, enumerate evaluator graph, execute ZERO native actions."""
    graph.verify_source()
    layouts = {}
    for layout in graph.LAYOUTS:
        world = graph._world_class()(
            size=graph.SIZE, view_size=graph.VIEW_SIZE,
            max_steps=CAP, start_mode="seeded_start_chamber", layout_mode=layout,
        )
        try:
            layouts[layout] = audit_random_policy(world)
        finally:
            world.close()
    return {
        "study": "G2-E5-P3",
        "status": "complete",
        "contribution": "exact evaluator-only uniform-random sparse-reward baseline",
        "source_git_blob_sha1": graph.SOURCE_GIT_BLOB_SHA1,
        "native_step_calls": 0,
        "native_reset_calls": 0,
        "teacher_queries": 0,
        "policy_updates": 0,
        "heldout_reads": 0,
        "promotion": False,
        "layouts": layouts,
        "limitations": [
            "Privileged evaluator-only finite graph; NEVER policy, teacher or actor input.",
            "Uniform random policy is not PPO/DQN and does not bound learned competence.",
            "Reset seeds choose 13 starts within two fixed mirrored geometries.",
            "Independent-episode illustrations are not actual native training outcomes.",
            "No native step or reset, no real rewarded episodes, no learned skills.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"study": result["study"], "status": result["status"],
                      "native_step_calls": 0, "promotion": False}))


if __name__ == "__main__":
    main()
