"""Evaluator-only Workshop reachability and public-aliasing audit (NO native steps).

This is a finite-graph *diagnostic*, never an actor, teacher, training
reward, planning module, or source of privileged policy inputs. Uses the
existing Workshop world's observation and collision implementations; mirrors
only its documented switch/key effects in a side-effect-free transition.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_GIT_BLOB_SHA1 = "3a91b058d3461ce4ecb489e93939c88f624557f4"
LAYOUTS = ("default", "mirror_x")
SIZE = 9
VIEW_SIZE = 5
CAP = 40
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
STARTS = tuple((row, col) for row in (1, 2) for col in range(1, 8)
               if (row, col) != (1, 3))

# (row, col, door_open, has_key) — evaluator-only latent state.
State = tuple[int, int, bool, bool]


def _world_class():
    from bagi.worlds.workshop import WorkshopEnv

    return WorkshopEnv


def verify_source() -> None:
    """Fail closed on a changed environment; graph is NOT a generic oracle."""
    path = Path(inspect.getfile(_world_class()))
    raw = path.read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()
    if blob != SOURCE_GIT_BLOB_SHA1:
        raise ValueError("Workshop native source changed; evaluator audit must be reviewed")
    if len(STARTS) != 13 or len(DIRECTIONS) != 4 or CAP != 40:
        raise ValueError("Frozen evaluator audit constants changed")


def _place(world: Any, state: State) -> None:
    """Set PRIVATE evaluator fields; never invoke env.reset/step or an actor."""
    world._pos = np.asarray(state[:2], dtype=np.int64)
    world._door_open = state[2]
    world._has_key = state[3]


def _view(world: Any, state: State) -> bytes:
    _place(world, state)
    public = np.asarray(world._observe())
    if public.shape != (VIEW_SIZE, VIEW_SIZE) or public.dtype != np.int8:
        raise ValueError("Unexpected Workshop native public observation")
    return public.tobytes()


def _next_state(world: Any, state: State, action: int) -> State:
    """Read-only evaluator graph edge using native collision + known effects."""
    if action not in range(4):
        raise ValueError("Invalid native action")
    _place(world, state)
    candidate = world._pos + np.asarray(DIRECTIONS[action], dtype=np.int64)
    if world._can_enter(candidate):
        pos = (int(candidate[0]), int(candidate[1]))
    else:
        pos = state[:2]
    opened = state[2] or pos == tuple(map(int, world._switch))
    has_key = state[3] or pos == tuple(map(int, world._key))
    return pos[0], pos[1], opened, has_key


def _initial_states(world: Any) -> list[State]:
    return [(*world._layout_coords(row, col), False, False) for row, col in STARTS]


def _shortest(world: Any, start: State, goal: tuple[int, int]) -> int | None:
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        state, distance = queue.popleft()
        if state[:2] == goal:
            return distance
        for action in range(4):
            after = _next_state(world, state, action)
            if after not in visited:
                visited.add(after)
                queue.append((after, distance + 1))
    return None


def audit_layout(world: Any) -> dict:
    """No simulator actions. Terminal goal states have no outgoing edges."""
    starts = _initial_states(world)
    goal = tuple(map(int, world._goal))
    queue = deque(starts)
    states = set(starts)
    while queue:
        state = queue.popleft()
        if state[:2] == goal:
            continue
        for action in range(4):
            after = _next_state(world, state, action)
            if after not in states:
                states.add(after)
                queue.append(after)
    public_to_latent: dict[bytes, set[State]] = defaultdict(set)
    public_action_to_next: dict[tuple[bytes, int], set[bytes]] = defaultdict(set)
    for state in states:
        view = _view(world, state)
        public_to_latent[view].add(state)
        if state[:2] == goal:
            continue
        for action in range(4):
            after = _next_state(world, state, action)
            public_action_to_next[(view, action)].add(_view(world, after))
    shortest = [_shortest(world, start, goal) for start in starts]
    if any(value is None for value in shortest):
        raise ValueError("Unreachable registered Workshop initial state")
    return {
        "legal_initial_starts": len(starts),
        "distinct_initial_public_views": len({_view(world, state) for state in starts}),
        "reachable_evaluator_states": len(states),
        "distinct_public_views": len(public_to_latent),
        "public_views_aliasing_multiple_latent_states": sum(
            len(states_for_view) > 1 for states_for_view in public_to_latent.values()
        ),
        "maximum_latent_states_per_public_view": max(map(len, public_to_latent.values())),
        "nonterminal_public_view_action_keys": len(public_action_to_next),
        "keys_with_ambiguous_next_public_view": sum(
            len(outcomes) > 1 for outcomes in public_action_to_next.values()
        ),
        "shortest_native_actions_from_each_start": [int(value) for value in shortest],
        "min_shortest_actions": min(shortest),
        "max_shortest_actions": max(shortest),
        "all_starts_reachable_within_40": all(value <= CAP for value in shortest),
    }


def run() -> dict:
    verify_source()
    layouts: dict[str, dict] = {}
    world_class = _world_class()
    for name in LAYOUTS:
        world = world_class(size=SIZE, view_size=VIEW_SIZE, max_steps=CAP,
                            start_mode="seeded_start_chamber", layout_mode=name)
        try:
            layouts[name] = audit_layout(world)
        finally:
            world.close()
    return {
        "study": "G2-E5-P2", "status": "complete",
        "contribution": "evaluator-only finite-state reachability and aliasing diagnostic",
        "source_git_blob_sha1": SOURCE_GIT_BLOB_SHA1,
        "layouts": layouts,
        "native_step_calls": 0, "native_reset_calls": 0,
        "teacher_queries": 0, "policy_updates": 0, "heldout_reads": 0,
        "promotion": False,
        "limitations": [
            "Uses privileged evaluator-only state; NEVER a policy input or teacher.",
            "Graph mirrors switch/key effects without native step calls; not a native rollout.",
            "Fixed default/mirror geometries and 13 starts; reset seeds do not create layouts.",
            "Latent-state aliasing does not prove memory is required for optimal control.",
            "Shortest-path reachability does not establish a learned public-only policy.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "native_step_calls": 0,
                      "promotion": result["promotion"]}))


if __name__ == "__main__":
    main()
