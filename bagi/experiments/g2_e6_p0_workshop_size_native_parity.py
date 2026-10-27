"""Evaluator-only native-source Workshop size feasibility, with ZERO env steps.

Reuse G2-E5-P2's pinned Workshop source, public view, and collision helpers.
The evaluator's latent graph is never an actor, teacher, or training label.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

from bagi.experiments import g2_e5_p2_workshop_reachability_audit as p2

SIZES = (9, 11, 13)
LAYOUTS = ("default", "mirror_x")
CAP = 40
EXPECTED_INITIAL_PUBLIC = {9: 13, 11: 16, 13: 16}
EXPECTED = {
    9: (13, 79, 57, 2, 15, 19),
    11: (17, 121, 83, 11, 19, 25),
    13: (21, 171, 89, 34, 23, 31),
}


def verify_registration() -> None:
    if SIZES != (9, 11, 13) or LAYOUTS != ("default", "mirror_x") or CAP != 40:
        raise ValueError("Workshop size feasibility registration changed")


def audit_size(world) -> dict:
    """Use native _observe/_can_enter, never native reset/step."""
    starts = [
        (*world._layout_coords(row, col), False, False)
        for row in (1, 2)
        for col in range(1, world.size - 1)
        if (row, col) != (1, 3)
    ]
    goal = tuple(map(int, world._goal))
    queue = deque(starts)
    states = set(starts)
    while queue:
        state = queue.popleft()
        if state[:2] == goal:
            continue
        for action in range(4):
            after = p2._next_state(world, state, action)
            if after not in states:
                states.add(after)
                queue.append(after)

    public_to_states: dict[bytes, set] = defaultdict(set)
    next_public: dict[tuple[bytes, int], set[bytes]] = defaultdict(set)
    for state in states:
        before = p2._view(world, state)
        public_to_states[before].add(state)
        if state[:2] == goal:
            continue
        for action in range(4):
            after = p2._next_state(world, state, action)
            next_public[(before, action)].add(p2._view(world, after))
    shortest = [p2._shortest(world, start, goal) for start in starts]
    if any(distance is None for distance in shortest):
        raise ValueError("Unreachable Workshop start")
    return {
        "size": int(world.size),
        "layout": world.layout_mode,
        "legal_initial_starts": len(starts),
        "distinct_initial_public_views": len({p2._view(world, s) for s in starts}),
        "reachable_evaluator_states": len(states),
        "distinct_public_views": len(public_to_states),
        "public_views_aliasing_multiple_latent_states": sum(
            len(v) > 1 for v in public_to_states.values()
        ),
        "maximum_latent_states_per_public_view": max(
            map(len, public_to_states.values())
        ),
        "nonterminal_public_view_action_keys": len(next_public),
        "keys_with_ambiguous_next_public_view": sum(
            len(v) > 1 for v in next_public.values()
        ),
        "min_shortest_actions": min(shortest),
        "max_shortest_actions": max(shortest),
        "all_starts_reachable_within_40": all(d <= CAP for d in shortest),
        "shortest_native_actions_from_each_start": shortest,
    }


def run() -> dict:
    verify_registration()
    p2.verify_source()
    world_class = p2._world_class()
    cases = []
    for size in SIZES:
        for layout in LAYOUTS:
            world = world_class(
                size=size,
                view_size=5,
                max_steps=CAP,
                start_mode="seeded_start_chamber",
                layout_mode=layout,
            )
            try:
                result = audit_size(world)
            finally:
                world.close()
            signature = (
                result["legal_initial_starts"],
                result["reachable_evaluator_states"],
                result["distinct_public_views"],
                result["keys_with_ambiguous_next_public_view"],
                result["min_shortest_actions"],
                result["max_shortest_actions"],
            )
            if (
                signature != EXPECTED[size]
                or not result["all_starts_reachable_within_40"]
            ):
                raise ValueError("Workshop size hypothesis did not replicate")
            if result["distinct_initial_public_views"] != EXPECTED_INITIAL_PUBLIC[size]:
                raise ValueError("Initial public-view diversity changed")
            cases.append(result)
        left, right = cases[-2:]
        for key in EXPECTED_KEYS:
            if left[key] != right[key]:
                raise ValueError("Mirrored Workshop size parity changed")
    return {
        "study": "G2-E6-P0 Workshop source-pinned size feasibility",
        "status": "complete_evaluator_only",
        "workshop_source_git_blob_sha1": p2.SOURCE_GIT_BLOB_SHA1,
        "cases": cases,
        "native_step_calls": 0,
        "native_reset_calls": 0,
        "model_fits": 0,
        "optimizer_steps": 0,
        "teacher_queries": 0,
        "promotion": False,
        "limitations": [
            "Privileged evaluator graph, never actor inputs or training labels.",
            "Uses native _observe/_can_enter but no native reset/step.",
            "Size variants change geometry and distance, not causal rules.",
            "Feasibility is not learned goal competence or cross-domain transfer.",
        ],
    }


EXPECTED_KEYS = (
    "legal_initial_starts",
    "distinct_initial_public_views",
    "reachable_evaluator_states",
    "distinct_public_views",
    "public_views_aliasing_multiple_latent_states",
    "maximum_latent_states_per_public_view",
    "nonterminal_public_view_action_keys",
    "keys_with_ambiguous_next_public_view",
    "min_shortest_actions",
    "max_shortest_actions",
    "all_starts_reachable_within_40",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing audit")
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "native_step_calls": 0}))


if __name__ == "__main__":
    main()
