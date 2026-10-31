"""G2-E6-R5: bounded native Workshop public-history observability witness.

Evaluator-only paired route probe. The two action programs are authored from
public Workshop geometry, NOT learned policies or expert training labels.
No hidden info fields enter recorded observations or predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from bagi.experiments import g2_e5_p2_workshop_reachability_audit as p2

SIZES = (9, 11)
LAYOUTS = ("default", "mirror_x")
VIEW_SIZES = (3, 5)
PROGRAM_ACTIONS = 11
MAX_NATIVE_ACTIONS = 176
MAX_NATIVE_RESETS = 16
SOURCE_GIT_BLOB_SHA1 = "3a91b058d3461ce4ec489e93939c88f624557f4"
# Absolute movements: up=0, right=1, down=2, left=3.
COMMON = (1, 1, 3, 3, 2, 2)  # Enter switch, then traverse the open door.
WITH_KEY = (2, 2, 1, 1)     # Collect visible key, then approach lock.
WITHOUT_KEY = (2, 1, 1, 2)  # Reach the same lock without touching key.
REFLECT_ACTION = (0, 3, 2, 1)
LOCK = 8


def action_program(layout: str, collect_key: bool) -> tuple[int, ...]:
    """A fixed diagnostic control, not an autonomous navigation policy."""
    if layout not in LAYOUTS:
        raise ValueError("Unregistered layout")
    route = COMMON + (WITH_KEY if collect_key else WITHOUT_KEY) + (1,)
    if layout == "mirror_x":
        route = tuple(REFLECT_ACTION[a] for a in route)
    if len(route) != PROGRAM_ACTIONS:
        raise ValueError("Registered native action count changed")
    return route


def _rollout(world, *, size: int, layout: str, view_size: int,
             collect_key: bool) -> tuple[dict, np.ndarray, np.ndarray]:
    """Execute actual native actions; consume PUBLIC observations only."""
    seed = 17000 + (100 if collect_key else 0) + size
    observation, _ = world.reset(seed=seed)
    actions = action_program(layout, collect_key)
    public_history = []
    for i, action in enumerate(actions):
        before = np.asarray(observation)
        if before.shape != (view_size, view_size) or before.dtype != np.int8:
            raise ValueError("Unexpected public input contract")
        public_history.append({
            "public_view_sha256": hashlib.sha256(before.tobytes()).hexdigest(),
            "executed_action": int(action),
        })
        after, reward, terminated, truncated, _ = world.step(action)
        if reward != 0.0 or terminated or truncated:
            raise ValueError("Unexpected goal/reward/terminal in diagnostic route")
        if np.asarray(after).shape != (view_size, view_size):
            raise ValueError("Unexpected next public view")
        observation = after
        if i == PROGRAM_ACTIONS - 2:
            pre_lock = np.asarray(observation).copy()
    post_lock = np.asarray(observation).copy()
    last_action = actions[-1]
    direction = ((-1, 0), (0, 1), (1, 0), (0, -1))[last_action]
    radius = view_size // 2
    if int(pre_lock[radius + direction[0], radius + direction[1]]) != LOCK:
        raise ValueError("Not a visible public lock-target action")
    moved = not np.array_equal(pre_lock, post_lock)
    if moved != collect_key:
        raise ValueError("Failed expected effect/blocked paired control")
    if collect_key and np.any(post_lock == LOCK):
        raise ValueError("Lock still visible after claimed physical entry")
    return ({
        "collect_key_route": collect_key,
        "seed": seed,
        "native_steps": len(actions),
        "actual_action_history": list(actions),
        "public_action_history": public_history,
        "pre_lock_public": pre_lock.tolist(),
        "post_lock_public": post_lock.tolist(),
        "publicly_verified_entry": bool(moved),
        "native_reward_sum": 0.0,
    }, pre_lock, post_lock)


def run() -> dict:
    """One frozen 16-reset, 176-action source-pinned native feasibility run."""
    p2.verify_source()
    if SOURCE_GIT_BLOB_SHA1 != p2.SOURCE_GIT_BLOB_SHA1:
        raise ValueError("Immutable Workshop source mismatch")
    cases = []
    actions_total = 0
    resets_total = 0
    for size in SIZES:
        for layout in LAYOUTS:
            for view_size in VIEW_SIZES:
                world = p2._world_class()(
                    size=size, view_size=view_size, max_steps=40,
                    start_mode="fixed", layout_mode=layout,
                )
                try:
                    negative, no_key_before, _ = _rollout(
                        world, size=size, layout=layout, view_size=view_size,
                        collect_key=False,
                    )
                    positive, with_key_before, _ = _rollout(
                        world, size=size, layout=layout, view_size=view_size,
                        collect_key=True,
                    )
                finally:
                    world.close()
                resets_total += 2
                actions_total += negative["native_steps"] + positive["native_steps"]
                same_public = bool(np.array_equal(no_key_before, with_key_before))
                if same_public != (view_size == 3):
                    raise ValueError("Frozen 3x3 witness / 5x5 control failed")
                if negative["actual_action_history"] == positive["actual_action_history"]:
                    raise ValueError("Different key histories must have different actions")
                cases.append({
                    "size": size, "layout": layout, "view_size": view_size,
                    "same_pre_lock_public": same_public,
                    "same_last_executed_action": (
                        negative["actual_action_history"][-1]
                        == positive["actual_action_history"][-1]
                    ),
                    "negative": negative, "positive": positive,
                })
    if actions_total != MAX_NATIVE_ACTIONS or resets_total != MAX_NATIVE_RESETS:
        raise ValueError("Native simulator step/reset budget mismatch")
    return {
        "study": "G2-E6-R5 public-observation lock-effect witness",
        "status": "bounded_native_witness_not_training",
        "source_git_blob_sha1": SOURCE_GIT_BLOB_SHA1,
        "native_step_returns": actions_total,
        "native_reset_calls": resets_total,
        "optimizer_updates": 0,
        "fitted_models": 0,
        "teacher_queries": 0,
        "cases": cases,
        "limitations": [
            "Hand-authored open-loop source routes, not a learned navigation skill.",
            "Mirroring is not an independent causal geometry family.",
            "Fixed starts and deterministic trajectories do not establish distinct data support.",
            "Only legal PUBLIC observations/actions are recorded; no hidden labels train a model.",
            "A 3x3 witness does not establish sample-efficient memory learning or transfer.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute-bounded-native", action="store_true")
    args = parser.parse_args()
    if not args.execute_bounded_native:
        raise SystemExit("Native action budget requires explicit --execute-bounded-native")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite existing result")
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "native_step_returns": result["native_step_returns"]}))


if __name__ == "__main__":
    main()
