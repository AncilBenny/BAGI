"""G2-E4-P0: bounded MiniGrid benchmark-diversity audit, NOT policy training.

Only PUBLIC image observations form the public observation fingerprints. Full
world-grid fingerprints are evaluator-only metadata for this BENCHMARK audit;
they never select actions or feed a learner. Each reset sees the identical
fixed random action sequence. Every actual simulator step is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

TASKS = (
    "MiniGrid-Empty-Random-6x6-v0",
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-DoorKey-6x6-v0",
    "MiniGrid-MemoryS13Random-v0",
)
AUDIT_SEEDS = tuple(range(21000, 21016))
PROBE_ACTIONS = 24
ACTION_RNG_SEED = 4260
REGISTERED_ACTIONS = 7
MAX_NATIVE_ACTIONS = len(TASKS) * len(AUDIT_SEEDS) * PROBE_ACTIONS  # 1536
MAX_RESETS = len(TASKS) * len(AUDIT_SEEDS)  # 64
MIN_DISTINCT_LAYOUTS = 8
MIN_DISTINCT_PUBLIC_SEQUENCES = 4
FUTURE_TEST_SEED_FLOOR = 26000  # none of these are touched by this audit


def verify_registration() -> None:
    """Reject any unpreregistered task, seed or budget before creating worlds."""
    if (
        TASKS != (
            "MiniGrid-Empty-Random-6x6-v0",
            "MiniGrid-DoorKey-5x5-v0",
            "MiniGrid-DoorKey-6x6-v0",
            "MiniGrid-MemoryS13Random-v0",
        )
        or AUDIT_SEEDS != tuple(range(21000, 21016))
        or PROBE_ACTIONS != 24
        or ACTION_RNG_SEED != 4260
        or REGISTERED_ACTIONS != 7
        or MAX_NATIVE_ACTIONS != 1536
        or MAX_RESETS != 64
        or MIN_DISTINCT_LAYOUTS != 8
        or MIN_DISTINCT_PUBLIC_SEQUENCES != 4
        or FUTURE_TEST_SEED_FLOOR != 26000
    ):
        raise ValueError("G2-E4-P0 preregistration or budget modified")


def fixed_actions() -> tuple[int, ...]:
    """Generate once; ALL tasks/seeds see an identical, nonadaptive script."""
    rng = np.random.default_rng(ACTION_RNG_SEED)
    values = tuple(int(x) for x in rng.integers(0, REGISTERED_ACTIONS, PROBE_ACTIONS))
    if len(values) != PROBE_ACTIONS or any(not 0 <= v < REGISTERED_ACTIONS for v in values):
        raise AssertionError("Fixed random diagnostic action sequence invalid")
    return values


def _hash_array(observation: Any) -> bytes:
    """Hash shape/dtype/content, not a cached pointer or evaluator state."""
    array = np.asarray(observation)
    if array.ndim != 1 or array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError("Expected finite flattened PUBLIC image observation")
    a = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(a.dtype).encode("ascii"))
    digest.update(json.dumps(a.shape).encode("ascii"))
    digest.update(a.tobytes())
    return digest.digest()


def _world_fingerprint(env: Any) -> str:
    """Evaluator ONLY: full reset layout, including agent position and facing."""
    native = env.unwrapped
    full = np.ascontiguousarray(native.grid.encode())
    if full.ndim != 3 or full.dtype.kind not in "biuf" or not np.isfinite(full).all():
        raise ValueError("Unexpected evaluator-only MiniGrid grid encoding")
    coordinates = tuple(int(x) for x in native.agent_pos)
    direction = int(native.agent_dir)
    if len(coordinates) != 2 or direction not in range(4):
        raise ValueError("Unexpected native reset geometry")
    digest = hashlib.sha256()
    digest.update(json.dumps([full.shape, str(full.dtype), coordinates, direction]).encode())
    digest.update(full.tobytes())
    return digest.hexdigest()


def inspect_one(env: Any, reset_seed: int, actions: tuple[int, ...]) -> dict[str, Any]:
    """Execute at most the registered native actions, no hidden-state decisions."""
    if reset_seed not in AUDIT_SEEDS or actions != fixed_actions():
        raise ValueError("Unregistered reset seed or diagnostic actions")
    if int(env.action_space.n) != REGISTERED_ACTIONS:
        raise ValueError("Unexpected benchmark action space")
    initial, _private = env.reset(seed=reset_seed)
    initial_fingerprint = _hash_array(initial)
    # Do not use any privileged state for actions or the PUBLIC hash.
    layout_fingerprint = _world_fingerprint(env)
    digest = hashlib.sha256()
    digest.update(initial_fingerprint)
    steps = 0
    reward_sum = 0.0
    terminated = truncated = False
    public_distinct = {initial_fingerprint}
    for action in actions:
        if not env.action_space.contains(action):
            raise ValueError("Invalid diagnostic action")
        next_public, raw_reward, terminated, truncated, _private = env.step(action)
        steps += 1
        reward = float(raw_reward)
        if not np.isfinite(reward):
            raise ValueError("Nonfinite native reward")
        reward_sum += reward
        obs_digest = _hash_array(next_public)
        digest.update(obs_digest)
        public_distinct.add(obs_digest)
        if terminated or truncated:
            break
    if not 1 <= steps <= PROBE_ACTIONS:
        raise AssertionError("Diagnostic real-step budget violated")
    return {
        "reset_seed": reset_seed,
        "evaluator_layout_sha256": layout_fingerprint,
        "initial_public_sha256": initial_fingerprint.hex(),
        "public_sequence_sha256": digest.hexdigest(),
        "distinct_public_views_in_episode": len(public_distinct),
        "native_actions": steps,
        "native_reward": round(reward_sum, 8),
        "native_terminated": bool(terminated),
        "native_truncated": bool(truncated),
    }


def _make_env(task: str) -> Any:
    """Only upstream registered standard environments, no privileged actor wrapper."""
    import gymnasium as gym
    import minigrid  # noqa: F401 - official environment registration
    from gymnasium.wrappers import FlattenObservation
    from minigrid.wrappers import ImgObsWrapper

    if task not in TASKS:
        raise ValueError("Unregistered task")
    return FlattenObservation(ImgObsWrapper(gym.make(task)))


def summarize_task(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if (
        len(episodes) != len(AUDIT_SEEDS)
        or [x["reset_seed"] for x in episodes] != list(AUDIT_SEEDS)
    ):
        raise ValueError("Missing or unregistered benchmark audit reset")
    layout_count = len({x["evaluator_layout_sha256"] for x in episodes})
    first_view_count = len({x["initial_public_sha256"] for x in episodes})
    sequence_count = len({x["public_sequence_sha256"] for x in episodes})
    action_count = sum(x["native_actions"] for x in episodes)
    if action_count > len(AUDIT_SEEDS) * PROBE_ACTIONS:
        raise ValueError("Over budget")
    return {
        "num_resets": len(episodes),
        "native_actions": action_count,
        "distinct_evaluator_reset_layouts": layout_count,
        "distinct_initial_public_views": first_view_count,
        "distinct_fixed_script_public_sequences": sequence_count,
        "qualified_diversity_only": (
            layout_count >= MIN_DISTINCT_LAYOUTS
            and sequence_count >= MIN_DISTINCT_PUBLIC_SEQUENCES
        ),
        "episodes": episodes,
    }


def run(output_dir: Path) -> dict[str, Any]:
    verify_registration()
    output_dir.mkdir(parents=True, exist_ok=True)
    actions = fixed_actions()
    summary: dict[str, Any] = {
        "study": "G2-E4-P0",
        "status": "running",
        "kind": "benchmark-validity diagnostic, NO training/evaluation/policy competence",
        "versions": {"minigrid": version("minigrid"), "gymnasium": version("gymnasium")},
        "tasks": list(TASKS),
        "audit_seeds": list(AUDIT_SEEDS),
        "diagnostic_action_script": list(actions),
        "max_resets": MAX_RESETS,
        "max_native_actions": MAX_NATIVE_ACTIONS,
        "actual_resets": 0,
        "actual_native_actions": 0,
        "teacher_queries": 0,
        "results": {},
        "limitations": [
            "Full-grid hashes are evaluator-only, never used for action selection.",
            "A common open-loop random script tests public visibility, not task competence.",
            "Different initial PUBLIC images alone are not required for different latent starts.",
            "Seed-audit conditions MUST NOT be used as future training or heldout evaluation.",
            "Diversity alone does NOT establish learnability, policy skill, transfer or retention.",
        ],
    }
    try:
        for task in TASKS:
            episodes: list[dict[str, Any]] = []
            for reset_seed in AUDIT_SEEDS:
                world = _make_env(task)
                try:
                    row = inspect_one(world, reset_seed, actions)
                finally:
                    world.close()
                summary["actual_resets"] += 1
                summary["actual_native_actions"] += row["native_actions"]
                if (
                    summary["actual_resets"] > MAX_RESETS
                    or summary["actual_native_actions"] > MAX_NATIVE_ACTIONS
                ):
                    raise AssertionError("Registered native audit resource ceiling exceeded")
                episodes.append(row)
            summary["results"][task] = summarize_task(episodes)
        if summary["actual_resets"] != MAX_RESETS:
            raise AssertionError("Incomplete registered benchmark audit")
        if sum(row["native_actions"] for row in summary["results"].values()) != summary[
            "actual_native_actions"
        ]:
            raise AssertionError("Actual native step ledger inconsistent")
        summary["status"] = "complete"
    except Exception as exc:
        summary["status"] = "failed"
        summary["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/g2_e4_p0_benchmark_validity")
    )
    outcome = run(parser.parse_args().output_dir)
    print(
        f"G2-E4-P0 status={outcome['status']} "
        f"resets={outcome['actual_resets']} "
        f"native_actions={outcome['actual_native_actions']}"
    )
    for name, row in outcome["results"].items():
        print(
            f"{name}: evaluator_layouts={row['distinct_evaluator_reset_layouts']}/16 "
            f"initial_public={row['distinct_initial_public_views']}/16 "
            f"public_traces={row['distinct_fixed_script_public_sequences']}/16 "
            f"diversity_gate={row['qualified_diversity_only']}"
        )
