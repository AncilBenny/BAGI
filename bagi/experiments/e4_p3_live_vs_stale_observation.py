"""One-variable G2-E4-P3 frozen-policy online-observation ablation.

Uses original verified P1 numeric PPO weights and native MiniGrid Empty-Random.
The ONLY intervention freezes the policy INPUT to the episode's first PUBLIC
observation; the actual environment always evolves normally. No training,
planner, oracle hints, privileged inputs, or new reward is introduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from bagi.experiments import e4_p2_frozen_contract as p2
from bagi.experiments import e4_p2_frozen_runner as loader
from bagi.experiments import learned_competence_e4 as p1

TASK = "MiniGrid-Empty-Random-6x6-v0"
MODEL_SEEDS = (17, 29)
TEST_SEEDS = tuple(range(28000, 28016))
ARMS = ("live_public", "stale_initial_public")
CAP = 144
MAX_TEST_ACTIONS = len(MODEL_SEEDS) * len(ARMS) * len(TEST_SEEDS) * CAP
RNG_BASE = 28_000_000
SOURCE_ARTIFACT = 12606845219
SOURCE_ZIP_SHA256 = "b3cecce2f9cb15092490f893ba8b0df16a9424a6d1b010747b07094cdccfa712"


def verify_registration() -> None:
    if (
        TASK != p2.TASKS[0]
        or MODEL_SEEDS != (17, 29)
        or TEST_SEEDS != tuple(range(28000, 28016))
        or set(TEST_SEEDS) & (set(p1.TEST_SEEDS) | set(p2.TEST_SEEDS))
        or ARMS != ("live_public", "stale_initial_public")
        or CAP != 144
        or MAX_TEST_ACTIONS != 9216
        or RNG_BASE != 28_000_000
        or SOURCE_ARTIFACT != p2.SOURCE_ARTIFACT_ID
        or SOURCE_ZIP_SHA256 != p2.ARCHIVE_SHA256
    ):
        raise ValueError("G2-E4-P3 prospective one-variable protocol has changed")


def paired_seed(model_seed: int, reset_seed: int) -> int:
    verify_registration()
    if model_seed not in MODEL_SEEDS or reset_seed not in TEST_SEEDS:
        raise ValueError("Unregistered model/start seed")
    return RNG_BASE + MODEL_SEEDS.index(model_seed) * 100_000 + reset_seed - TEST_SEEDS[0]


def evaluate_episode(
    env: Any, actor: Any, model_seed: int, reset_seed: int,
    arm: str, ledger: dict[str, int],
) -> tuple[dict, dict[str, np.ndarray]]:
    """Only first/current PUBLIC images reach the SAME frozen stochastic PPO."""
    import torch

    verify_registration()
    if arm not in ARMS:
        raise ValueError("Invalid preregistered observation condition")
    env.action_space.seed(reset_seed)
    observation, _private = env.reset(seed=reset_seed)
    initial = p1._public_array(observation).copy()
    initial_hash = p1._public_digest(initial).hex()
    layout_hash = p1._layout_digest(env)  # EVALUATOR only, never to actor
    # SAME torch sampling stream on corresponding starts in both arms.
    sampling_seed = paired_seed(model_seed, reset_seed)
    torch.manual_seed(sampling_seed)
    obs_rows: list[np.ndarray] = []
    actor_rows: list[np.ndarray] = []
    action_rows: list[int] = []
    reward_rows: list[float] = []
    end_rows: list[bool] = []
    trunc_rows: list[bool] = []
    terminated = truncated = False
    for _ in range(CAP):
        current = p1._public_array(observation).copy()
        actor_input = current if arm == "live_public" else initial
        prediction, _ = actor.predict(actor_input, deterministic=False)
        action = int(np.asarray(prediction).item())
        if not env.action_space.contains(action):
            raise ValueError("Actor chose invalid native action")
        if ledger["test_actions"] >= MAX_TEST_ACTIONS:
            raise ValueError("G2-E4-P3 true native action ceiling exceeded")
        observation, reward, terminated, truncated, _private = env.step(action)
        ledger["test_actions"] += 1  # one per actually EXECUTED native action
        if not np.isfinite(float(reward)):
            raise ValueError("Nonfinite native reward")
        obs_rows.append(current)
        actor_rows.append(actor_input.copy())
        action_rows.append(action)
        reward_rows.append(float(reward))
        end_rows.append(bool(terminated))
        trunc_rows.append(bool(truncated))
        if terminated or truncated:
            break
    if not (terminated or truncated):
        raise ValueError("Native episode did not stop by official 144-step cap")
    total_reward = float(sum(reward_rows))
    if total_reward > 0 and not terminated:
        raise ValueError("Goal reward without native termination")
    row = {
        "task": TASK,
        "model_seed": model_seed,
        "reset_seed": reset_seed,
        "arm": arm,
        "sampling_seed": sampling_seed,
        "evaluator_layout_sha256": layout_hash,
        "initial_public_sha256": initial_hash,
        "native_actions": len(action_rows),
        "native_reward": round(total_reward, 8),
        "native_terminated": bool(terminated),
        "native_truncated": bool(truncated),
        "native_goal_success": int(bool(terminated) and total_reward > 0),
        "action_histogram": np.bincount(action_rows, minlength=7).tolist(),
    }
    arrays = {
        "public_pre_action": np.stack(obs_rows),
        "actor_public_input": np.stack(actor_rows),
        "actions": np.asarray(action_rows, dtype=np.int16),
        "rewards": np.asarray(reward_rows, dtype=np.float64),
        "terminated": np.asarray(end_rows, dtype=np.bool_),
        "truncated": np.asarray(trunc_rows, dtype=np.bool_),
    }
    if arm == "stale_initial_public" and not np.all(
        arrays["actor_public_input"] == initial
    ):
        raise AssertionError("Stale actor input changed within episode")
    return row, arrays


def score(rows: dict[str, list[dict]]) -> dict:
    verify_registration()
    if set(rows) != set(ARMS) or any(
        len(rows[arm]) != len(TEST_SEEDS)
        or [x["reset_seed"] for x in rows[arm]] != list(TEST_SEEDS)
        for arm in ARMS
    ):
        raise ValueError("Missing or nonfrozen paired TEST condition")
    for index, seed in enumerate(TEST_SEEDS):
        live, stale = (rows[arm][index] for arm in ARMS)
        if (
            live["evaluator_layout_sha256"] != stale["evaluator_layout_sha256"]
            or live["initial_public_sha256"] != stale["initial_public_sha256"]
            or live["sampling_seed"] != stale["sampling_seed"]
            or live["sampling_seed"] != paired_seed(live["model_seed"], seed)
            or live["model_seed"] != stale["model_seed"]
        ):
            raise ValueError("Physical reset/public initial view/RNG not paired")
        for row in (live, stale):
            if (
                not 1 <= row["native_actions"] <= CAP
                or sum(row["action_histogram"]) != row["native_actions"]
                or row["native_goal_success"]
                != int(row["native_terminated"] and row["native_reward"] > 0)
            ):
                raise ValueError("Invalid native goal/action accounting")
    distinct_layouts = len({x["evaluator_layout_sha256"] for x in rows[ARMS[0]]})
    success = {arm: sum(x["native_goal_success"] for x in rows[arm]) for arm in ARMS}
    paired_wins = sum(
        live["native_goal_success"] > stale["native_goal_success"]
        for live, stale in zip(rows[ARMS[0]], rows[ARMS[1]], strict=True)
    )
    paired_losses = sum(
        live["native_goal_success"] < stale["native_goal_success"]
        for live, stale in zip(rows[ARMS[0]], rows[ARMS[1]], strict=True)
    )
    return {
        "distinct_physical_test_starts": distinct_layouts,
        "valid_start_diversity": distinct_layouts >= 12,
        "native_goals": success,
        "paired_live_wins": paired_wins,
        "paired_live_losses": paired_losses,
        "paired_ties": len(TEST_SEEDS) - paired_wins - paired_losses,
        "native_actions": {arm: sum(x["native_actions"] for x in rows[arm]) for arm in ARMS},
        "strong_online_feedback_signal": bool(
            distinct_layouts >= 12
            and success[ARMS[0]] - success[ARMS[1]] >= 3
            and paired_losses <= 1
        ),
    }


def save_numeric(path: Path, traces: list[dict[str, np.ndarray]]) -> str:
    arrays: dict[str, np.ndarray] = {
        "episode_offsets": np.asarray(
            [0] + list(np.cumsum([len(e["actions"]) for e in traces])), dtype=np.int64
        )
    }
    for key in (
        "public_pre_action", "actor_public_input", "actions", "rewards",
        "terminated", "truncated",
    ):
        arrays[key] = np.concatenate([e[key] for e in traces], axis=0)
    if any(v.dtype.kind not in "biuf" or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Unsafe or nonfinite numeric evidence")
    np.savez_compressed(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(source_archive: Path, output_dir: Path) -> dict:
    import torch

    verify_registration()
    # Previous E4-P1 source SHA and all 12 numeric inner NPZs verified here,
    # BEFORE constructing/resetting even one native environment.
    frozen = p2.archive_weights(source_archive)
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = {"train_actions": 0, "test_actions": 0}
    result: dict[str, Any] = {
        "study": "G2-E4-P3", "status": "running",
        "contribution_type": "single-variable live-public versus stale-public ablation",
        "source_artifact_id": SOURCE_ARTIFACT,
        "source_zip_sha256": SOURCE_ZIP_SHA256,
        "tasks": [TASK], "model_seeds": list(MODEL_SEEDS),
        "test_seeds": list(TEST_SEEDS), "arms": list(ARMS),
        "action_cap": MAX_TEST_ACTIONS,
        "native_action_ledger": ledger, "teacher_queries": 0,
        "versions": {x: version(x) for x in ("gymnasium", "minigrid", "stable-baselines3")},
        "results": [], "numeric_evidence": [],
        "limitations": [
            "This does not train or validate general task-independent competence.",
            "Stale arm retains initial public view; removes only later feedback.",
            "Coupled RNG does not imply identical later physical trajectories.",
            "One navigation task, 16 starts per seed: no transfer inference.",
        ],
    }
    try:
        for model_seed in MODEL_SEEDS:
            construction_env = p1._make_env(TASK)
            try:
                actor = loader._restore_policy(
                    TASK, model_seed, frozen[(TASK, model_seed, "trained")],
                    construction_env,
                )
                episodes: dict[str, list[dict]] = {}
                for arm in ARMS:
                    arm_rows: list[dict] = []
                    traces: list[dict[str, np.ndarray]] = []
                    for seed in TEST_SEEDS:
                        env = p1._make_env(TASK)
                        try:
                            row, trace = evaluate_episode(
                                env, actor, model_seed, seed, arm, ledger
                            )
                        finally:
                            env.close()
                        arm_rows.append(row)
                        traces.append(trace)
                    filename = f"E4_P3_EmptyRandom_seed{model_seed}_{arm}.npz"
                    sha = save_numeric(output_dir / filename, traces)
                    result["numeric_evidence"].append({
                        "model_seed": model_seed, "arm": arm, "file": filename,
                        "sha256": sha,
                    })
                    episodes[arm] = arm_rows
                result["results"].append({
                    "model_seed": model_seed, "episodes": episodes,
                    "comparison": score(episodes),
                })
            finally:
                construction_env.close()
        if ledger["train_actions"] != 0 or ledger["test_actions"] > MAX_TEST_ACTIONS:
            raise ValueError("Final native budget mismatch")
        if len(result["results"]) != len(MODEL_SEEDS):
            raise ValueError("Incomplete model-seed coverage")
        result["strong_online_feedback_signal_both_seeds"] = all(
            r["comparison"]["strong_online_feedback_signal"] for r in result["results"]
        )
        result["status"] = "complete"
    except Exception as exc:
        result["status"] = "failed"
        result["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output_dir / "summary.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-archive", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("runs/g2_e4_p3_public_feedback"))
    args = p.parse_args()
    outcome = run(args.source_archive, args.output_dir)
    print(
        f"G2-E4-P3 {outcome['status']} actions={outcome['native_action_ledger']} "
        f"online_feedback_both={outcome['strong_online_feedback_signal_both_seeds']}"
    )
