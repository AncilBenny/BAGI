"""G2-E4-P2: one frozen-weight, public-only stochastic-vs-greedy PPO test.

Only the preregistered independent starts are evaluated. No learning, new
rewards, teacher labels, privileged actor inputs, or simulator probes.
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
from bagi.experiments import learned_competence_e4 as p1


def _restore_policy(task: str, seed: int, archived: dict, env: Any) -> Any:
    """Build upstream PPO and load ONLY verified, exact numeric source weights."""
    import torch
    from stable_baselines3 import PPO

    kwargs = {
        "seed": seed, "device": "cpu", "n_steps": p1.ROLLOUT_STEPS,
        "batch_size": 64, "n_epochs": 2, "gamma": 0.99,
        "learning_rate": 3e-4, "ent_coef": 0.01,
        "policy_kwargs": p1.POLICY_KWARGS, "verbose": 0,
    }
    model = PPO("MlpPolicy", env, **kwargs)
    expected = model.policy.state_dict()
    safe = p2.validated_state_dict(archived, expected)
    model.policy.load_state_dict(
        {key: torch.from_numpy(value) for key, value in safe.items()}, strict=True,
    )
    if not all(
        torch.equal(model.policy.state_dict()[key], torch.from_numpy(value))
        for key, value in safe.items()
    ):
        raise AssertionError("Restored PPO tensors differ from immutable P1 source")
    model.policy.set_training_mode(False)
    return model


def evaluate_episode(
    env: Any, task: str, reset_seed: int, arm: str, model_seed: int,
    task_index: int, model_index: int, agent: Any, ledger: dict,
) -> tuple[dict, dict]:
    """Execute exactly one native trajectory; evaluator-only hashes stay outside actor."""
    import torch

    p2.verify_registration()
    if task != p2.TASKS[task_index] or model_seed != p2.MODEL_SEEDS[model_index]:
        raise ValueError("Unregistered task or model-seed condition")
    if arm not in p2.ARMS or reset_seed not in p2.TEST_SEEDS:
        raise ValueError("Unregistered evaluation arm or heldout seed")
    if (agent is None) != (arm == "uniform_random"):
        raise ValueError("Incorrect frozen policy assignment to evaluation arm")
    arm_index = p2.ARMS.index(arm)
    rng_seed = p2.episode_rng_seed(task_index, model_index, arm_index, reset_seed)
    if arm.startswith("stochastic_"):
        torch.manual_seed(rng_seed)  # one independent stream per episode, not per step
    rng = np.random.default_rng(rng_seed) if arm == "uniform_random" else None
    env.action_space.seed(reset_seed)
    obs, _info = env.reset(seed=reset_seed)
    first_hash = p1._public_digest(obs)
    layout_hash = p1._layout_digest(env)  # evaluator ONLY, never policy input
    trajectory_hash = hashlib.sha256(first_hash)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    terminations: list[bool] = []
    truncations: list[bool] = []
    terminated = truncated = False
    for _ in range(p2.CAPS[task]):
        public = p1._public_array(obs)
        if rng is not None:
            action = int(rng.integers(p1.N_ACTIONS))
        else:
            prediction, _state = agent.predict(
                public, deterministic=(arm == "greedy_trained"),
            )
            action = int(np.asarray(prediction).item())
        if not env.action_space.contains(action):
            raise ValueError("Invalid native action from frozen actor")
        if ledger["test_actions"] >= p2.MAX_TEST_ACTIONS:
            raise AssertionError("Preregistered real TEST action ceiling exceeded")
        obs, reward, terminated, truncated, _info = env.step(action)
        ledger["test_actions"] += 1  # exactly one increment per ACTUAL native step
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError("Nonfinite native reward")
        observations.append(public.copy())
        actions.append(action)
        rewards.append(reward)
        terminations.append(bool(terminated))
        truncations.append(bool(truncated))
        trajectory_hash.update(p1._public_digest(obs))
        if terminated or truncated:
            break
    if not terminated and not truncated:
        raise AssertionError("Native episode exceeded frozen task cap without ending")
    reward_sum = float(sum(rewards))
    if reward_sum > 0 and not terminated:
        raise AssertionError("Unexpected positive reward without native goal termination")
    row = {
        "reset_seed": reset_seed,
        "action_sampling_seed": rng_seed,
        "evaluator_layout_sha256": layout_hash,
        "initial_public_sha256": first_hash.hex(),
        "public_trajectory_sha256": trajectory_hash.hexdigest(),
        "native_actions": len(actions),
        "native_reward": round(reward_sum, 8),
        "native_terminated": bool(terminated),
        "native_truncated": bool(truncated),
        "goal_success": int(terminated and reward_sum > 0),
        "action_histogram": np.bincount(actions, minlength=p1.N_ACTIONS).tolist(),
    }
    trace = {
        "public_pre_action": np.stack(observations),
        "actions": np.asarray(actions, dtype=np.int16),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "terminated": np.asarray(terminations, dtype=np.bool_),
        "truncated": np.asarray(truncations, dtype=np.bool_),
    }
    return row, trace


def score(task: str, rows: dict[str, list[dict]]) -> dict:
    """Strict paired native-goal gate, separately for each model seed."""
    p2.verify_registration()
    if task not in p2.TASKS or set(rows) != set(p2.ARMS):
        raise ValueError("Incomplete P2 evaluation conditions")
    if any(
        len(rows[arm]) != len(p2.TEST_SEEDS)
        or [x["reset_seed"] for x in rows[arm]] != list(p2.TEST_SEEDS)
        for arm in p2.ARMS
    ):
        raise ValueError("P2 TEST starts are missing, reordered or duplicated")
    for index in range(len(p2.TEST_SEEDS)):
        paired = [rows[arm][index] for arm in p2.ARMS]
        if len({x["evaluator_layout_sha256"] for x in paired}) != 1:
            raise ValueError("Matched P2 arms did not share physical reset layout")
        if len({x["initial_public_sha256"] for x in paired}) != 1:
            raise ValueError("Matched P2 arms did not share initial PUBLIC observation")
        for arm_index, (arm, x) in enumerate(zip(p2.ARMS, paired, strict=True)):
            if (
                x["native_actions"] not in range(1, p2.CAPS[task] + 1)
                or sum(x["action_histogram"]) != x["native_actions"]
                or x["goal_success"] != int(x["native_terminated"] and x["native_reward"] > 0)
                or x["action_sampling_seed"] != p2.episode_rng_seed(
                    p2.TASKS.index(task), x["model_index"], arm_index,
                    p2.TEST_SEEDS[index],
                )
            ):
                raise ValueError("P2 episode action, outcome or RNG accounting mismatch")
    diversity = len({x["evaluator_layout_sha256"] for x in rows[p2.ARMS[0]]})
    successes = {arm: sum(x["goal_success"] for x in rows[arm]) for arm in p2.ARMS}
    gain = min(
        successes["stochastic_trained"] - successes[control]
        for control in p2.ARMS if control != "stochastic_trained"
    )
    threshold, margin = (5, 3) if task == p2.TASKS[0] else (2, 2)
    return {
        "distinct_test_layouts": diversity,
        "valid_test_diversity": diversity >= 8,
        "native_successes": successes,
        "trained_stochastic_min_gain_over_all_controls": gain,
        "native_eval_actions": {
            arm: sum(x["native_actions"] for x in rows[arm]) for arm in p2.ARMS
        },
        "per_seed_gate": bool(
            diversity >= 8 and successes["stochastic_trained"] >= threshold
            and gain >= margin
        ),
    }


def run(source_archive: Path, output_dir: Path) -> dict:
    """One no-retraining native P2 trial after complete pinned-source verification."""
    import torch

    p1.verify_registration()
    p2.verify_registration()
    # Source ZIP and ALL 12 inner numeric archives verified before any native reset.
    frozen = p2.archive_weights(source_archive)
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = {"train_actions": 0, "test_actions": 0}
    summary: dict[str, Any] = {
        "study": "G2-E4-P2", "status": "running",
        "contribution": "frozen upstream PPO action-selection evaluation",
        "source_run_id": p2.SOURCE_RUN_ID,
        "source_artifact_id": p2.SOURCE_ARTIFACT_ID,
        "source_zip_sha256": hashlib.sha256(source_archive.read_bytes()).hexdigest(),
        "tasks": list(p2.TASKS), "model_seeds": list(p2.MODEL_SEEDS),
        "test_seeds": list(p2.TEST_SEEDS), "arms": list(p2.ARMS),
        "max_test_actions": p2.MAX_TEST_ACTIONS, "max_train_actions": 0,
        "native_action_ledger": ledger, "teacher_queries": 0,
        "versions": {name: version(name) for name in
                     ("minigrid", "gymnasium", "stable-baselines3")},
        "results": [], "promotion": False,
        "limitations": [
            "Only two related MiniGrid tasks; not cross-domain transfer.",
            "All policies frozen; this is not continual or online learning.",
            "Random and untrained controls had zero P1 training actions.",
            "Native rewards measure task success, not learned belief or planning.",
            "Independent P2 TEST starts are used once, with no tuning or repeats.",
        ],
    }
    try:
        for task_index, task in enumerate(p2.TASKS):
            for model_index, seed in enumerate(p2.MODEL_SEEDS):
                # PPO construction itself is not training and does not call learn().
                policy_env = p1._make_env(task)
                try:
                    trained = _restore_policy(
                        task, seed, frozen[(task, seed, "trained")], policy_env,
                    )
                    untrained = _restore_policy(
                        task, seed, frozen[(task, seed, "untrained")], policy_env,
                    )
                    agents = {
                        "greedy_trained": trained,
                        "stochastic_trained": trained,
                        "stochastic_untrained": untrained,
                        "uniform_random": None,
                    }
                    arms: dict[str, list[dict]] = {}
                    for arm in p2.ARMS:
                        episodes: list[dict] = []
                        traces: list[dict] = []
                        for reset_seed in p2.TEST_SEEDS:
                            env = p1._make_env(task)
                            try:
                                row, trace = evaluate_episode(
                                    env, task, reset_seed, arm, seed,
                                    task_index, model_index, agents[arm], ledger,
                                )
                            finally:
                                env.close()
                            row["model_index"] = model_index
                            episodes.append(row)
                            traces.append(trace)
                        path = output_dir / (
                            f"{task.replace('-', '_')}_seed{seed}_{arm}.npz"
                        )
                        digest = p1._save_traces(path, traces)  # numeric-only; no pickle
                        arms[arm] = episodes
                        summary.setdefault("trace_artifacts", []).append({
                            "task": task, "model_seed": seed, "arm": arm,
                            "file": path.name, "sha256": digest,
                        })
                    outcome = score(task, arms)
                    summary["results"].append({
                        "task": task, "model_seed": seed,
                        "evaluation": outcome, "episodes": arms,
                    })
                finally:
                    policy_env.close()
        if ledger["train_actions"] != 0 or ledger["test_actions"] > p2.MAX_TEST_ACTIONS:
            raise AssertionError("P2 real-action budget violation")
        if len(summary["results"]) != len(p2.TASKS) * len(p2.MODEL_SEEDS):
            raise AssertionError("Incomplete P2 model-seed conditions")
        summary["empty_navigation_gate"] = all(
            row["evaluation"]["per_seed_gate"] for row in summary["results"]
            if row["task"] == p2.TASKS[0]
        )
        summary["doorkey_causal_gate"] = all(
            row["evaluation"]["per_seed_gate"] for row in summary["results"]
            if row["task"] == p2.TASKS[1]
        )
        summary["promotion"] = bool(summary["empty_navigation_gate"])
        summary["status"] = "complete"
    except Exception as exc:
        summary["status"] = "failed"
        summary["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8",
        )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs/g2_e4_p2_frozen_policy_selection"),
    )
    args = parser.parse_args()
    result = run(args.source_archive, args.output_dir)
    print(
        f"G2-E4-P2 {result['status']} native={result['native_action_ledger']} "
        f"navigation_gate={result['empty_navigation_gate']} "
        f"causal_gate={result['doorkey_causal_gate']}"
    )
