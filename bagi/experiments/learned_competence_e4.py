"""G2-E4-P1: bounded, public-only, maintained PPO competence replication.

Train only on explicit disjoint reset seeds. Evaluate frozen PPO, identically
initialized untrained PPO, and uniform random on the SAME untouched starts.
Evaluator-only world hashes never enter actions, learning, or intrinsic reward.
No algorithm or task-specific reward is implemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

TASKS = ("MiniGrid-Empty-Random-6x6-v0", "MiniGrid-DoorKey-5x5-v0")
MODEL_SEEDS = (17, 29)
TEST_SEEDS = tuple(range(26000, 26010))
ARMS = ("random", "untrained", "trained")
TRAIN_STEPS_PER_MODEL = 16384
ROLLOUT_STEPS = 128
N_ACTIONS = 7
TRAIN_SEED_BASE = 1_000_000
TRAIN_SEED_TASK_STRIDE = 200_000
TRAIN_SEED_MODEL_STRIDE = 100_000
NATIVE_EPISODE_CAPS = {TASKS[0]: 144, TASKS[1]: 250}
MAX_TRAIN_ACTIONS = len(TASKS) * len(MODEL_SEEDS) * TRAIN_STEPS_PER_MODEL  # 65,536
# Tight exact upper bound: 2 seeds x 3 arms x 10 starts x (144 + 250).
MAX_TEST_ACTIONS = len(MODEL_SEEDS) * len(ARMS) * len(TEST_SEEDS) * sum(
    NATIVE_EPISODE_CAPS.values()
)  # 23,640
MAX_ALL_ACTIONS = MAX_TRAIN_ACTIONS + MAX_TEST_ACTIONS  # 89,176
MIN_DISTINCT_TEST_LAYOUTS = 8
POLICY_KWARGS = {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}


def verify_registration() -> None:
    """Fail closed if any frozen scientific decision or bound is changed."""
    if (
        TASKS != ("MiniGrid-Empty-Random-6x6-v0", "MiniGrid-DoorKey-5x5-v0")
        or MODEL_SEEDS != (17, 29)
        or TEST_SEEDS != tuple(range(26000, 26010))
        or ARMS != ("random", "untrained", "trained")
        or TRAIN_STEPS_PER_MODEL != 16384
        or ROLLOUT_STEPS != 128
        or N_ACTIONS != 7
        or TRAIN_SEED_BASE != 1_000_000
        or TRAIN_SEED_TASK_STRIDE != 200_000
        or TRAIN_SEED_MODEL_STRIDE != 100_000
        or NATIVE_EPISODE_CAPS != {TASKS[0]: 144, TASKS[1]: 250}
        or MAX_TRAIN_ACTIONS != 65536
        or MAX_TEST_ACTIONS != 23640
        or MAX_ALL_ACTIONS != 89176
        or MIN_DISTINCT_TEST_LAYOUTS != 8
        or POLICY_KWARGS != {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}
    ):
        raise ValueError("G2-E4-P1 preregistration or resource ceiling modified")


def _public_array(obs: Any) -> np.ndarray:
    arr = np.asarray(obs)
    if arr.ndim != 1 or arr.dtype.kind not in "biuf" or not np.isfinite(arr).all():
        raise ValueError("Policy must receive finite one-dimensional PUBLIC image only")
    return np.ascontiguousarray(arr)


def _public_digest(obs: Any) -> bytes:
    arr = _public_array(obs)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(json.dumps(arr.shape).encode("ascii"))
    h.update(arr.tobytes())
    return h.digest()


def _layout_digest(env: Any) -> str:
    """Evaluator ONLY. Never called in a policy or training-reward function."""
    world = env.unwrapped
    grid = np.ascontiguousarray(world.grid.encode())
    pos = tuple(int(x) for x in world.agent_pos)
    direction = int(world.agent_dir)
    if grid.ndim != 3 or len(pos) != 2 or direction not in range(4):
        raise ValueError("Unexpected evaluator world geometry")
    h = hashlib.sha256()
    h.update(json.dumps([grid.shape, str(grid.dtype), pos, direction]).encode())
    h.update(grid.tobytes())
    return h.hexdigest()


def _make_env(task: str) -> Any:
    import gymnasium as gym
    import minigrid  # noqa: F401 - registers upstream Farama environments
    from gymnasium.wrappers import FlattenObservation
    from minigrid.wrappers import ImgObsWrapper

    if task not in TASKS:
        raise ValueError("Unregistered native task")
    env = FlattenObservation(ImgObsWrapper(gym.make(task)))
    if int(env.action_space.n) != N_ACTIONS:
        env.close()
        raise ValueError("Unexpected upstream action space")
    return env


def _make_train_env(task: str, task_index: int, model_index: int, ledger: dict) -> Any:
    import gymnasium as gym

    base = (TRAIN_SEED_BASE + task_index * TRAIN_SEED_TASK_STRIDE
            + model_index * TRAIN_SEED_MODEL_STRIDE)

    class NativeTrainRecorder(gym.Wrapper):
        def __init__(self, env: Any) -> None:
            super().__init__(env)
            self.native_actions = 0
            self.native_resets = 0
            self.native_goal_episodes = 0
            self.native_completed_episodes = 0
            self.native_positive_reward_steps = 0
            self.reset_layouts: list[str] = []
            self.reset_seeds: list[int] = []

        def reset(self, **_kwargs: Any) -> tuple[Any, dict]:
            # SB3 may pass a model seed; deliberately use the registered,
            # separate TRAIN reset stream on EVERY episode instead.
            seed = base + self.native_resets
            if self.native_resets >= TRAIN_SEED_MODEL_STRIDE:
                raise AssertionError("TRAIN seed stream exhausted")
            obs, _info = self.env.reset(seed=seed)
            self.native_resets += 1
            self.reset_seeds.append(seed)
            self.reset_layouts.append(_layout_digest(self.env))
            return _public_array(obs), {}

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
            value = int(np.asarray(action).item())
            if not self.action_space.contains(value):
                raise ValueError("Invalid TRAIN action")
            if (self.native_actions >= TRAIN_STEPS_PER_MODEL
                    or ledger["train_actions"] >= MAX_TRAIN_ACTIONS):
                raise AssertionError("TRAIN native-action ceiling exceeded")
            obs, reward, terminated, truncated, _info = self.env.step(value)
            self.native_actions += 1
            ledger["train_actions"] += 1
            reward = float(reward)
            if not np.isfinite(reward):
                raise ValueError("Nonfinite TRAIN native reward")
            self.native_positive_reward_steps += int(reward > 0)
            if terminated or truncated:
                self.native_completed_episodes += 1
                self.native_goal_episodes += int(terminated and reward > 0)
            return _public_array(obs), reward, bool(terminated), bool(truncated), {}

    return NativeTrainRecorder(_make_env(task))


def evaluate_episode(
    env: Any, task: str, reset_seed: int, arm: str, model_seed: int,
    agent: Any, ledger: dict,
) -> tuple[dict, dict]:
    """One real native episode; record PRE-action public inputs for safe replay."""
    if task not in TASKS or reset_seed not in TEST_SEEDS or arm not in ARMS:
        raise ValueError("Unregistered frozen evaluation")
    env.action_space.seed(reset_seed)
    obs, _info = env.reset(seed=reset_seed)
    first_hash = _public_digest(obs)
    layout_hash = _layout_digest(env)  # evaluator-only, not given to actor
    digest = hashlib.sha256(first_hash)
    rng = np.random.default_rng(50_000 + reset_seed * 100 + model_seed)
    obs_before: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    terminations: list[bool] = []
    truncations: list[bool] = []
    terminated = truncated = False
    for _ in range(NATIVE_EPISODE_CAPS[task]):
        public = _public_array(obs)
        if arm == "random":
            action = int(rng.integers(N_ACTIONS))
        else:
            prediction, _state = agent.predict(public, deterministic=True)
            action = int(np.asarray(prediction).item())
        if not env.action_space.contains(action):
            raise ValueError("Actor produced an invalid discrete action")
        if ledger["test_actions"] >= MAX_TEST_ACTIONS:
            raise AssertionError("TEST native-action ceiling exceeded")
        obs, reward, terminated, truncated, _info = env.step(action)
        ledger["test_actions"] += 1
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError("Nonfinite TEST native reward")
        obs_before.append(public.copy())
        actions.append(action)
        rewards.append(reward)
        terminations.append(bool(terminated))
        truncations.append(bool(truncated))
        digest.update(_public_digest(obs))
        if terminated or truncated:
            break
    if not terminated and not truncated:
        raise AssertionError("Upstream native episode did not finish within frozen cap")
    reward_sum = float(sum(rewards))
    if reward_sum > 0 and not terminated:
        raise AssertionError("Unexpected native positive reward without goal termination")
    row = {
        "reset_seed": reset_seed,
        "evaluator_layout_sha256": layout_hash,
        "initial_public_sha256": first_hash.hex(),
        "public_trajectory_sha256": digest.hexdigest(),
        "native_actions": len(actions),
        "native_reward": round(reward_sum, 8),
        "native_terminated": bool(terminated),
        "native_truncated": bool(truncated),
        "goal_success": int(terminated and reward_sum > 0),
        "action_histogram": np.bincount(actions, minlength=N_ACTIONS).tolist(),
    }
    trace = {
        "public_pre_action": np.stack(obs_before),
        "actions": np.asarray(actions, dtype=np.int16),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "terminated": np.asarray(terminations, dtype=np.bool_),
        "truncated": np.asarray(truncations, dtype=np.bool_),
    }
    return row, trace


def _save_traces(path: Path, traces: list[dict], agent: Any = None) -> str:
    """Only numeric NumPy arrays; never torch.save or pickle/checkpoint load."""
    offsets = np.asarray([0] + list(np.cumsum([len(x["actions"]) for x in traces])),
                         dtype=np.int64)
    arrays = {"episode_offsets": offsets}
    for field in ("public_pre_action", "actions", "rewards", "terminated", "truncated"):
        arrays[field] = np.concatenate([trace[field] for trace in traces], axis=0)
    if agent is not None:
        for key, tensor in agent.policy.state_dict().items():
            arrays["policy__" + key.replace(".", "__")] = tensor.detach().cpu().numpy()
    if any(np.asarray(x).dtype.kind not in "biuf" for x in arrays.values()):
        raise ValueError("Refusing unsafe/non-numeric evaluation artifact")
    np.savez_compressed(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(rows: dict[str, list[dict]]) -> dict:
    """Frozen per-seed promotion; no averaging away a failed model seed."""
    if set(rows) != set(ARMS) or any(len(rows[arm]) != len(TEST_SEEDS) for arm in ARMS):
        raise ValueError("Incomplete or mismatched heldout arms")
    if any([x["reset_seed"] for x in rows[arm]] != list(TEST_SEEDS) for arm in ARMS):
        raise ValueError("Different evaluation seed sets")
    for index in range(len(TEST_SEEDS)):
        if len({rows[arm][index]["evaluator_layout_sha256"] for arm in ARMS}) != 1:
            raise ValueError("Paired arms did not share identical physical reset layout")
    diversity = len({x["evaluator_layout_sha256"] for x in rows["trained"]})
    successes = {arm: sum(x["goal_success"] for x in rows[arm]) for arm in ARMS}
    trained_gain = min(successes["trained"] - successes["random"],
                       successes["trained"] - successes["untrained"])
    return {
        "distinct_test_layouts": diversity,
        "valid_test_diversity": diversity >= MIN_DISTINCT_TEST_LAYOUTS,
        "native_successes": successes,
        "trained_min_gain_over_both_controls": trained_gain,
        "native_eval_actions": {arm: sum(x["native_actions"] for x in rows[arm])
                                for arm in ARMS},
    }


def run(output_dir: Path) -> dict:
    import torch
    from stable_baselines3 import PPO

    verify_registration()
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = {"train_actions": 0, "test_actions": 0}
    summary: dict[str, Any] = {
        "study": "G2-E4-P1", "status": "running", "contribution": "upstream PPO replication",
        "tasks": list(TASKS), "model_seeds": list(MODEL_SEEDS),
        "test_seeds": list(TEST_SEEDS), "arms": list(ARMS),
        "train_steps_per_model": TRAIN_STEPS_PER_MODEL,
        "max_train_actions": MAX_TRAIN_ACTIONS, "max_test_actions": MAX_TEST_ACTIONS,
        "max_all_actions": MAX_ALL_ACTIONS,
        "native_action_ledger": ledger, "teacher_queries": 0,
        "versions": {name: version(name) for name in
                     ("minigrid", "gymnasium", "stable-baselines3")},
        "results": [], "promotion": False,
        "limitations": [
            "Only two related MiniGrid task families, not cross-domain transfer.",
            "Untrained/random controls use zero TRAIN actions; matched TEST starts and caps.",
            "Only public local images reach policies; evaluator-only full-grid hashes do not.",
            "A sparse-reward null at 16,384 train actions is not a PPO impossibility proof.",
            "Test starts are frozen once; no heldout-driven retuning or resampling.",
        ],
    }
    try:
        for task_index, task in enumerate(TASKS):
            for model_index, seed in enumerate(MODEL_SEEDS):
                train_env = _make_train_env(task, task_index, model_index, ledger)
                try:
                    kwargs = {"seed": seed, "device": "cpu", "n_steps": ROLLOUT_STEPS,
                              "batch_size": 64, "n_epochs": 2, "gamma": 0.99,
                              "learning_rate": 3e-4, "ent_coef": 0.01,
                              "policy_kwargs": POLICY_KWARGS, "verbose": 0}
                    model = PPO("MlpPolicy", train_env, **kwargs)
                    initial = {key: value.detach().clone()
                               for key, value in model.policy.state_dict().items()}
                    started = time.monotonic()
                    model.learn(total_timesteps=TRAIN_STEPS_PER_MODEL)
                    elapsed = round(time.monotonic() - started, 3)
                    if (int(model.num_timesteps) != TRAIN_STEPS_PER_MODEL
                            or train_env.native_actions != TRAIN_STEPS_PER_MODEL):
                        raise AssertionError("Native TRAIN budget not exactly exhausted")
                    training = {
                        "real_actions": train_env.native_actions,
                        "native_resets": train_env.native_resets,
                        "unique_train_reset_layouts": len(set(train_env.reset_layouts)),
                        "train_reset_seeds": train_env.reset_seeds,
                        "completed_episodes": train_env.native_completed_episodes,
                        "native_goal_episodes": train_env.native_goal_episodes,
                        "positive_native_reward_steps": train_env.native_positive_reward_steps,
                        "train_wall_seconds": elapsed,
                    }
                    # Reconstruct the exact initialized network AFTER training,
                    # without exposing heldout outcomes to the optimizer.
                    untrained_env = _make_env(task)
                    try:
                        untrained = PPO("MlpPolicy", untrained_env, **kwargs)
                        if not all(torch.equal(value, untrained.policy.state_dict()[key])
                                   for key, value in initial.items()):
                            raise AssertionError("Untrained control does not match initial weights")
                        arms: dict[str, list[dict]] = {}
                        for arm in ARMS:
                            agent = {"random": None, "untrained": untrained,
                                     "trained": model}[arm]
                            episodes = []
                            traces = []
                            for reset_seed in TEST_SEEDS:
                                env = _make_env(task)
                                try:
                                    row, trace = evaluate_episode(
                                        env, task, reset_seed, arm, seed, agent, ledger
                                    )
                                finally:
                                    env.close()
                                episodes.append(row)
                                traces.append(trace)
                            artifact_path = output_dir / (
                                f"{task.replace('-', '_')}_seed{seed}_{arm}.npz"
                            )
                            digest = _save_traces(artifact_path, traces, agent)
                            arms[arm] = episodes
                            summary.setdefault("trace_artifacts", []).append({
                                "task": task, "model_seed": seed, "arm": arm,
                                "file": artifact_path.name, "sha256": digest,
                            })
                        outcome = score(arms)
                        summary["results"].append({
                            "task": task, "model_seed": seed, "training": training,
                            "evaluation": outcome, "episodes": arms,
                        })
                    finally:
                        untrained_env.close()
                finally:
                    train_env.close()
        if ledger["train_actions"] != MAX_TRAIN_ACTIONS:
            raise AssertionError("Incomplete registered training action budget")
        if ledger["train_actions"] + ledger["test_actions"] > MAX_ALL_ACTIONS:
            raise AssertionError("Combined real native action budget exceeded")
        if len(summary["results"]) != len(TASKS) * len(MODEL_SEEDS):
            raise AssertionError("Incomplete preregistered task/seed conditions")
        # Both model seeds must pass; each trained policy must beat both controls.
        empty = [x for x in summary["results"] if x["task"] == TASKS[0]]
        door = [x for x in summary["results"] if x["task"] == TASKS[1]]
        def eligible(rows: list[dict], successes: int, gain: int) -> bool:
            return all(
                row["evaluation"]["valid_test_diversity"]
                and row["training"]["native_goal_episodes"] > 0
                and row["evaluation"]["native_successes"]["trained"] >= successes
                and row["evaluation"]["trained_min_gain_over_both_controls"] >= gain
                for row in rows
            )
        summary["empty_navigation_gate"] = eligible(empty, successes=5, gain=3)
        summary["doorkey_causal_gate"] = eligible(door, successes=2, gain=2)
        summary["promotion"] = bool(summary["empty_navigation_gate"])
        summary["status"] = "complete"
    except Exception as exc:
        summary["status"] = "failed"
        summary["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/g2_e4_p1_learned_competence"))
    result = run(parser.parse_args().output_dir)
    print(f"G2-E4-P1 {result['status']} native={result['native_action_ledger']} "
          f"navigation_gate={result['empty_navigation_gate']} "
          f"causal_gate={result['doorkey_causal_gate']}")
