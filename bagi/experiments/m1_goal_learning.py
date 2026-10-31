"""M1: first bounded learned goal-completion and transfer experiment.

Uses upstream PPO / RecurrentPPO and Farama MiniGrid. NOT executed by CI.
The existing metered BAGI research loop must explicitly admit this command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import deque
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
import yaml
from gymnasium.spaces import Box
from gymnasium.wrappers import FlattenObservation, FrameStackObservation
from minigrid.wrappers import ImgObsWrapper

TRAIN_TASK = "MiniGrid-MemoryS7-v0"
TRANSFER_TASK = "MiniGrid-MemoryS13Random-v0"
ARMS = ("reactive", "action4", "frame4", "recurrent")
TRAIN_STEPS = 8192
MAX_TRAIN_TOTAL = 65536
MAX_EVAL_TOTAL = 17280
MAX_EVAL_EPISODE_STEPS = 120
FROZEN_EVAL_SEEDS = (92001, 92002, 92003, 92004, 92005, 92006)
FROZEN_TRANSFER_SEEDS = (93001, 93002, 93003, 93004, 93005, 93006)


class PublicExecutedActions(gym.Wrapper):
    """Local public image and one-hot ACTUALLY executed actions only."""

    def __init__(self, env, history_length: int):
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("Requires a discrete action space")
        if len(env.observation_space.shape) != 1 or history_length not in (1, 4):
            raise ValueError("Requires flat image and frozen 1/4 history length")
        self.n_actions = int(env.action_space.n)
        self.n_history = history_length
        self.previous = deque([self.n_actions] * history_length, maxlen=history_length)
        width = int(env.observation_space.shape[0]) + history_length * (self.n_actions + 1)
        self.observation_space = Box(0, 255, shape=(width,), dtype=np.float32)

    def _encode(self, observation):
        image = np.asarray(observation, dtype=np.float32)
        if image.shape != self.env.observation_space.shape:
            raise ValueError("Unexpected local public observation")
        one_hot = np.zeros((self.n_history, self.n_actions + 1), dtype=np.float32)
        for i, action in enumerate(self.previous):
            one_hot[i, action] = 1
        return np.concatenate((image, one_hot.ravel()))

    def reset(self, *, seed=None, options=None):
        self.previous = deque([self.n_actions] * self.n_history, maxlen=self.n_history)
        observation, info = self.env.reset(seed=seed, options=options)
        return self._encode(observation), info

    def step(self, action):
        actual = int(action)
        if not 0 <= actual < self.n_actions:
            raise ValueError("Invalid executed action")
        observation, reward, terminated, truncated, info = self.env.step(actual)
        self.previous.append(actual)
        return self._encode(observation), reward, terminated, truncated, info


class CountNativeSteps(gym.Wrapper):
    """Records real simulator returns and resets, not model rollouts."""

    def __init__(self, env):
        super().__init__(env)
        self.native_steps = 0
        self.native_resets = 0

    def reset(self, **kwargs):
        self.native_resets += 1
        return self.env.reset(**kwargs)

    def step(self, action):
        result = self.env.step(action)
        self.native_steps += 1
        return result


def make_task(task: str, arm: str):
    if task not in (TRAIN_TASK, TRANSFER_TASK) or arm not in ARMS:
        raise ValueError("Unregistered task or model arm")
    # Local image only. Mission and hidden/evaluator information are excluded.
    env = gym.make(task)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EVAL_EPISODE_STEPS)
    env = FlattenObservation(ImgObsWrapper(env))
    if arm == "action4":
        env = PublicExecutedActions(env, 4)
    if arm == "frame4":
        env = FlattenObservation(FrameStackObservation(env, stack_size=4))
    return CountNativeSteps(env)


def validate_config(c):
    if (c.get("train_task"), c.get("transfer_task")) != (TRAIN_TASK, TRANSFER_TASK):
        raise ValueError("Task or transfer contract changed")
    if c.get("arms") != list(ARMS) or c.get("training_seeds") != [17, 29]:
        raise ValueError("Model comparison and seeds must remain frozen")
    for name, expected in (("train_steps_per_arm_seed", TRAIN_STEPS),
                           ("rollout_steps", 128), ("batch_size", 64),
                           ("epochs_per_rollout", 2),
                           ("max_eval_steps", MAX_EVAL_EPISODE_STEPS),
                           ("max_total_training_steps", MAX_TRAIN_TOTAL),
                           ("max_total_evaluation_steps", MAX_EVAL_TOTAL)):
        if type(c.get(name)) is not int or c[name] != expected:
            raise ValueError("Registered research budget changed: " + name)
    train = set(c["training_seeds"])
    first, transfer = c.get("evaluation_seeds"), c.get("transfer_seeds")
    if first != list(FROZEN_EVAL_SEEDS) or transfer != list(FROZEN_TRANSFER_SEEDS):
        raise ValueError("Frozen evaluation or transfer seed set changed")
    if any(not isinstance(s, list) or len(s) != 6 or
           any(type(n) is not int for n in s) or len(set(s)) != 6
           for s in (first, transfer)):
        raise ValueError("Require six distinct fixed evaluation seeds per split")
    if train & (set(first) | set(transfer)) or set(first) & set(transfer):
        raise ValueError("Training/validation/transfer seed overlap")
    if len(ARMS) * len(train) * TRAIN_STEPS != MAX_TRAIN_TOTAL:
        raise ValueError("Training arithmetic changed")
    if len(ARMS) * len(train) * 3 * len(first) * MAX_EVAL_EPISODE_STEPS != MAX_EVAL_TOTAL:
        raise ValueError("Evaluation arithmetic changed")
    if c.get("output_dir") != "runs/m1_goal_learning":
        raise ValueError("Results must remain inside the registered runs path")
    return c


def count_optimizer_steps(agent):
    """Instrument optimizer.step calls instead of approximating epochs."""
    count = {"calls": 0}
    original = agent.policy.optimizer.step

    def counted(*args, **kwargs):
        count["calls"] += 1
        return original(*args, **kwargs)

    agent.policy.optimizer.step = counted
    return count


def numeric_state(agent, path: Path):
    """Pure numeric state arrays; no unsafe pickle or SB3 archive deserialization."""
    if path.exists():
        raise FileExistsError("Refusing to overwrite a learned policy")
    arrays, entries = {}, {}
    for i, (name, tensor) in enumerate(agent.policy.state_dict().items()):
        a = tensor.detach().cpu().numpy()
        if a.dtype.kind not in "fiub":
            raise ValueError("Non-numeric policy state is forbidden")
        key = f"tensor_{i:03d}"
        arrays[key] = a
        entries[key] = {"name": name, "shape": list(a.shape), "dtype": str(a.dtype)}
    if not arrays:
        raise ValueError("Empty policy state")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "members": entries, "allow_pickle": False}


def evaluate_goal_agent(agent, task, evaluation_seeds, env_factory):
    """Native goal success requires positive reward AND actual termination."""
    results = []
    for seed in evaluation_seeds:
        env = env_factory(task)
        try:
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            lstm_states = None
            episode_start = np.array([True], dtype=bool)
            total_reward = 0.0
            terminated = truncated = False
            steps = 0
            for _ in range(MAX_EVAL_EPISODE_STEPS):
                action, lstm_states = agent.predict(
                    observation, state=lstm_states,
                    episode_start=episode_start, deterministic=True,
                )
                observation, reward, terminated, truncated, _ = env.step(
                    int(np.asarray(action).item())
                )
                total_reward += float(reward)
                steps += 1
                if terminated or truncated:
                    break
                episode_start[:] = False
            if not (terminated or truncated):
                raise ValueError("Evaluation did not terminate within the frozen cap")
            if env.native_steps != steps or env.native_resets != 1:
                raise ValueError("Native evaluation action/reset count disagreement")
            results.append({
                "seed": seed, "reward": total_reward,
                "goal_success": int(bool(terminated) and total_reward > 0.0),
                "terminated": bool(terminated), "truncated": bool(truncated),
                "evaluation_real_steps": steps,
                "evaluation_real_resets": env.native_resets,
            })
        finally:
            env.close()
    return {
        "episodes": len(results),
        "successes": sum(row["goal_success"] for row in results),
        "mean_reward": sum(row["reward"] for row in results) / len(results),
        "evaluation_real_steps": sum(row["evaluation_real_steps"] for row in results),
        "evaluation_real_resets": sum(row["evaluation_real_resets"] for row in results),
        "episode_results": results,
    }


def score_goal_milestone(records):
    """Post-hoc arithmetic on frozen goal metrics; no training/selection."""
    arms = {}
    for arm in ARMS:
        rows = [row for row in records if row["arm"] == arm]
        if len(rows) != 2 or {row["model_seed"] for row in rows} != {17, 29}:
            raise ValueError("Expected two independent model seeds for each arm")
        before = sum(row["before"]["successes"] for row in rows)
        after = sum(row["after"]["successes"] for row in rows)
        transfer = sum(row["transfer"]["successes"] for row in rows)
        arms[arm] = {
            "before_goals_of_12": before,
            "after_goals_of_12": after,
            "transfer_goals_of_12": transfer,
            "tentative_learning_pass": (
                after >= 6 and after - before >= 4
                and all(row["after"]["successes"] >= 1 for row in rows)
            ),
        }
    recurrent = arms["recurrent"]
    strongest_control = max(arms[arm]["after_goals_of_12"] for arm in ARMS[:-1])
    strongest_transfer = max(
        arms[arm]["transfer_goals_of_12"] for arm in ARMS[:-1]
    )
    return {
        "per_arm": arms,
        "recurrent_post_minus_best_control": (
            recurrent["after_goals_of_12"] - strongest_control
        ),
        "recurrent_transfer_minus_best_control": (
            recurrent["transfer_goals_of_12"] - strongest_transfer
        ),
        "tentative_memory_advantage": bool(
            recurrent["tentative_learning_pass"]
            and recurrent["after_goals_of_12"] - strongest_control >= 3
        ),
        "tentative_size_transfer": bool(
            recurrent["tentative_learning_pass"]
            and recurrent["transfer_goals_of_12"] >= 3
            and recurrent["transfer_goals_of_12"] - strongest_transfer >= 2
        ),
        "note": "Small N=12 pilot gates; independent replication required.",
    }


def run(config: Path, output: Path):
    if output.exists() or (output.parent / "progress.json").exists():
        raise FileExistsError("Refusing to overwrite previous M1 evidence")
    raw_config = config.read_bytes()
    c = validate_config(yaml.safe_load(raw_config))
    if output.parent.as_posix() != c["output_dir"]:
        raise ValueError("Unregistered M1 output path")
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    for arm in ARMS:
        a = make_task(TRAIN_TASK, arm)
        b = make_task(TRANSFER_TASK, arm)
        try:
            if a.observation_space.shape != b.observation_space.shape:
                raise ValueError("Train-to-transfer public observation mismatch")
            if a.action_space != b.action_space:
                raise ValueError("Train-to-transfer action contract mismatch")
        finally:
            a.close()
            b.close()

    records, training, evaluation, optimizer = [], 0, 0, 0
    started = time.monotonic()
    for seed in c["training_seeds"]:
        for arm in ARMS:
            env = make_task(TRAIN_TASK, arm)
            try:
                recurrent = arm == "recurrent"
                kwargs = {"net_arch": {"pi": [32], "vf": [32]}}
                if recurrent:
                    kwargs["lstm_hidden_size"] = 32
                agent = (RecurrentPPO if recurrent else PPO)(
                    "MlpLstmPolicy" if recurrent else "MlpPolicy", env,
                    seed=seed, device="cpu", n_steps=c["rollout_steps"],
                    batch_size=c["batch_size"], n_epochs=c["epochs_per_rollout"],
                    policy_kwargs=kwargs, verbose=0,
                )
                count = count_optimizer_steps(agent)
                factory = lambda task, arm=arm: make_task(task, arm)
                before = evaluate_goal_agent(agent, TRAIN_TASK, c["evaluation_seeds"], factory)
                agent.learn(total_timesteps=TRAIN_STEPS)
                if agent.num_timesteps != TRAIN_STEPS or env.native_steps != TRAIN_STEPS:
                    raise ValueError("Native training action budget mismatch")
                after = evaluate_goal_agent(agent, TRAIN_TASK, c["evaluation_seeds"], factory)
                transfer = evaluate_goal_agent(agent, TRANSFER_TASK, c["transfer_seeds"], factory)
                total_eval = sum(x["evaluation_real_steps"] for x in
                                 (before, after, transfer))
                state = numeric_state(agent, output.parent / f"{arm}-seed{seed}.npz")
                records.append({
                    "arm": arm, "model_seed": seed,
                    "parameters": sum(p.numel() for p in agent.policy.parameters()),
                    "training_real_actions": env.native_steps,
                    "training_resets": env.native_resets,
                    "optimizer_step_calls": count["calls"],
                    "evaluation_real_actions": total_eval,
                    "before": before, "after": after, "transfer": transfer,
                    "numeric_policy": state,
                })
                training += env.native_steps
                evaluation += total_eval
                optimizer += count["calls"]
                if training > MAX_TRAIN_TOTAL or evaluation > MAX_EVAL_TOTAL:
                    raise ValueError("Aggregate M1 native-action budget exceeded")
                progress = {
                    "status": "partial_not_a_completed_study",
                    "source_commit": os.environ.get("GITHUB_SHA", "unavailable"),
                    "completed_models": len(records), "training_real_actions": training,
                    "evaluation_real_actions": evaluation,
                    "optimizer_step_calls": optimizer, "records": records,
                }
                (output.parent / "progress.json").write_text(
                    json.dumps(progress, indent=2) + "\n", encoding="utf-8"
                )
                print(f"{arm} seed={seed}: {before['successes']}/6 -> "
                      f"{after['successes']}/6; transfer={transfer['successes']}/6")
            finally:
                env.close()
    if training != MAX_TRAIN_TOTAL:
        raise ValueError("Training under-ran frozen total")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "study": "BAGI-M1 goal learning using established PPO and recurrent PPO",
        "status": "exploratory_bounded_training",
        "source_commit": os.environ.get("GITHUB_SHA", "unavailable"),
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "versions": {name: version(name) for name in
                     ("gymnasium", "minigrid", "stable-baselines3", "sb3-contrib")},
        "records": records, "milestone_pilot_gates": score_goal_milestone(records),
        "training_real_actions": training,
        "evaluation_real_actions": evaluation,
        "optimizer_step_calls": optimizer, "teacher_queries": 0,
        "elapsed_seconds": round(time.monotonic()-started, 3),
        "limitations": [
            "This is public-image goal learning, not general intelligence.",
            "Training uses only MemoryS7; MemoryS13Random is a separate zero-shot test.",
            "Parameter counts differ; native interaction budgets match.",
            "No evaluation feedback is used for model selection or tuning.",
            "No Workshop P0B diagnostic/heldout/size13 data are accessed.",
        ],
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/m1_goal_learning.yaml"))
    p.add_argument("--output", type=Path,
                   default=Path("runs/m1_goal_learning/summary.json"))
    p.add_argument("--execute-budgeted-research", action="store_true")
    a = p.parse_args()
    if (not a.execute_budgeted_research or
            os.environ.get("GITHUB_WORKFLOW") != "BAGI Research Loop" or
            os.environ.get("GITHUB_JOB") != "research" or
            os.environ.get("BAGI_M1_ADMITTED") != "1"):
        raise SystemExit("M1 requires the existing audited, quota-admitted research job")
    run(a.config, a.output)


if __name__ == "__main__":
    main()
