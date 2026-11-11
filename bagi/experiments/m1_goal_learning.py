"""M1: first bounded learned goal-completion and transfer experiment.

Uses upstream PPO / RecurrentPPO and Farama MiniGrid. NOT executed by CI.
Run explicitly from the CLI or the existing source-verified research loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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
M2_EVAL_SEEDS = (94001, 94002, 94003, 94004, 94005, 94006)
M2_TRANSFER_SEEDS = (95001, 95002, 95003, 95004, 95005, 95006)
M2_TRAIN_STEPS = 65536
M3_TRAIN_SEEDS = (37, 53)
M3_EVAL_SEEDS = (96001, 96002, 96003, 96004, 96005, 96006)
M3_TRANSFER_SEEDS = (97001, 97002, 97003, 97004, 97005, 97006)


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

    def __init__(self, env, *, max_native_steps: int | None = None):
        super().__init__(env)
        if max_native_steps is not None and (
                type(max_native_steps) is not int or max_native_steps < 1):
            raise ValueError("Native action limit must be a positive integer")
        self.max_native_steps = max_native_steps
        self.native_steps = 0
        self.native_resets = 0
        self.native_public_reward_sum = 0.0
        self.native_positive_reward_steps = 0
        self.native_rewarded_goal_terminations = 0

    def reset(self, **kwargs):
        self.native_resets += 1
        return self.env.reset(**kwargs)

    def step(self, action):
        if (self.max_native_steps is not None
                and self.native_steps >= self.max_native_steps):
            raise RuntimeError("Native action limit reached before env.step")
        result = self.env.step(action)
        self.native_steps += 1
        _, reward, terminated, _, _ = result
        public_reward = float(reward)
        self.native_public_reward_sum += public_reward
        if public_reward > 0:
            self.native_positive_reward_steps += 1
            if terminated:
                self.native_rewarded_goal_terminations += 1
        return result


def make_task(task: str, arm: str, *, max_native_steps: int | None = None):
    if task not in (TRAIN_TASK, TRANSFER_TASK) or arm not in ARMS:
        raise ValueError("Unregistered task or model arm")
    # Local image only. Mission and hidden/evaluator information are excluded.
    env = gym.make(task)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EVAL_EPISODE_STEPS)
    env = FlattenObservation(ImgObsWrapper(env))
    # Give reactive and recurrent the SAME public observation/action inputs.
    # The recurrent state then tests learned memory beyond the last action.
    if arm in ("reactive", "recurrent"):
        env = PublicExecutedActions(env, 1)
    elif arm == "action4":
        env = PublicExecutedActions(env, 4)
    elif arm == "frame4":
        env = FlattenObservation(FrameStackObservation(env, stack_size=4))
    return CountNativeSteps(env, max_native_steps=max_native_steps)


def validate_config(c):
    if (c.get("train_task"), c.get("transfer_task")) != (TRAIN_TASK, TRANSFER_TASK):
        raise ValueError("Task or transfer contract changed")
    if c.get("arms") != list(ARMS):
        raise ValueError("Model comparison and arms must remain frozen")
    for name in ("train_steps_per_arm_seed", "rollout_steps", "batch_size",
                 "epochs_per_rollout"):
        if type(c.get(name)) is not int or c[name] < 1:
            raise ValueError("Require a positive training setting: " + name)
    if c["rollout_steps"] < 2 or c["batch_size"] < 2:
        raise ValueError("PPO requires at least two rollout/batch samples")
    if c.get("max_eval_steps") != MAX_EVAL_EPISODE_STEPS:
        raise ValueError("Evaluation episode definition changed")
    study = c.get("registered_study", "m1")
    if study == "m1":
        expected_eval, expected_transfer = FROZEN_EVAL_SEEDS, FROZEN_TRANSFER_SEEDS
        expected_train = (17, 29)
    elif study in ("m2", "m3"):
        expected_eval, expected_transfer = (
            (M2_EVAL_SEEDS, M2_TRANSFER_SEEDS) if study == "m2"
            else (M3_EVAL_SEEDS, M3_TRANSFER_SEEDS)
        )
        expected_train = (17, 29) if study == "m2" else M3_TRAIN_SEEDS
        if (c["train_steps_per_arm_seed"] != M2_TRAIN_STEPS
                or c["rollout_steps"] != 128 or c["batch_size"] != 64
                or c["epochs_per_rollout"] != 2
                or c.get("output_dir") != f"runs/{study}_goal_learning"):
            raise ValueError("Extended-study training and output contract changed")
    else:
        raise ValueError("Unknown registered research study")
    if c.get("training_seeds") != list(expected_train):
        raise ValueError("Model training seeds differ from registered study")
    train = set(c["training_seeds"])
    first, transfer = c.get("evaluation_seeds"), c.get("transfer_seeds")
    if first != list(expected_eval) or transfer != list(expected_transfer):
        raise ValueError("Frozen evaluation or transfer seed set changed")
    if any(not isinstance(s, list) or len(s) != 6 or
           any(type(n) is not int for n in s) or len(set(s)) != 6
           for s in (first, transfer)):
        raise ValueError("Require six distinct fixed evaluation seeds per split")
    if train & (set(first) | set(transfer)) or set(first) & set(transfer):
        raise ValueError("Training/validation/transfer seed overlap")
    output_dir = Path(c.get("output_dir", ""))
    if (output_dir.is_absolute() or ".." in output_dir.parts
            or len(output_dir.parts) < 2 or output_dir.parts[0] != "runs"):
        raise ValueError("Use an isolated output directory under runs")
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


def interrupted_model_measurement(env, evaluation_envs, optimizer_counter):
    """Account for observed, unfinished native work without calling it completed."""
    return {
        "status": "interrupted_not_completed",
        "training_real_actions": env.native_steps,
        "training_resets": env.native_resets,
        "training_positive_reward_steps": env.native_positive_reward_steps,
        "training_rewarded_goal_terminations": env.native_rewarded_goal_terminations,
        "training_public_reward_sum": env.native_public_reward_sum,
        "evaluation_real_actions": sum(item.native_steps for item in evaluation_envs),
        "evaluation_resets": sum(item.native_resets for item in evaluation_envs),
        "optimizer_step_calls": optimizer_counter["calls"],
    }


def save_progress(path: Path, payload: dict):
    """Atomically replace the same run's progress, never an earlier study run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def admitted_source_commit():
    """Bind results to checked-out source; verify an expected SHA when provided."""
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, timeout=5
    ).strip()
    expected = os.environ.get("BAGI_RESEARCH_SOURCE_SHA")
    if expected and actual != expected:
        raise ValueError("M1 source commit does not match admitted research HEAD")
    return actual


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


def evaluate_goal_agent(agent, task, evaluation_seeds, env_factory, *, deterministic=True):
    """Native goal success requires positive reward AND actual termination."""
    results = []
    for seed in evaluation_seeds:
        env = env_factory(task)
        try:
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            if not deterministic:
                torch.manual_seed(seed)
            lstm_states = None
            episode_start = np.array([True], dtype=bool)
            total_reward = 0.0
            terminated = truncated = False
            steps = 0
            for _ in range(MAX_EVAL_EPISODE_STEPS):
                action, lstm_states = agent.predict(
                    observation, state=lstm_states,
                    episode_start=episode_start, deterministic=deterministic,
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
                "deterministic": bool(deterministic),
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


def score_goal_milestone(records, training_seeds=(17, 29)):
    """Post-hoc arithmetic on frozen goal metrics; no training/selection."""
    arms = {}
    for arm in ARMS:
        rows = [row for row in records if row["arm"] == arm]
        if len(rows) != 2 or {row["model_seed"] for row in rows} != set(training_seeds):
            raise ValueError("Expected two independent registered model seeds for each arm")
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
    if (output.exists() or (output.parent / "progress.json").exists()
            or (output.parent / "progress.json.pending").exists()):
        raise FileExistsError("Refusing to overwrite previous M1 evidence")
    raw_config = config.read_bytes()
    c = validate_config(yaml.safe_load(raw_config))
    study = c.get("registered_study", "m1")
    is_m2 = study == "m2"
    is_m3 = study == "m3"
    extended = is_m2 or is_m3
    config_digest = hashlib.sha256(raw_config).hexdigest()
    source_commit = admitted_source_commit()
    requested_steps = c["train_steps_per_arm_seed"]
    # Upstream PPO finishes whole rollouts. Record its expected actual work
    # instead of imposing the old fixed 8,192-action training ceiling.
    rollout_steps = c["rollout_steps"]
    actual_steps_per_model = ((requested_steps + rollout_steps - 1)
                              // rollout_steps) * rollout_steps
    expected_training_total = len(ARMS) * len(c["training_seeds"]) * actual_steps_per_model
    evaluation_conditions = 5 if extended else 3
    expected_evaluation_total = (
        len(ARMS) * len(c["training_seeds"]) * evaluation_conditions
        * len(c["evaluation_seeds"]) * MAX_EVAL_EPISODE_STEPS
    )
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
    training_positive_rewards = 0
    training_rewarded_goals = 0
    training_public_reward_sum = 0.0
    started = time.monotonic()
    for seed in c["training_seeds"]:
        for arm in ARMS:
            env = make_task(TRAIN_TASK, arm,
                            max_native_steps=actual_steps_per_model)
            evaluation_envs = []
            count = {"calls": 0}
            phase = "agent_initialization"
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
                def factory(task, current_arm=arm, tracked_envs=evaluation_envs):
                    item = make_task(
                        task, current_arm, max_native_steps=MAX_EVAL_EPISODE_STEPS
                    )
                    tracked_envs.append(item)
                    return item

                phase = "before_evaluation"
                before = evaluate_goal_agent(agent, TRAIN_TASK, c["evaluation_seeds"], factory)
                phase = "training"
                agent.learn(total_timesteps=requested_steps)
                if agent.num_timesteps != actual_steps_per_model or env.native_steps != actual_steps_per_model:
                    raise ValueError("Native training action budget mismatch")
                phase = "after_evaluation"
                after = evaluate_goal_agent(agent, TRAIN_TASK, c["evaluation_seeds"], factory)
                phase = "transfer_evaluation"
                transfer = evaluate_goal_agent(agent, TRANSFER_TASK, c["transfer_seeds"], factory)
                sampled = {}
                if extended:
                    phase = "sampled_after_evaluation"
                    sampled["after_sampled"] = evaluate_goal_agent(
                        agent, TRAIN_TASK, c["evaluation_seeds"], factory,
                        deterministic=False,
                    )
                    phase = "sampled_transfer_evaluation"
                    sampled["transfer_sampled"] = evaluate_goal_agent(
                        agent, TRANSFER_TASK, c["transfer_seeds"], factory,
                        deterministic=False,
                    )
                total_eval = sum(
                    item["evaluation_real_steps"]
                    for item in (before, after, transfer, *sampled.values())
                )
                phase = "numeric_export"
                state = numeric_state(agent, output.parent / f"{arm}-seed{seed}.npz")
                phase = "completed_model_record"
                records.append({
                    "arm": arm, "model_seed": seed,
                    "parameters": sum(p.numel() for p in agent.policy.parameters()),
                    "training_real_actions": env.native_steps,
                    "training_resets": env.native_resets,
                    "training_positive_reward_steps": env.native_positive_reward_steps,
                    "training_rewarded_goal_terminations":
                        env.native_rewarded_goal_terminations,
                    "training_public_reward_sum": env.native_public_reward_sum,
                    "optimizer_step_calls": count["calls"],
                    "evaluation_real_actions": total_eval,
                    "before": before, "after": after, "transfer": transfer,
                    **sampled, "numeric_policy": state,
                })
                training += env.native_steps
                evaluation += total_eval
                optimizer += count["calls"]
                training_positive_rewards += env.native_positive_reward_steps
                training_rewarded_goals += env.native_rewarded_goal_terminations
                training_public_reward_sum += env.native_public_reward_sum
                if training > expected_training_total or evaluation > expected_evaluation_total:
                    raise ValueError("Actual work exceeded the configured study accounting")
                progress = {
                    "status": "partial_not_a_completed_study",
                    "source_commit": source_commit,
                    "completed_models": len(records), "training_real_actions": training,
                    "evaluation_real_actions": evaluation,
                    "training_positive_reward_steps": training_positive_rewards,
                    "training_rewarded_goal_terminations": training_rewarded_goals,
                    "training_public_reward_sum": training_public_reward_sum,
                    "optimizer_step_calls": optimizer, "records": records,
                }
                save_progress(output.parent / "progress.json", progress)
                print(f"{study} {arm} seed={seed}: {before['successes']}/6 -> "
                      f"{after['successes']}/6; transfer={transfer['successes']}/6"
                      + (f"; sampled={sampled['after_sampled']['successes']}/6"
                         f"; sampled-transfer={sampled['transfer_sampled']['successes']}/6"
                         if extended else ""), flush=True)
            except BaseException as failure:
                # The completed-model ledger intentionally excludes in-flight
                # work. Preserve separately observed actions/updates on failure.
                already_complete = any(
                    row["arm"] == arm and row["model_seed"] == seed
                    for row in records
                )
                pending = None if already_complete else interrupted_model_measurement(
                    env, evaluation_envs, count
                )
                partial = {
                    "status": "interrupted_not_a_completed_study",
                    "source_commit": source_commit,
                    "config_sha256": config_digest,
                    "failure_phase": phase,
                    "failure_type": type(failure).__name__,
                    "completed_models": len(records),
                    "training_real_actions": training,
                    "evaluation_real_actions": evaluation,
                    "optimizer_step_calls": optimizer,
                    "observed_training_real_actions": (
                        training + (pending["training_real_actions"] if pending else 0)
                    ),
                    "observed_evaluation_real_actions": (
                        evaluation + (pending["evaluation_real_actions"] if pending else 0)
                    ),
                    "observed_optimizer_step_calls": (
                        optimizer + (pending["optimizer_step_calls"] if pending else 0)
                    ),
                    "in_flight_model": (
                        {"arm": arm, "model_seed": seed, **pending} if pending else None
                    ),
                    "records": records,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "limitations": [
                        "In-flight measurements are observed counts, not a completed fit.",
                        "Abrupt process termination can prevent this exception handler.",
                    ],
                }
                save_progress(output.parent / "progress.json", partial)
                raise
            finally:
                env.close()
    if training != expected_training_total:
        raise ValueError("Training did not match configured study accounting")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "study": (
            "BAGI-M3 independent memory-transfer replication" if is_m3
            else ("BAGI-M2 extended learning and policy-sampling diagnostic"
                  if is_m2 else
                  "BAGI-M1 goal learning using established PPO and recurrent PPO")
        ),
        "status": "independent_replication" if is_m3 else (
            "exploratory_extended_training" if is_m2 else "exploratory_training"
        ),
        "registered_study": study,
        "requested_training_steps_per_model": requested_steps,
        "expected_actual_steps_per_model": actual_steps_per_model,
        "protocol_variant": (
            "preregistered_m3_independent_seed_replication" if is_m3
            else ("preregistered_m2_extended_sampled_action_diagnostic" if is_m2
                  else ("original_pilot_settings" if
                  (requested_steps, rollout_steps, c["batch_size"],
                   c["epochs_per_rollout"]) == (8192, 128, 64, 2)
                  else "modified_exploratory_settings"))
        ),
        "source_commit": source_commit,
        "config_sha256": config_digest,
        "versions": {name: version(name) for name in
                     ("gymnasium", "minigrid", "stable-baselines3", "sb3-contrib")},
        "records": records, "milestone_pilot_gates": score_goal_milestone(
            records, training_seeds=tuple(c["training_seeds"])
        ),
        "training_real_actions": training,
        "evaluation_real_actions": evaluation,
        "training_positive_reward_steps": training_positive_rewards,
        "training_rewarded_goal_terminations": training_rewarded_goals,
        "training_public_reward_sum": training_public_reward_sum,
        "optimizer_step_calls": optimizer, "teacher_queries": 0,
        "elapsed_seconds": round(time.monotonic()-started, 3),
        "observation_contract": {
            "reactive": "public_image_plus_one_executed_previous_action",
            "recurrent": "public_image_plus_one_executed_previous_action",
            "action4": "public_image_plus_four_executed_previous_actions",
            "frame4": "four_public_image_frames",
        },
        "limitations": [
            "This is public-image goal learning, not general intelligence.",
            "Training uses only MemoryS7; MemoryS13Random is a separate zero-shot test.",
            "Parameter counts differ; native interaction budgets match.",
            *([
                ("M2 is exploratory after viewing M1; no paired gain "
                 "can be concluded from different test seeds."),
                "Sampled actions are diagnostic, not policy selection.",
            ] if is_m2 else []),
            *([
                ("M3 independently changes training and evaluation seeds, not "
                 "the published M2 model/optimizer/training protocol."),
                ("Two initialization seeds and six episodes per condition "
                 "are insufficient for strong generalization claims."),
                ("M3 replication targets only the single MiniGrid Memory "
                 "task family; no task-independent continual learning."),
            ] if is_m3 else []),
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
    p.add_argument("--execute", "--execute-budgeted-research", dest="execute",
                   action="store_true")
    a = p.parse_args()
    if not a.execute:
        raise SystemExit("Use --execute to start training")
    run(a.config, a.output)


if __name__ == "__main__":
    main()
