"""G2-E3-P1: frozen public-only episodic count versus extrinsic PPO.

Reuses upstream stable_baselines3.PPO and native Gymnasium environments.
No privileged observation, oracle reward, new RL optimizer or scripted navigation.
Only exactly logged real native steps train the agent; evaluation uses native reward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import numpy as np

from bagi.exploration.public_count import EpisodicFirstVisit

SEEDS = (17, 29)
TASKS = ("workshop", "minigrid_empty_6x6")
ARMS = ("extrinsic", "first_visit")
TRAIN_STEPS = 4096
ROLLOUT_STEPS = 128
BATCH_SIZE = 64
EPOCHS = 4
BETA = 0.05
EVAL_EPISODES = 10
EVAL_SEED_START = {"workshop": 16000, "minigrid_empty_6x6": 17000}
TRAIN_SEED_START = {"workshop": 18000, "minigrid_empty_6x6": 19000}
EPISODE_CAP = {"workshop": 40, "minigrid_empty_6x6": 144}
MAX_TRAIN = 2 * 2 * 2 * TRAIN_STEPS
MAX_EVAL = 2 * 2 * EVAL_EPISODES * sum(EPISODE_CAP.values())
MAX_ACTIONS = 40128
WORKSHOP_EVENTS = ("switch_activated", "key_picked_up", "lock_opened", "goal_reached")


def verify_registration() -> None:
    """Fail before simulator creation if frozen methodological constants change."""
    if (
        SEEDS != (17, 29)
        or TASKS != ("workshop", "minigrid_empty_6x6")
        or ARMS != ("extrinsic", "first_visit")
        or TRAIN_STEPS != 4096
        or ROLLOUT_STEPS != 128
        or BATCH_SIZE != 64
        or EPOCHS != 4
        or BETA != 0.05
        or EVAL_EPISODES != 10
        or EVAL_SEED_START != {"workshop": 16000, "minigrid_empty_6x6": 17000}
        or TRAIN_SEED_START != {"workshop": 18000, "minigrid_empty_6x6": 19000}
        or EPISODE_CAP != {"workshop": 40, "minigrid_empty_6x6": 144}
        or TRAIN_STEPS % ROLLOUT_STEPS
        or MAX_TRAIN != 32768
        or MAX_EVAL != 7360
        or MAX_ACTIONS != MAX_TRAIN + MAX_EVAL
    ):
        raise ValueError("G2-E3-P1 preregistration or native budget changed")


def _public(observation: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(observation)
    if (
        array.ndim != 1
        or array.dtype.kind not in "biuf"
        or not np.isfinite(array).all()
        or (shape is not None and array.shape != shape)
    ):
        raise ValueError("Only finite, flat native PUBLIC image observations are permitted")
    return np.ascontiguousarray(array).copy()


def _new_native(task: str, *, evaluation: bool = False) -> gym.Env:
    """No reset/step is allowed here or in agent initialization."""
    if task == "workshop":
        from bagi.worlds.workshop import WorkshopEnv
        from gymnasium.wrappers import FlattenObservation

        world = WorkshopEnv(
            size=9, view_size=5, max_steps=40,
            start_mode="seeded_start_chamber",
            layout_mode="mirror_x" if evaluation else "default",
        )
        return FlattenObservation(world)
    if task == "minigrid_empty_6x6":
        import minigrid  # noqa: F401 - official benchmark registration
        from gymnasium.wrappers import FlattenObservation
        from minigrid.wrappers import ImgObsWrapper

        return FlattenObservation(ImgObsWrapper(gym.make("MiniGrid-Empty-6x6-v0")))
    raise ValueError("Unregistered experimental task")


class NativePublicRecorder(gym.Wrapper):
    """Count real steps and isolate private info from learner and novelty.

    ``reset`` uses a predeclared TRAIN seed sequence. SB3's initial seed is
    intentionally NOT the native environment seed. TEST needs an explicit
    heldout seed and never produces intrinsic rewards.
    """

    def __init__(
        self, env: gym.Env, task: str, arm: str, seed_index: int, training: bool
    ) -> None:
        if task not in TASKS or arm not in ARMS or seed_index not in (0, 1):
            raise ValueError("Unregistered task, arm or seed index")
        super().__init__(env)
        self.task = task
        self.arm = arm
        self.seed_index = seed_index
        self.training = training
        self.limit = TRAIN_STEPS if training else EVAL_EPISODES * EPISODE_CAP[task]
        self.native_actions = 0
        self.episodes: list[dict] = []
        self.rows: dict[str, list] = {key: [] for key in (
            "observation", "next_observation", "action", "native_reward",
            "intrinsic_reward", "terminated", "truncated", "episode_index",
        )}
        self._obs: np.ndarray | None = None
        self._current: dict | None = None
        self._novelty = EpisodicFirstVisit(BETA) if arm == "first_visit" and training else None

    def reset(self, *, seed=None, options=None):
        if self._current is not None and not self._current["ended"]:
            # A partial final TRAIN episode is recorded only once, after train.
            raise ValueError("Unexpected reset before native episode completion")
        if self.training:
            actual_seed = (
                TRAIN_SEED_START[self.task] + 1000 * self.seed_index + len(self.episodes)
            )
        else:
            if type(seed) is not int:
                raise ValueError("Heldout TEST reset requires an explicit registered seed")
            expected = EVAL_SEED_START[self.task] + len(self.episodes)
            if seed != expected:
                raise ValueError("Unexpected or repeated heldout evaluation seed")
            actual_seed = seed
        initial, _private = self.env.reset(seed=actual_seed, options=options)
        initial = _public(initial)
        if self._novelty is not None:
            self._novelty.reset(initial)
        self._obs = initial
        self._current = {
            "native_reset_seed": actual_seed,
            "steps": 0,
            "native_reward": 0.0,
            "intrinsic_reward": 0.0,
            "native_goal_success": False,
            "events": {k: 0 for k in WORKSHOP_EVENTS},
            "action_counts": {},
            "ended": False,
            "terminated": False,
            "truncated": False,
            "partial_final": False,
        }
        self._trace = hashlib.sha256()
        # Never expose environment diagnostic info through Gymnasium info.
        return initial.copy(), {}

    def step(self, action):
        if self._obs is None or self._current is None or self._current["ended"]:
            raise ValueError("No live native episode")
        if self.native_actions >= self.limit:
            raise ValueError("G2-E3-P1 real simulator step budget exceeded")
        if not isinstance(action, (int, np.integer)) or isinstance(action, (bool, np.bool_)):
            raise ValueError("Non-discrete action received")
        native_action = int(action)
        if not self.action_space.contains(native_action):
            raise ValueError("Action not in native registered action space")
        before = self._obs.copy()
        after_raw, raw_reward, native_done, native_trunc, privileged = self.env.step(native_action)
        after = _public(after_raw, shape=before.shape)
        reward = float(raw_reward)
        if not np.isfinite(reward):
            raise ValueError("Nonfinite native extrinsic reward")
        terminated, truncated = bool(native_done), bool(native_trunc)
        current = self._current
        current["steps"] += 1
        # The native simulator's own time limit is expected at these caps;
        # externally stop TEST without modifying the native reward/termination.
        cap = EPISODE_CAP[self.task]
        if current["steps"] >= cap and not (terminated or truncated):
            if self.training:
                raise ValueError("Native environment exceeded preregistered episode cap")
            truncated = True
        if reward > 0 and not terminated:
            raise ValueError("Positive native goal reward without native goal termination")
        intrinsic = self._novelty.observe(after) if self._novelty is not None else 0.0
        current["native_reward"] += reward
        current["intrinsic_reward"] += intrinsic
        current["terminated"] = terminated
        current["truncated"] = truncated
        current["native_goal_success"] = bool(terminated and reward > 0)
        counts = current["action_counts"]
        counts[str(native_action)] = counts.get(str(native_action), 0) + 1
        # Privileged simulator details exist ONLY here, after native action.
        # Report events but never pass info to PPO or the intrinsic reward.
        for event in privileged.get("events", ()):
            if event in current["events"]:
                current["events"][event] += 1
        self._trace.update(before.tobytes())
        self._trace.update(np.asarray([native_action], dtype="<i4").tobytes())
        self._trace.update(np.asarray([reward, intrinsic], dtype="<f8").tobytes())
        self._trace.update(after.tobytes())
        self.rows["observation"].append(before)
        self.rows["next_observation"].append(after)
        self.rows["action"].append(native_action)
        self.rows["native_reward"].append(reward)
        self.rows["intrinsic_reward"].append(intrinsic)
        self.rows["terminated"].append(int(terminated))
        self.rows["truncated"].append(int(truncated))
        self.rows["episode_index"].append(len(self.episodes))
        self.native_actions += 1
        self._obs = after
        if terminated or truncated:
            current["ended"] = True
            current["public_trajectory_sha256"] = self._trace.hexdigest()
            self.episodes.append(current)
        return after.copy(), reward + intrinsic, terminated, truncated, {}

    def finalize_training(self) -> None:
        if not self.training or self.native_actions != TRAIN_STEPS:
            raise ValueError("TRAIN actions must equal frozen native budget")
        if self._current is not None and not self._current["ended"]:
            self._current["partial_final"] = True
            self._current["public_trajectory_sha256"] = self._trace.hexdigest()
            self.episodes.append(self._current)
            self._current["ended"] = True
        if sum(row["steps"] for row in self.episodes) != TRAIN_STEPS:
            raise ValueError("TRAIN episode action accounting mismatch")

    def report(self) -> dict:
        if sum(x["steps"] for x in self.episodes) != self.native_actions:
            raise ValueError("Native episode action accounting mismatch")
        return {
            "native_actions": self.native_actions,
            "episodes": self.episodes,
            "native_goals": sum(int(e["native_goal_success"]) for e in self.episodes),
            "first_native_reward_at_action": next(
                (i + 1 for i, value in enumerate(self.rows["native_reward"]) if value > 0),
                None,
            ),
            "native_reward_sum": round(sum(e["native_reward"] for e in self.episodes), 6),
            "intrinsic_reward_sum": round(
                sum(e["intrinsic_reward"] for e in self.episodes), 6
            ),
            "unique_public_views": len({
                row.tobytes() for row in self.rows["observation"]
            } | {row.tobytes() for row in self.rows["next_observation"]}),
            "native_events": {
                event: sum(e["events"][event] for e in self.episodes)
                for event in WORKSHOP_EVENTS
            },
        }

    def numeric_arrays(
        self, weights: dict[str, np.ndarray] | None = None
    ) -> dict[str, np.ndarray]:
        n = self.native_actions
        if self.rows["observation"]:
            obs = np.stack(self.rows["observation"])
            nxt = np.stack(self.rows["next_observation"])
        else:
            obs = np.empty((0, *self.observation_space.shape), dtype=np.float32)
            nxt = obs.copy()
        arrays = {
            "observation": obs,
            "next_observation": nxt,
            "action": np.asarray(self.rows["action"], dtype=np.int16),
            "native_reward": np.asarray(self.rows["native_reward"], dtype=np.float32),
            "intrinsic_reward": np.asarray(self.rows["intrinsic_reward"], dtype=np.float32),
            "terminated": np.asarray(self.rows["terminated"], dtype=np.int8),
            "truncated": np.asarray(self.rows["truncated"], dtype=np.int8),
            "episode_index": np.asarray(self.rows["episode_index"], dtype=np.int32),
        }
        if any(x.shape[0] != n or x.dtype.kind == "O" for x in arrays.values()):
            raise ValueError("Unsafe numeric evidence or action-counter mismatch")
        if weights:
            for name, array in weights.items():
                if name in arrays or array.dtype.kind == "O":
                    raise ValueError("Invalid policy weight evidence")
                arrays[name] = array
        return arrays


def _evaluate(model, world: NativePublicRecorder) -> NativePublicRecorder:
    task = world.task
    for i in range(EVAL_EPISODES):
        observation, _ = world.reset(seed=EVAL_SEED_START[task] + i)
        done = False
        while not done:
            # Only the native PUBLIC observation enters the frozen policy.
            action, _ = model.predict(observation, deterministic=True)
            observation, _reward, terminated, truncated, _info = world.step(
                int(np.asarray(action).item())
            )
            done = bool(terminated or truncated)
    if len(world.episodes) != EVAL_EPISODES:
        raise ValueError("Missing registered TEST episodes")
    if world.native_actions > EVAL_EPISODES * EPISODE_CAP[task]:
        raise ValueError("TEST action budget exceeded")
    return world


def run(output_dir: Path) -> dict:
    verify_registration()
    from stable_baselines3 import PPO
    import torch

    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "status": "running",
        "study": "G2-E3-P1",
        "source_sha": os.environ.get("GITHUB_SHA", "local-unpinned"),
        "preregistered_max_actions": MAX_ACTIONS,
        "native_actions": 0,
        "native_train_actions": 0,
        "native_eval_actions": 0,
        "teacher_queries": 0,
        "model_seeds": list(SEEDS),
        "policy": "stable_baselines3.PPO:MlpPolicy",
        "versions": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "gymnasium": version("gymnasium"),
            "minigrid": version("minigrid"),
            "stable-baselines3": version("stable-baselines3"),
        },
        "results": {},
    }
    records = []
    policy_arrays: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    try:
        for task in TASKS:
            summary["results"][task] = {}
            for seed_index, seed in enumerate(SEEDS):
                summary["results"][task][str(seed)] = {}
                for arm in ARMS:
                    world = NativePublicRecorder(
                        _new_native(task), task, arm, seed_index, training=True
                    )
                    records.append((task, seed, arm, "train", world))
                    try:
                        before_init = world.native_actions
                        policy = PPO(
                            "MlpPolicy", world, seed=seed, device="cpu",
                            n_steps=ROLLOUT_STEPS, batch_size=BATCH_SIZE,
                            n_epochs=EPOCHS, learning_rate=3e-4,
                            gamma=0.99, verbose=0,
                        )
                        if world.native_actions != before_init:
                            raise ValueError("PPO initializer consumed hidden simulator actions")
                        start = time.monotonic()
                        policy.learn(total_timesteps=TRAIN_STEPS)
                        elapsed = time.monotonic() - start
                        if world.native_actions != TRAIN_STEPS or policy.num_timesteps != TRAIN_STEPS:
                            raise ValueError("PPO native training step accounting mismatch")
                        world.finalize_training()
                        # Save weights as verified numeric arrays, never torch.save/pickle.
                        policy_arrays[(task, seed, arm)] = {
                            "policy__" + name.replace(".", "_"): tensor.detach().cpu()
                            .numpy().copy()
                            for name, tensor in policy.policy.state_dict().items()
                        }
                        train_report = world.report()
                        summary["native_train_actions"] += world.native_actions
                        summary["native_actions"] += world.native_actions
                        eval_world = NativePublicRecorder(
                            _new_native(task, evaluation=True),
                            task, arm, seed_index, training=False,
                        )
                        records.append((task, seed, arm, "eval", eval_world))
                        _evaluate(policy, eval_world)
                        eval_report = eval_world.report()
                        summary["native_eval_actions"] += eval_world.native_actions
                        summary["native_actions"] += eval_world.native_actions
                        summary["results"][task][str(seed)][arm] = {
                            "train": train_report,
                            "train_wall_seconds": round(elapsed, 3),
                            "train_optimizer_timesteps": int(policy.num_timesteps),
                            "ppo_training_epochs": int(policy._n_updates),
                            "test": eval_report,
                        }
                    finally:
                        world.close()
        if summary["native_train_actions"] != MAX_TRAIN:
            raise ValueError("Missing exact matched TRAIN native action counts")
        if summary["native_eval_actions"] > MAX_EVAL or summary["native_actions"] > MAX_ACTIONS:
            raise ValueError("Real experiment exceeded preregistered native ceiling")
        decisions = {}
        for task in TASKS:
            for seed in SEEDS:
                row = summary["results"][task][str(seed)]
                paired = _paired_from_rows(row["extrinsic"]["test"], row["first_visit"]["test"])
                row["paired"] = paired
            decisions[task] = all(
                summary["results"][task][str(seed)]["paired"]["count_minus_extrinsic"] >= 2
                and summary["results"][task][str(seed)]["first_visit"]["train"]["native_goals"] >= 1
                for seed in SEEDS
            )
        # Promotion requires improvement in both seeds in one task AND
        # absence of regressions in either seed on the other task.
        summary["promotion"] = any(
            decisions[task]
            and all(
                summary["results"][other][str(seed)]["paired"]["count_minus_extrinsic"] >= 0
                for other in TASKS if other != task for seed in SEEDS
            ) for task in TASKS
        )
        summary["status"] = "complete"
    except BaseException as err:
        summary["status"] = "failed"
        summary["failure"] = f"{type(err).__name__}: {err}"
        raise
    finally:
        # Preserve exact accounted native actions even on partial failures.
        summary["observed_actions_including_partial_failure"] = sum(
            recorder.native_actions for _, _, _, _, recorder in records
        )
        if summary["observed_actions_including_partial_failure"] > MAX_ACTIONS:
            summary["status"] = "failed"
            summary["failure"] = "observed native action ceiling exceeded"
        # Native-only numeric evidence; never serialize policy objects/pickle.
        for task, seed, arm, phase, recorder in records:
            filename = f"{task}_seed{seed}_{arm}_{phase}.npz"
            weights = (
                policy_arrays.get((task, seed, arm), {}) if phase == "train" else {}
            )
            np.savez_compressed(output_dir / filename, **recorder.numeric_arrays(weights))
            recorder.close()
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return summary


def _paired_from_rows(extrinsic: dict, count: dict) -> dict:
    a, b = extrinsic["episodes"], count["episodes"]
    if len(a) != EVAL_EPISODES or len(b) != EVAL_EPISODES:
        raise ValueError("Paired TEST episodes incomplete")
    if any(x["native_reset_seed"] != y["native_reset_seed"]
           for x, y in zip(a, b, strict=True)):
        raise ValueError("Paired TEST reset seeds differ")
    wins = sum(x["native_goal_success"] < y["native_goal_success"]
               for x, y in zip(a, b, strict=True))
    losses = sum(x["native_goal_success"] > y["native_goal_success"]
                 for x, y in zip(a, b, strict=True))
    return {
        "count_wins": wins,
        "count_losses": losses,
        "ties": EVAL_EPISODES - wins - losses,
        "count_minus_extrinsic": wins - losses,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/g2_e3_p1_public_exploration"))
    args = parser.parse_args()
    summary = run(args.output_dir)
    print(
        f"G2-E3-P1 {summary['status']}: "
        f"train={summary['native_train_actions']} test={summary['native_eval_actions']} "
        f"native goals reported separately; promotion={summary['promotion']}"
    )
