"""REUSE-15: established imitation.SimpleDAggerTrainer on XLand train primitives.

Only a thin Gymnasium adapter and privileged TRAIN-ONLY expert policy are new.
The learned BC policy sees XLand local image + direction, never hidden rules,
grid, target coordinates, task IDs or an oracle action at inference time.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

from bagi.experiments.xland_dagger_teacher_feasibility import correction_plan
from bagi.experiments.xland_existing_bc import (
    ALL_TASKS,
    TRAIN_TASKS,
    evaluate,
    get_rules,
    validate_config,
    visible_features,
)


def training_task_for_episode(index):
    """Only frozen three primitive tasks in cyclic order, NEVER composed chain."""
    if not isinstance(index, int) or index < 0:
        raise ValueError("Nonnegative episode index required")
    task = TRAIN_TASKS[index % len(TRAIN_TASKS)]
    if task not in TRAIN_TASKS or task in (ALL_TASKS[-1],):
        raise AssertionError("Held-out task entered DAgger training")
    return task


class TrainingXLandGym(gym.Env):
    """Minimal Gymnasium bridge to upstream JAX dynamics for existing DAgger."""
    def __init__(self, seed_base, cap):
        import jax
        import xminigrid
        from xminigrid.wrappers import DirectionObservationWrapper

        super().__init__()
        self.metadata = {"render_modes": []}
        self.env, self.defaults = xminigrid.make("XLand-MiniGrid-R1-9x9")
        self.views = DirectionObservationWrapper(self.env)
        self._reset = jax.jit(self.views.reset)
        self._step = jax.jit(self.views.step)
        self.seed_base = int(seed_base)
        self.cap = int(cap)
        self.episode_index = 0
        self.params = None
        self.timestep = None
        self.task = None
        self.real_steps = 0
        self.goal_completions = Counter()
        self.episodes_by_task = Counter()
        self.executed_actions = Counter()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=255.0, shape=(54,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(6)

    def reset(self, *, seed=None, options=None):
        import jax

        super().reset(seed=seed)
        index = self.episode_index
        self.task = training_task_for_episode(index)
        self.params = self.defaults.replace(
            ruleset=get_rules(self.task), max_steps=self.cap
        )
        generator_seed = self.seed_base + index // len(TRAIN_TASKS)
        self.timestep = self._reset(self.params, jax.random.key(generator_seed))
        self.episode_index += 1
        self.episodes_by_task[self.task] += 1
        return visible_features(self.timestep.observation), {}

    def step(self, action):
        if self.timestep is None or self.task not in TRAIN_TASKS:
            raise RuntimeError("Reset training-only XLand env before stepping")
        if bool(self.timestep.last()):
            raise RuntimeError("Step attempted after terminal; VecEnv must reset")
        chosen = int(np.asarray(action).item())
        if chosen not in range(6):
            raise ValueError("Invalid XLand native action")
        self.timestep = self._step(self.params, self.timestep, chosen)
        self.real_steps += 1
        self.executed_actions[chosen] += 1
        reward = float(self.timestep.reward)
        terminal = bool(self.timestep.last())
        success = reward > 0
        if success:
            self.goal_completions[self.task] += 1
        return (
            visible_features(self.timestep.observation), reward,
            bool(terminal and success), bool(terminal and not success), {}
        )


def make_privileged_expert(gym_env):
    """Only DAgger's labeling policy can access privileged simulator state."""
    import torch
    from stable_baselines3.common.policies import BasePolicy

    class CurrentStateExpert(BasePolicy):
        def __init__(self):
            super().__init__(
                observation_space=gym_env.observation_space,
                action_space=gym_env.action_space,
            )
            self.teacher_queries = 0
            self.labels = Counter()

        def _predict(self, observation, deterministic=False):
            # Ignore observed tensor: this is the privileged TEACHER, not learner.
            if gym_env.task not in TRAIN_TASKS or gym_env.timestep is None:
                raise ValueError("Unapproved teacher query or uninitialized simulator")
            if bool(gym_env.timestep.last()):
                raise ValueError("Oracle queried after terminal state")
            planned, _stage = correction_plan(gym_env.timestep, gym_env.task)
            if not planned:
                raise ValueError("No corrective expert action available")
            action = int(planned[0])
            if action not in range(6):
                raise ValueError("Invalid expert native action")
            self.teacher_queries += 1
            self.labels[action] += 1
            return torch.tensor([action], dtype=torch.long, device=observation.device)

    return CurrentStateExpert()


def run(config_path):
    import torch
    import xminigrid
    from imitation.algorithms.bc import BC
    from imitation.algorithms.dagger import LinearBetaSchedule, SimpleDAggerTrainer
    from stable_baselines3.common.vec_env import DummyVecEnv
    from xminigrid.wrappers import DirectionObservationWrapper

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    original = yaml.safe_load(
        Path(cfg["baseline_config"]).read_text(encoding="utf-8")
    )
    _, cap, eval_seeds, model_seeds = validate_config(original)
    if (cfg["environment"] != original["environment"]
            or cfg["training_tasks"] != list(TRAIN_TASKS)
            or cfg["evaluation_seeds"] != eval_seeds
            or cfg["model_seeds"] != model_seeds
            or cfg["episode_cap"] != cap
            or int(cfg["rollout_timesteps"]) != 256
            or int(cfg["round_min_timesteps"]) != 128
            or int(cfg["round_min_episodes"]) != 3
            or int(cfg["bc_epochs_per_round"]) != 4
            or int(cfg["batch_size"]) != 32):
        raise ValueError("REUSE-15 frozen resource controls were changed")
    torch.set_num_threads(1)
    eval_env, eval_defaults = xminigrid.make(cfg["environment"])
    eval_views = DirectionObservationWrapper(eval_env)
    experiments = []
    for seed in model_seeds:
        gym_env = TrainingXLandGym(seed_base=int(cfg["training_seed_base"]), cap=cap)
        if int(gym_env.observation_space.shape[0]) != 54:
            raise AssertionError("Unexpected learner-visible feature count")
        venv = DummyVecEnv([lambda env=gym_env: env])
        expert = make_privileged_expert(gym_env)
        torch.manual_seed(seed)
        bc = BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            rng=np.random.default_rng(seed),
            batch_size=int(cfg["batch_size"]),
            device="cpu",
        )
        before = evaluate(bc.policy, eval_views, eval_defaults, eval_seeds, cap)
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="bagi_dagger_") as scratch:
            trainer = SimpleDAggerTrainer(
                venv=venv, scratch_dir=scratch, expert_policy=expert,
                bc_trainer=bc, rng=np.random.default_rng(seed),
                beta_schedule=LinearBetaSchedule(rampdown_rounds=1),
            )
            trainer.train(
                total_timesteps=int(cfg["rollout_timesteps"]),
                rollout_round_min_episodes=int(cfg["round_min_episodes"]),
                rollout_round_min_timesteps=int(cfg["round_min_timesteps"]),
                bc_train_kwargs={
                    "n_epochs": int(cfg["bc_epochs_per_round"]),
                    "progress_bar": False,
                    "log_rollouts_venv": None,
                    "log_rollouts_n_episodes": 0,
                },
            )
            trained_seconds = round(time.monotonic() - start, 3)
            after = evaluate(trainer.policy, eval_views, eval_defaults, eval_seeds, cap)
            rounds_completed = int(trainer.round_num)
        venv.close()
        experiments.append({
            "model_seed": seed,
            "dagger_rounds": rounds_completed,
            "collect_and_train_seconds": trained_seconds,
            "real_training_environment_steps": gym_env.real_steps,
            "teacher_label_queries": expert.teacher_queries,
            "teacher_label_action_counts": dict(expert.labels),
            "training_environment_action_counts": dict(gym_env.executed_actions),
            "training_episodes_by_task": dict(gym_env.episodes_by_task),
            "training_rollout_goal_events": dict(gym_env.goal_completions),
            "before_closed_loop": before,
            "after_closed_loop": after,
            "real_before_evaluation_steps": sum(
                x["evaluation_steps"] for x in before.values()
            ),
            "real_after_evaluation_steps": sum(
                x["evaluation_steps"] for x in after.values()
            ),
        })
        print(f"DAgger seed={seed} interactions={gym_env.real_steps} "
              f"queries={expert.teacher_queries} rounds={rounds_completed} "
              f"skills={[after[k]['successes'] for k in TRAIN_TASKS]} "
              f"heldout={after['hold_then_near']['successes']}/{len(eval_seeds)}")
    result = {
        "question": "Does established online DAgger correction acquire any XLand primitive?",
        "contribution": "Integration of existing imitation.SimpleDAggerTrainer, not novel learning",
        "packages": {"imitation": version("imitation"),
                     "xminigrid": version("xminigrid")},
        "config": cfg,
        "training_tasks": list(TRAIN_TASKS),
        "heldout_training_tasks_or_teacher_queries": 0,
        "evaluation_task_labels_used_for_training": 0,
        "learned_policy_features": "XLand partial image and direction only",
        "expert_policy_can_read_privileged_state": True,
        "dagger_beta_round0": 1.0,
        "dagger_beta_round1_and_later": 0.0,
        "dagger_minimum_timesteps_not_hard_upper_limit": True,
        "baseline_reference": "REUSE-10 class-balanced offline BC 0/3 on every task; data collection budgets differ",
        "results": experiments,
        "limitations": [
            "Online DAgger uses additional teacher queries and different sampled states from REUSE-10; not a matched-label controlled advantage claim.",
            "Single-room task family and two model seeds do not show broad learning or compositional transfer.",
            "Task cycling is known to the environment and expert, not visible as an input to the learned policy.",
            "DAgger initial beta=1 means training rollout reward can reflect expert actions, not learner competence.",
            "Teacher accesses current privileged grid/geometry only to provide labels, never runtime learner features.",
            "The native DAgger total_timesteps argument is a LOWER bound, actual interactions and queries reported.",
            "No DAgger model deserialization or unsafe external pickle benchmark was used.",
        ],
    }
    output = Path(cfg["output_dir"]) / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved DAgger result: {output}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xland_existing_dagger.yaml")
    run(parser.parse_args().config)
