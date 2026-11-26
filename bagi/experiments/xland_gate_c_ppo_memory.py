"""REUSE-32 Gate C: matched published PPO/RecurrentPPO on real XLand.

Both policies receive the SAME learner-legal REUSE-31b public features.
No custom RL optimizer, teacher labels, heldout composition or private policy
state. Privileged simulator state is inspected only AFTER native actions to
validate physical object movement during independent evaluations.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import ClassVar

import numpy as np
import yaml

from bagi.experiments.xland_gate_b_exact_collection import (
    CAP,
    FRESH_EVAL_LAYOUTS,
    INPUT_WIDTH,
    native_action,
    public_features,
)

MODEL_SEEDS = (17, 29)
TRAIN_TASKS = ("hold", "near", "putdown")
TEST_LAYOUTS = (9900, 9901, 9902, 9903, 9904)
SOURCE_SEED_BASE = 9400
TRAIN_STEPS_PER_MODEL = 1024
ROLLOUT = 128
BATCH_SIZE = 64
EPOCHS_PER_ROLLOUT = 2
ARCHITECTURES = ("PPO", "RecurrentPPO")


def validate_config(config: dict) -> None:
    """Stop if a predeclared information, budget or seed gate is changed."""
    training = config["training"]
    evaluation = config["evaluation"]
    if (config.get("environment") != "XLand-MiniGrid-R1-9x9"
            or config.get("training_tasks") != list(TRAIN_TASKS)
            or config.get("model_seeds") != list(MODEL_SEEDS)
            or int(training.get("native_steps_per_policy")) != TRAIN_STEPS_PER_MODEL
            or int(training.get("seed_base")) != SOURCE_SEED_BASE
            or int(training.get("rollout_steps")) != ROLLOUT
            or int(training.get("batch_size")) != BATCH_SIZE
            or int(training.get("epochs_per_rollout")) != EPOCHS_PER_ROLLOUT
            or int(evaluation.get("episode_cap")) != CAP
            or evaluation.get("layout_seeds") != list(TEST_LAYOUTS)
            or config.get("policy_input_width") != INPUT_WIDTH):
        raise ValueError("Unpreregistered Gate C native budget, task or public input")
    if (set(TEST_LAYOUTS) & set(FRESH_EVAL_LAYOUTS)
            or min(TEST_LAYOUTS) <= (SOURCE_SEED_BASE + TRAIN_STEPS_PER_MODEL // len(TRAIN_TASKS))
            or TRAIN_STEPS_PER_MODEL % ROLLOUT
            or ROLLOUT % BATCH_SIZE):
        raise ValueError("Overlapping evaluation layouts or invalid PPO rollout")


def next_executed_history(history: tuple[int, ...], action: int) -> tuple[int, ...]:
    """The policy's four-action history contains executed XLand actions only."""
    if len(history) != 4 or any(i not in range(7) for i in history):
        raise ValueError("Malformed four-action learner history")
    return (*history[1:], native_action(np.asarray(action, dtype=np.int64)))


def make_train_env(seed_base: int, cap: int):
    """Thin Gymnasium bridge around published XLand, not a custom RL model."""
    import gymnasium as gym
    import jax
    import xminigrid
    from xminigrid.wrappers import (
        DirectionObservationWrapper,
        RulesAndGoalsObservationWrapper,
    )

    from bagi.experiments.xland_existing_bc import get_rules

    class PublicXLandTraining(gym.Env):
        metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

        def __init__(self):
            super().__init__()
            room, self.defaults = xminigrid.make("XLand-MiniGrid-R1-9x9")
            view = RulesAndGoalsObservationWrapper(DirectionObservationWrapper(room))
            self.native_reset = jax.jit(view.reset)
            self.native_step = jax.jit(view.step)
            self.action_space = gym.spaces.Discrete(6)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(INPUT_WIDTH,), dtype=np.float32
            )
            self.episode_index = 0
            self.real_native_steps = 0
            self.steps_by_task = Counter()
            self.episodes_by_task = Counter()
            self.native_rewards_by_task = Counter()
            self.actions_by_task = {task: Counter() for task in TRAIN_TASKS}
            self.timestep = None
            self.params = None
            self.history: tuple[int, ...] = (6, 6, 6, 6)
            self.task = None
            self.seen_layouts: set[int] = set()

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            task = TRAIN_TASKS[self.episode_index % len(TRAIN_TASKS)]
            layout = seed_base + self.episode_index // len(TRAIN_TASKS)
            if layout in TEST_LAYOUTS or layout in FRESH_EVAL_LAYOUTS:
                raise ValueError("Evaluation layout reached by PPO training")
            self.episode_index += 1
            self.task = task
            self.seen_layouts.add(layout)
            self.episodes_by_task[task] += 1
            self.history = (6, 6, 6, 6)
            self.params = self.defaults.replace(
                ruleset=get_rules(task), max_steps=cap
            )
            self.timestep = self.native_reset(self.params, jax.random.key(layout))
            if bool(self.timestep.last()):
                raise RuntimeError("Native XLand training task terminal at reset")
            visible = public_features(self.timestep, self.history)
            if not self.observation_space.contains(visible):
                raise ValueError("Invalid 115-wide PUBLIC XLand observation")
            return visible, {}

        def step(self, action):
            if self.timestep is None or bool(self.timestep.last()):
                raise RuntimeError("PPO attempted step without active XLand state")
            taken = native_action(np.asarray(action))
            self.timestep = self.native_step(self.params, self.timestep, taken)
            self.real_native_steps += 1
            self.steps_by_task[self.task] += 1
            self.actions_by_task[self.task][taken] += 1
            reward = float(self.timestep.reward)
            if reward > 0:
                self.native_rewards_by_task[self.task] += 1
            terminal = bool(self.timestep.last())
            self.history = next_executed_history(self.history, taken)
            visible = public_features(self.timestep, self.history)
            if not self.observation_space.contains(visible):
                raise ValueError("Invalid PUBLIC XLand step observation")
            return (visible, reward, bool(terminal and reward > 0),
                    bool(terminal and reward <= 0), {})

        def close(self):
            self.timestep = None

        def audit(self):
            return {
                "actual_native_steps": self.real_native_steps,
                "episodes_by_task": dict(self.episodes_by_task),
                "native_steps_by_task": dict(self.steps_by_task),
                "native_rewards_by_task": dict(self.native_rewards_by_task),
                "executed_actions_by_task": {
                    task: dict(counts) for task, counts in self.actions_by_task.items()
                },
                "distinct_training_layouts": len(self.seen_layouts),
                "minimum_training_layout": min(self.seen_layouts),
                "maximum_training_layout": max(self.seen_layouts),
            }

    return PublicXLandTraining()


def run_closed_loop_episode(
    policy, timestep, *, task: str, native_step, observe, eligible: bool,
    on_physical_step=None, final_adjacent=None, cap: int = CAP,
) -> dict:
    """No expert controls; reset the LSTM on EVERY new independent episode."""
    from bagi.experiments.xland_putdown_reward_validity import physically_nonvacuous

    if task not in TRAIN_TASKS or cap != CAP:
        raise ValueError("Evaluation task or native episode cap changed")
    if not eligible:
        return {"eligible": False, "native_actions": 0,
                "native_reward": False, "physical_placement": False}
    if bool(timestep.last()):
        raise ValueError("Evaluation episode terminal at reset")
    history = (6, 6, 6, 6)
    recurrent_state = None
    episode_start = np.array([True], dtype=bool)
    native_reward = False
    pickups = drops = 0
    seen_pickup = ordered_drop = False
    steps = 0
    while steps < cap and not bool(timestep.last()):
        visible = np.asarray(observe(timestep, history), dtype=np.float32)
        if visible.shape != (INPUT_WIDTH,) or not np.isfinite(visible).all():
            raise ValueError("Non-public XLand eval observation")
        chosen, recurrent_state = policy.predict(
            visible, state=recurrent_state, episode_start=episode_start,
            deterministic=True,
        )
        taken = native_action(np.asarray(chosen))
        before = timestep
        timestep = native_step(timestep, taken)
        steps += 1
        native_reward |= float(timestep.reward) > 0
        if task == "putdown":
            if on_physical_step is None or final_adjacent is None:
                raise ValueError("Missing physical pickup/drop evaluator")
            pickup, drop = on_physical_step(before, timestep, taken)
            pickups += int(pickup)
            drops += int(drop)
            ordered_drop |= bool(drop and seen_pickup)
            seen_pickup |= bool(pickup)
        history = next_executed_history(history, taken)
        episode_start[:] = False
    actual = bool(task == "putdown" and physically_nonvacuous(
        native_reward, False, pickups, drops,
        bool(final_adjacent(timestep)) if task == "putdown" else False,
        ordered_drop,
    ))
    return {"eligible": True, "native_actions": steps,
            "native_reward": bool(native_reward),
            "effective_blue_pickups": pickups, "effective_blue_drops": drops,
            "ordered_blue_drop": bool(ordered_drop),
            "physical_placement": actual}


def evaluate_native(agent, config: dict) -> dict:
    """Fresh source-task layouts; only evaluator can inspect pocket and grid."""
    import jax
    import xminigrid
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
    from xminigrid.wrappers import (
        DirectionObservationWrapper,
        RulesAndGoalsObservationWrapper,
    )

    from bagi.experiments.xland_existing_bc import get_rules
    from bagi.experiments.xland_gate_b_native_collection import screen_source_layout
    from bagi.experiments.xland_putdown_reward_validity import (
        blue_square_transition,
        tiles_adjacent,
    )

    room, defaults = xminigrid.make(config["environment"])
    view = RulesAndGoalsObservationWrapper(DirectionObservationWrapper(room))
    reset, step = jax.jit(view.reset), jax.jit(view.step)
    blue = np.asarray(TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE])
    red = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.RED])
    empty_type = int(Tiles.EMPTY)
    by_task = {}
    for task in TRAIN_TASKS:
        params = defaults.replace(ruleset=get_rules(task), max_steps=CAP)
        cases = []
        for layout in TEST_LAYOUTS:
            ts = reset(params, jax.random.key(layout))
            eligible = (task != "putdown" or screen_source_layout(ts, blue, red))

            def native_step(timestep, taken, *, params=params):
                return step(params, timestep, taken)

            def check_blue(before, after, action):
                return blue_square_transition(
                    action, np.asarray(before.state.agent.pocket),
                    np.asarray(after.state.agent.pocket),
                    np.asarray(before.state.grid), np.asarray(after.state.grid),
                    blue, empty_type,
                )

            def at_goal(timestep):
                return tiles_adjacent(np.asarray(timestep.state.grid), blue, red)

            outcome = run_closed_loop_episode(
                agent, ts, task=task, native_step=native_step,
                observe=public_features, eligible=eligible,
                on_physical_step=check_blue if task == "putdown" else None,
                final_adjacent=at_goal if task == "putdown" else None,
            )
            cases.append({"layout": layout, **outcome})
        by_task[task] = {
            "cases": cases,
            "eligible_episodes": sum(x["eligible"] for x in cases),
            "native_rewarded": sum(int(x["native_reward"]) for x in cases),
            "verified_physical_placements": sum(
                int(x["physical_placement"]) for x in cases
            ),
            "native_evaluation_actions": sum(x["native_actions"] for x in cases),
            "effective_blue_pickups": sum(
                x.get("effective_blue_pickups", 0) for x in cases
            ),
            "effective_blue_drops": sum(x.get("effective_blue_drops", 0) for x in cases),
        }
    return by_task


def run(config_path: str) -> dict:
    """Run published algorithms only, with exact REAL native interaction counts."""
    import torch
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    validate_config(cfg)
    torch.set_num_threads(1)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / "summary.json"
    report = {
        "experiment": "REUSE-32 Gate C established PPO vs recurrent PPO XLand control",
        "status": "running",
        "information_contract": (
            "Same public image+heading+rule/goal+four truly executed actions: "
            "115 float32 learner features, never full grid or hidden pocket"
        ),
        "algorithms": ["stable_baselines3.PPO", "sb3_contrib.RecurrentPPO"],
        "policy_parameter_counts_matched": False,
        "predeclared_train_tasks": list(TRAIN_TASKS),
        "heldout_composition_teacher_queries": 0,
        "source_train_layout_seed_base": SOURCE_SEED_BASE,
        "predeclared_new_eval_layouts": list(TEST_LAYOUTS),
        "previous_reuse31_eval_layouts_not_reused": list(FRESH_EVAL_LAYOUTS),
        "per_policy_training_real_steps": TRAIN_STEPS_PER_MODEL,
        "episode_cap": CAP,
        "model_seeds": list(MODEL_SEEDS),
        "native_training_steps_recorded": 0,
        "native_evaluation_steps_recorded": 0,
        "new_teacher_queries": 0,
        "results": [],
        "limitations": [
            "Tiny source-skill smoke test; zero goal rewards cannot refute recurrent memory.",
            "PPO and RecurrentPPO share input and steps but differ in parameter count.",
            "Termination can vary task-layout exposure even with equal native budgets.",
            "Native reward without physical BLUE pickup/drop does not establish putdown skill.",
            "No heldout compositional train/eval or privileged teacher data.",
        ],
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    model_types = (("PPO", PPO, "MlpPolicy"),
                   ("RecurrentPPO", RecurrentPPO, "MlpLstmPolicy"))
    for seed in MODEL_SEEDS:
        for name, algorithm, policy_type in model_types:
            env = make_train_env(SOURCE_SEED_BASE, CAP)
            try:
                kwargs = {"net_arch": {"pi": [32], "vf": [32]}}
                if name == "RecurrentPPO":
                    kwargs["lstm_hidden_size"] = 32
                learner = algorithm(
                    policy_type, env, seed=seed, device="cpu", verbose=0,
                    n_steps=ROLLOUT, batch_size=BATCH_SIZE,
                    n_epochs=EPOCHS_PER_ROLLOUT,
                    learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                    policy_kwargs=kwargs,
                )
                t0 = time.monotonic()
                learner.learn(total_timesteps=TRAIN_STEPS_PER_MODEL)
                seconds = time.monotonic() - t0
                counted = env.real_native_steps
                if (int(learner.num_timesteps) != TRAIN_STEPS_PER_MODEL
                        or counted != TRAIN_STEPS_PER_MODEL):
                    raise AssertionError("PPO or RecurrentPPO exceeded native step budget")
                evaluation = evaluate_native(learner, cfg)
                eval_steps = sum(x["native_evaluation_actions"]
                                 for x in evaluation.values())
                report["results"].append({
                    "algorithm": name, "model_seed": seed,
                    "policy_parameters": sum(int(p.numel()) for p in learner.policy.parameters()),
                    "training_wall_seconds": round(seconds, 3),
                    "training_native_actions": counted,
                    "training_provenance": env.audit(),
                    "evaluation": evaluation,
                    "native_evaluation_actions": eval_steps,
                })
                report["native_training_steps_recorded"] += counted
                report["native_evaluation_steps_recorded"] += eval_steps
                path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"REUSE32 seed={seed} {name} train={counted} eval={eval_steps} "
                      f"real_BLUE_placements="
                      f"{evaluation['putdown']['verified_physical_placements']}/"
                      f"{evaluation['putdown']['eligible_episodes']}", flush=True)
            finally:
                env.close()
    if len(report["results"]) != 2 * len(MODEL_SEEDS):
        raise AssertionError("Incomplete Gate C paired comparison")
    report["status"] = "complete"
    report["package_versions"] = {
        "stable-baselines3": version("stable-baselines3"),
        "sb3-contrib": version("sb3-contrib"),
        "xminigrid": version("xminigrid"),
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xland_gate_c_ppo_memory.yaml")
    args = parser.parse_args()
    run(args.config)
