"""M10-P1: matched public image/direction representation ablation with upstream PPO.

Private MiniGrid state is accessed only by the independent physical-effect
auditor; policy observations, rewards and actions stay public and native.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
from minigrid.wrappers import OneHotPartialObsWrapper

from bagi.experiments.m1_goal_learning import (
    admitted_source_commit,
    count_optimizer_steps,
    numeric_state,
    save_progress,
)
from bagi.experiments.m5_continual_retention import (
    ACTION_HISTORY,
    DOORKEY,
    EVAL_HORIZON,
    MISSION_WIDTH,
    PublicGoalActions,
    mission_vector,
)
from bagi.experiments.m6_doorkey_continuity import BATCH, EPOCHS, ROLLOUT, NativeDoorTraining
from bagi.experiments.m10_native_audit import NativeDoorKeyAudit
from bagi.experiments.m10_train_only_diagnostic import aggregate

ARMS = ("raw_history", "raw_direction", "onehot_direction")
SEEDS = (211, 223)
MONITOR = tuple(range(119101, 119107))
HELDOUT = tuple(range(120001, 120013))
REPEATS = 3
TRAIN_STEPS = 131072
UPDATES_PER_MODEL = TRAIN_STEPS // ROLLOUT * EPOCHS * (ROLLOUT // BATCH)
EXPECTED_SHAPES = {
    "raw_history": (243,),
    "raw_direction": (247,),
    "onehot_direction": (1080,),
}
OUTPUT_DIR = "runs/m10_public_representation"


class PublicDirectionalActions(gym.Wrapper):
    """Append public heading while preserving genuine executed-action history.

    The image itself is encoded by the installed Farama MiniGrid wrapper.
    No private coordinates, inventory, door state, grid, info or evaluator
    attributes can enter this policy input.
    """

    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError("MiniGrid public observation must be a dictionary")
        image_shape = env.observation_space["image"].shape
        if image_shape not in ((7, 7, 3), (7, 7, 20)):
            raise ValueError("Unregistered MiniGrid public image representation")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("M10 requires discrete native MiniGrid actions")
        self.n_actions = int(env.action_space.n)
        if self.n_actions != 7:
            raise ValueError("M10 requires exactly seven native actions")
        self.image_shape = image_shape
        self.previous = deque([self.n_actions] * ACTION_HISTORY, maxlen=ACTION_HISTORY)
        width = int(np.prod(image_shape)) + MISSION_WIDTH + (
            ACTION_HISTORY * (self.n_actions + 1)
        ) + 4
        self.observation_space = Box(0.0, 255.0, shape=(width,), dtype=np.float32)

    def _encode(self, raw):
        if set(raw) != {"image", "direction", "mission"}:
            raise ValueError("Unexpected public observation fields")
        image = np.asarray(raw["image"], dtype=np.float32)
        if image.shape != self.image_shape:
            raise ValueError("Unexpected public image shape")
        heading = raw["direction"]
        if isinstance(heading, (bool, np.bool_)) or not isinstance(
            heading, (int, np.integer)
        ) or not 0 <= heading < 4:
            raise ValueError("Public direction must be an integer 0..3")
        directions = np.zeros(4, dtype=np.float32)
        directions[int(heading)] = 1
        history = np.zeros((ACTION_HISTORY, self.n_actions + 1), dtype=np.float32)
        for index, action in enumerate(self.previous):
            history[index, action] = 1
        return np.concatenate((
            image.ravel(), mission_vector(raw["mission"]), history.ravel(),
            directions,
        ))

    def reset(self, *, seed=None, options=None):
        self.previous = deque([self.n_actions] * ACTION_HISTORY, maxlen=ACTION_HISTORY)
        raw, info = self.env.reset(seed=seed, options=options)
        return self._encode(raw), info

    def step(self, action):
        executed = int(action)
        if not 0 <= executed < self.n_actions:
            raise ValueError("Invalid native MiniGrid action")
        raw, reward, terminated, truncated, info = self.env.step(executed)
        self.previous.append(executed)
        return self._encode(raw), reward, terminated, truncated, info


def public_env(env, arm: str):
    if arm not in ARMS:
        raise ValueError("Unknown M10 representation arm")
    if arm == "raw_history":
        return PublicGoalActions(env)
    if arm == "onehot_direction":
        env = OneHotPartialObsWrapper(env)
    return PublicDirectionalActions(env)


def validate_config(config: dict) -> dict:
    expected = {
        "registered_study": "m10p1",
        "task": DOORKEY,
        "arms": list(ARMS),
        "model_seeds": list(SEEDS),
        "train_steps_per_fit": TRAIN_STEPS,
        "rollout_steps": ROLLOUT,
        "batch_size": BATCH,
        "epochs_per_rollout": EPOCHS,
        "evaluation_horizon": EVAL_HORIZON,
        "monitor_seeds": list(MONITOR),
        "heldout_seeds": list(HELDOUT),
        "sampled_repeats": REPEATS,
        "output_dir": OUTPUT_DIR,
    }
    if config != expected:
        raise ValueError("Registered M10-P1 representation protocol changed")
    if set(MONITOR).intersection(HELDOUT):
        raise ValueError("M10 monitor/final layout contamination")
    return config


def make_training_env(arm: str, model_seed: int) -> NativeDoorTraining:
    if model_seed not in SEEDS or arm not in ARMS:
        raise ValueError("Unregistered M10 representation training arm or seed")
    native = gym.wrappers.TimeLimit(
        gym.make(DOORKEY), max_episode_steps=EVAL_HORIZON
    )
    return NativeDoorTraining(
        public_env(native, arm), model_seed=model_seed, allowed_seeds=SEEDS,
        max_native_steps=TRAIN_STEPS,
    )


def action_seed(arm: str, model_seed: int, layout: int, repeat: int) -> int:
    if arm not in ARMS or model_seed not in SEEDS:
        raise ValueError("Unregistered M10 model/representation")
    if layout not in MONITOR and layout not in HELDOUT:
        raise ValueError("Unregistered M10 monitor/final layout")
    if type(repeat) is not int or not 1 <= repeat <= REPEATS:
        raise ValueError("Unregistered M10 sampled repeat")
    return (90_000_000 + ARMS.index(arm) * 1_000_000
            + SEEDS.index(model_seed) * 100_000
            + (0 if layout in MONITOR else 10_000)
            + (MONITOR.index(layout) if layout in MONITOR else HELDOUT.index(layout))
            * 10 + repeat)


def evaluate_episode(agent, arm: str, model_seed: int, layout: int,
                     *, repeat: int | None, meter: dict, split: str) -> dict:
    """Pure policy input and separate native causal audit on one true episode."""
    if split not in ("monitor", "independent"):
        raise ValueError("Unknown M10 evaluation split")
    allowed = MONITOR if split == "monitor" else HELDOUT
    if layout not in allowed:
        raise ValueError("M10 evaluation accessed the wrong split")
    if repeat is not None:
        seed = action_seed(arm, model_seed, layout, repeat)
    else:
        seed = None
    raw = gym.make(DOORKEY)
    wrapped = public_env(
        gym.wrappers.TimeLimit(raw, max_episode_steps=EVAL_HORIZON), arm
    )
    audited = NativeDoorKeyAudit(wrapped)
    key = split + "_actions"
    meter.setdefault(key, 0)
    meter.setdefault(split + "_resets", 0)
    try:
        observation, _ = audited.reset(seed=layout)
        meter[split + "_resets"] += 1
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        terminated = truncated = False
        reward_sum = 0.0
        for _ in range(EVAL_HORIZON):
            choice, _ = agent.predict(observation, deterministic=seed is None)
            executed = int(np.asarray(choice).item())
            observation, reward, terminated, truncated, _ = audited.step(executed)
            meter[key] += 1
            reward_sum += float(reward)
            if terminated or truncated:
                break
        result = audited.summary()
        if not (terminated or truncated) or result["native_actions"] > EVAL_HORIZON:
            raise ValueError("Incomplete M10 native evaluation episode")
        if sum(result["action_counts"].values()) != result["native_actions"]:
            raise ValueError("M10 auditor action-count disagreement")
        return {
            "layout_seed": layout, "arm": arm, "model_seed": model_seed,
            "sampled_repeat": repeat, "action_rng_seed": seed,
            "native_reward_sum": reward_sum,
            "terminated": bool(terminated), "truncated": bool(truncated),
            **result,
        }
    finally:
        wrapped.close()


def suite(agent, arm: str, model_seed: int, *, split: str, sampled: bool,
          meter: dict) -> dict:
    seeds = MONITOR if split == "monitor" else HELDOUT
    rows = [
        evaluate_episode(
            agent, arm, model_seed, layout, repeat=repeat, split=split, meter=meter,
        )
        for layout in seeds
        for repeat in (range(1, REPEATS + 1) if sampled else [None])
    ]
    totals = aggregate(rows)
    expected = len(seeds) * (REPEATS if sampled else 1)
    if totals["episodes"] != expected:
        raise ValueError("M10 missing native evaluation episodes")
    if sum(totals["native_action_histogram"]) != totals["native_actions"]:
        raise ValueError("M10 evaluation native actions not conserved")
    return {"totals": totals, "episodes": rows}


def training_integrity(records: list[dict]) -> dict:
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    models = {(r["arm"], r["model_seed"]): r for r in records}
    if len(records) != len(expected) or set(models) != expected:
        raise ValueError("M10 requires six independently trained policies")
    checks = {}
    for arm, seed in expected:
        row = models[(arm, seed)]
        training = row["training"]
        checks[f"{arm}:{seed}"] = (
            row["training_real_actions"] == TRAIN_STEPS
            and row["optimizer_step_calls"] == UPDATES_PER_MODEL
            and training["native_actions"] == TRAIN_STEPS
            and training["incomplete_episode_resets"] == 0
            and training["max_episode_native_actions"] <= EVAL_HORIZON
            and sum(training["executed_action_histogram"]) == TRAIN_STEPS
            and training["first_reset_seed"] == 1_000_000 + seed * 500_000
            and row["policy_parameters"] > 0
            and row["numeric_policy"]["allow_pickle"] is False
            and bool(row["numeric_policy"]["sha256"])
        )
    if not all(checks.values()):
        raise ValueError("M10 training integrity failed before opening final split")
    return checks


def score(records: list[dict], checks: dict) -> dict:
    models = {(r["arm"], r["model_seed"]): r for r in records}

    def goals(arm, seed, mode):
        return models[(arm, seed)]["independent"][mode]["totals"][
            "verified_full_physical_goal_episodes"
        ]

    raw_sampled = sum(goals("raw_history", s, "sampled") for s in SEEDS)
    direct_sampled = sum(goals("raw_direction", s, "sampled") for s in SEEDS)
    onehot_sampled = sum(goals("onehot_direction", s, "sampled") for s in SEEDS)
    return {
        "raw_sampled_physical_goals_of_72": raw_sampled,
        "raw_direction_sampled_physical_goals_of_72": direct_sampled,
        "onehot_direction_sampled_physical_goals_of_72": onehot_sampled,
        "onehot_minus_raw_sampled": onehot_sampled - raw_sampled,
        "onehot_minus_raw_direction_sampled": onehot_sampled - direct_sampled,
        "onehot_each_model_seed_at_least_one_physical_goal": all(
            goals("onehot_direction", seed, "sampled") >= 1 for seed in SEEDS
        ),
        "training_integrity_passed": all(checks.values()),
        "primary_gate_pass": (
            onehot_sampled - raw_sampled >= 4
            and all(goals("onehot_direction", s, "sampled") >= 1 for s in SEEDS)
            and all(checks.values())
        ),
        "limitations": [
            "Three inputs differ in parameter count under equal 32-width MLPs.",
            "Sampled repeats on the same layout are correlated, not independent layouts.",
            "Single task family and two initializations do not establish generality.",
        ],
    }


def run(config_path: Path, output: Path) -> dict:
    if output.exists() or (output.parent / "progress.json").exists():
        raise FileExistsError("Never overwrite M10-P1 results or interrupted evidence")
    original = config_path.read_bytes()
    validate_config(yaml.safe_load(original))
    if output.parent.as_posix() != OUTPUT_DIR:
        raise ValueError("Unregistered M10-P1 output directory")
    source = admitted_source_commit()
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    started = time.monotonic()
    records = []
    agents = {}
    meter = {"monitor_actions": 0, "monitor_resets": 0,
             "independent_actions": 0, "independent_resets": 0}
    totals = {"training_real_actions": 0, "optimizer_step_calls": 0}
    stage = "preflight"
    in_flight_env = None
    in_flight_optimizer = None
    try:
        for arm in ARMS:
            for seed in SEEDS:
                stage = f"training-{arm}-{seed}"
                env = make_training_env(arm, seed)
                in_flight_env = env
                if env.action_space.n != 7 or env.observation_space.shape != (
                    EXPECTED_SHAPES[arm]
                ):
                    raise ValueError("M10 public image/direction input width changed")
                policy = PPO(
                    "MlpPolicy", env, seed=seed, device="cpu", n_steps=ROLLOUT,
                    batch_size=BATCH, n_epochs=EPOCHS, ent_coef=0.0,
                    policy_kwargs={"net_arch": {"pi": [32], "vf": [32]}}, verbose=0,
                )
                count = count_optimizer_steps(policy)
                in_flight_optimizer = count
                baseline = suite(
                    policy, arm, seed, split="monitor", sampled=False, meter=meter
                )
                policy.learn(total_timesteps=TRAIN_STEPS, reset_num_timesteps=False)
                if env.native_steps != TRAIN_STEPS or policy.num_timesteps != TRAIN_STEPS:
                    raise ValueError("M10 actual training interactions changed")
                if count["calls"] != UPDATES_PER_MODEL:
                    raise ValueError("M10 measured optimizer updates changed")
                measured = env.measured()
                after_det = suite(
                    policy, arm, seed, split="monitor", sampled=False, meter=meter
                )
                after_sampled = suite(
                    policy, arm, seed, split="monitor", sampled=True, meter=meter
                )
                checkpoint = numeric_state(
                    policy, output.parent / f"{arm}-seed{seed}.npz"
                )
                record = {
                    "arm": arm, "model_seed": seed,
                    "training_real_actions": env.native_steps,
                    "optimizer_step_calls": count["calls"],
                    "policy_parameters": sum(
                        p.numel() for p in policy.policy.parameters()
                    ),
                    "observation_width": EXPECTED_SHAPES[arm][0],
                    "training": measured, "numeric_policy": checkpoint,
                    "monitor_before": baseline,
                    "monitor_after_deterministic": after_det,
                    "monitor_after_sampled": after_sampled,
                }
                records.append(record)
                agents[(arm, seed)] = policy
                env.close()
                in_flight_env = None
                in_flight_optimizer = None
                totals["training_real_actions"] += record["training_real_actions"]
                totals["optimizer_step_calls"] += record["optimizer_step_calls"]
                save_progress(output.parent / "progress.json", {
                    "status": "partial_not_completed", "phase": stage,
                    "source_commit": source, **totals,
                    "monitor_actions": meter["monitor_actions"],
                    "monitor_resets": meter["monitor_resets"], "records": records,
                })
        stage = "preheldout_training_integrity"
        checks = training_integrity(records)
        if totals["training_real_actions"] != len(ARMS) * len(SEEDS) * TRAIN_STEPS:
            raise ValueError("M10 source training action total mismatch")
        if totals["optimizer_step_calls"] != (
            len(ARMS) * len(SEEDS) * UPDATES_PER_MODEL
        ):
            raise ValueError("M10 optimizer total mismatch")
        stage = "independent_heldout"
        for record in records:
            arm, seed = record["arm"], record["model_seed"]
            agent = agents[(arm, seed)]
            record["independent"] = {
                "deterministic": suite(
                    agent, arm, seed, split="independent", sampled=False,
                    meter=meter,
                ),
                "sampled": suite(
                    agent, arm, seed, split="independent", sampled=True,
                    meter=meter,
                ),
            }
            save_progress(output.parent / "progress.json", {
                "status": "partial_not_completed", "phase": stage,
                "source_commit": source, **totals,
                "monitor_actions": meter["monitor_actions"],
                "monitor_resets": meter["monitor_resets"],
                "independent_actions": meter["independent_actions"],
                "independent_resets": meter["independent_resets"], "records": records,
            })
    except BaseException as error:
        save_progress(output.parent / "progress.json", {
            "status": "interrupted_not_completed", "phase": stage,
            "failure_type": type(error).__name__, "source_commit": source,
            "completed_models": len(records),
            "in_flight_native_actions": (
                in_flight_env.native_steps if in_flight_env is not None else 0
            ),
            "in_flight_optimizer_calls": (
                in_flight_optimizer["calls"] if in_flight_optimizer is not None else 0
            ),
            "monitor_actions": meter["monitor_actions"],
            "monitor_resets": meter["monitor_resets"],
            "independent_actions": meter["independent_actions"],
            "independent_resets": meter["independent_resets"],
            **totals, "records": records,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        raise
    finally:
        if in_flight_env is not None:
            in_flight_env.close()
    result = {
        "study": "BAGI M10-P1 public representation controlled PPO comparison",
        "registered_study": "m10p1",
        "status": "completed_independent_public_representation_ablation",
        "source_commit": source,
        "config_sha256": hashlib.sha256(original).hexdigest(),
        "versions": {
            pkg: version(pkg) for pkg in (
                "gymnasium", "minigrid", "stable-baselines3"
            )
        },
        **totals,
        "monitor_actions": meter["monitor_actions"],
        "monitor_resets": meter["monitor_resets"],
        "evaluation_independent_actions": meter["independent_actions"],
        "evaluation_independent_resets": meter["independent_resets"],
        "model_fits": len(records),
        "teacher_queries": 0,
        "training_integrity_checks": checks,
        "primary_gate": score(records, checks),
        "records": records,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "limitations": [
            "One sparse MiniGrid task and two model seeds; no continual transfer.",
            "Parameter counts differ because the upstream one-hot image is wider.",
            "Repeated stochastic action samples do not create new independent layouts.",
            "Physical effects observed only by a separate native auditor.",
            "Final layouts are used once, not tuned on or recycled as monitor data.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/m10_public_representation.yaml")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/m10_public_representation/summary.json")
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("M10-P1 fresh training needs explicit --execute")
    result = run(args.config, args.output)
    print(json.dumps({
        "status": result["status"],
        "source_commit": result["source_commit"],
        "training_real_actions": result["training_real_actions"],
        "optimizer_step_calls": result["optimizer_step_calls"],
        "monitor_actions": result["monitor_actions"],
        "evaluation_independent_actions": result["evaluation_independent_actions"],
        "primary_gate": result["primary_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
