"""REUSE-9: use the existing imitation.BC learner on XLand expert primitives.

The expert's privileged map is used ONLY to generate action labels; learned
models receive XLand's standard partial image and orientation. Held-out chain
actions and states are never used to train.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from bagi.experiments.xland_action_coverage import (
    execute_putdown_oracle,
    putdown_ruleset,
)
from bagi.experiments.xland_structural_composition import (
    ORACLE_SUFFIXES,
    construct_ruleset,
    episode,
)

TRAIN_TASKS = ("hold", "near", "putdown")
ALL_TASKS = (*TRAIN_TASKS, "hold_then_near")


def visible_features(observation):
    """Only the upstream local observation and direction, no full grid or rules."""
    image = np.asarray(observation["img"], dtype=np.float32).reshape(-1)
    direction = np.asarray(observation["direction"], dtype=np.float32).reshape(-1)
    if direction.shape != (4,):
        raise ValueError("Expected upstream 4-way direction observation")
    return np.concatenate((image, direction))


def get_rules(name):
    if name == "putdown":
        return putdown_ruleset()
    if name in ("hold", "near", "hold_then_near"):
        return construct_ruleset(name)
    raise ValueError(f"Unapproved task {name}")


def expert_actions(env, params, name, seed, cap):
    if name == "putdown":
        outcome = execute_putdown_oracle(env, params, seed, cap)
    elif name in ("hold", "near"):
        outcome = episode(
            env, params, name, seed=seed,
            suffix=ORACLE_SUFFIXES[name], max_steps=cap,
        )
    else:
        raise ValueError("The held-out chain must never be demonstrated")
    if not outcome["goal_success"]:
        raise ValueError("Teacher must successfully solve the training task")
    return outcome["actions"]


def replay_visible(env, params, seed, actions):
    """Replay teacher actions, collecting ONLY what a learner would observe."""
    import jax

    reset, step = jax.jit(env.reset), jax.jit(env.step)
    timestep = reset(params, jax.random.key(seed))
    observations = []
    total_reward = 0.0
    for action in actions:
        if bool(timestep.last()):
            raise AssertionError("Expert continued after terminal state")
        observations.append(visible_features(timestep.observation))
        timestep = step(params, timestep, int(action))
        total_reward += float(timestep.reward)
    if total_reward <= 0 or not bool(timestep.last()):
        raise AssertionError("Teacher replay did not reproduce terminal reward")
    return observations


def collect(env, defaults, skills, *, budget, seed_base, cap, max_attempts):
    """Same number of expert labels across curricula; record real interactions."""
    from xminigrid.wrappers import DirectionObservationWrapper

    if not skills or not set(skills).issubset(TRAIN_TASKS):
        raise ValueError("Cannot demonstrate held-out or unknown tasks")
    if min(budget, cap, max_attempts) < 1:
        raise ValueError("Positive expert budget, episode cap and attempts required")
    wrapped = DirectionObservationWrapper(env)
    observations, labels = [], []
    attempts = rejected = generator_steps = replay_steps = 0
    per_skill_episodes = Counter()
    expert_action_support = {skill: set() for skill in skills}
    while len(labels) < budget and attempts < max_attempts:
        skill = skills[attempts % len(skills)]
        seed = seed_base + attempts // len(skills)
        attempts += 1
        params = defaults.replace(ruleset=get_rules(skill), max_steps=cap)
        try:
            actions = expert_actions(env, params, skill, seed, cap)
            views = replay_visible(wrapped, params, seed, actions)
        except ValueError:
            rejected += 1
            continue
        generator_steps += len(actions)
        replay_steps += len(actions)
        per_skill_episodes[skill] += 1
        expert_action_support[skill].update(actions)
        count = min(len(actions), budget - len(labels))
        observations.extend(views[:count])
        labels.extend(actions[:count])
    if len(labels) != budget:
        raise RuntimeError("Insufficient teacher demonstrations in fixed seed range")
    data = {
        "obs": np.asarray(observations, dtype=np.float32),
        "acts": np.asarray(labels, dtype=np.int64),
    }
    audit = {
        "used_labeled_examples": budget,
        "successful_expert_episodes": dict(per_skill_episodes),
        "successful_teacher_generator_real_steps": generator_steps,
        "successful_teacher_replay_real_steps": replay_steps,
        "collection_attempts": attempts,
        "rejected_seeds": rejected,
        "rejected_seed_interactions_not_counted": True,
        "expert_action_support": {
            k: sorted(values) for k, values in expert_action_support.items()
        },
        "used_label_action_counts": dict(Counter(labels)),
        "demonstration_seed_first": seed_base,
        "demonstration_seed_max_tried": seed_base + (attempts - 1) // len(skills),
    }
    return data, audit


def evaluate(policy, env, defaults, seeds, cap):
    """Closed-loop independent-seed evaluation without querying any teacher."""
    import jax

    reset, step = jax.jit(env.reset), jax.jit(env.step)
    output = {}
    for skill in ALL_TASKS:
        params = defaults.replace(ruleset=get_rules(skill), max_steps=cap)
        outcomes = []
        for seed in seeds:
            timestep = reset(params, jax.random.key(seed))
            steps = successes = 0
            action_counts = Counter()
            while steps < cap and not bool(timestep.last()):
                obs = visible_features(timestep.observation)
                choice, _ = policy.predict(obs, deterministic=True)
                action = int(np.asarray(choice).item())
                if action not in range(6):
                    raise AssertionError("Model output is not a native XLand action")
                timestep = step(params, timestep, action)
                action_counts[action] += 1
                successes |= int(float(timestep.reward) > 0)
                steps += 1
            outcomes.append({
                "seed": seed, "goal_success": successes,
                "real_steps": steps, "action_counts": dict(action_counts),
            })
        output[skill] = {
            "successes": sum(x["goal_success"] for x in outcomes),
            "episodes": len(outcomes),
            "evaluation_steps": sum(x["real_steps"] for x in outcomes),
            "details": outcomes,
        }
    return output


def validate_config(cfg):
    budget = int(cfg["demonstration_labels_per_curriculum"])
    cap = int(cfg["episode_cap"])
    eval_seeds = [int(x) for x in cfg["evaluation_seeds"]]
    model_seeds = [int(x) for x in cfg["policy_training_seeds"]]
    if (budget < int(cfg["batch_size"]) or cap < 1
            or int(cfg["epochs"]) < 1 or int(cfg["max_collection_attempts"]) < 1
            or not eval_seeds or not model_seeds
            or len(eval_seeds) != len(set(eval_seeds))
            or len(model_seeds) != len(set(model_seeds))):
        raise ValueError("Require positive sufficient budgets and distinct seeds")
    base = int(cfg["demonstration_seed_base"])
    if any(base <= seed <= base + int(cfg["max_collection_attempts"])
           for seed in eval_seeds):
        raise ValueError("Evaluation and demonstration seeds overlap")
    if cfg["environment"] != "XLand-MiniGrid-R1-9x9":
        raise ValueError("Only reviewed XLand single-room environment")
    return budget, cap, eval_seeds, model_seeds


def run(config_path):
    import gymnasium as gym
    import torch
    import xminigrid
    from imitation.algorithms.bc import BC
    from imitation.data.types import TransitionsMinimal
    from xminigrid.wrappers import DirectionObservationWrapper

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    budget, cap, eval_seeds, model_seeds = validate_config(cfg)
    torch.set_num_threads(1)
    env, defaults = xminigrid.make(cfg["environment"])
    wrapper = DirectionObservationWrapper(env)
    curricula = {
        "two_skills": ("hold", "near"),
        "three_skills": TRAIN_TASKS,
    }
    samples = {
        name: collect(
            env, defaults, skills, budget=budget,
            seed_base=int(cfg["demonstration_seed_base"]), cap=cap,
            max_attempts=int(cfg["max_collection_attempts"]),
        )
        for name, skills in curricula.items()
    }
    width = samples["two_skills"][0]["obs"].shape[1]
    if width != samples["three_skills"][0]["obs"].shape[1]:
        raise AssertionError("Policy observation shape differs across curricula")
    observation_space = gym.spaces.Box(
        low=0.0, high=255.0, shape=(width,), dtype=np.float32
    )
    action_space = gym.spaces.Discrete(env.num_actions(defaults))
    if action_space.n != 6:
        raise AssertionError("Unexpected XLand native action count")
    experiments = []
    for seed in model_seeds:
        for name, (dataset, audit) in samples.items():
            torch.manual_seed(seed)
            infos = np.empty(budget, dtype=object)
            infos.fill(None)
            trainer = BC(
                observation_space=observation_space, action_space=action_space,
                demonstrations=TransitionsMinimal(
                    obs=dataset["obs"], acts=dataset["acts"], infos=infos,
                ),
                rng=np.random.default_rng(seed),
                batch_size=int(cfg["batch_size"]), device="cpu",
            )
            before = evaluate(trainer.policy, wrapper, defaults, eval_seeds, cap)
            start = time.monotonic()
            trainer.train(n_epochs=int(cfg["epochs"]), progress_bar=False)
            seconds = round(time.monotonic() - start, 3)
            predicted, _ = trainer.policy.predict(dataset["obs"], deterministic=True)
            supervised_accuracy = float(np.mean(
                np.asarray(predicted).reshape(-1) == dataset["acts"]
            ))
            after = evaluate(trainer.policy, wrapper, defaults, eval_seeds, cap)
            experiments.append({
                "curriculum": name, "model_seed": seed,
                "training_examples": budget, "epochs": int(cfg["epochs"]),
                "training_seconds": seconds,
                "training_label_accuracy": supervised_accuracy,
                "before_training": before, "after_training": after,
            })
            print(f"{name} seed={seed} train_label_accuracy={supervised_accuracy:.3f} "
                  f"primitive_successes={[after[k]['successes'] for k in TRAIN_TASKS]} "
                  f"heldout={after['hold_then_near']['successes']}/{len(eval_seeds)}")
    result = {
        "question": "Does established BC acquire the individual training skills and transfer to a frozen unseen causal chain?",
        "established_method": "HumanCompatibleAI imitation.BC, not custom training",
        "packages": {"imitation": version("imitation"),
                     "xminigrid": version("xminigrid")},
        "training_task_sets": curricula,
        "held_out_never_demonstrated": "hold_then_near",
        "config": cfg,
        "dataset_audit": {name: meta for name, (_, meta) in samples.items()},
        "results": experiments,
        "limitations": [
            "Only a memoryless BC skill-acquisition comparison, not a recurrent memory test.",
            "Teacher uses privileged geometry only for ACTION LABELS; model inputs are visual image and heading.",
            "Matched labeled expert examples but unmatched expert acquisition simulator steps; both are reported.",
            "Rejected-source oracle generation interactions are not counted; do not claim matched real collection costs.",
            "Task descriptions and hidden rule IDs are not given to the model; ambiguity may prevent task-independent learning.",
            "Only a few evaluation seeds; no broad causal generalization claim.",
            "Offline BC may compound off-expert mistakes; established DAgger is a follow-up if training labels fit but success is poor.",
        ],
    }
    path = Path(cfg["output_dir"]) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved BC experiment: {path}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/xland_existing_bc.yaml")
    run(p.parse_args().config)
