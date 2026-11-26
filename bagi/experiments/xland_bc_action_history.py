"""REUSE-13: established class-balanced imitation.BC with 4 prior actions.

Only XLand's ordinary local image+direction and the agent's OWN past actions
reach its policy. Privileged expert control is used for TRAIN-ONLY labels.
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

from bagi.experiments.xland_bc_class_balance import action_recall, balanced_indices
from bagi.experiments.xland_existing_bc import (
    ALL_TASKS,
    TRAIN_TASKS,
    collect,
    get_rules,
    validate_config,
    visible_features,
)

HISTORY_LENGTH = 4
NATIVE_ACTIONS = 6


def encode_history(previous_actions):
    """Zero padding means no previous action, distinct from action zero."""
    if len(previous_actions) > HISTORY_LENGTH:
        raise ValueError("History exceeds fixed four-action memory")
    output = np.zeros((HISTORY_LENGTH, NATIVE_ACTIONS), dtype=np.float32)
    offset = HISTORY_LENGTH - len(previous_actions)
    for pos, action in enumerate(previous_actions):
        if int(action) not in range(NATIVE_ACTIONS):
            raise ValueError("Unexpected XLand action in history")
        output[offset + pos, int(action)] = 1.0
    return output.reshape(-1)


def augment_training_observations(original):
    """Encode only past actions for the current input, reset after episode end."""
    obs = np.asarray(original["obs"], dtype=np.float32)
    next_obs = np.asarray(original["next_obs"], dtype=np.float32)
    actions = np.asarray(original["acts"], dtype=np.int64)
    dones = np.asarray(original["dones"], dtype=bool)
    if (obs.ndim != 2 or next_obs.shape != obs.shape
            or actions.shape != (len(obs),) or dones.shape != (len(obs),)
            or not np.isfinite(obs).all() or not np.isfinite(next_obs).all()):
        raise ValueError("Expected aligned finite expert transitions")
    previous = []
    inputs, next_inputs = [], []
    for view, next_view, act, done in zip(obs, next_obs, actions, dones):
        inputs.append(np.concatenate([view, encode_history(previous)]))
        updated = (*previous, int(act))[-HISTORY_LENGTH:]
        next_inputs.append(np.concatenate([next_view, encode_history(updated)]))
        previous = [] if done else list(updated)
    return np.asarray(inputs, dtype=np.float32), np.asarray(next_inputs, dtype=np.float32)


def evaluate_history(policy, env, defaults, expert_inputs, seeds, cap):
    """Closed loop: augment local observations using learner actions only."""
    import jax

    reset, step = jax.jit(env.reset), jax.jit(env.step)
    known = {
        np.ascontiguousarray(row).tobytes()
        for row in np.asarray(expert_inputs, dtype=np.float32)
    }
    evaluations = {}
    for task in ALL_TASKS:
        params = defaults.replace(ruleset=get_rules(task), max_steps=cap)
        episodes = []
        for seed in seeds:
            timestep = reset(params, jax.random.key(seed))
            previous = []
            count = Counter()
            unseen = steps = goal = 0
            first_unseen_step = None
            while steps < cap and not bool(timestep.last()):
                current = np.concatenate([
                    visible_features(timestep.observation),
                    encode_history(previous),
                ])
                if np.ascontiguousarray(current).tobytes() not in known:
                    unseen += 1
                    if first_unseen_step is None:
                        first_unseen_step = steps
                chosen, _ = policy.predict(current, deterministic=True)
                action = int(np.asarray(chosen).item())
                if action not in range(NATIVE_ACTIONS):
                    raise AssertionError("BC selected an invalid native action")
                timestep = step(params, timestep, action)
                count[action] += 1
                previous = [*previous, action][-HISTORY_LENGTH:]
                goal |= int(float(timestep.reward) > 0)
                steps += 1
            episodes.append({
                "seed": seed, "goal_success": goal, "real_steps": steps,
                "action_counts": dict(count),
                "off_expert_exact_input_rows": unseen,
                "first_off_expert_step_zero_indexed": first_unseen_step,
            })
        evaluations[task] = {
            "successes": sum(ep["goal_success"] for ep in episodes),
            "episodes": len(episodes),
            "real_steps": sum(ep["real_steps"] for ep in episodes),
            "off_expert_exact_input_rows": sum(
                ep["off_expert_exact_input_rows"] for ep in episodes
            ),
            "details": episodes,
        }
    return evaluations


def run(config_path: str) -> dict:
    import gymnasium as gym
    import torch
    import xminigrid
    from imitation.algorithms.bc import BC
    from imitation.data.types import Transitions
    from xminigrid.wrappers import DirectionObservationWrapper

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    source_path = Path(cfg["source_config"])
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    budget, cap, seeds, model_seeds = validate_config(source)
    if (cfg["training_tasks"] != list(TRAIN_TASKS)
            or int(cfg["previous_action_count"]) != HISTORY_LENGTH
            or int(cfg["training_labels"]) != 256 or budget != 256):
        raise ValueError("REUSE-13 must use the frozen training-only curriculum and history")
    torch.set_num_threads(1)
    env, defaults = xminigrid.make(source["environment"])
    visible_env = DirectionObservationWrapper(env)
    examples, collection = collect(
        env, defaults, TRAIN_TASKS, budget=budget,
        seed_base=int(source["demonstration_seed_base"]), cap=cap,
        max_attempts=int(source["max_collection_attempts"]),
    )
    expected_action_counts = {0: 157, 1: 36, 2: 23, 3: 30, 4: 10}
    if dict(Counter(map(int, examples["acts"]))) != expected_action_counts:
        raise AssertionError("REUSE-10 frozen demonstration action pool changed")
    augmented, augmented_next = augment_training_observations(examples)
    if augmented.shape[1] != examples["obs"].shape[1] + 4 * NATIVE_ACTIONS:
        raise AssertionError("Incorrect observed-feature width")
    space = gym.spaces.Box(
        low=0.0, high=255.0, shape=augmented.shape[1:], dtype=np.float32
    )
    actions = gym.spaces.Discrete(env.num_actions(defaults))
    if actions.n != NATIVE_ACTIONS:
        raise AssertionError("Native XLand action count changed")
    runs = []
    for seed in model_seeds:
        indices = balanced_indices(examples["acts"], seed)
        torch.manual_seed(seed)
        batch = Transitions(
            obs=augmented[indices], acts=examples["acts"][indices],
            next_obs=augmented_next[indices], dones=examples["dones"][indices],
            infos=np.array([None] * budget, dtype=object),
        )
        trainer = BC(
            observation_space=space, action_space=actions,
            demonstrations=batch, rng=np.random.default_rng(seed),
            batch_size=int(source["batch_size"]), device="cpu",
        )
        started = time.monotonic()
        trainer.train(n_epochs=int(source["epochs"]), progress_bar=False)
        seconds = round(time.monotonic() - started, 3)
        predicted, _ = trainer.policy.predict(augmented, deterministic=True)
        predicted = np.asarray(predicted).reshape(-1)
        outcomes = evaluate_history(
            trainer.policy, visible_env, defaults, augmented, seeds, cap
        )
        scores = {
            "unweighted_original_expert_label_accuracy":
                float(np.mean(predicted == examples["acts"])),
            "unweighted_original_expert_action_recall":
                action_recall(predicted, examples["acts"]),
        }
        runs.append({
            "policy_seed": seed,
            "supervised_fit": scores,
            "distinct_demonstration_rows_sampled": len(np.unique(indices)),
            "optimization_exposures": budget * int(source["epochs"]),
            "train_seconds": seconds,
            "closed_loop": outcomes,
        })
        print(f"historyBC seed={seed}: "
              f"primitive={[outcomes[k]['successes'] for k in TRAIN_TASKS]} "
              f"heldout={outcomes['hold_then_near']['successes']}/{len(seeds)}")
    output = {
        "question": "Does established class-balanced BC with four prior own actions acquire training skills?",
        "contribution": "Feature integration with upstream imitation.BC; no new optimizer/model",
        "packages": {"imitation": version("imitation"),
                     "xminigrid": version("xminigrid")},
        "source_config": str(source_path),
        "heldout_training_examples": 0,
        "policy_features": "XLand local image+direction plus four one-hot past actions",
        "policy_history_source": "expert previous actions for training; OWN actions at evaluation",
        "collection": collection,
        "source_examples": budget,
        "evaluation_seeds": seeds,
        "model_seeds": model_seeds,
        "reference": "REUSE-10 memoryless class-balanced BC, 0/3 on each task and model seed",
        "models": runs,
        "limitations": [
            "History feature stacking is not a recurrent network and does not prove causal intelligence.",
            "Input width changes, so policy parameter count is not identical to memoryless BC.",
            "Exact source-state coverage is diagnostic; unseen evaluation seeds can start off-pool.",
            "No heldout expert labels or privileged runtime features enter the policy.",
            "Expert and evaluation simulator steps are reported, while policy optimization uses zero simulator steps.",
            "Only two model seeds and three evaluation seeds; no statistical generalization claim.",
        ],
    }
    dest = Path(cfg["output_dir"]) / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved history-based BC evidence: {dest}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xland_bc_action_history.yaml")
    run(parser.parse_args().config)
