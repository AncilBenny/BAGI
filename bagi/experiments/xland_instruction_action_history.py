"""REUSE-23: matched actual-action-history BC with upstream public instructions.

Both existing imitation.BC policies receive XLand image, direction, public
goal/rule tokens and four seven-way one-hot history slots. One condition fills
the history slots with real executed actions from a provenance-verified DAgger
pool and updates them with its OWN actions at deployment. The control receives
only four constant PAD actions, with the same input width/network/optimization.
No new teacher labels, DAgger rollouts, or custom learner architecture.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from bagi.experiments.xland_action_provenance import PAD_ACTION
from bagi.experiments.xland_bc_class_balance import action_recall, balanced_indices
from bagi.experiments.xland_existing_bc import ALL_TASKS, TRAIN_TASKS, get_rules, visible_features
from bagi.experiments.xland_upstream_instruction_bc import (
    INPUT_WIDTH,
    load_verified_source,
    task_instructions,
)

HISTORY_LENGTH = 4
ACTIONS_WITH_PAD = 7
WIDTH = INPUT_WIDTH + HISTORY_LENGTH * ACTIONS_WITH_PAD
CONDITIONS = ("sham_action_history", "real_executed_action_history")
EVAL_SEEDS = (9600, 9601, 9602, 9603, 9604)


def check_true_history(data):
    """Independently reconstruct preceding REAL actions, never expert labels."""
    history = []
    last_episode = None
    for i in range(len(data["expert_label"])):
        id_pair = (int(data["round"][i]), int(data["episode"][i]))
        if id_pair != last_episode:
            history = []
            last_episode = id_pair
        if int(data["step_in_episode"][i]) != len(history):
            raise ValueError("Non-contiguous source episode history")
        expectation = ([PAD_ACTION] * HISTORY_LENGTH + history)[-HISTORY_LENGTH:]
        if not np.array_equal(data["prior_executed_actions"][i], expectation):
            raise ValueError("Prior learner actions disagree with actual execution")
        history.append(int(data["executed_action"][i]))
        if bool(data["done"][i]):
            last_episode = None


def encode_context(histories, *, condition):
    """Both conditions use 28 history slots and identical policy dimensions."""
    if condition not in CONDITIONS:
        raise ValueError("Unapproved history information condition")
    past = np.asarray(histories, dtype=np.int64)
    if past.ndim != 2 or past.shape[1] != HISTORY_LENGTH:
        raise ValueError("Exactly four previous action IDs required")
    if np.any((past < 0) | (past > PAD_ACTION)):
        raise ValueError("Invalid native action or PAD in history")
    if condition == "sham_action_history":
        past = np.full_like(past, PAD_ACTION)
    return np.eye(ACTIONS_WITH_PAD, dtype=np.float32)[past].reshape(len(past), -1)


def assemble(local, instructions, histories, condition):
    base = np.asarray(local, dtype=np.float32)
    instruction = np.asarray(instructions, dtype=np.float32)
    if (base.ndim != 2 or base.shape[1] != 54
            or instruction.shape != (len(base), INPUT_WIDTH - 54)):
        raise ValueError("Wrong learner-visible local or task instructions")
    context = encode_context(histories, condition=condition)
    if len(context) != len(base):
        raise ValueError("History row count differs from training labels")
    result = np.concatenate((base, instruction, context), axis=1)
    if result.shape != (len(base), WIDTH) or not np.isfinite(result).all():
        raise ValueError("Invalid upstream BC input size or values")
    return result


def evaluate(policy, instructed, defaults, condition):
    import jax

    reset, step = jax.jit(instructed.reset), jax.jit(instructed.step)
    result = {}
    for task in ALL_TASKS:
        params = defaults.replace(ruleset=get_rules(task), max_steps=40)
        cases = []
        for seed in EVAL_SEEDS:
            time_step = reset(params, jax.random.key(seed))
            previous = [PAD_ACTION] * HISTORY_LENGTH
            steps = success = 0
            actions = Counter()
            while steps < 40 and not bool(time_step.last()):
                visible = visible_features(time_step.observation)[None, :]
                obs = time_step.observation
                instr = task_instructions(
                    obs["goal_encoding"], obs["rule_encoding"], enabled=True
                )[None, :]
                policy_input = assemble(
                    visible,
                    instr,
                    np.asarray([previous], dtype=np.int64),
                    condition,
                )
                selected, _ = policy.predict(policy_input[0], deterministic=True)
                action = int(np.asarray(selected).item())
                if action not in range(6):
                    raise AssertionError("Non-native XLand action from learned policy")
                time_step = step(params, time_step, action)
                previous = [*previous[1:], action]
                steps += 1
                actions[action] += 1
                success |= int(float(time_step.reward) > 0)
            cases.append({
                "seed": seed, "goal_success": success,
                "real_steps": steps, "executed_action_counts": dict(actions),
            })
        result[task] = {
            "successes": sum(x["goal_success"] for x in cases),
            "episodes": len(cases),
            "evaluation_simulator_steps": sum(x["real_steps"] for x in cases),
            "cases": cases,
        }
    return result


def run(config_path):
    import gymnasium as gym
    import torch
    import xminigrid
    from imitation.algorithms.bc import BC
    from imitation.data.types import Transitions
    from xminigrid.wrappers import (
        DirectionObservationWrapper,
        RulesAndGoalsObservationWrapper,
    )

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if (cfg["environment"] != "XLand-MiniGrid-R1-9x9"
            or cfg["train_tasks"] != list(TRAIN_TASKS)
            or cfg["model_seeds"] != [17, 29]
            or cfg["conditions"] != list(CONDITIONS)
            or cfg["evaluation_seeds"] != list(EVAL_SEEDS)
            or int(cfg["source_run"]) != 39637067506
            or int(cfg["history_length"]) != HISTORY_LENGTH
            or int(cfg["input_width"]) != WIDTH
            or int(cfg["batch_size"]) != 32
            or int(cfg["epochs"]) != 12
            or int(cfg["episode_cap"]) != 40):
        raise ValueError("Frozen REUSE-23 history control contract changed")
    source_dir = Path(cfg["source_dir"])
    summary = json.loads(
        (source_dir / "summary.json").read_text(encoding="utf-8")
    )
    if summary["heldout_teacher_labels"] or summary["total_expert_queries"] != 582:
        raise ValueError("Unreviewed REUSE-19 label provenance")
    torch.set_num_threads(1)
    room, defaults = xminigrid.make(cfg["environment"])
    instructed = RulesAndGoalsObservationWrapper(DirectionObservationWrapper(room))
    training_instruction = {
        name: task_instructions(
            get_rules(name).goal, get_rules(name).rules, enabled=True
        )
        for name in TRAIN_TASKS
    }
    all_models = []
    for seed in cfg["model_seeds"]:
        meta = [x for x in summary["model_results"] if x["model_seed"] == seed]
        if len(meta) != 1:
            raise ValueError("Missing unique verified REUSE-19 policy seed")
        data, source_tasks, checksum = load_verified_source(
            source_dir / f"action_provenance_seed{seed}.npz",
            meta[0], seed
        )
        if len(data["expert_label"]) != 291:
            raise ValueError("Source learner state pool does not match REUSE-19")
        check_true_history(data)
        codes = np.stack([training_instruction[name] for name in source_tasks])
        sampled = balanced_indices(data["expert_label"], seed)
        conditions = []
        for condition in CONDITIONS:
            features = assemble(
                data["obs"], codes, data["prior_executed_actions"], condition
            )
            torch.manual_seed(seed)
            bc = BC(
                observation_space=gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(WIDTH,), dtype=np.float32
                ),
                action_space=gym.spaces.Discrete(6),
                demonstrations=Transitions(
                    obs=features[sampled],
                    acts=data["expert_label"][sampled],
                    # BC supervised loss reads obs/acts only. No true successor
                    # was retained in REUSE-19, so this unused field is explicit.
                    next_obs=features[sampled].copy(),
                    dones=np.asarray(data["done"][sampled], dtype=bool),
                    infos=np.array([None] * len(sampled), dtype=object),
                ),
                rng=np.random.default_rng(seed),
                batch_size=32, device="cpu",
            )
            bc.train(n_epochs=12, progress_bar=False)
            prediction, _ = bc.policy.predict(features, deterministic=True)
            predicted = np.asarray(prediction, dtype=np.int64).reshape(-1)
            independent = evaluate(bc.policy, instructed, defaults, condition)
            conditions.append({
                "condition": condition,
                "data_sha256": checksum,
                "network_input_width": WIDTH,
                "dataset_rows": len(data["expert_label"]),
                "distinct_sampled_training_rows": len(np.unique(sampled)),
                "optimization_sample_exposures": 12 * len(sampled),
                "training_label_accuracy_original_pool": float(np.mean(
                    predicted == data["expert_label"]
                )),
                "training_action_recall_original_pool": action_recall(
                    predicted, data["expert_label"]
                ),
                "independent_evaluation": independent,
                "real_independent_evaluation_steps": sum(
                    item["evaluation_simulator_steps"]
                    for item in independent.values()
                ),
            })
            print(f"REUSE-23 seed {seed} {condition}: "
                  f"hold={independent['hold']['successes']}/5 "
                  f"near={independent['near']['successes']}/5 "
                  f"putdown={independent['putdown']['successes']}/5 "
                  f"heldout={independent['hold_then_near']['successes']}/5")
        if conditions[0]["data_sha256"] != conditions[1]["data_sha256"]:
            raise AssertionError("Comparison source pools must be identical")
        all_models.append({
            "model_seed": seed,
            "verified_numeric_label_pool_sha256": checksum,
            "conditions": conditions,
        })
    result = {
        "question": "Does true four-action execution history add closed-loop source-skill value on top of public XLand task instructions?",
        "established_method": "imitation.BC + PyTorch WeightedRandomSampler + upstream XLand instruction wrapper",
        "source_run": 39637067506,
        "same_model_architecture_and_training_rows": True,
        "both_receive_public_rules_and_goals": True,
        "only_controlled_difference": "Actual previous four executed actions versus constant four PADs",
        "new_privileged_teacher_queries": 0,
        "new_online_training_environment_steps": 0,
        "privileged_runtime_policy_features": 0,
        "heldout_expert_actions_used_for_training": 0,
        "eval_seeds": list(EVAL_SEEDS),
        "real_evaluation_simulator_steps": sum(
            row["real_independent_evaluation_steps"]
            for model in all_models for row in model["conditions"]
        ),
        "models": all_models,
        "limitations": [
            "History is four ACTUAL prior actions only; no hidden inventory state or full grid enters the learner.",
            "Action history may lose a pickup event that occurred more than four actions ago; this is not a general test of recurrence.",
            "Two model seeds and five independent layout seeds limit robustness and statistical claims.",
            "Original pooled imitation targets are expert labels, not learner actions; source provenance checks are mandatory.",
            "Both conditions get identical upstream public task instructions and identical-width 115-feature network architecture.",
            "Copied next_obs is explicitly nonphysical and solely for upstream supervised BC collator compatibility; never used as transitions.",
            "Heldout chain is evaluated without heldout teacher training actions; do not claim transfer before reliable source task skill.",
            "Rare-action recall is training-pool descriptive, not closed-loop causal competence.",
        ],
    }
    dest = Path(cfg["output_dir"]) / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Saved REUSE-23 matched action-history evidence")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/xland_instruction_action_history.yaml")
    run(p.parse_args().config)
