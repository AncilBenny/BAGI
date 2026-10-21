"""REUSE-21: upstream task-instruction input against same-width sham BC.

Reuses REUSE-19 verified DAgger numeric expert labels, imitation.BC,
torch WeightedRandomSampler and XLand RulesAndGoalsObservationWrapper.
No new teacher labels, altered XLand simulator, or model algorithm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from bagi.experiments.xland_bc_class_balance import action_recall, balanced_indices
from bagi.experiments.xland_existing_bc import ALL_TASKS, TRAIN_TASKS, get_rules, visible_features

MAX_RULES = 4
INPUT_WIDTH = 54 + 5 + MAX_RULES * 7
CONDITIONS = ("sham_instructions", "upstream_instructions")
EVAL_SEEDS = (9500, 9501, 9502, 9503, 9504)


def task_instructions(goal, rules, *, enabled):
    """Only upstream public task encodings; pad absent rules, never full grid."""
    goal = np.asarray(goal, dtype=np.float32)
    rules = np.asarray(rules, dtype=np.float32)
    if goal.shape != (5,) or rules.ndim != 2 or rules.shape[1] != 7:
        raise ValueError("Unexpected upstream task-instruction shape")
    if not 1 <= len(rules) <= MAX_RULES or not np.isfinite(rules).all():
        raise ValueError("Unreviewed task rule count or values")
    if not np.isfinite(goal).all():
        raise ValueError("Nonfinite task goal")
    out = np.concatenate((
        goal, np.pad(rules, ((0, MAX_RULES - len(rules)), (0, 0))).reshape(-1)
    ))
    if not enabled:
        out = np.zeros_like(out)
    assert out.shape == (5 + MAX_RULES * 7,)
    return out


def observation_with_instruction(local, encoding):
    local = np.asarray(local, dtype=np.float32)
    encoding = np.asarray(encoding, dtype=np.float32)
    if local.shape != (54,) or encoding.shape != (5 + MAX_RULES * 7,):
        raise ValueError("Wrong source observation or instruction size")
    obs = np.concatenate((local, encoding)).astype(np.float32)
    assert obs.shape == (INPUT_WIDTH,)
    return obs


def load_verified_source(path, metadata, seed):
    """Verify previous collection provenance before assigning task instructions."""
    with np.load(path, allow_pickle=False) as file:
        required = {"obs", "expert_label", "executed_action",
                    "prior_executed_actions", "round", "episode",
                    "step_in_episode", "done"}
        if set(file.files) != required:
            raise ValueError("Unexpected REUSE-19 numeric fields")
        data = {k: np.array(file[k], copy=True) for k in file.files}
    checksum = hashlib.sha256()
    for name in sorted(data):
        if data[name].dtype.kind not in ("b", "f", "i", "u"):
            raise ValueError("Untrusted record dtype")
        checksum.update(name.encode())
        checksum.update(np.ascontiguousarray(data[name]).tobytes())
    if (metadata["model_seed"] != seed
            or not metadata["stored_teacher_labels_match_logged_visits"]
            or checksum.hexdigest() != metadata["numeric_record_sha256"]):
        raise ValueError("Numeric source verification failed")
    if (len(data["expert_label"]) != 291
            or data["obs"].shape != (291, 54)
            or data["obs"].dtype != np.float32
            or data["episode"].shape != (291,)
            or data["expert_label"].shape != (291,)
            or np.any((data["expert_label"] < 0) | (data["expert_label"] >= 6))):
        raise ValueError("Unreviewed observed task source labels")
    # The upstream TrainingXLandGym source rotates exactly these train tasks,
    # with each episode index starting at 0. Never use heldout task in data.
    tasks = [TRAIN_TASKS[int(episode) % 3] for episode in data["episode"]]
    if set(tasks) != set(TRAIN_TASKS):
        raise ValueError("Train data missing reviewed task")
    return data, tasks, checksum.hexdigest()


def evaluate(policy, env, defaults, condition):
    import jax

    reset, step = jax.jit(env.reset), jax.jit(env.step)
    report = {}
    for task in ALL_TASKS:
        params = defaults.replace(ruleset=get_rules(task), max_steps=40)
        cases = []
        for seed in EVAL_SEEDS:
            ts = reset(params, jax.random.key(seed))
            actions = Counter()
            steps = success = 0
            while steps < 40 and not bool(ts.last()):
                o = ts.observation
                encoding = task_instructions(
                    o["goal_encoding"], o["rule_encoding"],
                    enabled=condition == "upstream_instructions"
                )
                features = observation_with_instruction(visible_features(o), encoding)
                choice, _ = policy.predict(features, deterministic=True)
                action = int(np.asarray(choice).item())
                if action not in range(6):
                    raise ValueError("Model returned non-native XLand action")
                ts = step(params, ts, action)
                steps += 1
                actions[action] += 1
                success |= int(float(ts.reward) > 0)
            cases.append({"seed": seed, "reward_success": success, "real_steps": steps,
                          "actions": dict(actions)})
        report[task] = {
            "successes": sum(r["reward_success"] for r in cases),
            "episodes": len(cases),
            "real_steps": sum(r["real_steps"] for r in cases),
            "cases": cases,
        }
    return report


def run(config_path):
    import gymnasium as gym
    import torch
    import xminigrid
    from imitation.algorithms.bc import BC
    from imitation.data.types import TransitionsMinimal
    from xminigrid.wrappers import (
        DirectionObservationWrapper,
        RulesAndGoalsObservationWrapper,
    )

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if (cfg["environment"] != "XLand-MiniGrid-R1-9x9"
            or cfg["training_tasks"] != list(TRAIN_TASKS)
            or cfg["model_seeds"] != [17, 29]
            or cfg["conditions"] != list(CONDITIONS)
            or cfg["evaluation_seeds"] != list(EVAL_SEEDS)
            or cfg["source_run_id"] != 39637067506
            or cfg["batch_size"] != 32 or cfg["epochs"] != 12
            or cfg["episode_cap"] != 40 or cfg["max_rule_slots"] != MAX_RULES):
        raise ValueError("Frozen task-instruction intervention was changed")
    with (Path(cfg["source_dir"]) / "summary.json").open(
        encoding="utf-8"
    ) as fh:
        summary = json.load(fh)
    if summary["heldout_teacher_labels"] or summary["total_expert_queries"] != 582:
        raise ValueError("Unreviewed source teacher interactions")
    torch.set_num_threads(1)
    base, defaults = xminigrid.make(cfg["environment"])
    instructed = RulesAndGoalsObservationWrapper(DirectionObservationWrapper(base))
    # The new learner's goal/rule values use only upstream PUBLIC encodings.
    training_codes = {}
    for task in TRAIN_TASKS:
        ruleset = get_rules(task)
        training_codes[task] = task_instructions(
            ruleset.goal, ruleset.rules, enabled=True
        )
    models = []
    for seed in (17, 29):
        candidates = [m for m in summary["model_results"] if m["model_seed"] == seed]
        if len(candidates) != 1:
            raise ValueError("Missing unique original action-provenance seed")
        data, names, digest = load_verified_source(
            Path(cfg["source_dir"]) / f"action_provenance_seed{seed}.npz",
            candidates[0], seed
        )
        drawn = balanced_indices(data["expert_label"], seed)
        per_condition = []
        for condition in CONDITIONS:
            active = condition == "upstream_instructions"
            observations = np.stack([
                observation_with_instruction(
                    data["obs"][i], training_codes[names[i]] if active
                    else np.zeros(5 + MAX_RULES * 7, dtype=np.float32)
                )
                for i in range(len(data["expert_label"]))
            ])
            torch.manual_seed(seed)
            trainer = BC(
                observation_space=gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(INPUT_WIDTH,), dtype=np.float32),
                action_space=gym.spaces.Discrete(6),
                demonstrations=TransitionsMinimal(
                    obs=observations[drawn], acts=data["expert_label"][drawn],
                    infos=np.array([None] * len(drawn), dtype=object)
                ),
                rng=np.random.default_rng(seed),
                batch_size=32, device="cpu",
            )
            trainer.train(n_epochs=12, progress_bar=False)
            predicted, _ = trainer.policy.predict(
                observations, deterministic=True
            )
            predicted = np.asarray(predicted).reshape(-1)
            rewards = evaluate(trainer.policy, instructed, defaults, condition)
            per_condition.append({
                "condition": condition,
                "expert_pool_sha256": digest,
                "pool_rows": len(drawn),
                "unique_sampled_rows": len(np.unique(drawn)),
                "optimization_exposures": len(drawn) * 12,
                "network_input_width": INPUT_WIDTH,
                "training_label_accuracy": float(
                    np.mean(predicted == data["expert_label"])
                ),
                "original_pool_action_recall": action_recall(
                    predicted, data["expert_label"]
                ),
                "closed_loop": rewards,
                "real_independent_evaluation_steps": sum(
                    item["real_steps"] for item in rewards.values()
                ),
            })
            print(f"seed={seed} {condition} source goals="
                  f"{[rewards[t]['successes'] for t in TRAIN_TASKS]} "
                  f"heldout={rewards['hold_then_near']['successes']}/5")
        models.append({"policy_seed": seed, "source_sha256": digest,
                       "conditions": per_condition})
    report = {
        "question": "Does STANDARD UPSTREAM task instruction information help basic source-skill acquisition?",
        "method": "Unchanged imitation.BC and WeightedRandomSampler, sham vs upstream rule/goal observation",
        "source_run_id": 39637067506,
        "new_teacher_queries": 0,
        "new_online_training_steps": 0,
        "heldout_expert_labels": 0,
        "privileged_grid_policy_inputs": 0,
        "task_instruction_wrapper": "xminigrid.wrappers.RulesAndGoalsObservationWrapper",
        "observation_width": INPUT_WIDTH,
        "max_rule_slots": MAX_RULES,
        "eval_seeds": list(EVAL_SEEDS),
        "models": models,
        "total_real_evaluation_steps": sum(
            c["real_independent_evaluation_steps"]
            for model in models for c in model["conditions"]
        ),
        "limitations": [
            "This changes the observation INFORMATION BUDGET vs previous 54-feature learners and is NOT a matched-input comparison with REUSE-16.",
            "Within this experiment both 87-feature models have identical architecture and training data; instruction slots are informative versus constant.",
            "Rule slot padding is fixed to 4 and contains upstream public task instructions, never full simulator grid.",
            "Episode index cycles only original training primitives to recover existing numeric pool task membership.",
            "Heldout chain is NEVER trained or teacher queried; heldout reward is descriptive until all primitives robust.",
            "Two model seeds and five fresh evaluation seeds are too small for population competence claims.",
        ],
    }
    path = Path(cfg["output_dir"]) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xland_upstream_instruction_bc.yaml")
    run(parser.parse_args().config)
