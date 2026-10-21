"""REUSE-16: matched ordinary-vs-class-balanced BC on ONE DAgger label pool.

Reuse upstream imitation.SimpleDAggerTrainer to collect the original train-only
XLand labels; then train independent ordinary and weighted-sampling BC policies
against the same in-memory observations/actions, without new expert queries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from bagi.experiments.xland_bc_class_balance import action_recall, balanced_indices
from bagi.experiments.xland_existing_bc import TRAIN_TASKS, evaluate, validate_config
from bagi.experiments.xland_existing_dagger import (
    TrainingXLandGym,
    make_privileged_expert,
)

CONDITIONS = ("ordinary", "class_balanced")


def checked_pool(obs, actions, next_obs, dones, *, expected_rows):
    """Hash raw learner inputs and expert labels, excluding privileged state."""
    x = np.asarray(obs, dtype=np.float32)
    y = np.asarray(actions, dtype=np.int64)
    nxt = np.asarray(next_obs, dtype=np.float32)
    done = np.asarray(dones, dtype=bool)
    if (x.ndim != 2 or x.shape[1] != 54 or nxt.shape != x.shape
            or y.shape != (len(x),) or done.shape != y.shape
            or len(x) != expected_rows or not np.isfinite(x).all()
            or not np.isfinite(nxt).all() or np.any((y < 0) | (y >= 6))):
        raise ValueError("DAgger expert pool shape, count, or native-action mismatch")
    if not np.any(y == 3) or not np.any(y == 4):
        raise ValueError("Missing causal pickup/putdown teacher labels")
    digest = hashlib.sha256()
    for arr in (x, y, nxt, done):
        contiguous = np.ascontiguousarray(arr)
        digest.update(str((contiguous.shape, str(contiguous.dtype))).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def fixed_sampler_indices(actions, seed, condition):
    """Both variants take len(pool) samples; balanced uses upstream sampler."""
    labels = np.asarray(actions, dtype=np.int64)
    if labels.ndim != 1 or len(labels) < 1:
        raise ValueError("Nonempty 1D action labels required")
    if condition == "ordinary":
        return np.arange(len(labels), dtype=np.int64)
    if condition == "class_balanced":
        return balanced_indices(labels, seed)
    raise ValueError("Only preregistered ordinary/class_balanced conditions")


def run(config_path):
    import torch
    import xminigrid
    from imitation.algorithms.bc import BC
    from imitation.algorithms.dagger import LinearBetaSchedule, SimpleDAggerTrainer
    from imitation.data import rollout
    from imitation.data.types import Transitions
    from stable_baselines3.common.vec_env import DummyVecEnv
    from xminigrid.wrappers import DirectionObservationWrapper

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    prev = yaml.safe_load(Path(cfg["dagger_config"]).read_text(encoding="utf-8"))
    baseline = yaml.safe_load(Path(cfg["bc_reference_config"]).read_text(encoding="utf-8"))
    _, cap, eval_seeds, model_seeds = validate_config(baseline)
    if (cfg["environment"] != prev["environment"]
            or cfg["training_tasks"] != list(TRAIN_TASKS)
            or tuple(cfg["conditions"]) != CONDITIONS
            or [int(v) for v in cfg["model_seeds"]] != model_seeds
            or [int(v) for v in cfg["evaluation_seeds"]] != eval_seeds
            or int(cfg["training_seed_base"]) != int(prev["training_seed_base"])
            or int(cfg["episode_cap"]) != cap
            or int(cfg["dagger_rollout_timesteps"]) != 256
            or int(cfg["dagger_round_min_timesteps"]) != 128
            or int(cfg["dagger_round_min_episodes"]) != 3
            or int(cfg["dagger_bc_epochs_per_round"]) != 4
            or int(cfg["replay_bc_epochs"]) != 12
            or int(cfg["batch_size"]) != 32
            or int(cfg["maximum_training_interactions"]) != 400):
        raise ValueError("Frozen REUSE-16 experiment contract changed")
    torch.set_num_threads(1)
    eval_env, defaults = xminigrid.make(cfg["environment"])
    eval_views = DirectionObservationWrapper(eval_env)
    experiments = []
    for seed in model_seeds:
        training = TrainingXLandGym(
            seed_base=int(cfg["training_seed_base"]), cap=cap
        )
        venv = DummyVecEnv([lambda environment=training: environment])
        expert = make_privileged_expert(training)
        torch.manual_seed(seed)
        initial_bc = BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            rng=np.random.default_rng(seed),
            batch_size=32, device="cpu",
        )
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="bagi_reuse16_") as temp_dir:
            dagger = SimpleDAggerTrainer(
                venv=venv, scratch_dir=temp_dir, expert_policy=expert,
                bc_trainer=initial_bc, rng=np.random.default_rng(seed),
                beta_schedule=LinearBetaSchedule(rampdown_rounds=1),
            )
            dagger.train(
                total_timesteps=256,
                rollout_round_min_episodes=3,
                rollout_round_min_timesteps=128,
                bc_train_kwargs={
                    "n_epochs": 4, "progress_bar": False,
                    "log_rollouts_venv": None, "log_rollouts_n_episodes": 0,
                },
            )
            collect_seconds = round(time.monotonic() - started, 3)
            # Pinned imitation 1.0.1 _try_load_demos() holds generated trajectory
            # objects in memory after each round. No NPZ/Pickle load is needed.
            # flatten_trajectories is upstream and preserves real next observations.
            trajectories = tuple(dagger._all_demos)
            pool = rollout.flatten_trajectories(trajectories)
            obs = np.asarray(pool.obs, dtype=np.float32)
            acts = np.asarray(pool.acts, dtype=np.int64)
            next_obs = np.asarray(pool.next_obs, dtype=np.float32)
            dones = np.asarray(pool.dones, dtype=bool)
            if int(dagger.round_num) != 2:
                raise AssertionError("Unexpected number of upstream DAgger rounds")
            if (training.real_steps != expert.teacher_queries
                    or training.real_steps > int(cfg["maximum_training_interactions"])):
                raise AssertionError("DAgger interaction/teacher-query budget changed")
            checksum = checked_pool(
                obs, acts, next_obs, dones, expected_rows=expert.teacher_queries
            )
            # Safe reproducibility artifact: only numeric observations and
            # expert labels, not serialized Python objects or privileged state.
            folder = Path(cfg["output_dir"])
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                folder / f"expert_pool_seed{seed}.npz",
                obs=obs, acts=acts, next_obs=next_obs, dones=dones,
            )
            per_condition = []
            for condition in CONDITIONS:
                indices = fixed_sampler_indices(acts, seed, condition)
                sampled = Transitions(
                    obs=obs[indices], acts=acts[indices],
                    next_obs=next_obs[indices], dones=dones[indices],
                    infos=np.array([None] * len(indices), dtype=object),
                )
                torch.manual_seed(seed)
                bc = BC(
                    observation_space=venv.observation_space,
                    action_space=venv.action_space,
                    demonstrations=sampled,
                    rng=np.random.default_rng(seed),
                    batch_size=32, device="cpu",
                )
                fit_start = time.monotonic()
                bc.train(n_epochs=12, progress_bar=False)
                fit_seconds = round(time.monotonic() - fit_start, 3)
                prediction, _ = bc.policy.predict(obs, deterministic=True)
                prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)
                after = evaluate(
                    bc.policy, eval_views, defaults, eval_seeds, cap
                )
                per_condition.append({
                    "condition": condition,
                    "source_pool_sha256": checksum,
                    "source_pool_size": len(acts),
                    "distinct_source_indices_sampled": int(len(np.unique(indices))),
                    "optimization_sample_exposures": len(acts) * 12,
                    "sampled_action_counts": dict(Counter(map(int, acts[indices]))),
                    "training_seconds": fit_seconds,
                    "expert_label_accuracy_original_pool": float(
                        np.mean(prediction == acts)
                    ),
                    "expert_action_recall_original_pool": action_recall(
                        prediction, acts
                    ),
                    "closed_loop": after,
                    "independent_evaluation_steps": sum(
                        item["evaluation_steps"] for item in after.values()
                    ),
                })
                print(f"REUSE16 seed={seed} {condition}: "
                      f"pickup={per_condition[-1]['expert_action_recall_original_pool']['3']} "
                      f"putdown={per_condition[-1]['expert_action_recall_original_pool']['4']} "
                      f"primitive_success={[after[k]['successes'] for k in TRAIN_TASKS]}")
            assert len({x["source_pool_sha256"] for x in per_condition}) == 1
        venv.close()
        experiments.append({
            "model_seed": seed,
            "dagger_rounds": 2,
            "collect_and_dagger_train_seconds": collect_seconds,
            "real_training_interactions": training.real_steps,
            "expert_queries": expert.teacher_queries,
            "expert_label_counts": dict(Counter(map(int, acts))),
            "training_rollout_goal_events_not_policy_success": dict(
                training.goal_completions
            ),
            "training_reset_counts": dict(training.episodes_by_task),
            "training_executed_action_counts": dict(training.executed_actions),
            "source_pool_sha256": checksum,
            "expert_trajectory_count": len(trajectories),
            "conditions": per_condition,
        })
    result = {
        "question": "Does rare-action class balancing repair BC on identical DAgger-aggregated data?",
        "contribution_type": "Existing DAgger/BC integration and controlled sampler ablation",
        "source": "HumanCompatibleAI imitation.SimpleDAggerTrainer and BC",
        "packages": {"imitation": version("imitation"),
                     "xminigrid": version("xminigrid")},
        "config": cfg,
        "heldout_teacher_queries": 0,
        "privileged_learned_policy_inputs": 0,
        "source_pool_access": "upstream in-memory trajectory objects only",
        "training_only_skills": list(TRAIN_TASKS),
        "frozen_heldout_task": "hold_then_near",
        "models": experiments,
        "limitations": [
            "This is a sampler comparison conditional on each seeded DAgger pool, not a matched-interaction comparison with REUSE-10 or a universal statement about DAgger.",
            "Each DAgger pool is collected once per seed; replayed BC conditions share the exact original rows and network architecture.",
            "Balanced sampling repeats scarce labels and omits other original indices; report their true counts and same optimization sample exposures.",
            "Expert-guided DAgger collection rewards cannot count as acquired learned-agent successes.",
            "Teacher can see privileged state solely for TRAIN-only labels; learner sees only image and direction.",
            "SimpleDAggerTrainer uses a 256-step minimum, not upper bound; actual simulator interactions and label queries are reported.",
            "Frozen independent eval seeds and 40-action caps; only two training seeds and one single-room task family.",
            "No untrusted model checkpoints or external benchmark pickles are loaded.",
            "Supervised recall is tested on the same collected training labels and is not generalization.",
        ],
    }
    path = Path(cfg["output_dir"]) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved REUSE-16 measurements: {path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xland_dagger_balance_control.yaml")
    run(parser.parse_args().config)
