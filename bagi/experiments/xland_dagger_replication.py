"""REUSE-17: frozen-label replication of REUSE-16 balanced DAgger-BC results.

Load ONLY the exact numeric expert pools emitted by the successful REUSE-16
GitHub run. No DAgger, expert, oracle, policy checkpoint or pickle load.
Retrain existing imitation.BC under the original seeds and evaluate on
independent new environment seeds while keeping the heldout chain untouched.
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

from bagi.experiments.xland_bc_class_balance import action_recall
from bagi.experiments.xland_dagger_balance_control import (
    CONDITIONS,
    checked_pool,
    fixed_sampler_indices,
)
from bagi.experiments.xland_existing_bc import TRAIN_TASKS, evaluate, validate_config

SOURCE_FINGERPRINTS = {
    17: "be72367678b28f8bd69572be6a6edb5a82ebde960db9354c65b3810d10974eda",
    29: "45a273de130a52dec8a94e31bd9081e862ed70f4f69f81bd15115f8248988c43",
}
EXPECTED_ANCHOR_SUCCESSES = {
    (17, "ordinary"): {"hold": (), "near": (), "putdown": (), "hold_then_near": ()},
    (17, "class_balanced"): {
        "hold": (9101,), "near": (9101,), "putdown": (), "hold_then_near": (),
    },
    (29, "ordinary"): {"hold": (), "near": (), "putdown": (), "hold_then_near": ()},
    (29, "class_balanced"): {
        "hold": (), "near": (), "putdown": (), "hold_then_near": (),
    },
}


def load_frozen_numeric_pool(folder, seed):
    """Load only trusted numeric arrays, reject changed provenance or labels."""
    path = Path(folder) / f"expert_pool_seed{seed}.npz"
    if seed not in SOURCE_FINGERPRINTS:
        raise ValueError("Seed is not a reviewed original REUSE-16 collection")
    with np.load(path, allow_pickle=False) as contents:
        if set(contents.files) != {"obs", "acts", "next_obs", "dones"}:
            raise ValueError("Unexpected fields in frozen numeric expert pool")
        obs, acts = contents["obs"], contents["acts"]
        next_obs, dones = contents["next_obs"], contents["dones"]
    if (obs.dtype != np.float32 or next_obs.dtype != np.float32
            or acts.dtype != np.int64 or dones.dtype != np.bool_):
        raise ValueError("Only reviewed numeric and bool dtypes are permitted")
    digest = checked_pool(
        obs, acts, next_obs, dones, expected_rows=291
    )
    if digest != SOURCE_FINGERPRINTS[seed]:
        raise ValueError("Frozen expert data differs from verified REUSE-16 SHA256")
    frequencies = Counter(map(int, acts))
    reference = (
        {0: 130, 1: 69, 2: 64, 3: 22, 4: 6} if seed == 17
        else {0: 132, 1: 83, 2: 49, 3: 21, 4: 6}
    )
    if frequencies != reference:
        raise ValueError("Source label distribution changed")
    return {
        "obs": obs, "acts": acts, "next_obs": next_obs, "dones": dones,
        "sha256": digest,
        "action_counts": dict(frequencies),
    }


def successful_seeds(evaluation):
    """Return actual environment seed IDs with rewarded terminal transitions."""
    return {
        task: tuple(
            detail["seed"] for detail in info["details"]
            if detail["goal_success"]
        )
        for task, info in evaluation.items()
    }


def run(config_path):
    import torch
    import xminigrid
    from imitation.algorithms.bc import BC
    from imitation.data.types import Transitions
    from xminigrid.wrappers import DirectionObservationWrapper

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    base_config = yaml.safe_load(
        Path(config["bc_reference_config"]).read_text(encoding="utf-8")
    )
    _, cap, anchor_seeds, training_seeds = validate_config(base_config)
    fresh = [int(x) for x in config["new_evaluation_seeds"]]
    if (config["environment"] != base_config["environment"]
            or config["training_tasks"] != list(TRAIN_TASKS)
            or tuple(config["conditions"]) != CONDITIONS
            or [int(x) for x in config["model_seeds"]] != training_seeds
            or [int(x) for x in config["anchor_evaluation_seeds"]] != anchor_seeds
            or int(config["episode_cap"]) != cap
            or int(config["replay_bc_epochs"]) != 12
            or int(config["batch_size"]) != 32
            or fresh != list(range(9200, 9206))
            or set(fresh) & set(anchor_seeds)
            or set(fresh) & set(range(7500, 7600))
            or int(config["original_source_run_id"]) != 39619756922):
        raise ValueError("Preregistered frozen-label replication contract changed")
    torch.set_num_threads(1)
    env, defaults = xminigrid.make(config["environment"])
    views = DirectionObservationWrapper(env)
    groups = []
    all_anchors_reproduce = True
    for seed in training_seeds:
        data = load_frozen_numeric_pool(config["source_artifact_dir"], seed)
        n = len(data["acts"])
        conditions = []
        for condition in CONDITIONS:
            indices = fixed_sampler_indices(data["acts"], seed, condition)
            pool = Transitions(
                obs=data["obs"][indices], acts=data["acts"][indices],
                next_obs=data["next_obs"][indices],
                dones=data["dones"][indices],
                infos=np.array([None] * n, dtype=object),
            )
            torch.manual_seed(seed)
            trainer = BC(
                observation_space=views.observation_space if False else
                    __import__("gymnasium").spaces.Box(
                        low=0, high=255, shape=(54,), dtype=np.float32
                    ),
                action_space=__import__("gymnasium").spaces.Discrete(6),
                demonstrations=pool, rng=np.random.default_rng(seed),
                batch_size=32, device="cpu",
            )
            started = time.monotonic()
            trainer.train(n_epochs=12, progress_bar=False)
            fit_seconds = round(time.monotonic() - started, 3)
            predicted, _ = trainer.policy.predict(
                data["obs"], deterministic=True
            )
            predicted = np.asarray(predicted, dtype=np.int64).reshape(-1)
            anchor = evaluate(
                trainer.policy, views, defaults, anchor_seeds, cap
            )
            reproduced = successful_seeds(anchor) == EXPECTED_ANCHOR_SUCCESSES[
                (seed, condition)
            ]
            all_anchors_reproduce &= reproduced
            new = evaluate(trainer.policy, views, defaults, fresh, cap)
            conditions.append({
                "condition": condition,
                "source_sha256": data["sha256"],
                "source_label_count": n,
                "distinct_training_indices": len(np.unique(indices)),
                "optimization_exposures": n * 12,
                "original_pool_expert_accuracy": float(np.mean(
                    predicted == data["acts"]
                )),
                "original_pool_action_recall": action_recall(
                    predicted, data["acts"]
                ),
                "fitting_seconds": fit_seconds,
                "anchor_successful_episode_seeds": successful_seeds(anchor),
                "expected_anchor_successful_episode_seeds": {
                    k: list(v) for k, v in EXPECTED_ANCHOR_SUCCESSES[
                        (seed, condition)
                    ].items()
                },
                "anchor_reproduced": reproduced,
                "anchor_evaluation": anchor,
                "new_evaluation": new,
                "new_successful_episode_seeds": successful_seeds(new),
                "new_evaluation_steps": sum(
                    t["evaluation_steps"] for t in new.values()
                ),
            })
            print(f"seed={seed} {condition} anchor={reproduced} "
                  f"new_training_successes="
                  f"{[new[k]['successes'] for k in TRAIN_TASKS]} "
                  f"new_heldout={new['hold_then_near']['successes']}/{len(fresh)}")
        groups.append({
            "policy_seed": seed,
            "source_sha256": data["sha256"],
            "original_expert_action_frequencies": data["action_counts"],
            "new_expert_queries": 0,
            "new_training_environment_steps": 0,
            "conditions": conditions,
        })
    result = {
        "question": "Do rare independent REUSE-16 BC successes recur on unseen eval seeds?",
        "type": "Frozen data BC reproduction and independent evaluation, no expert queries",
        "source_run_id": 39619756922,
        "source_model_commit": "4c2da82badf93ab2f40c3077d298e538e17c9b82",
        "packages": {
            "imitation": version("imitation"), "xminigrid": version("xminigrid")
        },
        "config": config,
        "anchor_reproduced_all": all_anchors_reproduce,
        "expert_queries": 0,
        "training_environment_interactions": 0,
        "heldout_teacher_queries": 0,
        "privileged_learner_inputs": 0,
        "models": groups,
        "limitations": [
            "Independent seeds measure repeated single-room skill robustness, NOT structural heldout task transfer.",
            "Reuses the same 291-label source pool per policy seed and same policy seed; no further training teacher queries.",
            "Two policy seeds, six additional evaluation seeds and deterministic BC are too small for population-level claims.",
            "One originally successful eval seed drove both prior positives; anchor verification prevents silent non-reproduction.",
            "Only learner-visible XLand image and orientation enter policy; no privileged oracle at evaluation.",
            "Evaluation results on the held-out chain cannot be considered transfer without robust source-skill competence.",
        ],
    }
    dst = Path(config["output_dir"]) / "summary.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Frozen-label replication written: {dst}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/xland_dagger_replication.yaml")
    run(p.parse_args().config)
