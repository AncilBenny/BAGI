"""REUSE-10: Class-balanced vs ordinary sampling for existing imitation.BC.

Exactly the same XLand expert transition pool, same BC algorithm, training
epochs, model seeds and evaluation seeds. Only PyTorch's standard sampler
changes. No training examples from the held-out causal chain.
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

from bagi.experiments.xland_existing_bc import (
    ALL_TASKS,
    TRAIN_TASKS,
    collect,
    evaluate,
    validate_config,
)


def balanced_indices(actions, seed):
    """Use upstream torch WeightedRandomSampler with inverse label frequency."""
    import torch
    from torch.utils.data import WeightedRandomSampler

    labels = np.asarray(actions, dtype=np.int64)
    if labels.ndim != 1 or len(labels) < 1:
        raise ValueError("Expected a nonempty 1D action-label vector")
    counts = np.bincount(labels, minlength=6)
    weights = np.array([1.0 / counts[a] for a in labels], dtype=np.float64)
    generator = torch.Generator().manual_seed(int(seed))
    drawn = WeightedRandomSampler(
        weights, num_samples=len(labels), replacement=True, generator=generator
    )
    return np.fromiter(drawn, dtype=np.int64, count=len(labels))


def action_recall(predictions, true_labels):
    """Report rare necessary pickup and put-down recall, not only average fit."""
    p = np.asarray(predictions, dtype=np.int64).reshape(-1)
    y = np.asarray(true_labels, dtype=np.int64).reshape(-1)
    if p.shape != y.shape:
        raise ValueError("Prediction/label count mismatch")
    return {
        str(action): (float(np.mean(p[y == action] == action))
                      if np.any(y == action) else None)
        for action in range(6)
    }


def run(config_path):
    import gymnasium as gym
    import torch
    import xminigrid
    from imitation.algorithms.bc import BC
    from imitation.data.types import Transitions
    from xminigrid.wrappers import DirectionObservationWrapper

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    budget, cap, eval_seeds, model_seeds = validate_config(cfg)
    torch.set_num_threads(1)
    env, defaults = xminigrid.make(cfg["environment"])
    views = DirectionObservationWrapper(env)
    original, collection = collect(
        env, defaults, TRAIN_TASKS, budget=budget,
        seed_base=int(cfg["demonstration_seed_base"]),
        cap=cap, max_attempts=int(cfg["max_collection_attempts"]),
    )
    if 4 not in original["acts"] or 3 not in original["acts"]:
        raise AssertionError("The third training skill must contain pickup and put-down")
    observation_space = gym.spaces.Box(
        low=0, high=255, dtype=np.float32, shape=original["obs"].shape[1:]
    )
    action_space = gym.spaces.Discrete(env.num_actions(defaults))
    if action_space.n != 6:
        raise AssertionError("Expected six native XLand actions")

    experiment = []
    for seed in model_seeds:
        for condition in ("ordinary", "class_balanced"):
            torch.manual_seed(seed)
            if condition == "ordinary":
                indices = np.arange(budget)
            else:
                indices = balanced_indices(original["acts"], seed)
            sampled = {key: value[indices] for key, value in original.items()}
            batch = Transitions(
                obs=sampled["obs"], acts=sampled["acts"],
                next_obs=sampled["next_obs"], dones=sampled["dones"],
                infos=np.array([None] * budget, dtype=object),
            )
            trainer = BC(
                observation_space=observation_space,
                action_space=action_space,
                demonstrations=batch,
                rng=np.random.default_rng(seed),
                batch_size=int(cfg["batch_size"]), device="cpu",
            )
            started = time.monotonic()
            trainer.train(n_epochs=int(cfg["epochs"]), progress_bar=False)
            train_seconds = round(time.monotonic() - started, 3)
            predicted, _ = trainer.policy.predict(
                original["obs"], deterministic=True
            )
            predicted = np.asarray(predicted).reshape(-1)
            score = {
                "unweighted_expert_label_accuracy": float(np.mean(
                    predicted == original["acts"]
                )),
                "unweighted_expert_action_recall": action_recall(
                    predicted, original["acts"]
                ),
            }
            outcomes = evaluate(
                trainer.policy, views, defaults, eval_seeds, cap
            )
            experiment.append({
                "sampling": condition,
                "model_seed": seed,
                "expert_example_pool_size": budget,
                "unique_expert_indices_sampled": len(np.unique(indices)),
                "optimization_sample_exposures": budget * int(cfg["epochs"]),
                "sampled_action_frequencies": dict(Counter(map(int, sampled["acts"]))),
                "train_seconds": train_seconds,
                "teacher_fit": score,
                "closed_loop": outcomes,
            })
            print(f"{condition} seed={seed}: "
                  f"pickup_recall={score['unweighted_expert_action_recall']['3']}, "
                  f"putdown_recall={score['unweighted_expert_action_recall']['4']}, "
                  f"training_goal={[outcomes[k]['successes'] for k in TRAIN_TASKS]}, "
                  f"heldout={outcomes['hold_then_near']['successes']}/{len(eval_seeds)}")
    report = {
        "question": "Does standard inverse-frequency sampling address rare causal-action collapse in imitation.BC without new demonstrations?",
        "established_algorithms": [
            "imitation.algorithms.bc.BC (unchanged)",
            "torch.utils.data.WeightedRandomSampler (unchanged)",
        ],
        "packages": {"imitation": version("imitation"), "xminigrid": version("xminigrid")},
        "training_tasks": list(TRAIN_TASKS),
        "held_out_never_demonstrated": ALL_TASKS[-1],
        "config": cfg,
        "expert_data_collection": collection,
        "experiments": experiment,
        "limitations": [
            "Equal unique expert pool and optimization sample exposures; inverse-frequency sampler repeats rare examples instead of collecting novel causal interactions.",
            "Reweighting may harm average imitation accuracy; monitor rare action recalls and closed-loop skill completion.",
            "No privileged map/rule/task-ID given to policy; geometry used only to label expert actions.",
            "BC is memoryless and closed-loop covariate shift remains unsolved; recurrent learner and DAgger comparisons remain future work.",
            "Small two-seed, three-evaluation-seed pilot; do not claim population-level generalization.",
            "No held-out chain state/action was included in any expert training dataset.",
        ],
    }
    output = Path(cfg["output_dir"]) / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved balanced BC ablation: {output}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xland_bc_class_balance.yaml")
    run(parser.parse_args().config)
