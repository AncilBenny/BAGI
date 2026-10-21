"""Gate B2: held-out memory lookup on diverse executed visible histories.

A transparent non-neural reference: for each visible history window, memorize
training-set majority outcomes, then evaluate on *different random action
trajectories*. Unknown keys are counted as abstentions, never as successes.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from bagi.experiments.history_ambiguity import (
    ambiguity,
    outcome,
    visible_input,
)
from bagi.worlds import WorkshopEnv


def generate(
    env: WorkshopEnv,
    guided_actions: list[int],
    prefixes: list[int],
    train_episodes: int,
    holdout_episodes: int,
    continuation_steps: int,
    seed: int,
    max_history: int,
) -> tuple[list[dict], dict]:
    """Execute real episodes. Only observation/action histories become inputs."""
    if min(train_episodes, holdout_episodes, continuation_steps, max_history) < 1:
        raise ValueError("All episode, continuation, and history budgets must be positive")
    if not prefixes or min(prefixes) < 0 or max(prefixes) > len(guided_actions):
        raise ValueError("Invalid causal-prefix selection")
    rows: list[dict] = []
    simulator_steps = 0
    prefix_steps = 0
    for prefix in prefixes:
        for episode_index in range(train_episodes + holdout_episodes):
            state_seed = seed + prefix * 100_000 + episode_index
            observation, _ = env.reset(seed=state_seed)
            history = []
            for action in guided_actions[:prefix]:
                prior = observation.tolist()
                observation, _, terminated, truncated, _ = env.step(action)
                simulator_steps += 1
                prefix_steps += 1
                history.append([prior, int(action)])
                history = history[-max_history:]
                if terminated or truncated:
                    raise ValueError("Guided prefix terminated before random continuation")
            rng = random.Random(state_seed)
            split = "train" if episode_index < train_episodes else "holdout"
            for _ in range(continuation_steps):
                action = rng.randrange(env.action_space.n)
                next_observation, reward, terminated, truncated, info = env.step(action)
                simulator_steps += 1
                rows.append({
                    "split": split,
                    "prefix_steps": prefix,
                    "episode": f"{prefix}:{episode_index}",
                    "history": history.copy(),
                    "observation": observation.tolist(),
                    "action": action,
                    "next_observation": next_observation.tolist(),
                    "reward": float(reward),
                    "events": list(info.get("events", [])),
                })
                history.append([observation.tolist(), action])
                history = history[-max_history:]
                observation = next_observation
                if terminated or truncated:
                    break
    return rows, {
        "episodes": len(prefixes) * (train_episodes + holdout_episodes),
        "training_episodes": len(prefixes) * train_episodes,
        "holdout_episodes": len(prefixes) * holdout_episodes,
        "simulator_real_steps": simulator_steps,
        "guided_prefix_steps": prefix_steps,
        "logged_continuation_steps": len(rows),
    }


def lookup_score(train: list[dict], holdout: list[dict], window: int) -> dict:
    """Report accuracy only where a training key exists, plus abstention rate."""
    if not train or not holdout:
        raise ValueError("Need separate nonempty training and held-out trajectories")
    by_key: dict[str, Counter] = defaultdict(Counter)
    for row in train:
        by_key[visible_input(row, window)][outcome(row)] += 1
    predictions = {
        key: labels.most_common(1)[0][0] for key, labels in by_key.items()
    }
    seen = 0
    correct = 0
    event_correct = 0
    test_event_positive = 0
    for row in holdout:
        actual = outcome(row)
        test_event_positive += len(row["events"])
        predicted = predictions.get(visible_input(row, window))
        if predicted is None:
            continue
        seen += 1
        correct += int(predicted == actual)
        event_correct += int(json.loads(predicted)[-1] == json.loads(actual)[-1])
    return {
        "history_window": window,
        "training_rows": len(train),
        "heldout_rows": len(holdout),
        "training_unique_keys": len(predictions),
        "heldout_keys_seen_in_training": seen,
        "heldout_key_coverage": seen / len(holdout),
        "heldout_whole_outcome_accuracy_given_seen": correct / seen if seen else None,
        "heldout_event_accuracy_given_seen": event_correct / seen if seen else None,
        "heldout_whole_outcome_accuracy_with_unknown_as_error": correct / len(holdout),
        "heldout_positive_event_labels": test_event_positive,
    }


def run(config_path: str) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    env = WorkshopEnv(**cfg["environment"])
    design = cfg["evaluation"]
    windows = [int(x) for x in design["history_windows"]]
    if windows != sorted(set(windows)) or not windows or windows[0] != 0:
        raise ValueError("History windows must begin with 0 and be sorted/unique")
    train_count = int(design["training_episodes_per_prefix"])
    holdout_count = int(design["holdout_episodes_per_prefix"])
    rows, counts = generate(
        env,
        [int(x) for x in design["guided_actions"]],
        [int(x) for x in design["prefix_steps"]],
        train_count,
        holdout_count,
        int(design["continuation_steps"]),
        int(design["seed"]),
        max(windows) if max(windows) > 0 else 1,
    )
    train = [row for row in rows if row["split"] == "train"]
    holdout = [row for row in rows if row["split"] == "holdout"]
    results = [lookup_score(train, holdout, window) for window in windows]
    result = {
        "question": "Does recent visible observation/action history improve held-out tabular next-outcome prediction across varied action paths, accounting for unknown-history abstentions?",
        "evaluation": counts,
        "design": design,
        "results": results,
        "empirical_in_sample_ambiguity": [
            ambiguity(rows, window) for window in windows
        ],
        "limitations": [
            "Train and holdout use independent action RNG but the same deterministic Workshop layout and repeated guided prefixes.",
            "Lookup is a transparent memory/control diagnostic, not a trained neural network or a general policy.",
            "Longer histories have more rare input keys, so conditional accuracy must be interpreted with heldout_key_coverage.",
            "Targets use observation, reward, and labeled events; those event labels are targets only, not predictor features.",
            "The experiment logs real simulator steps, including prefix replay, and does not add its transitions to model training.",
        ],
    }
    dest = Path(design["output_dir"]) / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for row in results:
        print(f"window={row['history_window']} "
              f"key_coverage={row['heldout_key_coverage']:.3f} "
              f"conditional_whole_accuracy="
              f"{row['heldout_whole_outcome_accuracy_given_seen']} "
              f"accuracy_with_abstention={row['heldout_whole_outcome_accuracy_with_unknown_as_error']:.3f}")
    print(f"Saved heldout memory lookup: {dest}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/history_rollout_robustness.yaml")
    run(parser.parse_args().config)
