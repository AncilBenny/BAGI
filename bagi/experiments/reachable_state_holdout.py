"""Gate A2: measure one-step generalization on the reachable Workshop graph.

The environment's full state is used only to deduplicate states when
constructing the evaluation set. The world model sees only observation+action.
No model training or planner tuning occurs in this study.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import yaml

from bagi.experiments.dataset_foundations import EVENTS, model_input
from bagi.models import WorldModel
from bagi.worlds import WorkshopEnv


def state_key(info: dict) -> tuple:
    """Diagnostic-only Markov state, never supplied to model.forward."""
    return (
        tuple(int(n) for n in info["position"]),
        bool(info["has_key"]),
        bool(info["door_open"]),
    )


def enumerate_reachable(env: WorkshopEnv, *, max_states: int = 512) -> tuple[list[dict], dict]:
    """BFS over executable action histories, replaying in the real environment."""
    observation, initial = env.reset(seed=0)
    del observation
    queue = deque([()])
    reached = {state_key(initial)}
    transitions = []
    replay_steps = 0
    while queue:
        path = queue.popleft()
        for action in range(env.action_space.n):
            observation, _ = env.reset(seed=0)
            for prior in path:
                observation, _, terminated, truncated, _ = env.step(prior)
                replay_steps += 1
                if terminated or truncated:
                    raise AssertionError("BFS attempted to replay terminal trajectory")
            next_obs, reward, terminated, truncated, info = env.step(action)
            replay_steps += 1
            transitions.append({
                "observation": observation.tolist(),
                "action": action,
                "next_observation": next_obs.tolist(),
                "reward": float(reward),
                "info": {"events": list(info["events"])},
            })
            key = state_key(info)
            if key not in reached and not (terminated or truncated):
                reached.add(key)
                if len(reached) > max_states:
                    raise ValueError("Reached state bound; raise max_states explicitly")
                queue.append((*path, action))
    return transitions, {
        "reachable_nonterminal_states": len(reached),
        "state_action_transitions": len(transitions),
        "simulator_replay_steps": replay_steps,
        "method": "breadth_first_search_over_action_prefixes_with_environment_replay",
    }


def split_training_keys(rows: list[dict], seed: int, fraction: float) -> set[str]:
    if not 0 < fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    if len(rows) < 2:
        raise ValueError("At least two logged transitions are required")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    cutoff = max(1, int(len(rows) * (1 - fraction)))
    return {model_input(rows[i]) for i in indices[:cutoff]}


def make_groups(transitions: list[dict], seen_keys: set[str]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in transitions:
        groups["seen_in_training" if model_input(row) in seen_keys else
               "unseen_observation_action"].append(row)
    return groups


@torch.no_grad()
def score_transitions(model, rows: list[dict], *, batch_size: int = 128) -> dict:
    """Classify event support before interpreting aggregate prediction accuracy."""
    if not rows:
        return {"comparisons": 0, "positive_event_labels": 0,
                "positive_event_brier": None, "negative_event_brier": None}
    cell_correct = 0.0
    cell_count = 0
    whole_correct = 0
    reward_abs_sum = 0.0
    event_sse_positive = 0.0
    event_sse_negative = 0.0
    positive_count = 0
    negative_count = 0
    event_true = {name: 0 for name in EVENTS}
    event_tp = {name: 0 for name in EVENTS}
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        observations = torch.as_tensor(np.array([x["observation"] for x in batch]),
                                       dtype=torch.long)
        actions = torch.tensor([x["action"] for x in batch], dtype=torch.long)
        targets = torch.as_tensor(np.array([x["next_observation"] for x in batch]),
                                  dtype=torch.long)
        rewards = torch.tensor([x["reward"] for x in batch], dtype=torch.float32)
        labels = torch.tensor([
            [float(name in x["info"]["events"]) for name in EVENTS] for x in batch
        ], dtype=torch.float32)
        # No hidden position/door/key flags are provided to the learned model.
        logits, predicted_rewards, event_logits = model.forward_with_events(
            observations, actions
        )
        predicted = logits.argmax(dim=-1)
        cell_correct += float((predicted == targets).sum())
        cell_count += targets.numel()
        whole_correct += int((predicted == targets).flatten(1).all(dim=1).sum())
        reward_abs_sum += float((predicted_rewards - rewards).abs().sum())
        probs = event_logits.sigmoid()
        event_sse_positive += float((((probs - labels) ** 2) * labels).sum())
        event_sse_negative += float((((probs - labels) ** 2) * (1 - labels)).sum())
        positive_count += int(labels.sum())
        negative_count += labels.numel() - int(labels.sum())
        for j, name in enumerate(EVENTS):
            event_true[name] += int(labels[:, j].sum())
            event_tp[name] += int(((probs[:, j] >= 0.5) & (labels[:, j] == 1)).sum())
    return {
        "comparisons": len(rows),
        "cell_accuracy": cell_correct / cell_count,
        "whole_observation_accuracy": whole_correct / len(rows),
        "reward_mae": reward_abs_sum / len(rows),
        "positive_event_labels": positive_count,
        "negative_event_labels": negative_count,
        "positive_event_brier": (
            event_sse_positive / positive_count if positive_count else None
        ),
        "negative_event_brier": (
            event_sse_negative / negative_count if negative_count else None
        ),
        "event_positive_counts": event_true,
        "event_true_positives_at_threshold_0_5": event_tp,
    }


def run(config_path: str) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data_path = Path(config["training_data"]["transitions"])
    checkpoint = Path(config["model"]["checkpoint"])
    if not data_path.is_file():
        raise FileNotFoundError(f"Training transitions unavailable: {data_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"World model unavailable: {checkpoint}")
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    seen = split_training_keys(
        rows,
        int(config["training_data"]["training_seed"]),
        float(config["training_data"]["validation_fraction"]),
    )
    env = WorkshopEnv(**config["environment"])
    transitions, enumeration = enumerate_reachable(
        env, max_states=int(config["evaluation"]["max_states"])
    )
    groups = make_groups(transitions, seen)
    model_cfg = config["model"]
    model = WorldModel(
        observation_shape=tuple(env.observation_space.shape),
        actions=int(model_cfg["actions"]),
        cell_values=int(model_cfg["cell_values"]),
        hidden=int(model_cfg["hidden"]),
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    all_unique_inputs = {model_input(x) for x in transitions}
    results = {
        "question": "Does the current world model predict reachable observation-action pairs absent from its actual training split?",
        "checkpoint": str(checkpoint),
        "training_dataset": str(data_path),
        "environment": config["environment"],
        "evaluation_scope": (
            "All reachable hidden-state/action combinations of the single fixed Workshop layout;"
            " not unseen layouts or transfer"
        ),
        "model_input_restriction": "visible observation and action only",
        "training_unique_observation_action_keys": len(seen),
        "enumeration": enumeration,
        "unique_evaluation_observation_action_keys": len(all_unique_inputs),
        "novel_evaluation_observation_action_keys": len(all_unique_inputs - seen),
        "groups": {
            key: score_transitions(model, groups.get(key, []))
            for key in ("seen_in_training", "unseen_observation_action")
        },
        "overall": score_transitions(model, transitions),
        "limits": [
            "Graph coverage is over simulator-reachable hidden states, but the world model never receives them.",
            "The reachable graph contains repeated visible observation/action keys when hidden state differs.",
            "The same fixed layout is used as in training; this does not measure transfer to new environments.",
            "Low positive event count limits event-specific conclusions.",
            "The BFS uses replayed real simulator transitions for evaluation only; it is not extra training data.",
        ],
    }
    destination = Path(config["evaluation"]["output_dir"]) / "summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, metric in results["groups"].items():
        print(f"{name}: n={metric['comparisons']}, cell_acc="
              f"{metric.get('cell_accuracy', float('nan')):.3f}, "
              f"whole_acc={metric.get('whole_observation_accuracy', float('nan')):.3f}, "
              f"positives={metric['positive_event_labels']}")
    print(f"Reachable states: {enumeration['reachable_nonterminal_states']}; "
          f"novel observation-action keys: {results['novel_evaluation_observation_action_keys']}")
    print(f"Saved reachable-state diagnostic to {destination}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reachable_state_holdout.yaml")
    run(parser.parse_args().config)
