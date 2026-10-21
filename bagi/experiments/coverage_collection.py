"""Gate A3: matched-budget collection diversity, without model retraining.

Compare the existing 50-guided/50-random dataset against two preregistered
alternatives: five guided episodes plus random exploration, or five guided
episodes plus count-based exploration of visible observation/action pairs.
A simulator is used for collection; policies see no privileged state.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import yaml

from bagi.experiments.collect_guided import GUIDED_ACTIONS
from bagi.experiments.dataset_foundations import (
    EVENTS,
    model_input,
    supervised_transition,
)
from bagi.experiments.reachable_state_holdout import enumerate_reachable
from bagi.worlds import WorkshopEnv


def select_action(observation, rng: random.Random, visits: Counter,
                  actions: int, *, policy: str, epsilon: float) -> int:
    """Choose an action from visible observation and past observed visit counts."""
    if policy == "random" or rng.random() < epsilon:
        return rng.randrange(actions)
    if policy != "novelty":
        raise ValueError(f"Unknown policy: {policy}")
    choices = [model_input({"observation": observation, "action": a})
               for a in range(actions)]
    minimum = min(visits[key] for key in choices)
    rarest = [a for a, key in enumerate(choices) if visits[key] == minimum]
    return rng.choice(rarest)


def collect(env, *, budget: int, guided_episodes: int, policy: str,
            seed: int, epsilon: float) -> list[dict]:
    """Collect exactly budget real transitions, resetting only between episodes."""
    if budget < 1 or guided_episodes < 0 or not 0 <= epsilon <= 1:
        raise ValueError("Positive budget, nonnegative guided episodes, epsilon in [0, 1]")
    if policy not in ("random", "novelty"):
        raise ValueError("policy must be random or novelty")
    rows: list[dict] = []
    visits: Counter = Counter()
    rng = random.Random(seed)
    episode_id = 0

    def step(observation, action: int):
        next_obs, reward, terminated, truncated, info = env.step(action)
        row = {
            "episode": episode_id,
            "observation": observation.tolist(),
            "action": int(action),
            "next_observation": next_obs.tolist(),
            "reward": float(reward),
            "info": {"events": list(info.get("events", []))},
        }
        rows.append(row)
        visits[model_input(row)] += 1
        return next_obs, terminated or truncated

    for ep in range(guided_episodes):
        if len(rows) >= budget:
            break
        observation, _ = env.reset(seed=seed + ep)
        for action in GUIDED_ACTIONS:
            if len(rows) >= budget:
                break
            observation, done = step(observation, action)
            if done:
                break
        episode_id += 1

    while len(rows) < budget:
        observation, _ = env.reset(seed=seed + 100_000 + episode_id)
        for _ in range(env.max_steps):
            if len(rows) >= budget:
                break
            action = select_action(
                observation.tolist(), rng, visits, env.action_space.n,
                policy=policy, epsilon=epsilon,
            )
            observation, done = step(observation, action)
            if done:
                break
        episode_id += 1
    assert len(rows) == budget
    return rows


def coverage(rows: list[dict], reachable: set[str],
             original_keys: set[str]) -> dict:
    keys = {model_input(x) for x in rows}
    distinct = {supervised_transition(x) for x in rows}
    event_keys = {
        name: {model_input(row) for row in rows
               if name in row.get("info", {}).get("events", [])}
        for name in EVENTS
    }
    return {
        "real_transitions": len(rows),
        "episodes_observed": len({int(x["episode"]) for x in rows}),
        "unique_observation_action_keys": len(keys),
        "unique_supervised_transitions": len(distinct),
        "reachable_input_keys_covered": len(keys & reachable),
        "reachable_input_fraction_covered": len(keys & reachable) / len(reachable),
        "new_input_keys_vs_existing_data": len(keys - original_keys),
        "missing_reachable_input_keys": len(reachable - keys),
        "positive_event_rows": {
            name: sum(name in row.get("info", {}).get("events", [])
                      for row in rows) for name in EVENTS
        },
        "unique_event_positive_input_keys": {
            name: len(event_keys[name]) for name in EVENTS
        },
        "unmatched_observable_input_keys": len(keys - reachable),
    }


def run(config_path: str) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    source = Path(cfg["existing_training_data"])
    if not source.is_file():
        raise FileNotFoundError(f"Missing original collection dataset: {source}")
    original = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    budget = len(original)
    if budget < 1:
        raise ValueError("Original dataset is empty")
    original_keys = {model_input(x) for x in original}

    env = WorkshopEnv(**cfg["environment"])
    reachable_rows, graph_info = enumerate_reachable(
        env, max_states=int(cfg["evaluation"]["max_states"])
    )
    reachable = {model_input(x) for x in reachable_rows}
    ecfg = cfg["evaluation"]
    arms = {"existing_50guided_50random": coverage(original, reachable, original_keys)}
    for policy in ("random", "novelty"):
        rows = collect(
            env, budget=budget, guided_episodes=int(ecfg["guided_episodes"]),
            policy=policy, seed=int(ecfg["seed"]),
            epsilon=float(ecfg["novelty_epsilon"]),
        )
        arms[f"five_guided_plus_{policy}"] = coverage(rows, reachable, original_keys)
    for name, result in arms.items():
        print(f"{name}: steps={result['real_transitions']}, "
              f"unique_input={result['unique_observation_action_keys']}, "
              f"reachable_coverage={result['reachable_input_fraction_covered']:.3f}, "
              f"new_keys={result['new_input_keys_vs_existing_data']}")
        print(f"  events={result['positive_event_rows']}")

    baseline = arms["existing_50guided_50random"]
    effects = {
        name: {
            "unique_input_gain": arm["unique_observation_action_keys"] -
                                 baseline["unique_observation_action_keys"],
            "reachable_coverage_gain": arm["reachable_input_fraction_covered"] -
                                       baseline["reachable_input_fraction_covered"],
            "unique_supervised_gain": arm["unique_supervised_transitions"] -
                                      baseline["unique_supervised_transitions"],
        }
        for name, arm in arms.items() if name != "existing_50guided_50random"
    }
    summary = {
        "question": "Can a fixed 5765-step budget yield more distinct visible state-action experience by replacing redundant guided episodes?",
        "existing_data": str(source),
        "all_arms_real_interaction_budget": budget,
        "new_simulator_interactions": budget * 2,
        "initial_guided_episodes_per_new_arm": int(ecfg["guided_episodes"]),
        "new_policy_seed": int(ecfg["seed"]),
        "novelty_epsilon": float(ecfg["novelty_epsilon"]),
        "reachable_graph": graph_info,
        "reachable_input_key_count": len(reachable),
        "results": arms,
        "effects_vs_existing_collection": effects,
        "limitations": [
            "Only the fixed Workshop layout; no new-layout generalization is measured.",
            "Novelty is the number of encounters with visible observation/action pairs, not information gain or epistemic uncertainty.",
            "The original 5765 training transitions are reused; the two prospective policies each collect exactly the same number of simulator interactions.",
            "The policies do not use hidden flags or diagnostic state, but graph coverage is measured post hoc using simulator state.",
            "This measures data coverage and event support, NOT model validation accuracy or downstream policy success.",
            "Single preregistered policy RNG seed; replicate across seeds if coverage gains matter.",
        ],
    }
    destination = Path(ecfg["output_dir"]) / "summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved coverage ablation: {destination}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/coverage_collection.yaml")
    run(parser.parse_args().config)
