"""Gate A5: seeded start-chamber diversity under a fixed interaction budget.

Explicit expert navigation uses reset position ONLY to construct demonstrations,
never as input to a learned controller. Random exploration uses no hidden state.
The fixed-start Workshop behavior remains the default for prior research.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import yaml

from bagi.experiments.dataset_foundations import EVENTS, model_input, supervised_transition
from bagi.worlds import WorkshopEnv


def expert_actions_from_start(position: list[int], guided_actions: list[int]) -> list[int]:
    """Privileged demonstration route to historical start, then fixed guided path."""
    row, col = (int(x) for x in position)
    if row not in (1, 2) or col < 1:
        raise ValueError("Expected pre-barrier starting cell")
    return [0] * (row - 1) + [3] * (col - 1) + guided_actions


def collect(
    env: WorkshopEnv, budget: int, expert_episodes: int, seed: int,
    guided_actions: list[int],
) -> tuple[list[dict], dict]:
    if budget < 1 or expert_episodes < 1:
        raise ValueError("Positive transition and demonstration budgets required")
    rows: list[dict] = []
    positions = Counter()
    event_counts = Counter()
    expert_steps = 0
    expert_goals = 0
    rng = random.Random(seed + 50_000)
    episode = 0

    def execute(observation, action: int) -> tuple:
        next_observation, reward, terminal, truncated, info = env.step(action)
        rows.append({
            "episode": episode,
            "observation": observation.tolist(),
            "action": int(action),
            "next_observation": next_observation.tolist(),
            "reward": float(reward),
            "info": {"events": list(info["events"])},
        })
        event_counts.update(event for event in info["events"] if event in EVENTS)
        return next_observation, terminal or truncated, info

    for episode in range(expert_episodes):
        if len(rows) >= budget:
            break
        observation, info = env.reset(seed=seed + episode)
        start = [int(x) for x in info["position"]]
        positions[tuple(start)] += 1
        plan = expert_actions_from_start(start, guided_actions)
        done = False
        for action in plan:
            if len(rows) >= budget or done:
                break
            observation, done, step_info = execute(observation, action)
            expert_steps += 1
            if "goal_reached" in step_info["events"]:
                expert_goals += 1
        if not done and len(rows) < budget:
            raise AssertionError("Demonstration path failed to reach a terminal state")

    episode = expert_episodes
    while len(rows) < budget:
        observation, info = env.reset(seed=seed + 200_000 + episode)
        positions[tuple(int(x) for x in info["position"])] += 1
        for _ in range(env.max_steps):
            if len(rows) >= budget:
                break
            action = rng.randrange(env.action_space.n)
            observation, done, _ = execute(observation, action)
            if done:
                break
        episode += 1
    assert len(rows) == budget
    keys = {model_input(row) for row in rows}
    return rows, {
        "real_interactions": budget,
        "distinct_starts": len(positions),
        "start_positions": {f"{r},{c}": n for (r, c), n in sorted(positions.items())},
        "expert_episodes": expert_episodes,
        "expert_interactions": expert_steps,
        "expert_goal_episodes": expert_goals,
        "unique_visible_state_action_keys": len(keys),
        "unique_supervised_transitions": len({
            supervised_transition(row) for row in rows
        }),
        "event_positive_rows": {
            event: event_counts[event] for event in EVENTS
        },
    }


def run(config_path: str) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    env_config = cfg["environment"]
    design = cfg["evaluation"]
    budget = int(design["real_interaction_budget"])
    experts = int(design["guided_episodes"])
    guided = [int(action) for action in design["guided_actions"]]
    runs = []
    for seed in [int(x) for x in design["collection_seeds"]]:
        arm_rows = {}
        for mode in ("fixed", "seeded_start_chamber"):
            env = WorkshopEnv(**env_config, start_mode=mode)
            rows, stats = collect(env, budget, experts, seed, guided)
            arm_rows[mode] = rows
            runs.append({"seed": seed, "start_mode": mode, **stats})
            print(f"seed={seed} start={mode} unique={stats['unique_visible_state_action_keys']} "
                  f"distinct_starts={stats['distinct_starts']} "
                  f"guided_goals={stats['expert_goal_episodes']}")

        fixed_keys = {model_input(row) for row in arm_rows["fixed"]}
        varied_keys = {model_input(row) for row in arm_rows["seeded_start_chamber"]}
        for item in runs[-2:]:
            item["unique_keys_vs_other_arm"] = len(
                fixed_keys - varied_keys if item["start_mode"] == "fixed"
                else varied_keys - fixed_keys
            )
    paired = []
    for seed in design["collection_seeds"]:
        base = next(x for x in runs if x["seed"] == int(seed) and x["start_mode"] == "fixed")
        varied = next(
            x for x in runs if x["seed"] == int(seed)
            and x["start_mode"] == "seeded_start_chamber"
        )
        paired.append({
            "seed": int(seed),
            "unique_input_key_difference": (
                varied["unique_visible_state_action_keys"] -
                base["unique_visible_state_action_keys"]
            ),
            "unique_transition_difference": (
                varied["unique_supervised_transitions"] -
                base["unique_supervised_transitions"]
            ),
            "goal_event_row_difference": (
                varied["event_positive_rows"]["goal_reached"] -
                base["event_positive_rows"]["goal_reached"]
            ),
        })
    result = {
        "question": "At matched total interactions and causal demonstrations, does seeded pre-barrier start variation improve distinct visible experience?",
        "environment": env_config,
        "design": design,
        "arm_results": runs,
        "paired_seed_differences": paired,
        "new_collection_simulator_steps": budget * len(design["collection_seeds"]) * 2,
        "limitations": [
            "Only starting positions vary within a single fixed Workshop layout; NO held-out layout or task transfer.",
            "A privileged position is used only by the scripted demonstration oracle to navigate to the historical start, NEVER by random exploration or a learned controller.",
            "Navigation costs differing expert-step budgets across arms, though each has identical total real interactions and expert goal demonstrations.",
            "Three paired seeds are exploratory; counts measure data diversity, not learned model improvement.",
            "The generated transitions are experiment data; the historical world model is not retrained.",
        ],
    }
    destination = Path(design["output_dir"]) / "summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved start-diversity ablation: {destination}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/start_diversity.yaml")
    run(parser.parse_args().config)
