"""XLand-MiniGrid compositional task-split smoke test using upstream primitives.

No pretrained weights, training, benchmark downloads or pickle deserialization.
This only establishes reproducible disjoint causal tasks for future research.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import yaml


def ruleset_for(spec: dict):
    """Compose published XLand RuleSet, AgentNearRule and AgentNearGoal primitives."""
    import jax.numpy as jnp
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
    from xminigrid.core.goals import AgentNearGoal
    from xminigrid.core.rules import AgentNearRule
    from xminigrid.types import RuleSet

    source = TILES_REGISTRY[
        getattr(Tiles, spec["source_tile"]), getattr(Colors, spec["source_color"])
    ]
    result = TILES_REGISTRY[
        getattr(Tiles, spec["result_tile"]), getattr(Colors, spec["result_color"])
    ]
    if bool(jnp.array_equal(source, result)):
        raise ValueError("Task must actually transform one tile into another")
    return RuleSet(
        goal=AgentNearGoal(tile=result).encode(),
        rules=AgentNearRule(tile=source, prod_tile=result).encode()[None, :],
        init_tiles=jnp.stack([source]),
    )


def fingerprint(ruleset) -> str:
    """Hash all executable rule, goal and initial-tile encodings."""
    import numpy as np

    payload = b"".join(
        np.asarray(value, dtype=np.uint8).tobytes()
        for value in (ruleset.goal, ruleset.rules, ruleset.init_tiles)
    )
    return hashlib.sha256(payload).hexdigest()


def disjoint_tasks(config: dict) -> tuple[list[dict], list[dict]]:
    groups = []
    used = set()
    for split in ("train", "held_out"):
        tasks = []
        for item in config["tasks"][split]:
            ruleset = ruleset_for(item)
            digest = fingerprint(ruleset)
            if digest in used:
                raise ValueError("Train/test rule-goal overlap or duplicate task")
            used.add(digest)
            tasks.append({"name": item["name"], "fingerprint": digest,
                          "specification": item, "ruleset": ruleset})
        if not tasks:
            raise ValueError("Each split needs one or more distinct tasks")
        groups.append(tasks)
    return groups[0], groups[1]


def evaluate_random(env, params, *, seed: int, episodes: int,
                    max_steps: int) -> dict:
    """Evaluation-only uniform random policy; no hidden simulator state input."""
    import jax
    import numpy as np

    if episodes < 1 or max_steps < 1:
        raise ValueError("Evaluation episode and step limits must be positive")
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    outcomes = []
    for episode in range(episodes):
        episode_seed = seed + episode
        timestep = reset(params, jax.random.key(episode_seed))
        action_rng = np.random.default_rng(episode_seed)
        transitions = 0
        total_reward = 0.0
        while transitions < max_steps and not bool(timestep.last()):
            action = int(action_rng.integers(0, env.num_actions(params)))
            timestep = step(params, timestep, action)
            total_reward += float(timestep.reward)
            transitions += 1
        outcomes.append({
            "episode_seed": episode_seed,
            "real_environment_steps": transitions,
            "reached_goal": bool(total_reward > 0),
            "return": total_reward,
        })
    return {
        "episodes": episodes,
        "real_environment_steps": sum(x["real_environment_steps"] for x in outcomes),
        "goal_successes": sum(x["reached_goal"] for x in outcomes),
        "episode_results": outcomes,
    }


def run(config_path: str) -> dict:
    import xminigrid

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    train, held_out = disjoint_tasks(config)
    budget = config["evaluation"]
    if budget["episodes_per_task"] < 1 or budget["max_steps_per_episode"] < 1:
        raise ValueError("Invalid fixed evaluation budget")
    if config["environment"] != "XLand-MiniGrid-R1-9x9":
        raise ValueError("Only the reviewed XLand environment is approved")
    env, default_params = xminigrid.make(config["environment"])
    rows = []
    for split, tasks in (("train", train), ("held_out", held_out)):
        for task in tasks:
            params = default_params.replace(
                ruleset=task["ruleset"],
                max_steps=int(budget["max_steps_per_episode"]),
            )
            stats = evaluate_random(
                env, params, seed=int(budget["evaluation_seed_base"]),
                episodes=int(budget["episodes_per_task"]),
                max_steps=int(budget["max_steps_per_episode"]),
            )
            rows.append({
                "split": split,
                "task": task["name"],
                "ruleset_sha256": task["fingerprint"],
                "goal_successes": stats["goal_successes"],
                "episodes": stats["episodes"],
                "real_environment_steps": stats["real_environment_steps"],
                "episode_results": stats["episode_results"],
            })
            print(f"{split}/{task['name']}: "
                  f"success={stats['goal_successes']}/{stats['episodes']}, "
                  f"steps={stats['real_environment_steps']}")
    output = {
        "research_stage": "reused benchmark contract; no learned-agent study",
        "existing_implementation": "dunnolab/xland-minigrid, xminigrid",
        "xminigrid_version": version("xminigrid"),
        "contribution_type": "existing causal goal/rule composition and split integration",
        "environment": config["environment"],
        "random_baseline": "uniform six-action random policy, no hidden state",
        "train_task_count": len(train),
        "heldout_task_count": len(held_out),
        "train_test_ruleset_hashes_disjoint": True,
        "evaluation_budget_max_real_steps": (
            (len(train) + len(held_out)) * int(budget["episodes_per_task"])
            * int(budget["max_steps_per_episode"])
        ),
        "real_steps_used": sum(x["real_environment_steps"] for x in rows),
        "results": rows,
        "no_external_pickle_loaded": True,
        "limitations": [
            "Compositional rulesets use only existing XLand primitives and human chosen pairs.",
            "Train/held-out are distinct goal+rule combinations, not a trained transfer comparison.",
            "No agent is trained here, and zero random-policy success would be expected.",
            "Do not provide the hidden ruleset to an agent unless specified by a future information-matched protocol.",
            "The upstream XLand benchmark loader currently unpickles externally downloaded task data and is NOT used.",
        ],
    }
    dest = Path(config["output_dir"]) / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved reproducible composition split: {dest}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xland_composition_smoke.yaml")
    run(parser.parse_args().config)
