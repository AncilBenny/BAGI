"""REUSE-18: diagnostic-only causal stage traces of the frozen balanced BC.

Replays the two verified REUSE-17 class-balanced models from original numeric
REUSE-16 DAgger pools. No new expert labels or learned algorithm. Full state
is inspected solely AFTER policy selects each action for evaluation telemetry.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from bagi.experiments.xland_dagger_balance_control import fixed_sampler_indices
from bagi.experiments.xland_dagger_replication import (
    EXPECTED_ANCHOR_SUCCESSES,
    load_frozen_numeric_pool,
)
from bagi.experiments.xland_existing_bc import (
    ALL_TASKS,
    TRAIN_TASKS,
    get_rules,
    visible_features,
)

FROZEN_FRESH_SUCCESSES = {
    17: {"hold": (9202, 9203, 9204), "near": (9203, 9204),
         "putdown": (), "hold_then_near": ()},
    29: {name: () for name in ALL_TASKS},
}


def inventory_and_grid(state):
    """Ground milestone probes in current XLand pocket and object tiles."""
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles

    grid = np.asarray(state.grid)
    pocket = np.asarray(state.agent.pocket)
    square = np.asarray(TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE])
    red = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.RED])
    green = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.GREEN])
    squares = np.argwhere(np.all(grid == square, axis=-1))
    reds = np.argwhere(np.all(grid == red, axis=-1))
    adjacent = any(
        int(np.abs(a - b).sum()) == 1 for a in squares for b in reds
    )
    return {
        "pocket_empty": int(pocket[0]) == int(Tiles.EMPTY),
        "carrying_square": bool(np.array_equal(pocket, square)),
        "carrying_red": bool(np.array_equal(pocket, red)),
        "carrying_green": bool(np.array_equal(pocket, green)),
        "green_on_grid": bool(np.any(np.all(grid == green, axis=-1))),
        "square_next_to_red": bool(adjacent),
    }


def evaluate_stages(policy, env, defaults, seeds, cap):
    """Policy sees local image+heading; evaluator inspects state after action."""
    import jax

    reset, step = jax.jit(env.reset), jax.jit(env.step)
    report = {}
    for task in ALL_TASKS:
        params = defaults.replace(ruleset=get_rules(task), max_steps=cap)
        cases = []
        for seed in seeds:
            ts = reset(params, jax.random.key(seed))
            action_counts = Counter()
            milestone_first = {}
            events = Counter()
            success = steps = 0
            while steps < cap and not bool(ts.last()):
                observation = visible_features(ts.observation)
                output, _ = policy.predict(observation, deterministic=True)
                action = int(np.asarray(output).item())
                if action not in range(6):
                    raise ValueError("Policy selected invalid native action")
                before = inventory_and_grid(ts.state)
                ts = step(params, ts, action)
                after = inventory_and_grid(ts.state)
                steps += 1
                action_counts[action] += 1
                success |= int(float(ts.reward) > 0)
                if action == 3 and before["pocket_empty"] and not after["pocket_empty"]:
                    events["effective_pickup"] += 1
                    milestone_first.setdefault("effective_pickup", steps)
                if action == 4 and not before["pocket_empty"] and after["pocket_empty"]:
                    events["effective_putdown"] += 1
                    milestone_first.setdefault("effective_putdown", steps)
                if after["green_on_grid"] and not before["green_on_grid"]:
                    events["new_green_transformation"] += 1
                    milestone_first.setdefault("new_green_transformation", steps)
                for key in ("carrying_square", "carrying_red", "carrying_green",
                            "green_on_grid", "square_next_to_red"):
                    if after[key]:
                        events[key] += 1
                        milestone_first.setdefault(key, steps)
            cases.append({
                "seed": seed, "goal_success": success, "real_steps": steps,
                "action_counts": dict(action_counts),
                "first_reached_milestone_steps": milestone_first,
                "milestone_observed_step_counts": dict(events),
            })
        keys = ("effective_pickup", "effective_putdown", "new_green_transformation",
                "carrying_square", "carrying_red", "carrying_green",
                "green_on_grid", "square_next_to_red")
        report[task] = {
            "successes": sum(x["goal_success"] for x in cases),
            "episodes": len(cases),
            "simulator_steps": sum(x["real_steps"] for x in cases),
            "episodes_reaching_milestone": {
                k: sum(k in x["first_reached_milestone_steps"] for x in cases)
                for k in keys
            },
            "details": cases,
        }
    return report


def successful_seed_ids(report):
    return {
        name: tuple(r["seed"] for r in item["details"] if r["goal_success"])
        for name, item in report.items()
    }


def run(config_path):
    import gymnasium as gym
    import torch
    import xminigrid
    from imitation.algorithms.bc import BC
    from imitation.data.types import Transitions
    from xminigrid.wrappers import DirectionObservationWrapper

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if (cfg["environment"] != "XLand-MiniGrid-R1-9x9"
            or cfg["training_tasks"] != list(TRAIN_TASKS)
            or cfg["model_seeds"] != [17, 29]
            or cfg["original_seeds"] != [9100, 9101, 9102]
            or cfg["replicated_seeds"] != list(range(9200, 9206))
            or int(cfg["source_run_id"]) != 39619756922
            or int(cfg["replication_run_id"]) != 39621044776
            or int(cfg["batch_size"]) != 32
            or int(cfg["epochs"]) != 12
            or int(cfg["episode_cap"]) != 40):
        raise ValueError("REUSE-18 frozen replay contract changed")
    torch.set_num_threads(1)
    room, defaults = xminigrid.make(cfg["environment"])
    local = DirectionObservationWrapper(room)
    models = []
    for seed in (17, 29):
        data = load_frozen_numeric_pool(cfg["source_dir"], seed)
        idx = fixed_sampler_indices(data["acts"], seed, "class_balanced")
        demos = Transitions(
            obs=data["obs"][idx], acts=data["acts"][idx],
            next_obs=data["next_obs"][idx], dones=data["dones"][idx],
            infos=np.array([None] * len(idx), dtype=object),
        )
        torch.manual_seed(seed)
        learner = BC(
            observation_space=gym.spaces.Box(
                low=0.0, high=255.0, shape=(54,), dtype=np.float32
            ),
            action_space=gym.spaces.Discrete(6),
            demonstrations=demos, rng=np.random.default_rng(seed),
            batch_size=32, device="cpu",
        )
        learner.train(n_epochs=12, progress_bar=False)
        anchor = evaluate_stages(
            learner.policy, local, defaults, cfg["original_seeds"], 40
        )
        fresh = evaluate_stages(
            learner.policy, local, defaults, cfg["replicated_seeds"], 40
        )
        if (successful_seed_ids(anchor) != EXPECTED_ANCHOR_SUCCESSES[
                (seed, "class_balanced")]
                or successful_seed_ids(fresh) != FROZEN_FRESH_SUCCESSES[seed]):
            raise AssertionError("Frozen model results diverged from REUSE-17")
        models.append({
            "model_seed": seed,
            "verified_source_sha256": data["sha256"],
            "reward_seed_reproduction": True,
            "reference": anchor, "replication": fresh,
        })
        print(f"seed {seed}: verified exact reward seeds; "
              f"putdown pickup={fresh['putdown']['episodes_reaching_milestone']['effective_pickup']}/6 "
              f"effective drop={fresh['putdown']['episodes_reaching_milestone']['effective_putdown']}/6")
    result = {
        "question": "Where do frozen learned XLand policies fail the causal stages?",
        "method": "Upstream BC replay and simulator phase telemetry; not new learning",
        "source_run_id": 39619756922,
        "replication_run_id": 39621044776,
        "new_teacher_queries": 0,
        "new_online_training_interactions": 0,
        "heldout_demonstrations": 0,
        "privileged_policy_inputs": 0,
        "privileged_evaluator_state_only": True,
        "models": models,
        "real_simulator_evaluation_steps": sum(
            model[part][task]["simulator_steps"]
            for model in models
            for part in ("reference", "replication")
            for task in ALL_TASKS
        ),
        "limitations": [
            "Same seeds as REUSE-17; post-hoc explanation, NOT an independent generalization result.",
            "Privileged grid and pocket inspect events only after learner action choice.",
            "Selecting action 3/4 does not guarantee pickup/drop has occurred.",
            "Two policies, one room; no robust causal transfer or new learning algorithm.",
            "Held-out chain is evaluated for reward, never expert labeled.",
        ],
    }
    dest = Path(cfg["output_dir"]) / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved REUSE-18 phase evidence: {dest}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xland_phase_diagnostics.yaml")
    run(parser.parse_args().config)
