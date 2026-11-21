"""REUSE-33: source-only complete physical expert episodes vs equal-budget BC.

Use established XLand TRAIN correction_plan and HumanCompatibleAI imitation.BC.
All policies see exactly the existing 115 public task+executed-history features.
Privileged state is confined to source-teacher labeling and evaluation gates.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from bagi.experiments.xland_gate_b_exact_collection import (
    CAP,
    EXPERT_STAGE_CODES,
    INPUT_WIDTH,
    native_action,
    public_features,
)
from bagi.experiments.xland_gate_b_native_collection import (
    ENVIRONMENT,
    MODEL_SEEDS,
    SOURCE_RUN,
    screen_source_layout,
    source_record_checks,
)

TRAIN_LAYOUTS = tuple(range(10400, 10424))
EVAL_LAYOUTS = tuple(range(10600, 10610))
MIN_COMPLETE_PHYSICAL = 12
TRAIN_EPOCHS = 12
TRAIN_BATCH = 32
SOURCE_BASE_ROWS = 291
ARM_NAMES = ("repeated_original_control", "complete_teacher_trajectories")


def verify_registration() -> None:
    """Immutable resource limits: never train on the independent layouts."""
    if (len(TRAIN_LAYOUTS) != 24 or len(EVAL_LAYOUTS) != 10
            or set(TRAIN_LAYOUTS) & set(EVAL_LAYOUTS)
            or len(set(TRAIN_LAYOUTS)) != len(TRAIN_LAYOUTS)
            or len(set(EVAL_LAYOUTS)) != len(EVAL_LAYOUTS)
            or MIN_COMPLETE_PHYSICAL != 12 or CAP != 40
            or INPUT_WIDTH != 115 or MODEL_SEEDS != (17, 29)):
        raise ValueError("Unregistered REUSE-33 source/eval/budget contract")


def actual_history(history: tuple[int, ...], action: int) -> tuple[int, ...]:
    if len(history) != 4 or any(i not in range(7) for i in history):
        raise ValueError("Corrupt preceding executed history")
    return (*history[1:], native_action(np.asarray(action, dtype=np.int64)))


def collect_one(reset, step, teacher, observe, check_physical, final_adjacent,
                eligible, layout: int) -> tuple[list[dict], dict]:
    """Use EVERY genuine teacher decision until terminal/cap; no stage cherry-pick.

    A trajectory is eligible to teach the COMPLETE placement skill only if
    pickup, later drop, final adjacency and native reward were all observed.
    Every attempted query and executed native action is counted regardless.
    """
    ts = reset(layout)
    if not eligible(ts):
        return [], {"layout": layout, "failure": "initially_ineligible",
                    "native_steps": 0, "queries": 0, "physical": False}
    history = (6, 6, 6, 6)
    rows: list[dict] = []
    pickup_count = drop_count = 0
    seen_pickup = ordered_drop = reward_seen = False
    queries = steps = 0
    failure = None
    while steps < CAP and not bool(ts.last()):
        try:
            feature = np.asarray(observe(ts, history), dtype=np.float32)
            if feature.shape != (INPUT_WIDTH,) or not np.isfinite(feature).all():
                raise ValueError("Invalid learner-legal public features")
            expected = np.eye(7, dtype=np.float32)[
                np.asarray(history, dtype=np.int64)
            ].reshape(-1)
            if not np.array_equal(feature[-28:], expected):
                raise ValueError("Inconsistent executed-action history features")
            queries += 1
            plan, stage = teacher(ts)
            if not plan or stage not in EXPERT_STAGE_CODES:
                raise ValueError("Unsupported privileged teacher stage or plan")
            taken = native_action(np.asarray(plan[0], dtype=np.int64))
            after = step(ts, taken)
            picked, dropped = check_physical(ts, after, taken)
            pickup_count += int(picked)
            drop_count += int(dropped)
            ordered_drop |= bool(dropped and seen_pickup)
            seen_pickup |= bool(picked)
            reward_seen |= float(after.reward) > 0
            rows.append({
                "public_obs": feature, "expert_label": taken,
                "done": bool(after.last()), "history": history,
                "layout": layout, "step": steps, "stage": stage,
            })
            ts = after
            history = actual_history(history, taken)
            steps += 1
        except (ValueError, IndexError) as exc:
            failure = type(exc).__name__ + ":" + str(exc)
            break
    physical = bool(reward_seen and pickup_count > 0 and drop_count > 0
                    and ordered_drop and final_adjacent(ts))
    audit = {
        "layout": layout, "failure": failure,
        "native_steps": steps, "queries": queries,
        "native_reward": bool(reward_seen),
        "blue_pickups": pickup_count, "blue_drops": drop_count,
        "ordered_blue_drop": bool(ordered_drop),
        "physical": physical,
        "labels_used": len(rows) if physical else 0,
        "terminal": bool(ts.last()),
    }
    return rows if physical else [], audit


def collect_train_demonstrations():
    import jax
    import xminigrid
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
    from xminigrid.wrappers import (
        DirectionObservationWrapper,
        RulesAndGoalsObservationWrapper,
    )

    from bagi.experiments.xland_dagger_teacher_feasibility import correction_plan
    from bagi.experiments.xland_existing_bc import get_rules
    from bagi.experiments.xland_putdown_reward_validity import (
        blue_square_transition,
        tiles_adjacent,
    )

    room, defaults = xminigrid.make(ENVIRONMENT)
    views = RulesAndGoalsObservationWrapper(DirectionObservationWrapper(room))
    params = defaults.replace(ruleset=get_rules("putdown"), max_steps=CAP)
    reset_jax, step_jax = jax.jit(views.reset), jax.jit(views.step)
    blue = np.asarray(TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE])
    red = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.RED])
    empty = int(Tiles.EMPTY)

    def reset(layout):
        return reset_jax(params, jax.random.key(layout))

    def step(ts, taken):
        return step_jax(params, ts, taken)

    def teacher(ts):
        return correction_plan(ts, "putdown")

    def movement(before, after, taken):
        return blue_square_transition(
            taken, np.asarray(before.state.agent.pocket),
            np.asarray(after.state.agent.pocket),
            np.asarray(before.state.grid), np.asarray(after.state.grid),
            blue, empty,
        )

    all_rows = []
    all_cases = []
    for layout in TRAIN_LAYOUTS:
        rows, case = collect_one(
            reset, step, teacher, public_features, movement,
            lambda ts: tiles_adjacent(np.asarray(ts.state.grid), blue, red),
            lambda ts: screen_source_layout(ts, blue, red), layout,
        )
        all_rows.extend(rows)
        all_cases.append(case)
    return all_rows, {
        "predeclared_source_layouts": list(TRAIN_LAYOUTS),
        "eligible_physical_episode_count": sum(int(x["physical"]) for x in all_cases),
        "attempted_teacher_queries": sum(x["queries"] for x in all_cases),
        "native_teacher_actions": sum(x["native_steps"] for x in all_cases),
        "label_rows": len(all_rows),
        "complete_teacher_stage_counts": dict(Counter(x["stage"] for x in all_rows)),
        "complete_teacher_action_counts": dict(Counter(x["expert_label"] for x in all_rows)),
        "cases": all_cases,
    }


def validate_demonstrations(rows: list[dict], audit: dict) -> dict:
    """Don't train a phase-deficient learner or conceal failed TRAIN layouts."""
    if (len(audit["cases"]) != len(TRAIN_LAYOUTS)
            or [c["layout"] for c in audit["cases"]] != list(TRAIN_LAYOUTS)
            or audit["native_teacher_actions"] > len(TRAIN_LAYOUTS) * CAP
            or audit["native_teacher_actions"] > audit["attempted_teacher_queries"]
            or audit["eligible_physical_episode_count"] < MIN_COMPLETE_PHYSICAL
            or len(rows) != audit["label_rows"]
            or not rows):
        raise ValueError("Insufficient valid complete physical TRAIN demonstrations")
    counts = Counter(x["expert_label"] for x in rows)
    stages = Counter(x["stage"] for x in rows)
    if not (counts[3] >= MIN_COMPLETE_PHYSICAL
            and counts[4] >= MIN_COMPLETE_PHYSICAL
            and stages["carrying_square"] >= MIN_COMPLETE_PHYSICAL):
        raise ValueError("Missing genuine pickup/carry/drop label coverage")
    for row in rows:
        if (row["layout"] not in TRAIN_LAYOUTS
                or row["stage"] not in EXPERT_STAGE_CODES
                or row["expert_label"] not in range(6)
                or row["public_obs"].shape != (INPUT_WIDTH,)):
            raise ValueError("Invalid source demonstration provenance")
    return {"complete_physical_train_episodes": audit["eligible_physical_episode_count"],
            "genuine_source_pickup_labels": counts[3],
            "genuine_source_drop_labels": counts[4],
            "source_stage_counts": dict(stages)}


def sampled_indices(labels, size: int, seed: int) -> np.ndarray:
    """Published PyTorch WeightedRandomSampler, same exposures in both arms."""
    import torch
    from torch.utils.data import WeightedRandomSampler

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) < 1 or size < 1:
        raise ValueError("Invalid class-balanced label sampling budget")
    freq = np.bincount(labels, minlength=6)
    weights = np.array([1.0 / freq[v] for v in labels], dtype=np.float64)
    sample = WeightedRandomSampler(
        weights, num_samples=size, replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return np.fromiter(sample, dtype=np.int64, count=size)


def train_two_arms(base: dict, extra: dict, seed: int):
    import gymnasium as gym
    import torch
    from imitation.algorithms.bc import BC
    from imitation.data.types import Transitions

    base_x = np.asarray(base["obs"], dtype=np.float32)
    base_y = np.asarray(base["expert_label"], dtype=np.int64)
    base_done = np.asarray(base["done"], dtype=bool)
    extra_x = np.asarray(extra["obs"], dtype=np.float32)
    extra_y = np.asarray(extra["expert_label"], dtype=np.int64)
    extra_done = np.asarray(extra["done"], dtype=bool)
    if (base_x.shape != (SOURCE_BASE_ROWS, INPUT_WIDTH)
            or base_y.shape != (SOURCE_BASE_ROWS,)
            or base_done.shape != (SOURCE_BASE_ROWS,)
            or extra_x.shape != (len(extra_y), INPUT_WIDTH)
            or extra_done.shape != (len(extra_y),)
            or len(extra_y) < MIN_COMPLETE_PHYSICAL * 2):
        raise ValueError("Unmatched base or completed teacher row shape")
    base_sample = sampled_indices(base_y, len(base_y), seed)
    control_extra = sampled_indices(base_y, len(extra_y), seed + 4100)
    demonstrated_extra = sampled_indices(extra_y, len(extra_y), seed + 4100)
    arms = {}
    torch.set_num_threads(1)
    for name, indices, x_extra, y_extra, done_extra in (
        (ARM_NAMES[0], control_extra, base_x, base_y, base_done),
        (ARM_NAMES[1], demonstrated_extra, extra_x, extra_y, extra_done),
    ):
        x = np.concatenate((base_x[base_sample], x_extra[indices]))
        y = np.concatenate((base_y[base_sample], y_extra[indices]))
        done = np.concatenate((base_done[base_sample], done_extra[indices]))
        torch.manual_seed(seed)
        trainer = BC(
            observation_space=gym.spaces.Box(low=-np.inf, high=np.inf,
                                              shape=(INPUT_WIDTH,), dtype=np.float32),
            action_space=gym.spaces.Discrete(6),
            demonstrations=Transitions(
                obs=x, acts=y, next_obs=x.copy(), dones=done,
                infos=np.array([None] * len(y), dtype=object),
            ),
            rng=np.random.default_rng(seed), batch_size=TRAIN_BATCH,
            device="cpu",
        )
        trainer.train(n_epochs=TRAIN_EPOCHS, progress_bar=False)
        arms[name] = {
            "policy": trainer.policy,
            "training": {"original_source_rows": SOURCE_BASE_ROWS,
                         "added_sampled_rows": len(extra_y),
                         "total_rows": len(y),
                         "epochs": TRAIN_EPOCHS,
                         "nominal_optimization_exposures": len(y) * TRAIN_EPOCHS,
                         "sampled_original_base_rows_identical_between_arms": True,
                         "sampling_method": "torch.WeightedRandomSampler",
                         "label_action_counts": dict(Counter(map(int, y)))},
        }
    return arms


def evaluate_policy(policy) -> dict:
    """Frozen source-disjoint native validation, no teacher or training labels."""
    import jax
    import xminigrid
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
    from xminigrid.wrappers import (
        DirectionObservationWrapper,
        RulesAndGoalsObservationWrapper,
    )

    from bagi.experiments.xland_existing_bc import get_rules
    from bagi.experiments.xland_gate_c_ppo_memory import run_closed_loop_episode
    from bagi.experiments.xland_putdown_reward_validity import (
        blue_square_transition,
        tiles_adjacent,
    )

    room, defaults = xminigrid.make(ENVIRONMENT)
    views = RulesAndGoalsObservationWrapper(DirectionObservationWrapper(room))
    reset, step = jax.jit(views.reset), jax.jit(views.step)
    blue = np.asarray(TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE])
    red = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.RED])
    empty = int(Tiles.EMPTY)
    result = {}
    for task in ("hold", "near", "putdown"):
        params = defaults.replace(ruleset=get_rules(task), max_steps=CAP)
        cases = []
        for layout in EVAL_LAYOUTS:
            ts = reset(params, jax.random.key(layout))
            eligible = task != "putdown" or screen_source_layout(ts, blue, red)

            def native_step(timestep, action, *, params=params):
                return step(params, timestep, action)

            def physical(before, after, action):
                return blue_square_transition(
                    action, np.asarray(before.state.agent.pocket),
                    np.asarray(after.state.agent.pocket),
                    np.asarray(before.state.grid), np.asarray(after.state.grid),
                    blue, empty,
                )

            def adjacent(timestep):
                return tiles_adjacent(np.asarray(timestep.state.grid), blue, red)

            measured = run_closed_loop_episode(
                policy, ts, task=task, native_step=native_step,
                observe=public_features, eligible=eligible,
                on_physical_step=physical if task == "putdown" else None,
                final_adjacent=adjacent if task == "putdown" else None,
            )
            cases.append({"layout": layout, **measured})
        result[task] = {
            "cases": cases,
            "eligible": sum(int(x["eligible"]) for x in cases),
            "native_rewarded": sum(int(x["native_reward"]) for x in cases),
            "verified_physical_placements": sum(
                int(x["physical_placement"]) for x in cases
            ),
            "effective_blue_pickups": sum(x.get("effective_blue_pickups", 0)
                                          for x in cases),
            "effective_blue_drops": sum(x.get("effective_blue_drops", 0)
                                        for x in cases),
            "native_evaluation_steps": sum(x["native_actions"] for x in cases),
        }
    return result


def run(source_dir: Path, output_dir: Path) -> dict:
    """Collect complete TRAIN teacher trajectories, then compare frozen BC."""
    from bagi.experiments.xland_existing_bc import TRAIN_TASKS, get_rules
    from bagi.experiments.xland_instruction_action_history import (
        assemble,
        check_true_history,
    )
    from bagi.experiments.xland_upstream_instruction_bc import (
        load_verified_source,
        task_instructions,
    )

    verify_registration()
    source_summary = json.loads((source_dir / "summary.json").read_text())
    source_record_checks(source_summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    result = {
        "experiment": "REUSE-33 full physical TRAIN teacher trajectories",
        "source_run": SOURCE_RUN,
        "source_artifact": 12362840074,
        "status": "collecting",
        "predeclared_train_layouts": list(TRAIN_LAYOUTS),
        "predeclared_fresh_eval_layouts": list(EVAL_LAYOUTS),
        "minimum_complete_physical_teacher_episodes": MIN_COMPLETE_PHYSICAL,
        "heldout_composition_teacher_queries": 0,
        "new_postintervention_training_updates": 0,
        "native_training_teacher_actions": 0,
        "native_eval_learner_actions": 0,
        "results": [],
    }
    path.write_text(json.dumps(result, indent=2))
    rows, audit = collect_train_demonstrations()
    result["source_teacher_provenance"] = audit
    result["native_training_teacher_actions"] = audit["native_teacher_actions"]
    result["source_teacher_query_attempts"] = audit["attempted_teacher_queries"]
    path.write_text(json.dumps(result, indent=2))
    evidence = validate_demonstrations(rows, audit)
    result["physical_source_teacher_evidence"] = evidence
    extra = {
        "obs": np.stack([x["public_obs"] for x in rows]),
        "expert_label": np.asarray([x["expert_label"] for x in rows], dtype=np.int64),
        "done": np.asarray([x["done"] for x in rows], dtype=bool),
    }
    if len(extra["obs"]) != audit["label_rows"]:
        raise AssertionError("Recorded teacher row count changed")
    # Numeric-only, no pickle, no teacher stage exposed as learned input.
    np.savez_compressed(output_dir / "train_source_complete_demos.npz", **extra)
    instructions = {task: task_instructions(get_rules(task).goal,
                                             get_rules(task).rules, enabled=True)
                    for task in TRAIN_TASKS}
    for seed in MODEL_SEEDS:
        models = [m for m in source_summary["model_results"]
                  if m["model_seed"] == seed]
        if len(models) != 1:
            raise ValueError("Missing unique original source BC seed")
        data, tasks, checksum = load_verified_source(
            source_dir / f"action_provenance_seed{seed}.npz", models[0], seed,
        )
        check_true_history(data)
        base_x = assemble(data["obs"], np.stack([instructions[t] for t in tasks]),
                          data["prior_executed_actions"],
                          "real_executed_action_history")
        training = train_two_arms(
            {"obs": base_x, "expert_label": data["expert_label"],
             "done": data["done"]}, extra, seed,
        )
        for arm in ARM_NAMES:
            record = training[arm]
            evaluated = evaluate_policy(record["policy"])
            actions = sum(v["native_evaluation_steps"] for v in evaluated.values())
            result["native_eval_learner_actions"] += actions
            result["results"].append({
                "model_seed": seed, "arm": arm,
                "original_source_sha256": checksum,
                "training": record["training"],
                "evaluation": evaluated,
                "native_eval_actions": actions,
                "teacher_queries_during_policy_eval": 0,
            })
            path.write_text(json.dumps(result, indent=2))
    result["status"] = "complete"
    result["limitations"] = [
        "Full expert trajectories are reset-state privileged teacher demos, not on-policy DAgger.",
        "Only 24 preregistered source layouts; missing successes are recorded, never replaced.",
        "The native putdown goal alone permits vacuous rewards; only ordered physical transitions count.",
        "Even positive physical skill on these 10 fresh layouts is not AGI or heldout causal transfer.",
        "Identical optimizer epochs and source base rows but added data has different label distribution.",
        "No heldout composition teacher labels, policy privileged features, or unsafe pickle archives.",
    ]
    path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/xland_complete_demo_bc"))
    arguments = parser.parse_args()
    run(arguments.source_dir, arguments.output_dir)
