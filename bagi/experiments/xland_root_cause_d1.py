"""G2-D1: preregistered XLand policy failure-funnel + memory-input ablation.

Only frozen public 115-wide imitation.BC features affect learner actions.
A separate discarded native-action branch probes *whether* a pickup was
possible, solely for diagnosis. Its simulator state never enters the policy,
training dataset, belief memory, or any future executed native step.

This is an interpretable intervention on a deployed policy's preceding-action
features, NOT a learned memory model, an oracle-corrected policy, or a general
intelligence benchmark.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from bagi.experiments.xland_gate_b_exact_collection import (
    CAP,
    INPUT_WIDTH,
    native_action,
)

MODEL_SEEDS = (17, 29)
TASKS = ("hold", "near", "putdown")
CONDITIONS = ("real_executed_history", "erased_history_control")
FRESH_LAYOUTS = tuple(range(11500, 11510))
PAD_HISTORY = np.eye(7, dtype=np.float32)[[6, 6, 6, 6]].reshape(-1)
SOURCE_ONLY_LAYOUTS = (
    set(range(9700, 9900)) | set(range(10400, 10424))
    | set(range(10600, 10810)) | set(range(10900, 10972))
    | set(range(11000, 11410))
)


def validate_registration() -> None:
    """Fail closed on changing the predeclared new evaluation experiment."""
    if (
        FRESH_LAYOUTS != tuple(range(11500, 11510))
        or set(FRESH_LAYOUTS) & SOURCE_ONLY_LAYOUTS
        or MODEL_SEEDS != (17, 29)
        or TASKS != ("hold", "near", "putdown")
        or CONDITIONS != ("real_executed_history", "erased_history_control")
        or INPUT_WIDTH != 115
        or CAP != 40
        or PAD_HISTORY.shape != (28,)
    ):
        raise ValueError("Unregistered root-cause layout/input/action budget")


def next_actual_history(history: tuple[int, ...], action: int) -> tuple[int, ...]:
    """Public history ALWAYS advances from the real learner-executed action."""
    if len(history) != 4 or any(int(a) not in range(7) for a in history):
        raise ValueError("Malformed true executed-action history")
    return (*history[1:], native_action(np.asarray(action, dtype=np.int64)))


def policy_features(public_obs: np.ndarray, history: tuple[int, ...],
                    condition: str) -> np.ndarray:
    """Ablate ONLY the last 28 history bits; leave all public task data intact."""
    if condition not in CONDITIONS:
        raise ValueError("Unknown intervention condition")
    features = np.asarray(public_obs, dtype=np.float32)
    expected = np.eye(7, dtype=np.float32)[list(history)].reshape(-1)
    if (
        features.shape != (INPUT_WIDTH,)
        or not np.isfinite(features).all()
        or not np.array_equal(features[-28:], expected)
    ):
        raise ValueError("True public observation/history mismatch")
    chosen = features.copy()
    if condition == CONDITIONS[1]:
        chosen[-28:] = PAD_HISTORY
    return chosen


def visible_target(public_image: object, target_tile: np.ndarray) -> bool:
    """Passive evidence only: target tile appears in the local public image."""
    image = np.asarray(public_image)
    target = np.asarray(target_tile)
    if image.ndim != 3 or image.shape[-1] != target.size:
        raise ValueError("Unexpected public local-image tile shape")
    return bool(np.any(np.all(image == target, axis=-1)))


def failure_stage(case: dict) -> str:
    """Descriptive failure funnel, never a claimed causal attribution."""
    if not case["eligible"]:
        return "initially_ineligible"
    if case["physical_placement"]:
        return "complete_physical_sequence"
    if case["target_visible_steps"] == 0:
        return "target_never_publicly_seen"
    if case["pickup_opportunity_steps"] == 0:
        return "target_seen_but_no_pickup_precondition"
    if case["effective_blue_pickups"] == 0:
        return "feasible_pickup_not_executed_successfully"
    if case["effective_blue_drops"] == 0:
        return "pickup_succeeded_but_no_effective_drop"
    return "drop_occurred_but_no_valid_final_goal"


def summarize_cases(cases: list[dict], task: str) -> dict:
    if task not in TASKS:
        raise ValueError("Unknown task")
    for item in cases:
        if item["task"] != task:
            raise ValueError("Mixed task source evaluation cases")
    eligible = [c for c in cases if c["eligible"]]
    result = {
        "eligible": len(eligible),
        "ineligible": len(cases) - len(eligible),
        "native_rewarded": sum(bool(c["native_reward"]) for c in eligible),
        "native_learner_actions": sum(int(c["native_actions"]) for c in cases),
        "discarded_counterfactual_native_probes": sum(
            int(c["counterfactual_native_probes"]) for c in cases
        ),
        "cases": cases,
    }
    if task == "putdown":
        result.update({
            "physical_placements": sum(bool(c["physical_placement"]) for c in eligible),
            "target_seen_episodes": sum(c["target_visible_steps"] > 0 for c in eligible),
            "pickup_opportunity_episodes": sum(
                c["pickup_opportunity_steps"] > 0 for c in eligible
            ),
            "effective_pickup_episodes": sum(
                c["effective_blue_pickups"] > 0 for c in eligible
            ),
            "effective_drop_episodes": sum(
                c["effective_blue_drops"] > 0 for c in eligible
            ),
            "pickup_opportunity_steps": sum(
                c["pickup_opportunity_steps"] for c in eligible
            ),
            "missed_feasible_pickup_steps": sum(
                c["missed_feasible_pickup_steps"] for c in eligible
            ),
            "failed_pickup_action_attempts": sum(
                c["failed_pickup_action_attempts"] for c in eligible
            ),
            "wrong_object_pickups": sum(c["wrong_object_pickups"] for c in eligible),
            "failure_funnel": dict(Counter(failure_stage(c) for c in cases)),
        })
    return result


def execute_diagnostic_episode(
    policy, timestep, *, layout: int, task: str, condition: str,
    native_step, observe, eligible: bool, target_tile,
    effective_transition, final_adjacent,
) -> dict:
    """Run unchanged native policy; measure discarded pickup counterfactuals.

    ``effective_transition`` and ``final_adjacent`` are EVALUATOR-ONLY. They
    are not provided to the chosen-action policy or used to change rollouts.
    ``native_step`` is called once for each real action, and at most once
    more for each distinct discarded hypothetical pickup.
    """
    if task not in TASKS or condition not in CONDITIONS or layout not in FRESH_LAYOUTS:
        raise ValueError("Unregistered evaluation task, condition or layout")
    record = {
        "layout": layout, "task": task, "condition": condition,
        "eligible": bool(eligible),
        "native_actions": 0, "counterfactual_native_probes": 0,
        "native_reward": False, "physical_placement": False,
        "target_visible_steps": 0, "pickup_opportunity_steps": 0,
        "missed_feasible_pickup_steps": 0,
        "pickup_action_attempts": 0, "failed_pickup_action_attempts": 0,
        "effective_blue_pickups": 0, "effective_blue_drops": 0,
        "wrong_object_pickups": 0,
        "first_target_seen_action": None, "first_pickup_opportunity_action": None,
        "first_effective_pickup_action": None, "first_effective_drop_action": None,
        "ordered_blue_drop": False,
        "native_action_histogram": {},
    }
    if not eligible:
        record["descriptive_failure_stage"] = "initially_ineligible"
        return record
    if bool(timestep.last()):
        raise ValueError("Eligible diagnostic already terminal at reset")
    history = (6, 6, 6, 6)
    seen_blue_pickup = False
    actions = Counter()
    blue_tile = np.asarray(target_tile)
    while record["native_actions"] < CAP and not bool(timestep.last()):
        public = np.asarray(observe(timestep, history), dtype=np.float32)
        inputs = policy_features(public, history, condition)
        predicted, _ = policy.predict(inputs, deterministic=True)
        action = native_action(np.asarray(predicted))
        actions[action] += 1
        if task == "putdown":
            can_see_blue = visible_target(timestep.observation["img"], blue_tile)
            if can_see_blue:
                record["target_visible_steps"] += 1
                if record["first_target_seen_action"] is None:
                    record["first_target_seen_action"] = record["native_actions"]
            if action == 3:
                record["pickup_action_attempts"] += 1
        before = timestep
        after = native_step(before, action)
        record["native_actions"] += 1
        record["native_reward"] |= float(after.reward) > 0
        if task == "putdown":
            blue_pick, blue_drop, wrong_pick = effective_transition(before, after, action)
            blue_pick, blue_drop, wrong_pick = bool(blue_pick), bool(blue_drop), bool(wrong_pick)
            opportunity = blue_pick
            # Probe the very same state *only when the learner did not issue pickup*.
            # This branch is discarded and has zero causal influence on the learner.
            # Probe even if BLUE is not visibly present; that avoids hiding a
            # real pickup opportunity behind a mistaken visibility detector.
            if action != 3 and not seen_blue_pickup:
                hypothetical = native_step(before, 3)
                record["counterfactual_native_probes"] += 1
                opportunity = bool(effective_transition(before, hypothetical, 3)[0])
            if opportunity:
                record["pickup_opportunity_steps"] += 1
                if record["first_pickup_opportunity_action"] is None:
                    record["first_pickup_opportunity_action"] = record["native_actions"] - 1
                if action != 3:
                    record["missed_feasible_pickup_steps"] += 1
            if action == 3 and not blue_pick:
                record["failed_pickup_action_attempts"] += 1
            if blue_pick:
                if record["first_effective_pickup_action"] is None:
                    record["first_effective_pickup_action"] = record["native_actions"] - 1
                record["effective_blue_pickups"] += 1
            if blue_drop:
                if record["first_effective_drop_action"] is None:
                    record["first_effective_drop_action"] = record["native_actions"] - 1
                record["effective_blue_drops"] += 1
                record["ordered_blue_drop"] |= seen_blue_pickup
            seen_blue_pickup |= blue_pick
            record["wrong_object_pickups"] += int(wrong_pick)
        history = next_actual_history(history, action)
        timestep = after
    record["native_action_histogram"] = dict(sorted(actions.items()))
    if task == "putdown":
        record["physical_placement"] = bool(
            record["native_reward"]
            and record["effective_blue_pickups"] > 0
            and record["effective_blue_drops"] > 0
            and record["ordered_blue_drop"]
            and final_adjacent(timestep)
        )
    record["descriptive_failure_stage"] = (
        failure_stage(record) if task == "putdown" else "not_a_putdown_task"
    )
    return record


def validate_paired_runs(results: list[dict]) -> dict:
    """Require both histories on each model seed and EVERY preregistered layout."""
    if len(results) != len(MODEL_SEEDS) * len(CONDITIONS):
        raise ValueError("Incomplete policy-seed/history-condition comparison")
    pairs = {}
    for row in results:
        key = (row["model_seed"], row["condition"])
        if key in pairs or key[0] not in MODEL_SEEDS or key[1] not in CONDITIONS:
            raise ValueError("Unregistered or duplicate model/condition")
        pairs[key] = row
        for task in TASKS:
            cases = row["tasks"][task]["cases"]
            if [c["layout"] for c in cases] != list(FRESH_LAYOUTS):
                raise ValueError("Missing, replaced, or reordered fixed evaluation layout")
            if any(c["native_actions"] > CAP for c in cases):
                raise ValueError("Native action cap exceeded")
    paired = {}
    for seed in MODEL_SEEDS:
        a = pairs[(seed, CONDITIONS[0])]["tasks"]
        b = pairs[(seed, CONDITIONS[1])]["tasks"]
        for task in TASKS:
            if [c["eligible"] for c in a[task]["cases"]] != [
                c["eligible"] for c in b[task]["cases"]
            ]:
                raise ValueError("Paired initial eligibility changed")
        paired[str(seed)] = {
            task: {
                "native_reward_real_history": a[task]["native_rewarded"],
                "native_reward_erased_history": b[task]["native_rewarded"],
                **({
                    "physical_real_history": a[task]["physical_placements"],
                    "physical_erased_history": b[task]["physical_placements"],
                    "pickup_opportunity_real_history": a[task]["pickup_opportunity_episodes"],
                    "pickup_opportunity_erased_history": b[task]["pickup_opportunity_episodes"],
                } if task == "putdown" else {}),
            } for task in TASKS
        }
    return paired


def run(source_dir: Path, old_dir: Path, diverse_dir: Path,
        output_dir: Path) -> dict:
    """Rebuild source-frozen published BC and use only NEW native eval layouts."""
    import jax
    import xminigrid
    from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
    from xminigrid.wrappers import (
        DirectionObservationWrapper,
        RulesAndGoalsObservationWrapper,
    )

    from bagi.experiments import xland_diverse_physical_demos as diversity
    from bagi.experiments.general_brain_native_g1 import source_policies
    from bagi.experiments.xland_existing_bc import get_rules
    from bagi.experiments.xland_gate_b_exact_collection import public_features
    from bagi.experiments.xland_gate_b_native_collection import screen_source_layout
    from bagi.experiments.xland_putdown_reward_validity import (
        blue_square_transition,
        tiles_adjacent,
    )

    validate_registration()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "summary.json"
    report = {
        "experiment": "G2-D1 frozen learned-policy root-cause failure funnel",
        "status": "running",
        "diagnostic_not_a_learned_new_algorithm": True,
        "public_input": "115 float32 image+heading+goal/rules+four executed actions",
        "one_controlled_ablation": "erase ONLY policy-input true four-action history",
        "counterfactual_probe_policy_influence": False,
        "source_model_seeds": list(MODEL_SEEDS),
        "predeclared_fresh_evaluation_layouts": list(FRESH_LAYOUTS),
        "tasks": list(TASKS), "conditions": list(CONDITIONS),
        "new_teacher_queries": 0, "new_train_native_interactions": 0,
        "real_learner_native_actions": 0,
        "discarded_counterfactual_native_action_calls": 0,
        "results": [],
    }
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    models, training = source_policies(source_dir, old_dir, diverse_dir)
    room, defaults = xminigrid.make("XLand-MiniGrid-R1-9x9")
    wrapped = RulesAndGoalsObservationWrapper(DirectionObservationWrapper(room))
    reset, step = jax.jit(wrapped.reset), jax.jit(wrapped.step)
    blue = np.asarray(TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE])
    red = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.RED])
    empty_kind = int(Tiles.EMPTY)
    for seed in MODEL_SEEDS:
        trained_policy = models[seed][diversity.ARMS[1]]
        for condition in CONDITIONS:
            task_results = {}
            for task in TASKS:
                params = defaults.replace(ruleset=get_rules(task), max_steps=CAP)
                cases = []
                for layout in FRESH_LAYOUTS:
                    ts = reset(params, jax.random.key(layout))
                    eligible = task != "putdown" or screen_source_layout(ts, blue, red)

                    def native_step(before, taken, *, params=params):
                        return step(params, before, taken)

                    def evaluator_transition(before, after, taken, *, task=task):
                        if task != "putdown":
                            return False, False, False
                        prior_pocket = np.asarray(before.state.agent.pocket)
                        next_pocket = np.asarray(after.state.agent.pocket)
                        prior_grid = np.asarray(before.state.grid)
                        after_grid = np.asarray(after.state.grid)
                        picked, dropped = blue_square_transition(
                            taken, prior_pocket, next_pocket,
                            prior_grid, after_grid, blue, empty_kind,
                        )
                        wrong_pick = (
                            taken == 3 and not picked
                            and int(prior_pocket[0]) == empty_kind
                            and int(next_pocket[0]) != empty_kind
                        )
                        return picked, dropped, bool(wrong_pick)

                    def final_adjacent(ts):
                        return tiles_adjacent(np.asarray(ts.state.grid), blue, red)

                    case = execute_diagnostic_episode(
                        trained_policy, ts, layout=layout, task=task,
                        condition=condition, native_step=native_step,
                        observe=public_features, eligible=eligible,
                        target_tile=blue, effective_transition=evaluator_transition,
                        final_adjacent=final_adjacent,
                    )
                    cases.append(case)
                task_results[task] = summarize_cases(cases, task)
            item = {"model_seed": seed, "condition": condition,
                    "training": training[seed][diversity.ARMS[1]],
                    "tasks": task_results}
            report["results"].append(item)
            report["real_learner_native_actions"] += sum(
                task_results[t]["native_learner_actions"] for t in TASKS
            )
            report["discarded_counterfactual_native_action_calls"] += sum(
                task_results[t]["discarded_counterfactual_native_probes"] for t in TASKS
            )
            target.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"G2-D1 seed={seed} {condition} "
                  f"physical={task_results['putdown']['physical_placements']}/"
                  f"{task_results['putdown']['eligible']} "
                  f"opportunity={task_results['putdown']['pickup_opportunity_episodes']} "
                  f"learner_native={report['real_learner_native_actions']} "
                  f"discarded_probes={report['discarded_counterfactual_native_action_calls']}",
                  flush=True)
    report["matched_paired_comparison"] = validate_paired_runs(report["results"])
    report["status"] = "complete"
    report["limitations"] = [
        ("Only published BC and two initial model seeds; no policy retraining "
         "or oracle actions in learner path."),
        ("Counterfactual pickup branches used privileged evaluator state "
         "but are discarded and separately counted."),
        ("Erased-action-history is an out-of-distribution input intervention, "
         "not a matched newly trained memoryless learner."),
        ("Pickup feasibility on current states isolates opportunity vs action timing, "
         "not arbitrary long-horizon navigation causality."),
        "Initially adjacent putdown layouts are excluded without replacing preregistered seeds.",
        ("Hold/near use original upstream native rewards, physical putdown requires "
         "ordered BLUE transitions and final adjacency."),
        ("One same-task XLand family does not establish a general AGI root cause; "
         "Workshop and cross-domain tests follow after this gate."),
    ]
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--old-teacher-dir", type=Path, required=True)
    parser.add_argument("--diverse-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/g2_d1_root_cause"))
    args = parser.parse_args()
    run(args.source_dir, args.old_teacher_dir, args.diverse_dir, args.output_dir)
