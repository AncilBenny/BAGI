"""G2-D4: independent heldout XLand confirmation of public-stagnation one-step escape.

This is a bounded, prospective diagnostic on frozen published imitation.BC:
no hidden evaluator state, label, corrective teacher or modified neural weight
may affect the acting policy. A public view failing to change is NOT a proof
of a native no-op. Physical BLUE placement is separately externally verified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bagi.agents.public_stagnation import PublicStagnationEscape

MODEL_SEEDS = (17, 29)
TASKS = ("hold", "near", "putdown")
CONDITIONS = ("direct", "stagnation_escape")
FRESH_LAYOUTS = tuple(range(11800, 11820))
CAP = 40
INPUT_WIDTH = 115
PATIENCE = 3
MAX_NATIVE_LEARNER_ACTIONS = 9600  # 2 model seeds x2 conditions x3 tasks x20 layouts x40.


def validate_registration() -> None:
    if (
        MODEL_SEEDS != (17, 29) or TASKS != ("hold", "near", "putdown")
        or CONDITIONS != ("direct", "stagnation_escape")
        or FRESH_LAYOUTS != tuple(range(11800, 11820))
        or CAP != 40 or INPUT_WIDTH != 115 or PATIENCE != 3
        or MAX_NATIVE_LEARNER_ACTIONS != 2 * 2 * 3 * 20 * CAP
    ):
        raise ValueError("Registered G2-D4 confirmation model/tasks/layouts/native budget changed")
    used = (
        set(range(9700, 9900)) | set(range(10400, 10424))
        | set(range(10600, 10810)) | set(range(10900, 10972))
        | set(range(11000, 11800))
    )
    if used & set(FRESH_LAYOUTS):
        raise ValueError("TRAIN/previous-evaluation seed overlap")


def validated_public_features(data: object, history: tuple[int, ...]) -> np.ndarray:
    x = np.asarray(data, dtype=np.float32)
    if x.shape != (INPUT_WIDTH,) or not np.isfinite(x).all():
        raise ValueError("Expected exactly 115 finite learner-legal public features")
    if len(history) != 4 or any(type(v) is not int or v not in range(7) for v in history):
        raise ValueError("Malformed prior executed-action history")
    expected = np.eye(7, dtype=np.float32)[list(history)].reshape(-1)
    if not np.array_equal(x[-28:], expected):
        raise ValueError("Public history does not match actual executed actions")
    return x


def public_signature(obs: object) -> bytes:
    x = np.asarray(obs, dtype=np.float32)
    if x.shape != (INPUT_WIDTH,) or not np.isfinite(x).all():
        raise ValueError("Invalid signature public feature contract")
    # The last 28 features are true action-history bits and should not
    # prevent detecting repeated public image/heading/task contexts.
    return np.ascontiguousarray(x[:-28]).tobytes()


def ranked_frozen_policy(policy, observation: np.ndarray) -> list[tuple[int, float]]:
    """Require exact argmax parity with SAME published SB3 categorical policy."""
    import torch

    proposal, _ = policy.predict(observation, deterministic=True)
    proposal = np.asarray(proposal)
    if proposal.shape not in ((), (1,)) or proposal.dtype.kind not in "iu":
        raise ValueError("Unexpected non-discrete upstream deterministic policy action")
    expected = int(proposal.reshape(-1)[0])
    if expected not in range(6):
        raise ValueError("Unrecognized native six-action upstream policy")
    with torch.no_grad():
        batch = torch.as_tensor(observation[None, :], dtype=torch.float32,
                                device=policy.device)
        distribution = policy.get_distribution(batch)
        probabilities = np.asarray(distribution.distribution.probs.detach().cpu().numpy())
    if probabilities.shape != (1, 6) or not np.isfinite(probabilities).all():
        raise ValueError("Published categorical policy returned invalid probabilities")
    if np.any(probabilities < 0) or abs(float(probabilities.sum()) - 1.0) > 1e-4:
        raise ValueError("Invalid upstream probabilities")
    ordered = sorted(((int(i), float(p)) for i, p in enumerate(probabilities[0])),
                     key=lambda pair: (-pair[1], pair[0]))
    if ordered[0][0] != expected:
        raise ValueError("Frozen policy argmax and distribution disagree")
    return ordered


def physical_transition_changed(before, after) -> bool:
    """EVALUATOR ONLY; called exclusively AFTER the policy already acted."""
    a, b = before.state, after.state
    return bool(
        not np.array_equal(np.asarray(a.grid), np.asarray(b.grid))
        or not np.array_equal(np.asarray(a.agent.position), np.asarray(b.agent.position))
        or not np.array_equal(np.asarray(a.agent.pocket), np.asarray(b.agent.pocket))
        or int(np.asarray(a.agent.direction)) != int(np.asarray(b.agent.direction))
    )


def execute_episode(
    policy, timestep, *, layout: int, seed: int, task: str, condition: str,
    eligible: bool, observe, native_step, public_target_seen,
    evaluator_transition, final_adjacent, physical_changed,
) -> dict:
    if layout not in FRESH_LAYOUTS or seed not in MODEL_SEEDS:
        raise ValueError("Off-preregistered eval model/layout")
    if task not in TASKS or condition not in CONDITIONS:
        raise ValueError("Unregistered source goal or policy intervention")
    tracker = PublicStagnationEscape(public_signature,
                                     enabled=condition == CONDITIONS[1],
                                     patience=PATIENCE)
    row = {
        "layout": layout, "model_seed": seed, "task": task, "condition": condition,
        "eligible": bool(eligible), "native_actions": 0,
        "native_reward": False, "physical_placement": False,
        "effective_blue_pickups": 0, "effective_blue_drops": 0,
        "ordered_blue_drop": False, "target_publicly_seen": False,
        "native_drop_action_attempts": 0,
        "suspected_stagnation_steps": 0, "escaped_actions": 0,
        "false_public_stagnation_alerts": 0,
        "public_unchanged_steps": 0,
        "actual_executed_actions": [], "trailing_drop_action_run": 0,
    }
    if not eligible:
        return row
    if bool(timestep.last()):
        raise ValueError("Eligible initial state is already native-terminal")
    from bagi.experiments.xland_gate_b_exact_collection import native_action

    history = (6, 6, 6, 6)
    has_picked_up = False
    while row["native_actions"] < CAP and not bool(timestep.last()):
        current = validated_public_features(observe(timestep, history), history)
        ranked = ranked_frozen_policy(policy, current)
        action = native_action(np.asarray(tracker.propose(current, ranked)))
        if task == "putdown":
            row["target_publicly_seen"] |= bool(public_target_seen(timestep))
        before = timestep
        after = native_step(before, action)
        row["native_actions"] += 1
        row["actual_executed_actions"].append(action)
        row["native_reward"] |= float(after.reward) > 0
        history = (*history[1:], action)
        next_view = validated_public_features(observe(after, history), history)
        event = tracker.observe(action, next_view, float(after.reward))
        row["public_unchanged_steps"] += int(event.unchanged_public_view)
        row["suspected_stagnation_steps"] += int(event.suspected_stagnation)
        row["escaped_actions"] += int(event.escaped_last_choice)
        if event.suspected_stagnation:
            row["false_public_stagnation_alerts"] += int(physical_changed(before, after))
        if task == "putdown":
            picked, dropped = evaluator_transition(before, after, action)
            row["ordered_blue_drop"] |= bool(dropped and has_picked_up)
            has_picked_up |= bool(picked)
            row["effective_blue_pickups"] += int(bool(picked))
            row["effective_blue_drops"] += int(bool(dropped))
            row["native_drop_action_attempts"] += int(action == 4)
        timestep = after
    for action in reversed(row["actual_executed_actions"]):
        if action != 4:
            break
        row["trailing_drop_action_run"] += 1
    row["physical_placement"] = bool(
        task == "putdown" and row["native_reward"]
        and row["effective_blue_pickups"] and row["effective_blue_drops"]
        and row["ordered_blue_drop"] and final_adjacent(timestep)
    )
    if row["suspected_stagnation_steps"] != tracker.suspected_stalls:
        raise AssertionError("Stagnation event reconciliation failed")
    if row["escaped_actions"] != tracker.escape_actions:
        raise AssertionError("Public escape intervention accounting failed")
    return row


def summarize_cases(cases: list[dict], task: str) -> dict:
    if [c["layout"] for c in cases] != list(FRESH_LAYOUTS):
        raise ValueError("Missing/replaced prospective source layout")
    if any(c["task"] != task or c["native_actions"] > CAP for c in cases):
        raise ValueError("Mixed tasks or real native action cap exceeded")
    eligible = [c for c in cases if c["eligible"]]
    return {
        "eligible": len(eligible), "native_rewarded": sum(int(c["native_reward"]) for c in eligible),
        "physical_placements": sum(int(c["physical_placement"]) for c in eligible),
        "actual_learner_native_actions": sum(c["native_actions"] for c in cases),
        "target_seen_episodes": sum(int(c["target_publicly_seen"]) for c in eligible),
        "effective_pickup_episodes": sum(c["effective_blue_pickups"] > 0 for c in eligible),
        "effective_drop_episodes": sum(c["effective_blue_drops"] > 0 for c in eligible),
        "drop_action_attempts": sum(c["native_drop_action_attempts"] for c in eligible),
        "suspected_stagnation_steps": sum(c["suspected_stagnation_steps"] for c in eligible),
        "escape_actions": sum(c["escaped_actions"] for c in eligible),
        "evaluator_hidden_change_during_suspected_stagnation": sum(
            c["false_public_stagnation_alerts"] for c in eligible
        ),
        "episodes_with_trailing_20_drops": sum(
            c["trailing_drop_action_run"] >= 20 for c in eligible
        ),
        "cases": cases,
    }


def run(source_dir: Path, old_dir: Path, diverse_dir: Path, output_dir: Path) -> dict:
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
    saved = output_dir / "summary.json"
    report = {
        "experiment": "G2-D4 prospective independent replication of identical G2-D3 frozen-policy intervention",
        "status": "running", "model_seeds": list(MODEL_SEEDS),
        "conditions": list(CONDITIONS), "tasks": list(TASKS),
        "preregistered_layouts": list(FRESH_LAYOUTS),
        "episode_cap": CAP, "max_native_actions": MAX_NATIVE_LEARNER_ACTIONS,
        "zero_new_teacher_queries": True,
        "new_teacher_queries": 0, "new_training_native_actions": 0,
        "native_real_learner_actions": 0,
        "evaluator_only_discarded_native_probes": 0,
        "results": [],
    }
    saved.write_text(json.dumps(report, indent=2), encoding="utf-8")
    policies, training = source_policies(source_dir, old_dir, diverse_dir)
    room, params0 = xminigrid.make("XLand-MiniGrid-R1-9x9")
    visible = RulesAndGoalsObservationWrapper(DirectionObservationWrapper(room))
    reset, step = jax.jit(visible.reset), jax.jit(visible.step)
    blue = np.asarray(TILES_REGISTRY[Tiles.SQUARE, Colors.BLUE])
    red = np.asarray(TILES_REGISTRY[Tiles.BALL, Colors.RED])
    empty_kind = int(Tiles.EMPTY)
    for model_seed in MODEL_SEEDS:
        policy = policies[model_seed][diversity.ARMS[1]]
        for condition in CONDITIONS:
            task_metrics = {}
            for task in TASKS:
                params = params0.replace(ruleset=get_rules(task), max_steps=CAP)
                cases = []
                for layout in FRESH_LAYOUTS:
                    ts = reset(params, jax.random.key(layout))
                    eligible = task != "putdown" or screen_source_layout(ts, blue, red)

                    def native_step(before, taken, *, params=params):
                        return step(params, before, taken)

                    def see_target(before):
                        image = np.asarray(before.observation["img"])
                        return bool(np.any(np.all(image == blue, axis=-1)))

                    def effective_transition(before, after, taken, *, task=task):
                        if task != "putdown":
                            return False, False
                        return blue_square_transition(
                            taken, np.asarray(before.state.agent.pocket),
                            np.asarray(after.state.agent.pocket),
                            np.asarray(before.state.grid),
                            np.asarray(after.state.grid), blue, empty_kind,
                        )

                    def final_adjacent(after):
                        return tiles_adjacent(np.asarray(after.state.grid), blue, red)

                    case = execute_episode(
                        policy, ts, layout=layout, seed=model_seed,
                        task=task, condition=condition, eligible=eligible,
                        observe=public_features, native_step=native_step,
                        public_target_seen=see_target,
                        evaluator_transition=effective_transition,
                        final_adjacent=final_adjacent,
                        physical_changed=physical_transition_changed,
                    )
                    cases.append(case)
                    report["native_real_learner_actions"] += case["native_actions"]
                task_metrics[task] = summarize_cases(cases, task)
                report["results"] = [r for r in report["results"] if (
                    r["model_seed"], r["condition"]
                ) != (model_seed, condition)]
                report["results"].append({
                    "model_seed": model_seed, "condition": condition,
                    "training": training[model_seed][diversity.ARMS[1]],
                    "tasks": dict(task_metrics),
                })
                saved.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"G2-D4 model_seed={model_seed} {condition} "
                  f"putdown={task_metrics['putdown']['physical_placements']}/"
                  f"{task_metrics['putdown']['eligible']} "
                  f"hold={task_metrics['hold']['native_rewarded']} "
                  f"near={task_metrics['near']['native_rewarded']}", flush=True)
    if len(report["results"]) != len(MODEL_SEEDS) * len(CONDITIONS):
        raise ValueError("Incomplete two-arm native study")
    if report["native_real_learner_actions"] > MAX_NATIVE_LEARNER_ACTIONS:
        raise ValueError("Over real native simulator step budget")
    for seed in MODEL_SEEDS:
        paired = [r for r in report["results"] if r["model_seed"] == seed]
        if len(paired) != 2 or paired[0]["training"] != paired[1]["training"]:
            raise ValueError("Frozen training recipe mismatch")
        for task in TASKS:
            left, right = (r["tasks"][task]["cases"] for r in paired)
            if [c["eligible"] for c in left] != [c["eligible"] for c in right]:
                raise ValueError("Matched layout eligibility changed")
    report["status"] = "complete"
    report["limitations"] = [
        "Public unchanged-view/action streak is not proof the hidden physical state remained unchanged.",
        "The intervention chooses one alternate action from the SAME frozen policy; no learned new policy or teacher labels.",
        "No evaluator-only native counterfactual probes or teacher-controlled actions were executed.",
        "The original 40 step cap yields up to 9600 real native learner steps for 240 fixed episodes.",
        "Two model seeds and twenty source-disjoint layouts cannot prove robust general learning.",
        "Native reward alone never counts as nonvacuous BLUE physical placement.",
    ]
    saved.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--old-teacher-dir", type=Path, required=True)
    p.add_argument("--diverse-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/g2_d4_independent_stagnation"))
    a = p.parse_args()
    run(a.source_dir, a.old_teacher_dir, a.diverse_dir, a.output_dir)
