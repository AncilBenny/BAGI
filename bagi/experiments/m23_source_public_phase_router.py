"""Stage18 public-witnessed door-open routing between two learned source GRUs.

The switch is an explicit hand-engineered public phase boundary, NOT a learned
task-independent planner, source teacher fallback, or private physics oracle.
Both GRU memories update every real step from the SAME executed action history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from bagi.experiments.m1_goal_learning import admitted_source_commit
from bagi.experiments.m13_public_interaction_coverage import Controller
from bagi.experiments.m15_public_action_bc import class_weights, save_numeric
from bagi.experiments.m23_source_feasibility import HORIZON
from bagi.experiments.m23_source_mission_bc import make_native_audit, make_source_actor
from bagi.experiments.m23_source_post_open_continuation_bc import (
    ARMS as FIT_ARMS,
)
from bagi.experiments.m23_source_post_open_continuation_bc import (
    MODEL_SEEDS,
    PostOpenGRU,
    fit_model,
    public_action_features,
    public_door_open_transition,
)
from bagi.experiments.m23_study_contract import (
    PHASE_ROUTER_MONITOR_LAYOUTS,
    PHASE_ROUTER_TRAIN_LAYOUTS,
    validate_split_contract,
)

OUT = "runs/m23_source_public_phase_router"
TASKS = ("doorkey", "fetch")
EVAL_ARMS = (*FIT_ARMS, "public_phase_router")
BOOTSTRAP_SEED = 231801
MIN_PRIMARY_GOAL_GAIN = 12
MIN_SECONDARY_GAIN = 8
MIN_ROUTER_DOORKEY_GOALS = 16
MIN_DOORKEY_ACTION_DIFFERENCES = 16
MIN_PUBLIC_SWITCHES = 16
ACTOR_MARGIN = 24


def select_executed_action(
    arm: str, in_post_open_phase: bool, control_proposal: int, post_proposal: int
) -> int:
    """Exactly ONE proposal is sent to env.step; the other is never executed."""
    if arm not in EVAL_ARMS or arm == "actor_only":
        raise ValueError("Stage18 learned evaluation arm not registered")
    if (type(control_proposal) is not int or type(post_proposal) is not int
            or control_proposal not in range(7) or post_proposal not in range(7)):
        raise ValueError("Stage18 proposed native actions invalid")
    if arm == "interaction_context":
        return control_proposal
    if arm == "post_open_continuation":
        return post_proposal
    return post_proposal if in_post_open_phase else control_proposal


def advance_public_phase(
    task: str, current_phase: bool, before: dict, executed: int, after: dict
) -> bool:
    """Switch only *after* a public visible door opening caused by a real toggle."""
    if task not in TASKS or type(current_phase) is not bool:
        raise ValueError("Stage18 public phase task/state malformed")
    if task == "fetch":
        if current_phase:
            raise ValueError("Stage18 Fetch must never enter a DoorKey phase")
        return False
    return current_phase or public_door_open_transition(before, executed, after)


def train_episode(task: str, layout: int) -> dict:
    if task not in TASKS or layout not in PHASE_ROUTER_TRAIN_LAYOUTS[task]:
        raise ValueError("Stage18 only fresh registered TRAIN source layouts")
    actor = make_source_actor(task)
    rng = np.random.default_rng(8_800_000 + layout)
    audit = make_native_audit(task)
    vectors: list[np.ndarray] = []
    actions: list[int] = []
    first_public_open: int | None = None
    finished = False
    try:
        obs, _ = audit.reset(seed=layout)
        previous: int | None = None
        for _ in range(HORIZON):
            vectors.append(public_action_features(
                obs, actor.key_observed_after_pickup, previous,
                arm="interaction_context",
            ))
            action = int(actor.choose(obs, rng))
            if action not in range(7):
                raise ValueError("Stage18 illegal native source-teacher action")
            nxt, _reward, done, truncated, _ = audit.step(action)
            if (task == "doorkey" and first_public_open is None
                    and public_door_open_transition(obs, action, nxt)):
                first_public_open = len(actions)
            actor.update(obs, action, nxt)
            actions.append(action)
            previous = action
            obs = nxt
            if done or truncated:
                finished = True
                break
        truth = audit.summary()
        if (not finished or not actions or len(vectors) != len(actions)
                or truth["native_actions"] != len(actions)
                or sum(truth["action_counts"].values()) != len(actions)):
            raise ValueError("Stage18 TRAIN source native actions not conserved")
        x = np.stack(vectors).astype(np.float32, copy=False)
        y = np.asarray(actions, dtype=np.int64)
        return {
            "task": task, "layout_seed": layout, "x": x, "y": y,
            "public_first_witnessed_open_action_index": first_public_open,
            "public_post_open_source_action_count": (
                len(actions) - first_public_open - 1
                if first_public_open is not None else 0
            ),
            "actual_native_steps": len(actions),
            "actual_executed_action_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
            "public_episode_x_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            "true_executed_episode_y_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
            "native_action_histogram": [
                truth["action_counts"][str(i)] for i in range(7)
            ],
            "external_effective_door_open_events_train": (
                truth["effective_door_open_events"] if task == "doorkey" else 0
            ),
            "external_strict_full_physical_goal": bool(truth["full_physical_goal"]),
        }
    finally:
        audit.env.close()


def collect_train() -> tuple[list[dict], np.ndarray, dict]:
    validate_split_contract()
    episodes = [
        train_episode(task, layout)
        for task in TASKS for layout in PHASE_ROUTER_TRAIN_LAYOUTS[task]
    ]
    if len(episodes) != 192:
        raise ValueError("Stage18 TRAIN requires 192 genuine native episodes")
    ys = np.concatenate([e["y"] for e in episodes])
    native_steps = sum(e["actual_native_steps"] for e in episodes)
    histogram = np.bincount(ys, minlength=7).tolist()
    if len(ys) != native_steps or sum(histogram) != native_steps:
        raise ValueError("Stage18 TRAIN source native action histogram mismatch")
    digest = hashlib.sha256()
    for e in episodes:
        digest.update(e["x"].tobytes())
    return episodes, ys, {
        "real_native_train_episodes": len(episodes),
        "actual_new_native_train_steps": native_steps,
        "native_source_train_executed_action_histogram": histogram,
        "actual_public_train_witnessed_open_episodes": sum(
            e["public_first_witnessed_open_action_index"] is not None
            for e in episodes
        ),
        "external_train_effective_open_episodes": sum(
            e["external_effective_door_open_events_train"] > 0
            for e in episodes
        ),
        "genuine_post_open_source_train_actions": sum(
            e["public_post_open_source_action_count"] for e in episodes
        ),
        "source_public_x_sha256": digest.hexdigest(),
        "source_true_executed_actions_sha256": hashlib.sha256(ys.tobytes()).hexdigest(),
        "per_episode": [
            {k: v for k, v in e.items() if k not in ("x", "y")}
            for e in episodes
        ],
    }


def monitor_episode(
    task: str, layout: int, arm: str, seed: int | None,
    models: dict[tuple[str, int], PostOpenGRU] | None,
) -> dict:
    if task not in TASKS or layout not in PHASE_ROUTER_MONITOR_LAYOUTS[task]:
        raise ValueError("Stage18 MONITOR outside independent fresh source split")
    if arm == "actor_only":
        if seed is not None or models is not None:
            raise ValueError("Stage18 source actor must be measured once")
        actor = make_source_actor(task)
    elif arm in EVAL_ARMS and seed in MODEL_SEEDS and models is not None:
        if set(models) != {(a, s) for a in FIT_ARMS for s in MODEL_SEEDS}:
            raise ValueError("Stage18 must reuse exactly six fitted policies")
        if any(model.training for model in models.values()):
            raise ValueError("Stage18 learned source policies must be eval only")
        actor = Controller("public_memory")
    else:
        raise ValueError("Stage18 unregistered model/seed evaluation")
    audit = make_native_audit(task)
    rng = np.random.default_rng(8_800_000 + layout)
    trace: list[int] = []
    control_hidden: torch.Tensor | None = None
    post_hidden: torch.Tensor | None = None
    previous_action: int | None = None
    public_phase = False
    public_first_switch_action: int | None = None
    selected_post_steps = 0
    reward_sum = 0.0
    finished = False
    try:
        obs, _ = audit.reset(seed=layout)
        for _ in range(HORIZON):
            if models is None:
                action = int(actor.choose(obs, rng))
            else:
                vector = public_action_features(
                    obs, actor.key_observed_after_pickup, previous_action,
                    arm="interaction_context",
                )
                pre_action, control_hidden = models[
                    ("interaction_context", seed)
                ].native_step(vector, control_hidden)
                post_action, post_hidden = models[
                    ("post_open_continuation", seed)
                ].native_step(vector, post_hidden)
                action = select_executed_action(
                    arm, public_phase, pre_action, post_action
                )
                if arm == "post_open_continuation" or (
                    arm == "public_phase_router" and public_phase
                ):
                    selected_post_steps += 1
            nxt, reward, done, truncated, _ = audit.step(action)
            next_phase = advance_public_phase(task, public_phase, obs, action, nxt)
            if next_phase and not public_phase:
                public_first_switch_action = len(trace)
            public_phase = next_phase
            actor.update(obs, action, nxt)
            trace.append(action)
            previous_action = action
            reward_sum += float(reward)
            obs = nxt
            if done or truncated:
                finished = True
                break
        truth = audit.summary()
        if (not finished or len(trace) != truth["native_actions"]
                or sum(truth["action_counts"].values()) != len(trace)
                or (task == "fetch" and public_first_switch_action is not None)):
            raise ValueError("Stage18 native physical MONITOR and phase invalid")
        return {
            "task": task, "layout_seed": layout, "arm": arm, "model_seed": seed,
            "actual_native_steps": len(trace),
            "actual_native_executed_action_sha256":
                hashlib.sha256(bytes(trace)).hexdigest(),
            "native_executed_action_histogram": [
                truth["action_counts"][str(i)] for i in range(7)
            ],
            "public_witnessed_door_open": public_first_switch_action is not None,
            "public_first_switch_action_index": public_first_switch_action,
            "actually_executed_post_policy_steps": (
                selected_post_steps if arm == "public_phase_router" else 0
            ),
            "actual_reward_total": reward_sum,
            "external_strict_full_physical_goal": bool(truth["full_physical_goal"]),
            "external_rewarded_terminations": truth["rewarded_terminations"],
            "external_effective_door_open_events": (
                truth["effective_door_open_events"] if task == "doorkey" else 0
            ),
            "external_effective_pickup_events": truth["effective_pickup_events"],
            "external_effective_forward_movements":
                truth.get("effective_forward_movements", 0),
            "external_wrong_target_pickup_events":
                truth.get("wrong_target_pickup_events", 0),
        }
    finally:
        audit.env.close()


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Stage18 source MONITOR has no real episodes")
    actions = sum(r["actual_native_steps"] for r in rows)
    hist = [
        sum(r["native_executed_action_histogram"][i] for r in rows)
        for i in range(7)
    ]
    if sum(hist) != actions:
        raise ValueError("Stage18 native MONITOR actions do not conserve")
    return {
        "actual_native_episodes": len(rows), "actual_native_steps": actions,
        "native_action_histogram": hist,
        "strict_native_full_physical_goals": sum(
            int(r["external_strict_full_physical_goal"]) for r in rows
        ),
        "public_witnessed_open_episodes": sum(
            int(r["public_witnessed_door_open"]) for r in rows
        ),
        "independent_native_door_open_episodes": sum(
            int(r["external_effective_door_open_events"] > 0) for r in rows
        ),
        "reward_only_nonqualifying_episodes": sum(
            int(r["external_rewarded_terminations"] > 0
                and not r["external_strict_full_physical_goal"]) for r in rows
        ),
        "actual_executed_post_policy_steps": sum(
            r["actually_executed_post_policy_steps"] for r in rows
        ),
    }


def acquire_monitor(models: dict[tuple[str, int], PostOpenGRU]) -> dict:
    if set(models) != {(a, s) for a in FIT_ARMS for s in MODEL_SEEDS}:
        raise ValueError("Stage18 requires six exactly paired trained models")
    result = {}
    for task in TASKS:
        layouts = PHASE_ROUTER_MONITOR_LAYOUTS[task]
        original = [
            monitor_episode(task, layout, "actor_only", None, None)
            for layout in layouts
        ]
        result[f"{task}:actor_only"] = {
            "episodes": original, "summary": summarize(original)
        }
        for arm in EVAL_ARMS:
            for seed in MODEL_SEEDS:
                rows = [
                    monitor_episode(task, layout, arm, seed, models)
                    for layout in layouts
                ]
                result[f"{task}:{arm}:{seed}"] = {
                    "episodes": rows, "summary": summarize(rows)
                }
    return result


def decide(monitor: dict) -> dict:
    required = {f"{task}:actor_only" for task in TASKS} | {
        f"{task}:{arm}:{seed}"
        for task in TASKS for arm in EVAL_ARMS for seed in MODEL_SEEDS
    }
    if set(monitor) != required:
        raise ValueError("Stage18 independent native MONITOR arms incomplete")
    seed_gain = {s: 0 for s in MODEL_SEEDS}
    task_gains = {}
    door_diffs = {}
    pairs = {}
    actor_goals = 0
    router_goals = router_doorkey_goals = post_goals = 0
    router_public_switches = 0
    for task in TASKS:
        layouts = list(PHASE_ROUTER_MONITOR_LAYOUTS[task])
        actor = monitor[f"{task}:actor_only"]["episodes"]
        if [r["layout_seed"] for r in actor] != layouts:
            raise ValueError("Stage18 actor MONITOR layouts unpaired")
        actor_goals += sum(int(r["external_strict_full_physical_goal"]) for r in actor)
        task_pairs = []
        door_diffs[task] = {}
        for seed in MODEL_SEEDS:
            control = monitor[f"{task}:interaction_context:{seed}"]["episodes"]
            post = monitor[f"{task}:post_open_continuation:{seed}"]["episodes"]
            routed = monitor[f"{task}:public_phase_router:{seed}"]["episodes"]
            if any([r["layout_seed"] for r in group] != layouts
                   for group in (control, post, routed)):
                raise ValueError("Stage18 native MONITOR layouts unpaired")
            if task == "fetch" and any(
                a["actual_native_executed_action_sha256"]
                != b["actual_native_executed_action_sha256"]
                or a["actual_native_steps"] != b["actual_native_steps"]
                or a["external_strict_full_physical_goal"]
                != b["external_strict_full_physical_goal"]
                or b["public_witnessed_door_open"]
                for a, b in zip(control, routed, strict=True)
            ):
                raise ValueError("Stage18 Fetch router/control source integrity failed")
            gains = np.asarray([
                int(b["external_strict_full_physical_goal"])
                - int(a["external_strict_full_physical_goal"])
                for a, b in zip(control, routed, strict=True)
            ], dtype=np.int64)
            task_pairs.append(gains)
            seed_gain[seed] += int(gains.sum())
            router_goals += sum(
                int(r["external_strict_full_physical_goal"]) for r in routed
            )
            post_goals += sum(
                int(r["external_strict_full_physical_goal"]) for r in post
            )
            if task == "doorkey":
                router_doorkey_goals += sum(
                    int(r["external_strict_full_physical_goal"]) for r in routed
                )
                router_public_switches += sum(
                    int(r["public_witnessed_door_open"]) for r in routed
                )
            door_diffs[task][str(seed)] = sum(
                a["actual_native_executed_action_sha256"]
                != b["actual_native_executed_action_sha256"]
                for a, b in zip(control, routed, strict=True)
            )
        pairs[task] = np.asarray(task_pairs, dtype=np.int64)
        task_gains[task] = int(pairs[task].sum())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.zeros(20000, dtype=np.int64)
    for task in TASKS:
        values = pairs[task]
        seed_indices = rng.integers(0, 3, size=(20000, 3))
        layout_indices = rng.integers(0, 40, size=(20000, 3, 40))
        draws += values[seed_indices[:, :, None], layout_indices].sum(axis=(1, 2))
    interval = [float(n) for n in np.quantile(draws, (0.025, 0.975))]
    diff = int(sum(task_gains.values()))
    router_minus_post = router_goals - post_goals
    noninferior = router_goals >= 3 * actor_goals - ACTOR_MARGIN
    distinct = (
        sum(door_diffs["doorkey"][str(seed)] >= MIN_DOORKEY_ACTION_DIFFERENCES
            for seed in MODEL_SEEDS) >= 2
        and router_public_switches >= MIN_PUBLIC_SWITCHES
    )
    numeric = bool(
        diff >= MIN_PRIMARY_GOAL_GAIN
        and task_gains["doorkey"] >= MIN_PRIMARY_GOAL_GAIN
        and task_gains["fetch"] == 0
        and router_minus_post >= MIN_SECONDARY_GAIN
        and sum(seed_gain[s] > 0 for s in MODEL_SEEDS) >= 2
        and router_doorkey_goals >= MIN_ROUTER_DOORKEY_GOALS
        and noninferior
    )
    scientific = (
        "INSUFFICIENT" if not distinct
        else "PASS" if numeric and interval[0] > 0 else "FAIL"
    )
    return {
        "registered_decision": scientific,
        "necessary_door_action_distinction_and_public_switch_nonvacuity": distinct,
        "registered_physical_numeric_gate": numeric,
        "per_source_task_router_minus_control_strict_goals": task_gains,
        "per_model_seed_router_minus_control_strict_goals": {
            str(s): int(v) for s, v in seed_gain.items()
        },
        "paired_router_minus_control_strict_goals": diff,
        "router_minus_post_only_strict_goals": router_minus_post,
        "source_actor_once_measured_strict_goals": actor_goals,
        "actor_tripled_arithmetic_only": 3 * actor_goals,
        "minimum_actor_margin_router_goal_count": 3 * actor_goals - ACTOR_MARGIN,
        "router_native_strict_goals": router_goals,
        "post_only_native_strict_goals": post_goals,
        "router_native_doorkey_strict_goals": router_doorkey_goals,
        "public_routed_door_open_episode_count": router_public_switches,
        "actual_native_path_hash_differences_router_vs_control": door_diffs,
        "paired_bootstrap_95pct_router_minus_control_interval": interval,
        "bootstrap_lower_bound_above_zero": interval[0] > 0,
        "actor_noninferiority": noninferior,
        "fetch_router_control_action_and_goal_identical": True,
        "original_unlockpickup_final_touched": False,
    }


def execute(output: str) -> dict:
    if output != f"{OUT}/summary.json" or Path(output).exists():
        raise ValueError("Stage18 requires fresh isolated source output")
    validate_split_contract()
    source_sha = admitted_source_commit()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    episodes, true_actions, source_train = collect_train()
    weights = class_weights(true_actions)
    models: dict[tuple[str, int], PostOpenGRU] = {}
    manifests = []
    for seed in MODEL_SEEDS:
        for arm in FIT_ARMS:
            model, profile = fit_model(episodes, seed, arm, weights)
            profile["numeric_checkpoint"] = save_numeric(
                model, path.parent / f"{arm}-{seed}.npz"
            )
            models[(arm, seed)] = model
            manifests.append(profile)
    if (len(manifests) != 6
            or len({p["num_parameters"] for p in manifests}) != 1
            or len({p["actual_optimizer_step_calls"] for p in manifests}) != 1
            or len({p["cached_source_label_presentations_not_native_steps"]
                    for p in manifests}) != 1):
        raise ValueError("Stage18 matched fitted model capacity/exposure failed")
    monitor = acquire_monitor(models)
    verdict = decide(monitor)
    monitor_steps = sum(
        row["summary"]["actual_native_steps"] for row in monitor.values()
    )
    result = {
        "study": "m23_stage18_source_public_phase_router",
        "execution_source_commit": source_sha,
        "train_source_layouts": {
            task: list(ids) for task, ids in PHASE_ROUTER_TRAIN_LAYOUTS.items()
        },
        "monitor_source_layouts": {
            task: list(ids) for task, ids in PHASE_ROUTER_MONITOR_LAYOUTS.items()
        },
        "model_seeds": list(MODEL_SEEDS),
        "fitted_arms": list(FIT_ARMS),
        "learned_monitor_arms": list(EVAL_ARMS),
        "public_input_width": 1056,
        "trained_model_parameters_each": manifests[0]["num_parameters"],
        "physical_source_train": source_train,
        "actual_native_source_train_steps": source_train["actual_new_native_train_steps"],
        "real_source_independent_monitor": monitor,
        "actual_native_source_monitor_steps": monitor_steps,
        "actual_native_source_total_steps":
            source_train["actual_new_native_train_steps"] + monitor_steps,
        "actual_fitted_models": len(manifests),
        "real_optimizer_updates": sum(
            m["actual_optimizer_step_calls"] for m in manifests
        ),
        "cached_label_presentations_not_physical_steps": sum(
            m["cached_source_label_presentations_not_native_steps"]
            for m in manifests
        ),
        "checkpoint_manifests": manifests,
        "registered_scientific_decision": verdict,
        "original_target_unlockpickup_native_steps": 0,
        "original_target_final_episodes": 0,
        "imagined_physical_steps": 0,
        "actual_study_wall_seconds": time.perf_counter() - started,
        "versions": {
            name: version(name)
            for name in ("numpy", "torch", "gymnasium", "minigrid")
        },
    }
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source": source_sha, "study": result["study"],
        "train_native_steps": result["actual_native_source_train_steps"],
        "monitor_native_steps": monitor_steps,
        "real_models": len(manifests),
        "true_optimizer_steps": result["real_optimizer_updates"],
        "router_gain": verdict["paired_router_minus_control_strict_goals"],
        "router_vs_post": verdict["router_minus_post_only_strict_goals"],
        "decision": verdict["registered_decision"],
    }), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=f"{OUT}/summary.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Stage18 requires --execute native physical research")
    execute(args.output)


if __name__ == "__main__":
    main()
