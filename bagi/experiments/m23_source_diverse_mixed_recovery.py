"""Stage25 learned post-open rescue specialist with a preserved source policy.

A hand-engineered public observed door event switches from a frozen learned
source GRU to an independently trained learned post-open GRU. The matched
two-model replay control isolates executed-recovery labels from extra capacity.
No teacher or private auditor observation is used during learned MONITOR.
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
from bagi.experiments.m23_executed_recovery_train import (
    FOLLOWUP_UPDATES,
    TOKENS_PER_UPDATE,
)
from bagi.experiments.m23_mixed_recovery_acquisition import (
    ACQUIRE_BUDGET,
    TASKS,
    collect_acquired,
    collect_teacher,
)
from bagi.experiments.m23_mixed_recovery_train import (
    FIT_ARMS,
    fit_stage25_followup,
)
from bagi.experiments.m23_source_feasibility import HORIZON
from bagi.experiments.m23_source_interaction_context_bc import (
    MODEL_SEEDS,
    InteractionContextGRU,
    fit_model,
    public_action_features,
)
from bagi.experiments.m23_source_mission_bc import make_native_audit, make_source_actor
from bagi.experiments.m23_source_post_open_continuation_bc import (
    public_door_open_transition,
)
from bagi.experiments.m23_study_contract import (
    MIXED_RECOVERY_ACQUIRE_LAYOUTS,
    MIXED_RECOVERY_MONITOR_LAYOUTS,
    MIXED_RECOVERY_TRAIN_LAYOUTS,
    validate_split_contract,
)

OUT = "runs/m23_source_diverse_mixed_recovery"
ARMS = ("frozen_source", "source_replay_option", "mixed_rehearsal_option")
BOOTSTRAP_SEED = 232501
MIN_TRUE_RECOVERY_SUFFIX_PER_SEED = 128
MIN_UNIQUE_ACQUISITION_OPEN_LAYOUTS_PER_SEED = 8
MIN_PUBLIC_MONITOR_SWITCHES = 16
MIN_DOORKEY_NATIVE_ACTION_DIFFERENCES = 16
MIN_PRIMARY_DOOR_GOAL_GAIN = 12
MIN_SECONDARY_DOOR_GOAL_GAIN = 12
MIN_TREATMENT_DOOR_GOALS = 16
ACTOR_MARGIN = 24


def enough_diverse_recovery_evidence(acquisition: dict) -> bool:
    """Never open Stage25 MONITOR without necessary actual TRAIN handoffs."""
    groups = acquisition["per_seed_task"]
    if set(groups) != {
        f"{task}:{seed}" for task in TASKS for seed in MODEL_SEEDS
    }:
        raise ValueError("Stage25 missing real native source TRAIN acquisition strata")
    return all(
        groups[f"doorkey:{seed}"]["actually_executed_teacher_suffix_native_actions"]
        >= MIN_TRUE_RECOVERY_SUFFIX_PER_SEED
        and groups[f"doorkey:{seed}"]["unique_public_open_layout_seeds"]
        >= MIN_UNIQUE_ACQUISITION_OPEN_LAYOUTS_PER_SEED
        for seed in MODEL_SEEDS
    )


def select_public_mixed_action(
    arm: str, already_open: bool,
    source_proposal: int, replay_proposal: int, rescue_proposal: int,
) -> int:
    """Select one real action; unused recurrent proposals are never native steps."""
    if arm not in ARMS or type(already_open) is not bool:
        raise ValueError("Stage25 unregistered learned source mode or phase")
    if (type(source_proposal) is not int or type(replay_proposal) is not int
            or type(rescue_proposal) is not int
            or any(x not in range(7)
                   for x in (source_proposal, replay_proposal, rescue_proposal))):
        raise ValueError("Stage25 invalid native learned action proposal")
    if not already_open or arm == "frozen_source":
        return source_proposal
    if arm == "source_replay_option":
        return replay_proposal
    return rescue_proposal


def monitor_episode(
    task: str, layout: int, arm: str, seed: int | None,
    frozen: dict[int, InteractionContextGRU] | None,
    specialists: dict[tuple[str, int], InteractionContextGRU] | None,
) -> dict:
    if task not in TASKS or layout not in MIXED_RECOVERY_MONITOR_LAYOUTS[task]:
        raise ValueError("Stage25 native MONITOR outside fresh independent source split")
    if arm == "actor_only":
        if seed is not None or frozen is not None or specialists is not None:
            raise ValueError("Stage25 original actor baseline measured once only")
        actor = make_source_actor(task)
    elif arm in ARMS and seed in MODEL_SEEDS:
        if (frozen is None or specialists is None
                or set(frozen) != set(MODEL_SEEDS)
                or set(specialists) != {
                    (a, s) for a in FIT_ARMS for s in MODEL_SEEDS
                }
                or any(m.training for m in frozen.values())
                or any(m.training for m in specialists.values())):
            raise ValueError("Stage25 requires exactly nine frozen learned GRUs")
        actor = Controller("public_memory")
    else:
        raise ValueError("Stage25 learned source policy arm/seed not registered")
    rng = np.random.default_rng(8_800_000 + layout)
    audit = make_native_audit(task)
    trace: list[int] = []
    hidden_base = None
    hidden_replay = None
    hidden_rescue = None
    previous: int | None = None
    opened = False
    first_open: int | None = None
    selected_specialist_steps = 0
    reward_total = 0.0
    finished = False
    try:
        obs, _ = audit.reset(seed=layout)
        for _ in range(HORIZON):
            if arm == "actor_only":
                action = int(actor.choose(obs, rng))
            else:
                vector = public_action_features(
                    obs, actor.key_observed_after_pickup, previous,
                    arm="interaction_context",
                )
                base_action, hidden_base = frozen[seed].native_step(
                    vector, hidden_base
                )
                # Both specialist histories follow precisely the SAME
                # observation / previous actually executed action stream.
                replay_action, hidden_replay = specialists[
                    ("source_replay", seed)
                ].native_step(vector, hidden_replay)
                rescue_action, hidden_rescue = specialists[
                    ("mixed_rehearsal", seed)
                ].native_step(vector, hidden_rescue)
                action = select_public_mixed_action(
                    arm, opened, base_action, replay_action, rescue_action
                )
                selected_specialist_steps += int(
                    opened and arm != "frozen_source"
                )
            if action not in range(7):
                raise ValueError("Stage25 executed native action invalid")
            nxt, reward, done, truncated, _ = audit.step(action)
            just_opened = (
                task == "doorkey"
                and not opened
                and public_door_open_transition(obs, action, nxt)
            )
            if just_opened:
                first_open = len(trace)
            opened |= just_opened
            actor.update(obs, action, nxt)
            trace.append(action)
            previous = action
            reward_total += float(reward)
            obs = nxt
            if done or truncated:
                finished = True
                break
        truth = audit.summary()
        if (not finished or not trace or len(trace) != truth["native_actions"]
                or sum(truth["action_counts"].values()) != len(trace)):
            raise ValueError("Stage25 genuine native MONITOR actions not conserved")
        effectful = (
            bool(truth["effective_door_open_events"] > 0)
            if task == "doorkey" else False
        )
        if (first_open is not None) != effectful:
            raise ValueError("Stage25 public real opening disagrees with independent audit")
        if task == "fetch" and (first_open is not None or selected_specialist_steps):
            raise ValueError("Stage25 Fetch cannot route to DoorKey specialist")
        prefix = trace if first_open is None else trace[:first_open + 1]
        return {
            "task": task, "layout_seed": layout, "arm": arm, "model_seed": seed,
            "actual_native_steps": len(trace),
            "full_true_native_action_sha256": hashlib.sha256(bytes(trace)).hexdigest(),
            "prefix_true_native_action_sha256":
                hashlib.sha256(bytes(prefix)).hexdigest(),
            "true_native_prefix_step_count": len(prefix),
            "first_real_public_door_open_action_index": first_open,
            "actual_executed_specialist_post_open_actions": selected_specialist_steps,
            "native_action_histogram": [
                truth["action_counts"][str(i)] for i in range(7)
            ],
            "external_strict_full_physical_goal": bool(truth["full_physical_goal"]),
            "external_effective_door_open_episodes": effectful,
            "external_reward_positive_episodes": int(
                truth["rewarded_terminations"] > 0
            ),
            "external_wrong_target_pickup_events": truth.get(
                "wrong_target_pickup_events", 0
            ),
            "actual_reward_total": reward_total,
        }
    finally:
        audit.env.close()


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Stage25 requires physically executed MONITOR episodes")
    actual = sum(int(r["actual_native_steps"]) for r in rows)
    histogram = [
        sum(int(r["native_action_histogram"][i]) for r in rows)
        for i in range(7)
    ]
    if actual != sum(histogram):
        raise ValueError("Stage25 true native action histogram mismatch")
    return {
        "actual_native_episodes": len(rows), "actual_native_steps": actual,
        "native_action_histogram": histogram,
        "strict_full_physical_goals": sum(
            int(r["external_strict_full_physical_goal"]) for r in rows
        ),
        "independent_effectful_open_episodes": sum(
            int(r["external_effective_door_open_episodes"]) for r in rows
        ),
        "actual_executed_post_open_specialist_actions": sum(
            r["actual_executed_specialist_post_open_actions"] for r in rows
        ),
        "reward_only_nonqualifying_episodes": sum(
            int(r["external_reward_positive_episodes"] > 0
                and not r["external_strict_full_physical_goal"]) for r in rows
        ),
    }


def acquire_monitor(
    frozen: dict[int, InteractionContextGRU],
    specialists: dict[tuple[str, int], InteractionContextGRU],
) -> dict:
    if (set(frozen) != set(MODEL_SEEDS)
            or set(specialists) != {
                (arm, seed) for arm in FIT_ARMS for seed in MODEL_SEEDS
            }
            or any(m.training for m in frozen.values())
            or any(m.training for m in specialists.values())):
        raise ValueError("Stage25 needs exactly three frozen and six specialist GRUs")
    result = {}
    for task in TASKS:
        layouts = MIXED_RECOVERY_MONITOR_LAYOUTS[task]
        actor = [
            monitor_episode(task, layout, "actor_only", None, None, None)
            for layout in layouts
        ]
        result[f"{task}:actor_only"] = {
            "episodes": actor, "summary": summarize(actor)
        }
        for seed in MODEL_SEEDS:
            for arm in ARMS:
                rows = [
                    monitor_episode(task, layout, arm, seed, frozen, specialists)
                    for layout in layouts
                ]
                result[f"{task}:{arm}:{seed}"] = {
                    "episodes": rows, "summary": summarize(rows)
                }
    return result


def decide(monitor: dict, acquisition: dict) -> dict:
    required = {f"{task}:actor_only" for task in TASKS} | {
        f"{task}:{arm}:{seed}" for task in TASKS
        for arm in ARMS for seed in MODEL_SEEDS
    }
    if set(monitor) != required:
        raise ValueError("Stage25 missing required native comparator groups")
    if not enough_diverse_recovery_evidence(acquisition):
        raise ValueError("Stage25 MONITOR opened without necessary true TRAIN evidence")
    actor_goals = 0
    learned_goals = {arm: 0 for arm in ARMS}
    door_goals = {arm: 0 for arm in ARMS}
    path_diffs = {}
    seed_gains = {s: 0 for s in MODEL_SEEDS}
    real_routed_door_episodes = 0
    pairs_primary: list[np.ndarray] = []
    pairs_secondary: list[np.ndarray] = []
    all_task_full_goal_counts = {}
    for task in TASKS:
        layouts = list(MIXED_RECOVERY_MONITOR_LAYOUTS[task])
        actor = monitor[f"{task}:actor_only"]["episodes"]
        if [r["layout_seed"] for r in actor] != layouts:
            raise ValueError("Stage25 original public actor MONITOR unpaired")
        actor_goals += sum(
            int(r["external_strict_full_physical_goal"]) for r in actor
        )
        all_task_full_goal_counts[f"{task}:actor_only"] = sum(
            int(r["external_strict_full_physical_goal"]) for r in actor
        )
        path_diffs[task] = {}
        for seed in MODEL_SEEDS:
            base = monitor[f"{task}:frozen_source:{seed}"]["episodes"]
            replay = monitor[f"{task}:source_replay_option:{seed}"]["episodes"]
            rescue = monitor[f"{task}:mixed_rehearsal_option:{seed}"]["episodes"]
            if any([r["layout_seed"] for r in group] != layouts
                   for group in (base, replay, rescue)):
                raise ValueError("Stage25 learned native MONITOR pairing invalid")
            for a, b, c in zip(base, replay, rescue, strict=True):
                event = a["first_real_public_door_open_action_index"]
                if any(
                    x["first_real_public_door_open_action_index"] != event
                    or x["true_native_prefix_step_count"]
                    != a["true_native_prefix_step_count"]
                    or x["prefix_true_native_action_sha256"]
                    != a["prefix_true_native_action_sha256"]
                    for x in (b, c)
                ):
                    raise ValueError("Stage25 real source learned PRE-OPEN native prefixes differ")
                all_identical = all(
                    x["full_true_native_action_sha256"]
                    == a["full_true_native_action_sha256"]
                    and x["actual_native_steps"] == a["actual_native_steps"]
                    and x["external_strict_full_physical_goal"]
                    == a["external_strict_full_physical_goal"]
                    for x in (b, c)
                )
                if (task == "fetch" or event is None) and not all_identical:
                    raise ValueError("Stage25 Fetch and no-event learned paths must be identical")
                if task == "doorkey":
                    real_routed_door_episodes += int(
                        c["actual_executed_specialist_post_open_actions"] > 0
                    )
                elif any(
                    x["actual_executed_specialist_post_open_actions"] > 0
                    for x in (a, b, c)
                ):
                    raise ValueError("Stage25 Fetch used specialist despite no public event")
            groups = {
                "frozen_source": base,
                "source_replay_option": replay,
                "mixed_rehearsal_option": rescue,
            }
            for arm, rows in groups.items():
                goals = sum(int(r["external_strict_full_physical_goal"]) for r in rows)
                learned_goals[arm] += goals
                all_task_full_goal_counts[f"{task}:{arm}:{seed}"] = goals
                if task == "doorkey":
                    door_goals[arm] += goals
            differences = [
                int(c["external_strict_full_physical_goal"])
                - int(b["external_strict_full_physical_goal"])
                for b, c in zip(replay, rescue, strict=True)
            ]
            if task == "doorkey":
                seed_gains[seed] = sum(differences)
                pairs_primary.append(np.asarray(differences, dtype=np.int64))
                pairs_secondary.append(np.asarray([
                    int(c["external_strict_full_physical_goal"])
                    - int(a["external_strict_full_physical_goal"])
                    for a, c in zip(base, rescue, strict=True)
                ], dtype=np.int64))
            path_diffs[task][str(seed)] = {
                "rescue_vs_replay": sum(
                    a["full_true_native_action_sha256"]
                    != b["full_true_native_action_sha256"]
                    for a, b in zip(replay, rescue, strict=True)
                ),
                "rescue_vs_frozen": sum(
                    a["full_true_native_action_sha256"]
                    != b["full_true_native_action_sha256"]
                    for a, b in zip(base, rescue, strict=True)
                ),
            }
    primary_array = np.asarray(pairs_primary, dtype=np.int64)
    secondary_array = np.asarray(pairs_secondary, dtype=np.int64)
    if primary_array.shape != (3, 40) or secondary_array.shape != (3, 40):
        raise ValueError("Stage25 independent paired DoorKey layouts missing")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seed_indices = rng.integers(0, 3, size=(20000, 3))
    layout_indices = rng.integers(0, 40, size=(20000, 3, 40))
    ci_primary = [
        float(x) for x in np.quantile(
            primary_array[seed_indices[:, :, None], layout_indices].sum(axis=(1, 2)),
            (.025, .975),
        )
    ]
    ci_secondary = [
        float(x) for x in np.quantile(
            secondary_array[seed_indices[:, :, None], layout_indices].sum(axis=(1, 2)),
            (.025, .975),
        )
    ]
    gain_primary = int(primary_array.sum())
    gain_secondary = int(secondary_array.sum())
    nonvacuous = bool(
        real_routed_door_episodes >= MIN_PUBLIC_MONITOR_SWITCHES
        and sum(
            path_diffs["doorkey"][str(s)]["rescue_vs_replay"]
            >= MIN_DOORKEY_NATIVE_ACTION_DIFFERENCES for s in MODEL_SEEDS
        ) >= 2
    )
    noninferior = (
        learned_goals["mixed_rehearsal_option"]
        >= 3 * actor_goals - ACTOR_MARGIN
    )
    numeric = bool(
        gain_primary >= MIN_PRIMARY_DOOR_GOAL_GAIN
        and gain_secondary >= MIN_SECONDARY_DOOR_GOAL_GAIN
        and sum(seed_gains[s] > 0 for s in MODEL_SEEDS) >= 2
        and door_goals["mixed_rehearsal_option"] >= MIN_TREATMENT_DOOR_GOALS
        and ci_primary[0] > 0
        and ci_secondary[0] > 0
        and noninferior
    )
    return {
        "registered_decision": (
            "INSUFFICIENT" if not nonvacuous
            else "PASS" if numeric else "FAIL"
        ),
        "actual_native_behavior_nonvacuity": nonvacuous,
        "registered_full_physical_numeric_gate": numeric,
        "paired_rescue_minus_replay_doorkey_strict_goals": gain_primary,
        "paired_rescue_minus_frozen_doorkey_strict_goals": gain_secondary,
        "per_seed_rescue_minus_replay_doorkey_full_goals": {
            str(s): int(seed_gains[s]) for s in MODEL_SEEDS
        },
        "paired_doorkey_seed_layout_bootstrap_rng": BOOTSTRAP_SEED,
        "paired_20000_bootstrap_95pct_rescue_minus_replay": ci_primary,
        "paired_20000_bootstrap_95pct_rescue_minus_frozen": ci_secondary,
        "learned_full_physical_goals": learned_goals,
        "learned_doorkey_full_goals": door_goals,
        "strict_full_physical_goals_by_native_task_arm_seed": all_task_full_goal_counts,
        "real_native_action_sha256_differences": path_diffs,
        "real_post_open_specialist_routed_door_episodes": real_routed_door_episodes,
        "once_measured_original_source_actor_goals": actor_goals,
        "descriptive_actor_tripled_arithmetic": 3 * actor_goals,
        "original_actor_margin_goal_count": 3 * actor_goals - ACTOR_MARGIN,
        "source_actor_noninferiority": noninferior,
        "fetch_all_three_learned_arms_true_native_action_identical": True,
        "pre_open_all_learned_actions_paired_identical": True,
        "hand_engineered_phase_boundary_not_learned_selector": True,
        "original_unlockpickup_final_touched": False,
    }


def execute(output: str) -> dict:
    if output != f"{OUT}/summary.json" or Path(output).exists():
        raise ValueError("Stage25 true native output path must be isolated and fresh")
    validate_split_contract()
    source_sha = admitted_source_commit()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    teacher, teacher_y, teacher_meta = collect_teacher()
    weights = class_weights(teacher_y)
    frozen = {}
    frozen_profiles = []
    for seed in MODEL_SEEDS:
        model, profile = fit_model(teacher, seed, "interaction_context", weights)
        profile["numeric_checkpoint"] = save_numeric(
            model, path.parent / f"frozen-source-{seed}.npz"
        )
        frozen[seed] = model
        frozen_profiles.append(profile)
    if len(frozen_profiles) != 3 or len({
        p["num_parameters"] for p in frozen_profiles
    }) != 1:
        raise ValueError("Stage25 original frozen learned source capacity mismatch")
    acquired, acquired_meta = collect_acquired(frozen)
    if acquired_meta["real_shared_native_acquisition_steps"] != 6 * ACQUIRE_BUDGET:
        raise ValueError("Stage25 source acquisition real steps not equal73728")
    if not enough_diverse_recovery_evidence(acquired_meta):
        report = {
            "study": "m23_stage25_diverse_mixed_recovery_rehearsal",
            "execution_source_commit": source_sha,
            "registered_scientific_decision": {
                "registered_decision": "INSUFFICIENT",
                "reason": "preregistered genuine executed TRAIN recovery label/open gate unmet",
                "independent_monitor_not_opened": True,
            },
            "real_source_teacher_train": teacher_meta,
            "real_shared_native_train_acquisition": acquired_meta,
            "frozen_source_numeric_checkpoints": frozen_profiles,
            "real_teacher_native_steps": teacher_meta["real_teacher_native_steps"],
            "actual_native_acquisition_steps": acquired_meta[
                "real_shared_native_acquisition_steps"
            ],
            "actual_native_monitor_steps": 0,
            "actual_total_real_native_steps": (
                teacher_meta["real_teacher_native_steps"]
                + acquired_meta["real_shared_native_acquisition_steps"]
            ),
            "original_unlockpickup_target_steps": 0,
            "original_target_final_episodes": 0,
        }
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "source_sha": source_sha, "decision": "INSUFFICIENT",
            "actual_native_monitor_steps": 0,
        }), flush=True)
        return report
    specialists = {}
    final_profiles = []
    for seed in MODEL_SEEDS:
        for arm in FIT_ARMS:
            model, profile = fit_stage25_followup(
                teacher, acquired[seed], frozen[seed], seed, arm, weights
            )
            profile["numeric_checkpoint"] = save_numeric(
                model, path.parent / f"{arm}-{seed}.npz"
            )
            specialists[(arm, seed)] = model
            final_profiles.append(profile)
    if (len(final_profiles) != 6
            or len({p["model_parameters"] for p in final_profiles}) != 1
            or any(
                a["frozen_base_model_sha256"] != b["frozen_base_model_sha256"]
                or a["real_optimizer_updates"] != b["real_optimizer_updates"]
                or a["actual_cached_label_presentations_not_native_steps"]
                != b["actual_cached_label_presentations_not_native_steps"]
                for a, b in zip(
                    final_profiles[::2], final_profiles[1::2], strict=True
                )
            )
            or any(
                p["real_optimizer_updates"] != FOLLOWUP_UPDATES
                or p["fixed_supervised_native_labels_per_update"]
                != TOKENS_PER_UPDATE for p in final_profiles
            )):
        raise ValueError("Stage25 specialist and replay capacity/optimizer not matched")
    monitor = acquire_monitor(frozen, specialists)
    verdict = decide(monitor, acquired_meta)
    monitor_steps = sum(
        group["summary"]["actual_native_steps"] for group in monitor.values()
    )
    actual_total = (
        teacher_meta["real_teacher_native_steps"]
        + acquired_meta["real_shared_native_acquisition_steps"]
        + monitor_steps
    )
    result = {
        "study": "m23_stage25_diverse_mixed_recovery_rehearsal",
        "execution_source_commit": source_sha,
        "source_train_layouts": {
            task: list(ids) for task, ids in MIXED_RECOVERY_TRAIN_LAYOUTS.items()
        },
        "source_acquisition_layouts": {
            task: list(ids) for task, ids in MIXED_RECOVERY_ACQUIRE_LAYOUTS.items()
        },
        "independent_source_monitor_layouts": {
            task: list(ids) for task, ids in MIXED_RECOVERY_MONITOR_LAYOUTS.items()
        },
        "model_seeds": list(MODEL_SEEDS), "evaluation_arms": list(ARMS),
        "specialist_training_arms": list(FIT_ARMS),
        "real_source_teacher_train": teacher_meta,
        "real_shared_native_train_acquisition": acquired_meta,
        "real_source_independent_monitor": monitor,
        "real_teacher_native_steps": teacher_meta["real_teacher_native_steps"],
        "actual_native_acquisition_steps": acquired_meta[
            "real_shared_native_acquisition_steps"
        ],
        "actual_native_monitor_steps": monitor_steps,
        "actual_total_real_native_steps": actual_total,
        "frozen_source_numeric_checkpoints": frozen_profiles,
        "final_specialist_numeric_checkpoints": final_profiles,
        "actual_frozen_source_fits": len(frozen_profiles),
        "actual_final_specialist_fits": len(final_profiles),
        "actual_optimizer_step_calls": sum(
            x["actual_optimizer_step_calls"] for x in frozen_profiles
        ) + sum(x["real_optimizer_updates"] for x in final_profiles),
        "cached_label_presentations_not_new_native_steps": sum(
            x["cached_source_label_presentations_not_native_steps"]
            for x in frozen_profiles
        ) + sum(
            x["actual_cached_label_presentations_not_native_steps"]
            for x in final_profiles
        ),
        "registered_scientific_decision": verdict,
        "original_unlockpickup_target_steps": 0,
        "original_target_final_episodes": 0,
        "imagined_native_steps": 0,
        "actual_study_wall_seconds": time.perf_counter() - started,
        "versions": {
            name: version(name)
            for name in ("numpy", "torch", "gymnasium", "minigrid")
        },
    }
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_sha": source_sha,
        "real_teacher_steps": result["real_teacher_native_steps"],
        "real_acquisition_steps": result["actual_native_acquisition_steps"],
        "real_monitor_steps": monitor_steps,
        "rescue_vs_replay_goals":
            verdict["paired_rescue_minus_replay_doorkey_strict_goals"],
        "rescue_vs_frozen_goals":
            verdict["paired_rescue_minus_frozen_doorkey_strict_goals"],
        "decision": verdict["registered_decision"],
    }), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=f"{OUT}/summary.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Stage25 requires CI-audited explicit --execute")
    execute(args.output)


if __name__ == "__main__":
    main()
