"""Stage23 public-only real expert suffixes on learner-generated TRAIN states.

Only genuinely executed native teacher actions after learner-witnessed door
opening become recovery supervision. Acquisition is shared by both final arms.
"""
from __future__ import annotations

import hashlib

import numpy as np

from bagi.experiments.m13_public_interaction_coverage import Controller
from bagi.experiments.m23_source_executed_rescue_diagnostic import (
    actual_public_open,
    live_teacher_action,
)
from bagi.experiments.m23_source_feasibility import HORIZON
from bagi.experiments.m23_source_interaction_context_bc import (
    MODEL_SEEDS,
    InteractionContextGRU,
    public_action_features,
)
from bagi.experiments.m23_source_mission_bc import make_native_audit, make_source_actor
from bagi.experiments.m23_study_contract import (
    RECOVERY_ACQUIRE_LAYOUTS,
    RECOVERY_TRAIN_LAYOUTS,
    validate_split_contract,
)

TASKS = ("doorkey", "fetch")
ACQUIRE_BUDGET = 6144


def validate_recovery_row(row: dict, recovery: bool) -> None:
    x, y = row["x"], row["y"]
    mask = row["expert_executed_suffix_mask"]
    if (row["task"] not in TASKS or not isinstance(x, np.ndarray)
            or not isinstance(y, np.ndarray) or not isinstance(mask, np.ndarray)
            or x.ndim != 2 or x.shape[1] != 1056 or y.ndim != 1
            or mask.shape != y.shape or mask.dtype != bool
            or len(x) != len(y) or not 1 <= len(y) <= HORIZON
            or not np.isfinite(x).all() or set(np.unique(y)) - set(range(7))):
        raise ValueError("Stage23 source public true-action sequence/mask invalid")
    if not np.array_equal(x[0, -7:], np.zeros(7, dtype=np.float32)):
        raise ValueError("Stage23 previous executed action leaked across episodes")
    if len(y) > 1 and not np.array_equal(
        x[1:, -7:], np.eye(7, dtype=np.float32)[y[:-1]]
    ):
        raise ValueError("Stage23 previous action must be actually executed")
    if recovery:
        if row["origin"] != "real_learner_then_actually_executed_expert":
            raise ValueError("Stage23 acquisition must come from native executed recovery")
        first = row["public_first_open_action_index"]
        if first is None:
            if mask.any():
                raise ValueError("Stage23 expert suffix without actual public open")
        elif (row["task"] != "doorkey" or type(first) is not int
              or not 0 <= first < len(y) or int(y[first]) != 5):
            raise ValueError("Stage23 public real door-open index not executed toggle")
        expected = np.zeros(len(y), dtype=bool)
        if first is not None:
            expected[first + 1:] = True
        if not np.array_equal(mask, expected):
            raise ValueError("Stage23 only truly executed next-step expert actions qualify")
    elif mask.any():
        raise ValueError("Stage23 original source teacher is not takeover data")


def teacher_episode(task: str, layout: int) -> dict:
    if task not in TASKS or layout not in RECOVERY_TRAIN_LAYOUTS[task]:
        raise ValueError("Stage23 original teacher TRAIN unregistered source layout")
    actor = make_source_actor(task)
    audit = make_native_audit(task)
    rng = np.random.default_rng(8_800_000 + layout)
    vectors: list[np.ndarray] = []
    actions: list[int] = []
    finished = False
    try:
        obs, _ = audit.reset(seed=layout)
        previous = None
        for _ in range(HORIZON):
            x = public_action_features(
                obs, actor.key_observed_after_pickup, previous,
                arm="interaction_context",
            )
            action = int(actor.choose(obs, rng))
            if action not in range(7):
                raise ValueError("Stage23 illegal source actor native action")
            nxt, _reward, terminated, truncated, _ = audit.step(action)
            actor.update(obs, action, nxt)
            vectors.append(x)
            actions.append(action)
            previous = action
            obs = nxt
            if terminated or truncated:
                finished = True
                break
        truth = audit.summary()
        if (not finished or not actions or truth["native_actions"] != len(actions)
                or sum(truth["action_counts"].values()) != len(actions)):
            raise ValueError("Stage23 source actor true physical TRAIN actions not conserved")
        x = np.stack(vectors).astype(np.float32, copy=False)
        y = np.asarray(actions, dtype=np.int64)
        row = {
            "task": task, "layout_seed": layout, "x": x, "y": y,
            "expert_executed_suffix_mask": np.zeros(len(y), dtype=bool),
            "origin": "original_source_teacher_actually_executed",
            "native_steps": len(y),
            "native_action_histogram": [
                truth["action_counts"][str(i)] for i in range(7)
            ],
            "true_executed_action_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
            "public_x_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            "external_strict_full_goal": bool(truth["full_physical_goal"]),
        }
        validate_recovery_row(row, recovery=False)
        return row
    finally:
        audit.env.close()


def collect_teacher() -> tuple[list[dict], np.ndarray, dict]:
    validate_split_contract()
    rows = [
        teacher_episode(task, layout)
        for task in TASKS for layout in RECOVERY_TRAIN_LAYOUTS[task]
    ]
    if len(rows) != 192:
        raise ValueError("Stage23 source teacher needs 192 real TRAIN episodes")
    labels = np.concatenate([r["y"] for r in rows])
    steps = sum(row["native_steps"] for row in rows)
    histogram = np.bincount(labels, minlength=7).tolist()
    if steps != len(labels) or sum(histogram) != steps:
        raise ValueError("Stage23 source actor true native label accounting failed")
    return rows, labels, {
        "real_teacher_train_episodes": len(rows),
        "real_teacher_native_steps": steps,
        "real_teacher_native_action_histogram": histogram,
        "per_episode": [
            {k: v for k, v in row.items()
             if k not in ("x", "y", "expert_executed_suffix_mask")}
            for row in rows
        ],
    }


def executed_recovery_episode(
    task: str, layout: int, seed: int, frozen: InteractionContextGRU,
    budget: int, reset_index: int,
) -> dict:
    """One real native episode, potentially partial at the fixed TRAIN budget."""
    if (task not in TASKS or layout not in RECOVERY_ACQUIRE_LAYOUTS[task]
            or seed not in MODEL_SEEDS or frozen.training
            or not 1 <= budget <= ACQUIRE_BUDGET or reset_index < 0):
        raise ValueError("Stage23 real source acquisition registration invalid")
    observer = Controller("public_memory")
    teacher = make_source_actor(task)
    rng = np.random.default_rng(8_800_000 + layout + 10_000 * seed + reset_index)
    audit = make_native_audit(task)
    vectors: list[np.ndarray] = []
    actions: list[int] = []
    expert_mask: list[bool] = []
    public_open = False
    first_open: int | None = None
    hidden = None
    previous: int | None = None
    finished = False
    try:
        obs, _ = audit.reset(seed=layout)
        for _ in range(min(HORIZON, budget)):
            vector = public_action_features(
                obs, observer.key_observed_after_pickup, previous,
                arm="interaction_context",
            )
            if public_open and task == "doorkey":
                proposed = 0  # no unused learner proposal counted as physical
            else:
                proposed, hidden = frozen.native_step(vector, hidden)
            action, was_teacher = live_teacher_action(
                "teacher_after_open" if task == "doorkey" else "learner_continues",
                public_open, proposed, teacher, obs, rng,
            )
            nxt, _reward, terminated, truncated, _ = audit.step(action)
            opened_next = actual_public_open(task, public_open, obs, action, nxt)
            if opened_next and not public_open:
                first_open = len(actions)
            public_open = opened_next
            observer.update(obs, action, nxt)
            teacher.update(obs, action, nxt)
            vectors.append(vector)
            actions.append(action)
            expert_mask.append(was_teacher)
            previous = action
            obs = nxt
            if terminated or truncated:
                finished = True
                break
        truth = audit.summary()
        if (not actions or len(vectors) != len(actions)
                or len(expert_mask) != len(actions)
                or truth["native_actions"] != len(actions)
                or sum(truth["action_counts"].values()) != len(actions)
                or (not finished and len(actions) != min(HORIZON, budget))):
            raise ValueError("Stage23 actual learner/rescue action budget not conserved")
        if (first_open is not None) != (
            truth["effective_door_open_events"] > 0 if task == "doorkey" else False
        ):
            raise ValueError("Stage23 real public opening and physical audit disagree")
        x = np.stack(vectors).astype(np.float32, copy=False)
        y = np.asarray(actions, dtype=np.int64)
        mask = np.asarray(expert_mask, dtype=bool)
        row = {
            "task": task, "layout_seed": layout, "base_model_seed": seed,
            "reset_index": reset_index, "x": x, "y": y,
            "expert_executed_suffix_mask": mask,
            "origin": "real_learner_then_actually_executed_expert",
            "public_first_open_action_index": first_open,
            "native_steps": len(y),
            "native_episode_terminated": finished,
            "partial_at_fixed_budget": not finished,
            "real_executed_expert_suffix_steps": int(mask.sum()),
            "native_action_histogram": [
                truth["action_counts"][str(i)] for i in range(7)
            ],
            "true_executed_action_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
            "public_x_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            "external_strict_full_goal": (
                bool(truth["full_physical_goal"]) if finished else False
            ),
            "physical_open_episode": bool(
                truth["effective_door_open_events"] > 0
            ) if task == "doorkey" else False,
        }
        validate_recovery_row(row, recovery=True)
        return row
    finally:
        audit.env.close()


def collect_acquired(
    frozen: dict[int, InteractionContextGRU],
) -> tuple[dict[int, list[dict]], dict]:
    """Single shared native acquisition stream, 6144 steps/task/model seed."""
    if (set(frozen) != set(MODEL_SEEDS)
            or any(model.training for model in frozen.values())):
        raise ValueError("Stage23 requires three genuinely frozen source fits")
    acquired: dict[int, list[dict]] = {}
    per_seed_task = {}
    total = 0
    for seed in MODEL_SEEDS:
        rows = []
        for task in TASKS:
            pool = RECOVERY_ACQUIRE_LAYOUTS[task]
            if len(pool) != 24:
                raise ValueError("Stage23 acquisition source layout pool must be24")
            remaining = ACQUIRE_BUDGET
            reset_index = 0
            task_rows = []
            while remaining:
                layout = pool[reset_index % len(pool)]
                row = executed_recovery_episode(
                    task, layout, seed, frozen[seed], remaining, reset_index
                )
                remaining -= row["native_steps"]
                reset_index += 1
                task_rows.append(row)
                if reset_index > ACQUIRE_BUDGET:
                    raise ValueError("Stage23 implausible repeated native resets")
            used = sum(row["native_steps"] for row in task_rows)
            if used != ACQUIRE_BUDGET:
                raise ValueError("Stage23 exact genuine per-task acquisition budget broken")
            per_seed_task[f"{task}:{seed}"] = {
                "actual_native_steps": used,
                "actual_real_episode_resets": len(task_rows),
                "actual_budget_partial_episodes": sum(
                    int(r["partial_at_fixed_budget"]) for r in task_rows
                ),
                "public_witnessed_effectful_door_open_episodes": sum(
                    int(r["public_first_open_action_index"] is not None)
                    for r in task_rows
                ),
                "actually_executed_teacher_suffix_native_actions": sum(
                    r["real_executed_expert_suffix_steps"] for r in task_rows
                ),
                "source_teacher_rescued_completed_physical_goals": sum(
                    int(r["external_strict_full_goal"])
                    for r in task_rows if r["real_executed_expert_suffix_steps"] > 0
                ),
            }
            rows.extend(task_rows)
            total += used
        acquired[seed] = rows
    if total != 6 * ACQUIRE_BUDGET:
        raise ValueError("Stage23 acquired real native actions not exactly36864")
    return acquired, {
        "real_shared_native_acquisition_steps": total,
        "per_seed_task": per_seed_task,
        "real_native_acquisition_episode_resets": sum(
            len(rows) for rows in acquired.values()
        ),
        "per_seed_native_episode_metadata": {
            str(seed): [
                {k: v for k, v in row.items()
                 if k not in ("x", "y", "expert_executed_suffix_mask")}
                for row in rows
            ]
            for seed, rows in acquired.items()
        },
    }
