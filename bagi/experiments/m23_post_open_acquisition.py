"""Stage20 real native source demonstrations and shared on-policy public advice.

The frozen learner executes every acquisition env.step; the public teacher's
suggestion is NEVER counted as an executed action. Evaluator-only physics is
returned for external scoring, never passed to policy/features/advice.
"""
from __future__ import annotations

import hashlib

import numpy as np

from bagi.experiments.m13_public_interaction_coverage import Controller
from bagi.experiments.m23_source_feasibility import HORIZON
from bagi.experiments.m23_source_interaction_context_bc import (
    MODEL_SEEDS,
    InteractionContextGRU,
    public_action_features,
)
from bagi.experiments.m23_source_mission_bc import (
    make_native_audit,
    make_source_actor,
)
from bagi.experiments.m23_source_post_open_continuation_bc import (
    public_door_open_transition,
)
from bagi.experiments.m23_study_contract import (
    POST_CORRECT_ACQUIRE_LAYOUTS,
    POST_CORRECT_TRAIN_LAYOUTS,
    validate_split_contract,
)

TASKS = ("doorkey", "fetch")
ACQUIRE_BUDGET = 6144


def _digest(actions: list[int]) -> str:
    if any(x not in range(7) for x in actions):
        raise ValueError("Stage20 digest requires true native 0..6 actions")
    return hashlib.sha256(bytes(actions)).hexdigest()


def teacher_episode(task: str, layout: int) -> dict:
    """A teacher action is a genuine source label ONLY after audit.step executes."""
    if task not in TASKS or layout not in POST_CORRECT_TRAIN_LAYOUTS[task]:
        raise ValueError("Stage20 source teacher TRAIN is prospectively restricted")
    actor = make_source_actor(task)
    rng = np.random.default_rng(8_800_000 + layout)
    audit = make_native_audit(task)
    vectors: list[np.ndarray] = []
    actions: list[int] = []
    finished = False
    try:
        obs, _ = audit.reset(seed=layout)
        previous: int | None = None
        for _ in range(HORIZON):
            vector = public_action_features(
                obs, actor.key_observed_after_pickup, previous,
                arm="interaction_context",
            )
            action = int(actor.choose(obs, rng))
            if action not in range(7):
                raise ValueError("Stage20 illegal source teacher native action")
            nxt, _reward, terminated, truncated, _ = audit.step(action)
            actor.update(obs, action, nxt)
            vectors.append(vector)
            actions.append(action)
            previous = action
            obs = nxt
            if terminated or truncated:
                finished = True
                break
        truth = audit.summary()
        if (not finished or not actions or len(vectors) != len(actions)
                or truth["native_actions"] != len(actions)
                or sum(truth["action_counts"].values()) != len(actions)):
            raise ValueError("Stage20 original teacher native execution invalid")
        x = np.stack(vectors).astype(np.float32, copy=False)
        y = np.asarray(actions, dtype=np.int64)
        return {
            "task": task, "layout_seed": layout,
            "x": x, "y": y, "actually_executed": y.copy(),
            "origin": "genuine_original_teacher_execution",
            "native_steps": len(actions),
            "native_histogram": [truth["action_counts"][str(k)] for k in range(7)],
            "true_executed_action_sha256": _digest(actions),
            "public_source_x_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            "physical_full_goal_external_only": bool(truth["full_physical_goal"]),
        }
    finally:
        audit.env.close()


def collect_teacher() -> tuple[list[dict], np.ndarray, dict]:
    validate_split_contract()
    rows = [
        teacher_episode(task, layout)
        for task in TASKS for layout in POST_CORRECT_TRAIN_LAYOUTS[task]
    ]
    if len(rows) != 192:
        raise ValueError("Stage20 base TRAIN must contain 192 true expert episodes")
    labels = np.concatenate([r["y"] for r in rows])
    actions = sum(r["native_steps"] for r in rows)
    histogram = np.bincount(labels, minlength=7).tolist()
    if len(labels) != actions or sum(histogram) != actions:
        raise ValueError("Stage20 true source expert actions not conserved")
    return rows, labels, {
        "native_teacher_train_episodes": len(rows),
        "real_teacher_executed_native_steps": actions,
        "real_teacher_executed_histogram": histogram,
        "per_episode": [
            {k: v for k, v in r.items() if k not in ("x", "y", "actually_executed")}
            for r in rows
        ],
    }


def learner_advice_episode(
    task: str, layout: int, model_seed: int,
    base: InteractionContextGRU, budget: int, reset_index: int,
) -> dict:
    """One physical native episode, possibly PARTIAL at the exact action budget."""
    if (task not in TASKS or layout not in POST_CORRECT_ACQUIRE_LAYOUTS[task]
            or model_seed not in MODEL_SEEDS or budget < 1
            or budget > ACQUIRE_BUDGET or base.training
            or reset_index < 0):
        raise ValueError("Stage20 on-policy TRAIN acquisition registration invalid")
    observer = Controller("public_memory")
    teacher = make_source_actor(task)
    rng = np.random.default_rng(8_800_000 + layout + 10_000 * model_seed + reset_index)
    audit = make_native_audit(task)
    vectors: list[np.ndarray] = []
    advice: list[int] = []
    executed: list[int] = []
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
            suggestion = int(teacher.choose(obs, rng))
            action, hidden = base.native_step(vector, hidden)
            if suggestion not in range(7) or action not in range(7):
                raise ValueError("Stage20 invalid public advice/learner-native action")
            nxt, _reward, terminated, truncated, _ = audit.step(action)
            # Public phase evidence is verified only AFTER the REAL action.
            if (task == "doorkey" and first_open is None
                    and public_door_open_transition(obs, action, nxt)):
                first_open = len(executed)
            teacher.update(obs, action, nxt)
            observer.update(obs, action, nxt)
            vectors.append(vector)
            advice.append(suggestion)
            executed.append(action)
            previous = action
            obs = nxt
            if terminated or truncated:
                finished = True
                break
        truth = audit.summary()
        if (not executed or len(vectors) != len(executed)
                or len(advice) != len(executed)
                or truth["native_actions"] != len(executed)
                or sum(truth["action_counts"].values()) != len(executed)
                or (not finished and len(executed) != min(HORIZON, budget))):
            raise ValueError("Stage20 real learner acquisition/native accounting invalid")
        x = np.stack(vectors).astype(np.float32, copy=False)
        labels = np.asarray(advice, dtype=np.int64)
        true_actions = np.asarray(executed, dtype=np.int64)
        if (len(executed) > 1 and not np.array_equal(
                x[1:, -7:], np.eye(7, dtype=np.float32)[true_actions[:-1]])):
            raise ValueError("Stage20 public previous action is not ACTUALLY executed")
        if not np.array_equal(x[0, -7:], np.zeros(7, dtype=np.float32)):
            raise ValueError("Stage20 acquisition state leaked across episodes")
        post = slice(first_open + 1, None) if first_open is not None else slice(0, 0)
        return {
            "task": task, "layout_seed": layout, "base_model_seed": model_seed,
            "reset_index": reset_index, "x": x, "y": labels,
            "actually_executed": true_actions, "origin": "frozen_learner_public_advice",
            "public_first_witnessed_open_action_index": first_open,
            "native_steps": len(executed),
            "native_terminal": finished,
            "native_partial_at_acquisition_budget": not finished,
            "public_advice_disagreements": int(np.count_nonzero(labels != true_actions)),
            "post_open_advice_disagreements": int(np.count_nonzero(
                labels[post] != true_actions[post]
            )),
            "true_native_post_open_advice_labels": (
                len(executed) - first_open - 1 if first_open is not None else 0
            ),
            "external_effective_open_events": (
                int(truth["effective_door_open_events"]) if task == "doorkey" else 0
            ),
            "native_histogram": [truth["action_counts"][str(k)] for k in range(7)],
            "public_advice_histogram": np.bincount(labels, minlength=7).tolist(),
            "true_executed_action_sha256": _digest(executed),
            "public_advice_label_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            "public_learner_visited_x_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            "external_strict_full_physical_goal": bool(truth["full_physical_goal"]),
        }
    finally:
        audit.env.close()


def collect_acquired(
    frozen: dict[int, InteractionContextGRU],
) -> tuple[dict[int, list[dict]], dict]:
    """Exactly 6,144 real env.step actions per model seed and task, SHARED by arms."""
    if set(frozen) != set(MODEL_SEEDS):
        raise ValueError("Stage20 requires exactly three true frozen base fits")
    acquired: dict[int, list[dict]] = {}
    per_seed_task = {}
    total_actions = opens = post_advice_differences = 0
    for model_seed in MODEL_SEEDS:
        collected = []
        for task in TASKS:
            pool = POST_CORRECT_ACQUIRE_LAYOUTS[task]
            if len(pool) != 24:
                raise ValueError("Stage20 frozen acquisition layout pool changed")
            remaining = ACQUIRE_BUDGET
            episode_index = 0
            task_rows = []
            while remaining:
                layout = pool[episode_index % len(pool)]
                row = learner_advice_episode(
                    task, layout, model_seed, frozen[model_seed],
                    remaining, episode_index,
                )
                remaining -= row["native_steps"]
                episode_index += 1
                task_rows.append(row)
                if episode_index > ACQUIRE_BUDGET:
                    raise ValueError("Stage20 exceeded plausible real acquisition resets")
            used = sum(row["native_steps"] for row in task_rows)
            if used != ACQUIRE_BUDGET:
                raise ValueError("Stage20 acquisition budget not exactly true native actions")
            per_seed_task[f"{task}:{model_seed}"] = {
                "native_source_actions": used,
                "real_native_episode_resets": len(task_rows),
                "revisited_layout_resets": max(0, len(task_rows) - len(pool)),
                "partial_budget_episodes": sum(
                    row["native_partial_at_acquisition_budget"] for row in task_rows
                ),
                "physical_effective_open_episodes": sum(
                    row["external_effective_open_events"] > 0 for row in task_rows
                ),
                "public_witnessed_open_episodes": sum(
                    row["public_first_witnessed_open_action_index"] is not None
                    for row in task_rows
                ),
                "post_open_actual_advice_tokens": sum(
                    row["true_native_post_open_advice_labels"] for row in task_rows
                ),
                "post_open_teacher_disagreements": sum(
                    row["post_open_advice_disagreements"] for row in task_rows
                ),
            }
            total_actions += used
            opens += per_seed_task[f"{task}:{model_seed}"]["public_witnessed_open_episodes"]
            post_advice_differences += per_seed_task[
                f"{task}:{model_seed}"
            ]["post_open_teacher_disagreements"]
            collected.extend(task_rows)
        acquired[model_seed] = collected
    if total_actions != 6 * ACQUIRE_BUDGET:
        raise ValueError("Stage20 equal shared genuine native acquisition budget violated")
    return acquired, {
        "actual_new_acquisition_native_actions": total_actions,
        "per_seed_task": per_seed_task,
        "public_door_open_episode_count": opens,
        "post_open_public_teacher_advice_disagreements": post_advice_differences,
        "real_acquisition_native_episode_count": sum(
            len(rows) for rows in acquired.values()
        ),
        "per_seed_real_episode_metadata": {
            str(seed): [
                {k: v for k, v in row.items()
                 if k not in ("x", "y", "actually_executed")}
                for row in rows
            ]
            for seed, rows in acquired.items()
        },
    }
