"""REUSE-31: bounded, learner-legal XLand teacher-label collection adapter.

This is an experiment PREREQUISITE, not a trained agent or an evaluated skill.
Reuses established imitation.BC, XLand, existing BAGI teacher and feature code.
No XLand import at module import time, so deterministic synthetic tests can run.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

CAP = 40
QUERY_BUDGET = 20
INPUT_WIDTH = 115
TRAIN_TASK = "putdown"
# TRAIN layouts, previously used in earlier diagnostics; NOT fresh evaluation.
SOURCE_LAYOUTS = (9701, 9702, 9708, 9710, 9711, 9712, 9713, 9714)
# Do not inspect or collect held-out expert labels on these untouched eval layouts.
FRESH_EVAL_LAYOUTS = (9810, 9811, 9812, 9813, 9814)
ARMS = ("reset_expert", "learner_correction")


@dataclass(frozen=True)
class Collection:
    arm: str
    rows: tuple[dict, ...]
    native_steps: int
    teacher_queries: int
    reset_calls: int
    screened_seeds: tuple[int, ...]
    expert_failures: tuple[dict, ...]
    complete: bool

    def summary(self) -> dict:
        return {
            "arm": self.arm,
            "complete": self.complete,
            "native_steps": self.native_steps,
            "teacher_queries": self.teacher_queries,
            "reset_calls": self.reset_calls,
            "screened_source_layouts": list(self.screened_seeds),
            "expert_failures": list(self.expert_failures),
            "sampled_label_counts": dict(sorted(Counter(
                int(r["expert_label"]) for r in self.rows
            ).items())),
            "executed_action_counts": dict(sorted(Counter(
                int(r["executed_action"]) for r in self.rows
            ).items())),
            "state_origins": dict(sorted(Counter(
                str(r["state_origin"]) for r in self.rows
            ).items())),
            "label_rows": len(self.rows),
            "public_input_width": INPUT_WIDTH,
        }


def native_action(action: object) -> int:
    """Reject invalid labels/actions rather than silently repairing them."""
    a = np.asarray(action)
    if a.shape not in ((), (1,)) or a.dtype.kind not in "iu":
        raise ValueError("Expected one integer native action")
    result = int(a.reshape(-1)[0])
    if result not in range(6):
        raise ValueError("Non-native XLand action")
    return result


def public_features(timestep: object, history: tuple[int, ...]) -> np.ndarray:
    """Use ONLY the existing learner-visible feature and instruction contract.

    No privileged grid, pocket, expert stage, geometry, or true goal state enters
    the learned policy. Their use is restricted to teacher and diagnostics.
    """
    from bagi.experiments.xland_existing_bc import visible_features
    from bagi.experiments.xland_instruction_action_history import assemble
    from bagi.experiments.xland_upstream_instruction_bc import task_instructions

    obs = timestep.observation
    tokens = task_instructions(
        obs["goal_encoding"], obs["rule_encoding"], enabled=True
    )[None, :]
    return assemble(
        visible_features(obs)[None, :], tokens,
        np.asarray([history], dtype=np.int64),
        condition="real_executed_action_history",
    )[0]


def collect_arm(
    *,
    arm: str,
    reset: Callable[[int], object],
    step: Callable[[object, int], object],
    screen_nonadjacent: Callable[[object], bool],
    feature_fn: Callable[[object, tuple[int, ...]], np.ndarray],
    teacher: Callable[[object], tuple[int, str]],
    learner: Callable[[np.ndarray], object] | None,
    seeds: tuple[int, ...] = SOURCE_LAYOUTS,
    budget: int = QUERY_BUDGET,
    cap: int = CAP,
) -> Collection:
    """Query the teacher once per REAL executed step, never cherry-pick labels.

    * reset_expert: the privileged expert controls the source trajectory.
    * learner_correction: the fixed learned policy controls the source
      trajectory; teacher *only labels* that currently learner-visited state.
    Teacher labels and executed commands have separate provenance.
    Nonadjacent screening is diagnostic only; an invalid predeclared seed
    aborts the arm, with no silent replacement or off-list seed search.
    """
    if arm not in ARMS or budget != QUERY_BUDGET or cap != CAP:
        raise ValueError("REUSE-31 frozen collection contract changed")
    if arm == "learner_correction" and learner is None:
        raise ValueError("Correction arm requires a frozen learner")
    if not seeds or len(set(seeds)) != len(seeds) or any(
        not isinstance(x, int) for x in seeds
    ):
        raise ValueError("Predeclared unique native source seeds required")
    if any(seed in FRESH_EVAL_LAYOUTS for seed in seeds):
        raise ValueError("Evaluation-layout contamination")

    rows: list[dict] = []
    errors: list[dict] = []
    screened: list[int] = []
    steps = queries = resets = 0

    for seed in seeds:
        if queries >= budget or errors:
            break
        ts = reset(seed)
        resets += 1
        screened.append(seed)
        if not bool(screen_nonadjacent(ts)):
            errors.append({"seed": seed, "reason": "ineligible_adjacent_reset"})
            break
        if bool(ts.last()):
            errors.append({"seed": seed, "reason": "terminal_at_reset"})
            break
        history: tuple[int, ...] = (6, 6, 6, 6)
        episode_step = 0
        while episode_step < cap and queries < budget and not bool(ts.last()):
            feature = np.asarray(feature_fn(ts, history), dtype=np.float32)
            if feature.shape != (INPUT_WIDTH,) or not np.isfinite(feature).all():
                raise ValueError("Invalid 115-feature PUBLIC learner input")
            try:
                # A teacher *attempt* counts even if the oracle rejects this state.
                queries += 1
                label, stage = teacher(ts)
                y = native_action(label)
            except (ValueError, IndexError, RuntimeError) as exc:
                errors.append({"seed": seed, "step": episode_step,
                               "reason": "unsupported_teacher", "detail": str(exc)})
                break
            if not isinstance(stage, str) or not stage:
                raise ValueError("Teacher stage is diagnostic provenance only")
            act = y if arm == "reset_expert" else native_action(learner(feature.copy()))
            origin = "reset" if episode_step == 0 else (
                "teacher_continuation" if arm == "reset_expert"
                else "learner_continuation"
            )
            nxt = step(ts, act)
            steps += 1
            rows.append({
                "source_seed": int(seed), "episode_step": episode_step,
                "state_origin": origin, "expert_stage": stage,
                "public_obs": feature.copy(),
                "prior_executed_actions": tuple(history),
                "expert_label": y, "executed_action": act,
                "reward": float(nxt.reward), "done": bool(nxt.last()),
            })
            history = (*history[1:], act)
            ts = nxt
            episode_step += 1
        if errors:
            break
    return Collection(
        arm=arm, rows=tuple(rows), native_steps=steps,
        teacher_queries=queries, reset_calls=resets,
        screened_seeds=tuple(screened), expert_failures=tuple(errors),
        complete=(len(rows) == budget and queries == budget and steps == budget
                  and not errors),
    )


def validate_pair(reset: Collection, correction: Collection) -> dict:
    """Do not train on partial/unmatched budgets or privileged observations."""
    if (reset.arm, correction.arm) != ARMS:
        raise ValueError("Wrong ordered comparison arms")
    if not reset.complete or not correction.complete:
        raise ValueError("Incomplete collection; no valid REUSE-31 training comparison")
    if (reset.native_steps != correction.native_steps
            or reset.teacher_queries != correction.teacher_queries
            or len(reset.rows) != len(correction.rows)
            or len(reset.rows) != QUERY_BUDGET):
        raise ValueError("Mismatched native-step or teacher-query budgets")
    for arm in (reset, correction):
        old_seed = None
        history: tuple[int, ...] = (6, 6, 6, 6)
        for row in arm.rows:
            if row["source_seed"] != old_seed:
                old_seed = row["source_seed"]
                history = (6, 6, 6, 6)
            if tuple(row["prior_executed_actions"]) != history:
                raise ValueError("Non-executed or leaked action history")
            if row["public_obs"].shape != (INPUT_WIDTH,):
                raise ValueError("Non-public observation input width")
            if arm.arm == "reset_expert" and row["executed_action"] != row["expert_label"]:
                raise ValueError("Reset-expert control must execute the expert label")
            history = (*history[1:], int(row["executed_action"]))
    return {"matched": True, "labels_per_arm": QUERY_BUDGET,
            "native_steps_per_arm": QUERY_BUDGET,
            "reset_arm": reset.summary(), "correction_arm": correction.summary(),
            "privileged_teacher_policy_input": False,
            "heldout_teacher_queries": 0,
            "comparison_status": "collection_only_no_training_or_evaluation"}


def save_numeric(path: str, collection: Collection) -> None:
    """Safe numeric arrays only; never pickle full state or arbitrary objects."""
    data = collection.rows
    np.savez_compressed(
        path,
        obs=np.stack([row["public_obs"] for row in data]),
        expert_label=np.array([row["expert_label"] for row in data], dtype=np.int64),
        executed_action=np.array([row["executed_action"] for row in data], dtype=np.int64),
        prior_executed_actions=np.asarray(
            [row["prior_executed_actions"] for row in data], dtype=np.int64),
        source_seed=np.array([row["source_seed"] for row in data], dtype=np.int64),
        episode_step=np.array([row["episode_step"] for row in data], dtype=np.int64),
        done=np.array([row["done"] for row in data], dtype=np.bool_),
        reward=np.array([row["reward"] for row in data], dtype=np.float32),
    )