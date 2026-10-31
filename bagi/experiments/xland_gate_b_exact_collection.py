"""REUSE-31: bounded, learner-legal XLand teacher-label collection adapter.

This is an experiment PREREQUISITE, not a trained agent or an evaluated skill.
Reuses established imitation.BC, XLand, existing BAGI teacher and feature code.
No XLand import at module import time, so deterministic synthetic tests can run.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CAP = 40
QUERY_BUDGET = 20
INPUT_WIDTH = 115
TRAIN_TASK = "putdown"
# TRAIN layouts, previously used in earlier diagnostics; NOT fresh evaluation.
SOURCE_LAYOUTS = (9701, 9702, 9708, 9710, 9711, 9712, 9713, 9714)
# REUSE-31b: prospectively fixed genuine native labels from two TRAIN layouts.
BALANCED_LAYOUT_QUOTAS = ((9701, 10), (9702, 10))
# Do not inspect or collect held-out expert labels on these untouched eval layouts.
FRESH_EVAL_LAYOUTS = (9810, 9811, 9812, 9813, 9814)
ARMS = ("reset_expert", "learner_correction")
# Diagnostic-only codes; never present the teacher stage or provenance to the policy.
EXPERT_STAGE_CODES = {"seek_square": 0, "carrying_square": 1,
                      "release_wrong_red": 2}
STATE_ORIGIN_CODES = {"reset": 0, "teacher_continuation": 1,
                      "learner_continuation": 2}


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
            "accepted_teacher_labels": len(self.rows),
            "rejected_teacher_queries": self.teacher_queries - len(self.rows),
            "expert_stage_label_counts": dict(sorted(Counter(
                str(r["expert_stage"]) for r in self.rows
            ).items())),
            "expert_stage_codebook": dict(EXPERT_STAGE_CODES),
            "state_origin_codebook": dict(STATE_ORIGIN_CODES),
            "reset_calls": self.reset_calls,
            "screened_source_layouts": list(self.screened_seeds),
            "source_layout_label_counts": dict(sorted(Counter(
                int(r["source_seed"]) for r in self.rows
            ).items())),
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
    per_layout_quota: tuple[int, ...] | None = None,
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
    if tuple(seeds) != SOURCE_LAYOUTS[:len(seeds)]:
        raise ValueError("Unregistered or out-of-order source layout")
    if per_layout_quota is not None and (
        tuple(zip(seeds, per_layout_quota)) != BALANCED_LAYOUT_QUOTAS
        or sum(per_layout_quota) != budget
    ):
        raise ValueError("Unregistered fixed source-layout quota contract")

    rows: list[dict] = []
    errors: list[dict] = []
    screened: list[int] = []
    steps = queries = resets = 0

    for seed_index, seed in enumerate(seeds):
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
        layout_rows = 0
        layout_target = (
            per_layout_quota[seed_index]
            if per_layout_quota is not None else budget
        )
        while (episode_step < cap and queries < budget
               and layout_rows < layout_target and not bool(ts.last())):
            feature = np.asarray(feature_fn(ts, history), dtype=np.float32)
            if feature.shape != (INPUT_WIDTH,) or not np.isfinite(feature).all():
                raise ValueError("Invalid 115-feature PUBLIC learner input")
            try:
                # A teacher *attempt* counts even if the oracle rejects this state.
                queries += 1
                label, stage = teacher(ts)
                y = native_action(label)
                if not isinstance(stage, str) or stage not in EXPERT_STAGE_CODES:
                    raise ValueError("Unsupported train-putdown expert stage")
            except (ValueError, IndexError, RuntimeError) as exc:
                errors.append({"seed": seed, "step": episode_step,
                               "reason": "unsupported_teacher", "detail": str(exc)})
                break
            act = y if arm == "reset_expert" else native_action(learner(feature.copy()))
            origin = "reset" if episode_step == 0 else (
                "teacher_continuation" if arm == "reset_expert"
                else "learner_continuation"
            )
            nxt = step(ts, act)
            steps += 1
            layout_rows += 1
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
        if per_layout_quota is not None and not errors and layout_rows != layout_target:
            errors.append({
                "seed": seed, "reason": "source_layout_quota_unmet",
                "expected_rows": layout_target, "collected_rows": layout_rows,
            })
        if errors:
            break
    return Collection(
        arm=arm, rows=tuple(rows), native_steps=steps,
        teacher_queries=queries, reset_calls=resets,
        screened_seeds=tuple(screened), expert_failures=tuple(errors),
        complete=(len(rows) == budget and queries == budget and steps == budget
                  and not errors),
    )


def validate_pair(
    reset: Collection, correction: Collection, *,
    per_layout_quota: tuple[int, ...] | None = None,
) -> dict:
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
    # Fixed total labels do not ensure the arms came from the same layouts.
    # A different layout mixture confounds the correction-vs-reset contrast.
    reset_seeds = Counter(int(row["source_seed"]) for row in reset.rows)
    correction_seeds = Counter(int(row["source_seed"]) for row in correction.rows)
    if reset_seeds != correction_seeds:
        raise ValueError("Mismatched source-layout label allocation between arms")
    if per_layout_quota is not None and (
        per_layout_quota != tuple(count for _, count in BALANCED_LAYOUT_QUOTAS)
        or reset_seeds != Counter(dict(BALANCED_LAYOUT_QUOTAS))
    ):
        raise ValueError("Preregistered fixed source-layout quota not met")
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
            # Guard actual final 28 learner feature bits, not merely side metadata.
            true_history_bits = np.eye(7, dtype=np.float32)[
                np.asarray(history, dtype=np.int64)
            ].reshape(-1)
            if not np.array_equal(row["public_obs"][-28:], true_history_bits):
                raise ValueError("Encoded action history differs from execution")
            if row["expert_stage"] not in EXPERT_STAGE_CODES:
                raise ValueError("Unexpected expert stage in supervised row")
            expected_origin = "reset" if row["episode_step"] == 0 else (
                "teacher_continuation" if arm.arm == "reset_expert"
                else "learner_continuation"
            )
            if row["state_origin"] != expected_origin:
                raise ValueError("False learner/teacher state provenance")
            if arm.arm == "reset_expert" and row["executed_action"] != row["expert_label"]:
                raise ValueError("Reset-expert control must execute the expert label")
            history = (*history[1:], int(row["executed_action"]))
    return {"matched": True, "labels_per_arm": QUERY_BUDGET,
            "native_steps_per_arm": QUERY_BUDGET,
            "reset_arm": reset.summary(), "correction_arm": correction.summary(),
            "privileged_teacher_policy_input": False,
            "heldout_teacher_queries": 0,
            "comparison_status": "collection_only_no_training_or_evaluation",
            **({"fixed_source_layout_quotas": {
                str(seed): count for seed, count in BALANCED_LAYOUT_QUOTAS
            }} if per_layout_quota is not None else {})}


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
        expert_stage_code=np.asarray([
            EXPERT_STAGE_CODES[row["expert_stage"]] for row in data
        ], dtype=np.int8),
        state_origin_code=np.asarray([
            STATE_ORIGIN_CODES[row["state_origin"]] for row in data
        ], dtype=np.int8),
    )
def load_verified_collection_pair(
    directory: Path, *, model_seed: int, source_checksum: str,
    model_record: dict,
) -> dict[str, dict[str, np.ndarray]]:
    """Fail closed on persisted 20+20 native labels before matched BC training.

    The original REUSE-19 source SHA must be verified independently by the
    caller with load_verified_source. Diagnostic stage/origin codes are checked
    but are never included in policy inputs.
    """
    if (model_seed not in (17, 29)
            or model_record.get("model_seed") != model_seed
            or model_record.get("numeric_source_sha256") != source_checksum
            or model_record.get("matched") is not True
            or set(model_record.get("arms", {})) != set(ARMS)):
        raise ValueError("Unmatched or unverified native collection provenance")
    accounting = model_record.get("pair_accounting", {})
    if (accounting.get("matched") is not True
            or accounting.get("labels_per_arm") != QUERY_BUDGET
            or accounting.get("native_steps_per_arm") != QUERY_BUDGET
            or accounting.get("heldout_teacher_queries") != 0):
        raise ValueError("Missing matched collection accounting")
    quota = accounting.get("fixed_source_layout_quotas")
    expected_quota = {str(seed): count for seed, count in BALANCED_LAYOUT_QUOTAS}
    if quota is not None and quota != expected_quota:
        raise ValueError("Unregistered source-layout quota in paired accounting")
    expected_fields = {
        "obs", "expert_label", "executed_action",
        "prior_executed_actions", "source_seed", "episode_step",
        "done", "reward", "expert_stage_code", "state_origin_code",
    }
    result = {}
    for arm in ARMS:
        meta = model_record["arms"][arm]
        if (meta.get("arm") != arm or meta.get("complete") is not True
                or any(meta.get(key) != QUERY_BUDGET for key in (
                    "native_steps", "teacher_queries",
                    "accepted_teacher_labels", "label_rows"
                ))
                or meta.get("rejected_teacher_queries") != 0
                or meta.get("expert_failures")
                or meta.get("expert_stage_codebook") != EXPERT_STAGE_CODES
                or meta.get("state_origin_codebook") != STATE_ORIGIN_CODES
                or meta.get("public_input_width") != INPUT_WIDTH
                or accounting.get(
                    "reset_arm" if arm == "reset_expert" else "correction_arm"
                ) != meta):
            raise ValueError("Incomplete or inconsistent collection arm metadata")
        path = Path(directory) / f"seed{model_seed}_{arm}.npz"
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_fields:
                raise ValueError("Unreviewed numeric collection fields")
            data = {name: np.array(archive[name], copy=True)
                    for name in expected_fields}
        if (data["obs"].shape != (QUERY_BUDGET, INPUT_WIDTH)
                or data["obs"].dtype != np.float32
                or data["prior_executed_actions"].shape != (QUERY_BUDGET, 4)
                or data["expert_stage_code"].dtype != np.int8
                or data["state_origin_code"].dtype != np.int8
                or data["done"].dtype != np.bool_
                or any(data[name].shape != (QUERY_BUDGET,) for name in (
                    "expert_label", "executed_action", "source_seed",
                    "episode_step", "done", "reward",
                    "expert_stage_code", "state_origin_code"
                ))
                or any(data[name].dtype.kind not in "iu" for name in (
                    "expert_label", "executed_action", "source_seed",
                    "episode_step", "prior_executed_actions"
                ))
                or data["reward"].dtype.kind != "f"
                or not np.isfinite(data["obs"]).all()
                or not np.isfinite(data["reward"]).all()):
            raise ValueError("Unsafe or malformed native numeric collection")
        stage_counts = Counter()
        origin_counts = Counter()
        label_counts = Counter()
        action_counts = Counter()
        layout_counts = Counter()
        seen: list[int] = []
        previous_seed = None
        previous_step = -1
        previous_done = False
        history = (6, 6, 6, 6)
        for i in range(QUERY_BUDGET):
            seed = int(data["source_seed"][i])
            step = int(data["episode_step"][i])
            if seed != previous_seed:
                quota_boundary = (
                    quota == expected_quota
                    and previous_seed == BALANCED_LAYOUT_QUOTAS[0][0]
                    and previous_step == BALANCED_LAYOUT_QUOTAS[0][1] - 1
                )
                if (previous_seed is not None and not previous_done
                        and previous_step != CAP - 1 and not quota_boundary):
                    raise ValueError("Unjustified episode boundary in saved labels")
                if (len(seen) >= len(SOURCE_LAYOUTS)
                        or seed != SOURCE_LAYOUTS[len(seen)] or step != 0):
                    raise ValueError("Reordered or substituted source layout")
                seen.append(seed)
                history = (6, 6, 6, 6)
            elif previous_done or step != previous_step + 1:
                raise ValueError("Noncontiguous or postterminal saved episode")
            if not 0 <= step < CAP:
                raise ValueError("Episode exceeded native 40-action cap")
            if not np.array_equal(data["prior_executed_actions"][i], history):
                raise ValueError("Saved history differs from executed actions")
            encoded = np.eye(7, dtype=np.float32)[
                np.asarray(history, dtype=np.int64)
            ].reshape(-1)
            if not np.array_equal(data["obs"][i, -28:], encoded):
                raise ValueError("Public feature history differs from execution")
            label = int(data["expert_label"][i])
            action = int(data["executed_action"][i])
            if (label not in range(6) or action not in range(6)
                    or (arm == "reset_expert" and label != action)):
                raise ValueError("Invalid or falsely executed expert label")
            stage_code = int(data["expert_stage_code"][i])
            origin_code = int(data["state_origin_code"][i])
            expected_origin = STATE_ORIGIN_CODES[
                "reset" if step == 0 else (
                    "teacher_continuation" if arm == "reset_expert"
                    else "learner_continuation"
                )
            ]
            if (stage_code not in EXPERT_STAGE_CODES.values()
                    or origin_code != expected_origin):
                raise ValueError("Untrusted teacher-stage or state-origin code")
            stage_counts.update(
                name for name, code in EXPERT_STAGE_CODES.items()
                if code == stage_code
            )
            origin_counts.update(
                name for name, code in STATE_ORIGIN_CODES.items()
                if code == origin_code
            )
            label_counts[label] += 1
            action_counts[action] += 1
            layout_counts[seed] += 1
            history = (*history[1:], action)
            previous_seed, previous_step = seed, step
            previous_done = bool(data["done"][i])
        if quota == expected_quota and dict(layout_counts) != dict(BALANCED_LAYOUT_QUOTAS):
            raise ValueError("Saved fixed source-layout quota is not met")
        if (meta.get("screened_source_layouts") != seen
                or {int(k): v for k, v in meta.get(
                    "source_layout_label_counts", {}
                ).items()} != dict(layout_counts)
                or meta.get("expert_stage_label_counts") != dict(stage_counts)
                or meta.get("state_origins") != dict(origin_counts)
                or {int(k): v for k, v in meta.get(
                    "sampled_label_counts", {}
                ).items()} != dict(label_counts)
                or {int(k): v for k, v in meta.get(
                    "executed_action_counts", {}
                ).items()} != dict(action_counts)):
            raise ValueError("Saved labels disagree with provenance histograms")
        result[arm] = data
    if not np.array_equal(
        result["reset_expert"]["source_seed"],
        result["learner_correction"]["source_seed"],
    ):
        raise ValueError("Mismatched source-layout label allocation in numeric pair")
    return result
