"""Stage23 real executed recovery labels and teacher-free MONITOR source gates."""
import copy

import numpy as np
import pytest
import torch

from bagi.experiments.m23_executed_recovery_acquisition import (
    ACQUIRE_BUDGET,
    validate_recovery_row,
)
from bagi.experiments.m23_executed_recovery_train import (
    ARMS,
    FOLLOWUP_UPDATES,
    TOKENS_PER_UPDATE,
    fit_followup,
    model_sha256,
    prepare_label_pool,
)
from bagi.experiments.m23_source_executed_recovery_learning import (
    MODEL_SEEDS,
    RECOVERY_MONITOR_LAYOUTS,
    RECOVERY_TRAIN_LAYOUTS,
    TASKS,
    decide,
    monitor_episode,
)
from bagi.experiments.m23_source_interaction_context_bc import (
    PUBLIC_WIDTH,
    WIDTH,
    InteractionContextGRU,
)
from bagi.experiments.m23_study_contract import (
    FINAL_LAYOUTS,
    POST_CORRECT_ACQUIRE_LAYOUTS,
    POST_CORRECT_MONITOR_LAYOUTS,
    RESCUE_MONITOR_LAYOUTS,
    validate_split_contract,
)


def executed_row(recovery=False, seed=941, suffix=True):
    y = np.asarray([5, 2, 1, 2], dtype=np.int64)
    x = np.zeros((len(y), WIDTH), dtype=np.float32)
    x[:, 0] = 1
    for ix, previous in enumerate(y[:-1], start=1):
        x[ix, PUBLIC_WIDTH + int(previous)] = 1
    mask = np.asarray([False, suffix, suffix, suffix], dtype=bool)
    return {
        "task": "doorkey", "x": x, "y": y,
        "origin": (
            "real_learner_then_actually_executed_expert"
            if recovery else "original_source_teacher_actually_executed"
        ),
        "expert_executed_suffix_mask": (
            mask if recovery else np.zeros(len(y), dtype=bool)
        ),
        "public_first_open_action_index": 0 if recovery and suffix else None,
        "base_model_seed": seed,
    }


def test_public_true_actions_and_suffix_mask_reject_advisory_or_stale_history():
    teacher = executed_row()
    recovery = executed_row(recovery=True)
    validate_recovery_row(teacher, recovery=False)
    validate_recovery_row(recovery, recovery=True)
    bad = copy.deepcopy(recovery)
    bad["y"][0] = 2
    with pytest.raises(ValueError, match="executed toggle"):
        validate_recovery_row(bad, recovery=True)
    bad = copy.deepcopy(recovery)
    bad["expert_executed_suffix_mask"][0] = True
    with pytest.raises(ValueError, match="next-step"):
        validate_recovery_row(bad, recovery=True)
    bad = copy.deepcopy(recovery)
    bad["x"][1, PUBLIC_WIDTH + 5] = 0
    with pytest.raises(ValueError, match="actually executed"):
        validate_recovery_row(bad, recovery=True)
    bad = copy.deepcopy(recovery)
    bad["origin"] = "teacher_advice_only"
    with pytest.raises(ValueError, match="native executed recovery"):
        validate_recovery_row(bad, recovery=True)
    bad = copy.deepcopy(teacher)
    bad["expert_executed_suffix_mask"][1] = True
    with pytest.raises(ValueError, match="not takeover"):
        validate_recovery_row(bad, recovery=False)


def test_real_suffix_pool_never_supervises_learner_prefix_or_advice():
    teachers = [executed_row() for _ in range(192)]
    acquired = [executed_row(recovery=True, seed=941) for _ in range(9)]
    replay, regular = prepare_label_pool(
        teachers, acquired, 941, "source_replay"
    )
    suffix, recovery = prepare_label_pool(
        teachers, acquired, 941, "executed_rescue"
    )
    assert regular["unique_source_real_supervised_tokens"] == 192 * 4
    assert regular["labels_are_only_actually_executed_recovery_suffix"] is False
    assert recovery["unique_source_real_supervised_tokens"] == 9 * 3
    assert recovery["labels_are_only_actually_executed_recovery_suffix"] is True
    assert all(not mask[0] for _, mask in suffix)
    assert all(mask.all() for _, mask in replay)
    wrong = [executed_row(recovery=True, seed=947)]
    with pytest.raises(ValueError, match="wrong frozen learner seed"):
        prepare_label_pool(teachers, wrong, 941, "executed_rescue")
    with pytest.raises(ValueError, match="no genuinely executed"):
        prepare_label_pool(
            teachers, [executed_row(recovery=True, suffix=False)],
            941, "executed_rescue",
        )


def test_equal_frozen_start_optimizer_and_128_real_label_presentations():
    torch.set_num_threads(1)
    teachers = [executed_row() for _ in range(192)]
    acquired = [executed_row(recovery=True) for _ in range(10)]
    torch.manual_seed(941)
    original = InteractionContextGRU()
    original.eval()
    sha = model_sha256(original)
    control, a = fit_followup(
        teachers, acquired, original, 941, "source_replay", torch.ones(7)
    )
    rescue, b = fit_followup(
        teachers, acquired, original, 941, "executed_rescue", torch.ones(7)
    )
    assert not control.training and not rescue.training and not original.training
    assert model_sha256(original) == sha
    assert a["frozen_base_model_sha256"] == b["frozen_base_model_sha256"] == sha
    assert a["model_parameters"] == b["model_parameters"] == 215879
    assert a["real_optimizer_updates"] == b["real_optimizer_updates"] == 144
    assert a["fixed_supervised_native_labels_per_update"] == 128
    assert b["fixed_supervised_native_labels_per_update"] == 128
    assert a["actual_cached_label_presentations_not_native_steps"] == (
        b["actual_cached_label_presentations_not_native_steps"]
    ) == 18432
    assert a["unique_source_real_supervised_tokens"] != (
        b["unique_source_real_supervised_tokens"]
    )
    assert ARMS == ("source_replay", "executed_rescue")
    assert FOLLOWUP_UPDATES * TOKENS_PER_UPDATE == 18432


def test_new_training_acquisition_monitor_layouts_sealed_and_disjoint():
    validate_split_contract()
    from bagi.experiments.m23_study_contract import RECOVERY_ACQUIRE_LAYOUTS

    def all_layouts(split):
        return {k for ids in split.values() for k in ids}
    train, acquired, monitor = (
        all_layouts(x) for x in (
            RECOVERY_TRAIN_LAYOUTS, RECOVERY_ACQUIRE_LAYOUTS,
            RECOVERY_MONITOR_LAYOUTS,
        )
    )
    older = set().union(*(
        all_layouts(x) for x in (
            FINAL_LAYOUTS, POST_CORRECT_ACQUIRE_LAYOUTS,
            POST_CORRECT_MONITOR_LAYOUTS, RESCUE_MONITOR_LAYOUTS,
        )
    ))
    assert len(train) == 192 and len(acquired) == 48 and len(monitor) == 80
    assert not train.intersection(acquired | monitor | older)
    assert not acquired.intersection(monitor | older)
    assert not monitor.intersection(older | {453001})
    assert ACQUIRE_BUDGET == 6144
    assert len(MODEL_SEEDS) == 3
    with pytest.raises(ValueError, match="independent unopened"):
        monitor_episode("doorkey", 1052001, "actor_only", None, None)


def sample_monitor(ctrl=3, treatment=20, distinct=True):
    groups = {}
    for task in TASKS:
        layouts = RECOVERY_MONITOR_LAYOUTS[task]
        groups[f"{task}:actor_only"] = {"episodes": [
            {"layout_seed": layout, "external_strict_full_physical_goal": i < 4}
            for i, layout in enumerate(layouts)
        ]}
        for seed in MODEL_SEEDS:
            for arm in ARMS:
                n = treatment if arm == "executed_rescue" else ctrl
                if task == "fetch":
                    n = 10
                items = []
                for i, layout in enumerate(layouts):
                    action = f"{task}:{seed}:{layout}:control"
                    if arm == "executed_rescue" and distinct:
                        action = f"{task}:{seed}:{layout}:treatment"
                    items.append({
                        "layout_seed": layout,
                        "external_strict_full_physical_goal": i < n,
                        "full_native_executed_action_sha256": action,
                    })
                groups[f"{task}:{arm}:{seed}"] = {"episodes": items}
    return groups


def sample_acquired(teacher_labels=140, public_opens=10):
    return {"per_seed_task": {
        f"doorkey:{seed}": {
            "actually_executed_teacher_suffix_native_actions": teacher_labels,
            "public_witnessed_effectful_door_open_episodes": public_opens,
        }
        for seed in MODEL_SEEDS
    }}


def test_preregistered_true_physical_gate_and_uncertainty_recomputed():
    z = decide(sample_monitor(), sample_acquired())
    assert z["registered_decision"] == "PASS"
    assert z["paired_executed_rescue_minus_source_replay_strict_goal_gain"] == 51
    assert z["paired_strict_goal_gain_by_task"] == {
        "doorkey": 51, "fetch": 0,
    }
    assert z["bootstrap_lower_strictly_positive"]
    assert z["evaluation_teacher_free"]
    assert z["source_actor_noninferiority"]
    assert decide(
        sample_monitor(distinct=False), sample_acquired()
    )["registered_decision"] == "INSUFFICIENT"
    assert decide(
        sample_monitor(), sample_acquired(teacher_labels=50)
    )["registered_decision"] == "INSUFFICIENT"
    assert decide(
        sample_monitor(ctrl=20, treatment=3), sample_acquired()
    )["registered_decision"] == "FAIL"


def test_monitor_fail_closed_on_unpaired_or_missing_native_layout():
    groups = sample_monitor()
    groups["doorkey:executed_rescue:941"]["episodes"][0][
        "layout_seed"
    ] = 1052002
    with pytest.raises(ValueError, match="independent layout mismatch"):
        decide(groups, sample_acquired())
    groups = sample_monitor()
    del groups["fetch:source_replay:953"]
    with pytest.raises(ValueError, match="groups incomplete"):
        decide(groups, sample_acquired())
