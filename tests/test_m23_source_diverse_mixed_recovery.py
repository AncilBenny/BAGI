"""Stage25: actual public post-open specialist safety and physical criteria."""
import numpy as np
import pytest
import torch

from bagi.experiments.m23_executed_recovery_train import model_sha256
from bagi.experiments.m23_mixed_recovery_acquisition import (
    ACQUIRE_BUDGET,
    validate_recovery_row,
)
from bagi.experiments.m23_mixed_recovery_train import FIT_ARMS, fit_stage25_followup
from bagi.experiments.m23_source_diverse_mixed_recovery import (
    ARMS,
    MODEL_SEEDS,
    MIXED_RECOVERY_MONITOR_LAYOUTS,
    MIXED_RECOVERY_TRAIN_LAYOUTS,
    TASKS,
    decide,
    enough_diverse_recovery_evidence,
    monitor_episode,
    select_public_mixed_action,
)
from bagi.experiments.m23_source_interaction_context_bc import (
    PUBLIC_WIDTH,
    WIDTH,
    InteractionContextGRU,
)
from bagi.experiments.m23_study_contract import (
    FINAL_LAYOUTS,
    RECOVERY_MONITOR_LAYOUTS,
    MIXED_RECOVERY_ACQUIRE_LAYOUTS,
    validate_split_contract,
)


def acquisition(suffix=256, openings=12, unique=12):
    return {
        "per_seed_task": {
            f"{task}:{seed}": {
                "actually_executed_teacher_suffix_native_actions": (
                    suffix if task == "doorkey" else 0
                ),
                "public_witnessed_effectful_door_open_episodes": (
                    openings if task == "doorkey" else 0
                ),
                "unique_public_open_layout_seeds": (
                    unique if task == "doorkey" else 0
                ),
            }
            for task in TASKS for seed in MODEL_SEEDS
        }
    }


def monitor(rescue_goals=20, different=True):
    out = {}
    for task in TASKS:
        layouts = MIXED_RECOVERY_MONITOR_LAYOUTS[task]
        out[f"{task}:actor_only"] = {"episodes": [
            {"layout_seed": layout, "external_strict_full_physical_goal": i < 5}
            for i, layout in enumerate(layouts)
        ]}
        for seed in MODEL_SEEDS:
            for arm in ARMS:
                rows = []
                for i, layout in enumerate(layouts):
                    opened = task == "doorkey" and i < 30
                    cutoff = (
                        {"frozen_source": 3, "source_replay_option": 5,
                         "mixed_rehearsal_option": rescue_goals}[arm]
                        if task == "doorkey" else 5
                    )
                    digest = f"{task}:{seed}:{layout}:identical"
                    if opened and arm != "frozen_source":
                        digest += ":replay" if arm == "source_replay_option" or not different else ":rescue"
                    rows.append({
                        "layout_seed": layout,
                        "first_real_public_door_open_action_index": 5 if opened else None,
                        "true_native_prefix_step_count": 6 if opened else 20,
                        "prefix_true_native_action_sha256": f"{task}:{seed}:{layout}:prefix",
                        "full_true_native_action_sha256": digest,
                        "actual_native_steps": 20,
                        "actual_executed_specialist_post_open_actions": (
                            10 if opened and arm != "frozen_source" else 0
                        ),
                        "external_strict_full_physical_goal": i < cutoff,
                    })
                out[f"{task}:{arm}:{seed}"] = {"episodes": rows}
    return out


def test_public_router_chooses_one_actual_action_and_rejects_invalid():
    assert ARMS == ("frozen_source", "source_replay_option", "mixed_rehearsal_option")
    for arm in ARMS:
        assert select_public_mixed_action(arm, False, 2, 5, 3) == 2
    assert select_public_mixed_action("frozen_source", True, 2, 5, 3) == 2
    assert select_public_mixed_action("source_replay_option", True, 2, 5, 3) == 5
    assert select_public_mixed_action("mixed_rehearsal_option", True, 2, 5, 3) == 3
    with pytest.raises(ValueError):
        select_public_mixed_action("invalid", True, 2, 5, 3)
    with pytest.raises(ValueError):
        select_public_mixed_action("mixed_rehearsal_option", False, 2, 7, 3)


def test_fresh_train_acquisition_monitor_and_protected_final():
    validate_split_contract()
    train = {n for group in MIXED_RECOVERY_TRAIN_LAYOUTS.values() for n in group}
    acq = {n for group in MIXED_RECOVERY_ACQUIRE_LAYOUTS.values() for n in group}
    evaluation = {n for group in MIXED_RECOVERY_MONITOR_LAYOUTS.values() for n in group}
    prior = {n for group in RECOVERY_MONITOR_LAYOUTS.values() for n in group}
    final = {n for group in FINAL_LAYOUTS.values() for n in group}
    assert len(train) == 192 and len(acq) == 96 and len(evaluation) == 80
    assert len(MIXED_RECOVERY_ACQUIRE_LAYOUTS["doorkey"]) == 48
    assert ACQUIRE_BUDGET == 12288
    assert train.isdisjoint(acq | evaluation | prior | final)
    assert evaluation.isdisjoint(acq | prior | final | {453001})
    with pytest.raises(ValueError, match="outside fresh independent source split"):
        monitor_episode("unlockpickup", 453001, "actor_only", None, None, None)


def test_real_recovery_acquisition_gate_must_precede_monitor():
    assert enough_diverse_recovery_evidence(acquisition())
    assert not enough_diverse_recovery_evidence(acquisition(suffix=127))
    assert not enough_diverse_recovery_evidence(acquisition(openings=7, unique=7))
    assert not enough_diverse_recovery_evidence(acquisition(openings=20, unique=7))
    assert enough_diverse_recovery_evidence(acquisition(openings=8, unique=8))
    bad = acquisition()
    del bad["per_seed_task"]["fetch:947"]
    with pytest.raises(ValueError, match="missing real native"):
        enough_diverse_recovery_evidence(bad)


def test_preregistered_two_paired_bootstraps_and_strict_goals():
    result = decide(monitor(), acquisition())
    assert result["registered_decision"] == "PASS"
    assert result["paired_rescue_minus_replay_doorkey_strict_goals"] == 45
    assert result["strict_full_physical_goals_by_native_task_arm_seed"]["doorkey:mixed_rehearsal_option:941"] == 20
    assert result["paired_rescue_minus_frozen_doorkey_strict_goals"] == 51
    assert result["paired_20000_bootstrap_95pct_rescue_minus_replay"][0] > 0
    assert result["paired_20000_bootstrap_95pct_rescue_minus_frozen"][0] > 0
    assert result["real_post_open_specialist_routed_door_episodes"] == 90


def test_insufficient_vs_fail_and_no_early_monitor():
    assert decide(monitor(different=False), acquisition())[
        "registered_decision"
    ] == "INSUFFICIENT"
    assert decide(monitor(rescue_goals=5), acquisition())[
        "registered_decision"
    ] == "FAIL"
    with pytest.raises(ValueError, match="true TRAIN evidence"):
        decide(monitor(), acquisition(suffix=127))


def test_identical_preopen_and_fetch_actions_are_mandatory():
    data = monitor()
    data["doorkey:mixed_rehearsal_option:941"]["episodes"][0][
        "prefix_true_native_action_sha256"
    ] = "changed earlier than actual door opening"
    with pytest.raises(ValueError, match="PRE-OPEN"):
        decide(data, acquisition())
    data = monitor()
    data["fetch:mixed_rehearsal_option:947"]["episodes"][0][
        "full_true_native_action_sha256"
    ] = "unexplained different Fetch path"
    with pytest.raises(ValueError, match="Fetch"):
        decide(data, acquisition())


def actual_row(recovery=False, seed=941):
    y = np.asarray([5, 2, 1, 2], dtype=np.int64)
    x = np.zeros((len(y), WIDTH), dtype=np.float32)
    x[:, 0] = 1
    for ix, previous in enumerate(y[:-1], start=1):
        x[ix, PUBLIC_WIDTH + int(previous)] = 1
    mask = np.asarray([False, recovery, recovery, recovery], dtype=bool)
    return {
        "task": "doorkey", "x": x, "y": y,
        "origin": (
            "real_learner_then_actually_executed_expert"
            if recovery else "original_source_teacher_actually_executed"
        ),
        "expert_executed_suffix_mask": mask,
        "public_first_open_action_index": 0 if recovery else None,
        "base_model_seed": seed,
    }


def test_stage25_truly_executed_suffix_mask_excludes_prefix():
    row = actual_row(recovery=True)
    validate_recovery_row(row, recovery=True)
    assert row["expert_executed_suffix_mask"].tolist() == [
        False, True, True, True
    ]
    bad = actual_row(recovery=True)
    bad["expert_executed_suffix_mask"][0] = True
    with pytest.raises(ValueError, match="next-step"):
        validate_recovery_row(bad, recovery=True)
    bad = actual_row(recovery=True)
    bad["origin"] = "advisory_teacher_not_executed"
    with pytest.raises(ValueError, match="native executed recovery"):
        validate_recovery_row(bad, recovery=True)


def test_stage25_equal_frozen_weights_and_exact_64_64_real_mixed_labels():
    torch.set_num_threads(1)
    original = InteractionContextGRU()
    original.eval()
    before = model_sha256(original)
    teachers = [actual_row() for _ in range(192)]
    acquired = [actual_row(recovery=True) for _ in range(10)]
    replay, profile_a = fit_stage25_followup(
        teachers, acquired, original, 941, "source_replay", torch.ones(7)
    )
    mixed, profile_b = fit_stage25_followup(
        teachers, acquired, original, 941, "mixed_rehearsal", torch.ones(7)
    )
    assert not replay.training and not mixed.training and not original.training
    assert model_sha256(original) == before
    assert profile_a["frozen_base_model_sha256"] == before
    assert profile_b["frozen_base_model_sha256"] == before
    assert profile_a["model_parameters"] == profile_b["model_parameters"] == 215879
    assert profile_a["real_optimizer_updates"] == profile_b["real_optimizer_updates"] == 144
    assert profile_a["actual_cached_label_presentations_not_native_steps"] == (
        profile_b["actual_cached_label_presentations_not_native_steps"]
    ) == 18432
    assert profile_a["source_replay_label_presentations"] == 18432
    assert profile_a["executed_suffix_label_presentations"] == 0
    assert profile_b["source_replay_label_presentations"] == 9216
    assert profile_b["executed_suffix_label_presentations"] == 9216
    assert profile_b["genuine_executed_recovery_unique_labels"] == 30
    assert FIT_ARMS == ("source_replay", "mixed_rehearsal")
