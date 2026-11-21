"""Stage24: actual public post-open specialist safety and physical criteria."""
import pytest

from bagi.experiments.m23_source_learned_recovery_specialist import (
    ARMS,
    MODEL_SEEDS,
    SPECIALIST_MONITOR_LAYOUTS,
    SPECIALIST_TRAIN_LAYOUTS,
    TASKS,
    decide,
    enough_executed_specialist_evidence,
    monitor_episode,
    select_public_specialist_action,
)
from bagi.experiments.m23_study_contract import (
    FINAL_LAYOUTS,
    RECOVERY_MONITOR_LAYOUTS,
    SPECIALIST_ACQUIRE_LAYOUTS,
    validate_split_contract,
)


def acquisition(suffix=256, openings=12):
    return {
        "per_seed_task": {
            f"{task}:{seed}": {
                "actually_executed_teacher_suffix_native_actions": (
                    suffix if task == "doorkey" else 0
                ),
                "public_witnessed_effectful_door_open_episodes": (
                    openings if task == "doorkey" else 0
                ),
            }
            for task in TASKS for seed in MODEL_SEEDS
        }
    }


def monitor(rescue_goals=20, different=True):
    out = {}
    for task in TASKS:
        layouts = SPECIALIST_MONITOR_LAYOUTS[task]
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
                         "executed_rescue_option": rescue_goals}[arm]
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
    assert ARMS == ("frozen_source", "source_replay_option", "executed_rescue_option")
    for arm in ARMS:
        assert select_public_specialist_action(arm, False, 2, 5, 3) == 2
    assert select_public_specialist_action("frozen_source", True, 2, 5, 3) == 2
    assert select_public_specialist_action("source_replay_option", True, 2, 5, 3) == 5
    assert select_public_specialist_action("executed_rescue_option", True, 2, 5, 3) == 3
    with pytest.raises(ValueError):
        select_public_specialist_action("invalid", True, 2, 5, 3)
    with pytest.raises(ValueError):
        select_public_specialist_action("executed_rescue_option", False, 2, 7, 3)


def test_fresh_train_acquisition_monitor_and_protected_final():
    validate_split_contract()
    train = {n for group in SPECIALIST_TRAIN_LAYOUTS.values() for n in group}
    acq = {n for group in SPECIALIST_ACQUIRE_LAYOUTS.values() for n in group}
    evaluation = {n for group in SPECIALIST_MONITOR_LAYOUTS.values() for n in group}
    prior = {n for group in RECOVERY_MONITOR_LAYOUTS.values() for n in group}
    final = {n for group in FINAL_LAYOUTS.values() for n in group}
    assert len(train) == 192 and len(acq) == 48 and len(evaluation) == 80
    assert train.isdisjoint(acq | evaluation | prior | final)
    assert evaluation.isdisjoint(acq | prior | final | {453001})
    with pytest.raises(ValueError, match="outside fresh independent source split"):
        monitor_episode("unlockpickup", 453001, "actor_only", None, None, None)


def test_real_recovery_acquisition_gate_must_precede_monitor():
    assert enough_executed_specialist_evidence(acquisition())
    assert not enough_executed_specialist_evidence(acquisition(suffix=127))
    assert not enough_executed_specialist_evidence(acquisition(openings=7))
    bad = acquisition()
    del bad["per_seed_task"]["fetch:947"]
    with pytest.raises(ValueError, match="missing real native"):
        enough_executed_specialist_evidence(bad)


def test_preregistered_two_paired_bootstraps_and_strict_goals():
    result = decide(monitor(), acquisition())
    assert result["registered_decision"] == "PASS"
    assert result["paired_rescue_minus_replay_doorkey_strict_goals"] == 45
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
    data["doorkey:executed_rescue_option:941"]["episodes"][0][
        "prefix_true_native_action_sha256"
    ] = "changed earlier than actual door opening"
    with pytest.raises(ValueError, match="PRE-OPEN"):
        decide(data, acquisition())
    data = monitor()
    data["fetch:executed_rescue_option:947"]["episodes"][0][
        "full_true_native_action_sha256"
    ] = "unexplained different Fetch path"
    with pytest.raises(ValueError, match="Fetch"):
        decide(data, acquisition())
