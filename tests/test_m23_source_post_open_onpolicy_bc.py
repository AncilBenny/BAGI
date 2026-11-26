"""Stage20 prospective equal-native-action, public-only on-policy correction guards."""
import numpy as np
import pytest

from bagi.experiments.m23_post_open_acquisition import (
    ACQUIRE_BUDGET,
    learner_advice_episode,
    teacher_episode,
)
from bagi.experiments.m23_post_open_corrected_train import (
    acquired_token_masks,
    validate_public_train_episode,
)
from bagi.experiments.m23_source_interaction_context_bc import WIDTH
from bagi.experiments.m23_source_post_open_onpolicy_bc import (
    ARMS,
    MODEL_SEEDS,
    TASKS,
    decide,
    monitor_episode,
)
from bagi.experiments.m23_study_contract import (
    FINAL_LAYOUTS,
    PHASE_ROUTER_MONITOR_LAYOUTS,
    POST_CORRECT_ACQUIRE_LAYOUTS,
    POST_CORRECT_MONITOR_LAYOUTS,
    POST_CORRECT_TRAIN_LAYOUTS,
    PUBLIC_EVENT_MEMORY_MONITOR_LAYOUTS,
    validate_split_contract,
)


def test_stage20_fresh_teacher_acquisition_and_monitor_remain_disjoint():
    validate_split_contract()
    current = {
        seed for split in (
            POST_CORRECT_TRAIN_LAYOUTS,
            POST_CORRECT_ACQUIRE_LAYOUTS,
            POST_CORRECT_MONITOR_LAYOUTS,
        )
        for ids in split.values() for seed in ids
    }
    previous = {
        seed for split in (
            FINAL_LAYOUTS, PUBLIC_EVENT_MEMORY_MONITOR_LAYOUTS,
            PHASE_ROUTER_MONITOR_LAYOUTS,
        )
        for ids in split.values() for seed in ids
    }
    assert len(current) == 2 * (96 + 24 + 40)
    assert not current & previous
    assert not current & {932001, 942001, 453001}
    assert len(POST_CORRECT_ACQUIRE_LAYOUTS["doorkey"]) == 24
    assert len(POST_CORRECT_MONITOR_LAYOUTS["fetch"]) == 40
    with pytest.raises(ValueError, match="prospectively restricted"):
        teacher_episode("doorkey", 932001)
    with pytest.raises(ValueError, match="registration invalid"):
        learner_advice_episode(
            "doorkey", 453001, 941, None, 1, 0
        )
    with pytest.raises(ValueError, match="independently fresh"):
        monitor_episode("unlockpickup", 453001, "actor_only", None, None)


def fake_acquired():
    """48 synthetic 256-step sequences; ZERO native env.step calls."""
    rows = []
    for task in TASKS:
        for episode in range(24):
            y = np.full(256, 2, dtype=np.int64)
            actual = y.copy()
            witness = 20 if task == "doorkey" and episode < 12 else None
            if witness is not None:
                actual[witness] = 5
            rows.append({
                "task": task, "layout_seed": 1 + episode, "base_model_seed": 941,
                "x": np.zeros((256, 1), dtype=np.float32), "y": y,
                "actually_executed": actual,
                "public_first_witnessed_open_action_index": witness,
                "origin": "frozen_learner_public_advice",
            })
    assert sum(len(row["y"]) for row in rows) == 2 * ACQUIRE_BUDGET
    return rows


def test_equal_real_advice_boost_count_and_post_open_next_step_not_toggle():
    rows = fake_acquired()
    control, post, counts = acquired_token_masks(rows, 941)
    assert len(control) == len(post) == len(rows)
    assert counts["doorkey"]["public_post_open_true_teacher_advice_labels"] == 12 * 235
    assert counts["fetch"]["public_post_open_true_teacher_advice_labels"] == 0
    assert sum(int(x.sum()) for x in control) == sum(int(x.sum()) for x in post)
    assert all(not mask.any() for mask in post[24:])
    assert all(not mask.any() for mask in control[24:])
    assert not post[0][20]
    assert bool(post[0][21])
    assert post[0][21:].all()
    # Same prereg seed gives the exact same random control mask, independent
    # of final-arm training, but different native-scientific seed can vary.
    again, treatment, other_counts = acquired_token_masks(rows, 941)
    assert all(np.array_equal(a, b) for a, b in zip(control, again, strict=True))
    assert all(np.array_equal(a, b) for a, b in zip(post, treatment, strict=True))
    assert counts == other_counts
    malformed = fake_acquired()
    malformed[0]["actually_executed"][20] = 2
    with pytest.raises(ValueError, match="impossible learner public door"):
        acquired_token_masks(malformed, 941)


def test_public_advice_is_not_relabelled_as_actually_executed_action():
    x = np.zeros((3, WIDTH), dtype=np.float32)
    actual = np.asarray([2, 0, 5], dtype=np.int64)
    advice = np.asarray([5, 5, 0], dtype=np.int64)
    x[1, -7 + actual[0]] = 1
    x[2, -7 + actual[1]] = 1
    episode = {
        "task": "doorkey", "origin": "frozen_learner_public_advice",
        "x": x, "y": advice, "actually_executed": actual,
    }
    validate_public_train_episode(episode, acquisition=True)
    with pytest.raises(ValueError, match="original expert source label was not executed"):
        validate_public_train_episode(episode, acquisition=False)
    corrupted = {**episode, "x": x.copy()}
    corrupted["x"][1, -7:] = 0
    with pytest.raises(ValueError, match="actually executed"):
        validate_public_train_episode(corrupted, acquisition=True)


def fake_monitor(goals_actor=4, goals_control=5, goals_post=20, distinct=True):
    groups = {}
    for task in TASKS:
        layouts = POST_CORRECT_MONITOR_LAYOUTS[task]
        groups[f"{task}:actor_only"] = {"episodes": [
            {"layout_seed": layout, "external_strict_full_physical_goal": idx < goals_actor}
            for idx, layout in enumerate(layouts)
        ]}
        for arm in ARMS:
            for seed in MODEL_SEEDS:
                positive = (
                    goals_post if arm == "post_open_corrected" and task == "doorkey"
                    else goals_control
                )
                groups[f"{task}:{arm}:{seed}"] = {"episodes": [
                    {
                        "layout_seed": layout,
                        "external_strict_full_physical_goal": idx < positive,
                        "actual_native_executed_action_sha256": (
                            f"{task}:{arm}:{seed}:{layout}"
                            if distinct else f"same:{layout}"
                        ),
                    }
                    for idx, layout in enumerate(layouts)
                ]}
    return groups


def test_decision_requires_real_acquisition_novelty_and_full_physical_goal():
    informative = {
        "public_door_open_episode_count": 17,
        "post_open_public_teacher_advice_disagreements": 130,
    }
    out = decide(fake_monitor(), informative)
    assert out["registered_decision"] == "PASS"
    assert out["physical_goal_gain_post_open_minus_uniform"] == 45
    assert out["real_acquisition_public_advice_nonvacuity"]
    assert out["bootstrap_lower_bound_above_zero"]
    assert out["actor_noninferior"]
    not_informative = {**informative, "public_door_open_episode_count": 2}
    assert decide(fake_monitor(), not_informative)["registered_decision"] == "INSUFFICIENT"
    assert decide(fake_monitor(distinct=False), informative)["registered_decision"] == (
        "INSUFFICIENT"
    )
    failed = decide(fake_monitor(goals_post=2), informative)
    assert failed["registered_decision"] == "FAIL"


def test_independent_native_monitor_layout_pairing_fail_closed():
    informative = {
        "public_door_open_episode_count": 17,
        "post_open_public_teacher_advice_disagreements": 130,
    }
    groups = fake_monitor()
    groups["doorkey:post_open_corrected:953"]["episodes"][0]["layout_seed"] = 1
    with pytest.raises(ValueError, match="MONITOR layouts unpaired"):
        decide(groups, informative)
