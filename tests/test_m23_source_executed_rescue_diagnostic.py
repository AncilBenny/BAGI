"""Stage22 tests: actual native public-event rescue is teacher-assisted, not AGI."""
import numpy as np
import pytest
import torch
from minigrid.core.constants import OBJECT_TO_IDX

from bagi.experiments.m23_source_executed_rescue_diagnostic import (
    ARMS,
    MODEL_SEEDS,
    RESCUE_MONITOR_LAYOUTS,
    RESCUE_TRAIN_LAYOUTS,
    TASKS,
    actual_public_open,
    decide,
    live_teacher_action,
)
from bagi.experiments.m23_source_interaction_context_bc import (
    PUBLIC_WIDTH,
    WIDTH,
    fit_model,
)
from bagi.experiments.m23_study_contract import (
    FINAL_LAYOUTS,
    POST_CORRECT_MONITOR_LAYOUTS,
    PUBLIC_EVENT_MEMORY_MONITOR_LAYOUTS,
    validate_split_contract,
)


def public_obs():
    return {
        "image": np.zeros((7, 7, 3), dtype=np.uint8),
        "mission": "use a key to open the door and reach the goal",
        "direction": 1,
    }


class PublicTeacherStub:
    def __init__(self):
        self.calls = 0

    def choose(self, obs, rng):
        assert set(obs) == {"image", "mission", "direction"}
        self.calls += 1
        return 2


def test_public_door_witness_requires_executed_action5_and_real_next_obs():
    before, after = public_obs(), public_obs()
    door = OBJECT_TO_IDX["door"]
    before["image"][3, 5] = [door, 0, 2]
    after["image"][3, 5] = [door, 0, 0]
    assert actual_public_open("doorkey", False, before, 5, after)
    assert not actual_public_open("doorkey", False, before, 2, after)
    assert not actual_public_open("doorkey", False, before, 3, after)
    assert actual_public_open("doorkey", True, after, 2, after)
    assert not actual_public_open("fetch", False, before, 5, after)
    with pytest.raises(ValueError):
        actual_public_open("fetch", True, before, 5, after)
    with pytest.raises(ValueError):
        actual_public_open("doorkey", 1, before, 5, after)


def test_shadow_teacher_never_chooses_until_after_executed_public_open():
    teacher = PublicTeacherStub()
    rng = np.random.default_rng(401)
    action, chosen = live_teacher_action(
        "learner_continues", True, 1, teacher, public_obs(), rng
    )
    assert (action, chosen, teacher.calls) == (1, False, 0)
    action, chosen = live_teacher_action(
        "teacher_after_open", False, 5, teacher, public_obs(), rng
    )
    assert (action, chosen, teacher.calls) == (5, False, 0)
    action, chosen = live_teacher_action(
        "teacher_after_open", True, 0, teacher, public_obs(), rng
    )
    assert (action, chosen, teacher.calls) == (2, True, 1)
    with pytest.raises(ValueError):
        live_teacher_action("teacher_after_open", True, 7, teacher, public_obs(), rng)
    with pytest.raises(ValueError):
        live_teacher_action("bad", False, 2, teacher, public_obs(), rng)
    assert ARMS == ("learner_continues", "teacher_after_open")


def test_fresh_registered_native_source_splits_and_sealed_target():
    validate_split_contract()
    train = set().union(*map(set, RESCUE_TRAIN_LAYOUTS.values()))
    monitor = set().union(*map(set, RESCUE_MONITOR_LAYOUTS.values()))
    prior = set().union(*map(set, POST_CORRECT_MONITOR_LAYOUTS.values()))
    prior |= set().union(*map(set, PUBLIC_EVENT_MEMORY_MONITOR_LAYOUTS.values()))
    prior |= set().union(*map(set, FINAL_LAYOUTS.values()))
    assert len(train) == 192
    assert len(monitor) == 80
    assert train.isdisjoint(monitor | prior)
    assert monitor.isdisjoint(prior | {453001})
    assert 1052001 in RESCUE_MONITOR_LAYOUTS["doorkey"]
    assert 1062001 in RESCUE_MONITOR_LAYOUTS["fetch"]


def synthetic_source_train():
    rows = []
    for i in range(192):
        n = 3
        y = np.asarray([(i + t) % 7 for t in range(n)], dtype=np.int64)
        x = np.zeros((n, WIDTH), dtype=np.float32)
        x[:, i % PUBLIC_WIDTH] = 1.0
        for t in range(1, n):
            x[t, PUBLIC_WIDTH + int(y[t - 1])] = 1.0
        rows.append({"x": x, "y": y})
    return rows


def test_single_same_capacity_frozen_model_reused_without_extra_fit():
    torch.set_num_threads(1)
    rows = synthetic_source_train()
    model, profile = fit_model(rows, MODEL_SEEDS[0], "interaction_context",
                               torch.ones(7))
    assert not model.training
    assert profile["num_parameters"] == 215879
    assert profile["actual_optimizer_step_calls"] == 144
    assert profile["cached_source_label_presentations_not_native_steps"] == 12 * 576
    assert len(MODEL_SEEDS) == 3


def synthetic_monitor(
    event=True, different=True, control_doorkey=3, rescue_doorkey=20
):
    groups = {}
    for task in TASKS:
        layouts = RESCUE_MONITOR_LAYOUTS[task]
        groups[f"{task}:actor_only"] = {"episodes": [
            {"layout_seed": layout, "external_strict_full_physical_goal": i < 4}
            for i, layout in enumerate(layouts)
        ]}
        for seed in MODEL_SEEDS:
            for arm in ARMS:
                rows = []
                for i, layout in enumerate(layouts):
                    triggered = bool(event and task == "doorkey")
                    n = (rescue_doorkey if arm == "teacher_after_open"
                         else control_doorkey) if triggered else 3
                    action_sha = f"{task}:{seed}:{layout}:identical"
                    if triggered and arm == "teacher_after_open" and different:
                        action_sha += ":real-teacher-takeover"
                    rows.append({
                        "layout_seed": layout,
                        "external_strict_full_physical_goal": i < n,
                        "first_public_open_executed_action_index":
                            3 if triggered else None,
                        "actual_common_prefix_steps": 4 if triggered else 20,
                        "actual_common_prefix_sha256":
                            f"{task}:{seed}:{layout}:prefix",
                        "actual_native_action_sha256": action_sha,
                        "actual_native_steps": 20,
                        "teacher_took_over_next_native_step": (
                            triggered and arm == "teacher_after_open"
                        ),
                        "actually_executed_teacher_post_open_actions": (
                            16 if triggered and arm == "teacher_after_open" else 0
                        ),
                    })
                groups[f"{task}:{arm}:{seed}"] = {"episodes": rows}
    return groups


def test_diagnostic_requires_executed_recovery_goals_and_uncertainty():
    verdict = decide(synthetic_monitor())
    assert verdict["registered_decision"] == "DIAGNOSTIC_SUPPORT"
    assert verdict["paired_rescue_minus_learner_strict_goals_by_task"] == {
        "doorkey": 51, "fetch": 0,
    }
    assert verdict["paired_rescue_minus_learner_strict_goals_total"] == 51
    assert verdict["first_public_open_paired_door_episodes"] == 120
    assert verdict["actual_expert_took_over_door_episodes"] == 120
    assert verdict["actual_expert_executed_post_open_native_actions"] == 1920
    assert verdict["paired_bootstrap_95pct_rescue_minus_learner_interval"][0] > 0
    assert verdict["expert_assisted_not_autonomous_learning"]
    assert decide(synthetic_monitor(different=False))[
        "registered_decision"
    ] == "INSUFFICIENT"
    assert decide(synthetic_monitor(event=False))[
        "registered_decision"
    ] == "INSUFFICIENT"
    assert decide(synthetic_monitor(
        rescue_doorkey=3
    ))["registered_decision"] == "NOT_SUPPORTED"


def test_public_prefix_pairing_and_fetch_strict_identity_fail_closed():
    data = synthetic_monitor()
    data["doorkey:teacher_after_open:941"]["episodes"][0][
        "actual_common_prefix_sha256"
    ] = "changed-earlier-than-open"
    with pytest.raises(ValueError, match="BEFORE rescue"):
        decide(data)
    data = synthetic_monitor()
    data["doorkey:teacher_after_open:947"]["episodes"][0][
        "first_public_open_executed_action_index"
    ] = None
    with pytest.raises(ValueError, match="BEFORE rescue"):
        decide(data)
    data = synthetic_monitor()
    data["fetch:teacher_after_open:953"]["episodes"][0][
        "actual_native_action_sha256"
    ] = "unexplained-fetch-trajectory"
    with pytest.raises(ValueError, match="non-triggered"):
        decide(data)
    data = synthetic_monitor()
    data["doorkey:teacher_after_open:941"]["episodes"][0][
        "actually_executed_teacher_post_open_actions"
    ] = 0
    with pytest.raises(ValueError, match="true native action"):
        decide(data)
