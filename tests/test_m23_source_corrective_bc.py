"""Stage12 source on-policy public correction versus matched expert controls."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bagi.experiments.m23_source_corrective_bc import (
    ACQUIRE_ARMS,
    ARMS,
    MODEL_SEEDS,
    TASKS,
    acquire_episode,
    decide,
    fit_from_scratch,
    monitor_episode,
)
from bagi.experiments.m23_source_public_vision_bc import WIDTH
from bagi.experiments.m23_study_contract import (
    BC_MONITOR_LAYOUTS,
    CORRECT_ACQUIRE_LAYOUTS,
    CORRECT_MONITOR_LAYOUTS,
    CORRECT_TRAIN_LAYOUTS,
    FINAL_LAYOUTS,
    VISION_MONITOR_LAYOUTS,
    validate_split_contract,
)


def test_stage12_fresh_layouts_disjoint_and_unlockpickup_sealed():
    validate_split_contract()
    current = {
        seed for group in (
            CORRECT_TRAIN_LAYOUTS, CORRECT_ACQUIRE_LAYOUTS, CORRECT_MONITOR_LAYOUTS
        ) for layout in group.values() for seed in layout
    }
    old = {
        seed for group in (BC_MONITOR_LAYOUTS, VISION_MONITOR_LAYOUTS, FINAL_LAYOUTS)
        for layout in group.values() for seed in layout
    }
    assert len(current) == 320
    assert len(current & old) == 0
    assert len(MODEL_SEEDS) == 3
    assert ACQUIRE_ARMS == ("extra_expert", "corrective_onpolicy")
    with pytest.raises(ValueError, match="fresh source native"):
        monitor_episode("unlockpickup", 453001, "actor_only", None, None)


def test_model_fits_have_exact_same_seed_weights_steps_and_exposures():
    from bagi.experiments.m23_source_corrective_bc import VisualBC

    x = np.zeros((8, WIDTH), dtype=np.float32)
    y = np.asarray([0, 1, 2, 3, 5, 0, 1, 2], dtype=np.int64)
    class_weights = torch.ones(7, dtype=torch.float32)
    model1, expert = fit_from_scratch(x, y, 941, "extra_expert", class_weights)
    model2, corrective = fit_from_scratch(
        x.copy(), y.copy(), 941, "corrective_onpolicy", class_weights
    )
    assert isinstance(model1, VisualBC)
    assert all(torch.equal(a, b) for a, b in zip(
        model1.parameters(), model2.parameters(), strict=True
    ))
    assert expert["actual_optimizer_step_calls"] == 12
    assert corrective["actual_optimizer_step_calls"] == 12
    assert expert["cached_source_label_presentations"] == 96
    assert corrective["cached_source_label_presentations"] == 96
    assert expert["common_class_weights"] == corrective["common_class_weights"]
    with pytest.raises(ValueError):
        fit_from_scratch(x, np.full(len(y), 8, dtype=np.int64),
                         941, "corrective_onpolicy", class_weights)


def test_advisory_teacher_is_not_the_executed_native_action(monkeypatch):
    from bagi.experiments import m23_source_corrective_bc as study

    class FakeAudit:
        def __init__(self):
            self.actions = []
            self.env = SimpleNamespace(close=lambda: None)

        def reset(self, **kwargs):
            self.actions = []
            return {
                "image": np.zeros((7, 7, 3), dtype=np.uint8),
                "direction": 0,
                "mission": "use the key to open the door and then get to the goal",
            }, {}

        def step(self, action):
            self.actions.append(action)
            obs = {
                "image": np.zeros((7, 7, 3), dtype=np.uint8),
                "direction": 0,
                "mission": "use the key to open the door and then get to the goal",
            }
            return obs, 0.0, False, len(self.actions) >= 4, {}

        def summary(self):
            return {
                "native_actions": len(self.actions),
                "action_counts": {str(i): self.actions.count(i) for i in range(7)},
                "full_physical_goal": False,
                "effective_door_open_events": 0,
            }

    class FakeTeacher:
        def __init__(self):
            self.key_observed_after_pickup = False

        def choose(self, observation, rng):
            return 3

        def update(self, before, executed, after):
            assert executed in (2, 3)

    class AlwaysForward(study.VisualBC):
        def forward(self, x):
            out = torch.zeros((len(x), 7))
            out[:, 2] = 100.0
            return out

    monkeypatch.setattr(study, "native_acquire_audit", lambda task: FakeAudit())
    monkeypatch.setattr(study, "make_source_actor", lambda task: FakeTeacher())
    model = AlwaysForward()
    model.eval()
    correct = acquire_episode(
        "doorkey", CORRECT_ACQUIRE_LAYOUTS["doorkey"][0], 941,
        "corrective_onpolicy", model
    )
    extra = acquire_episode(
        "doorkey", CORRECT_ACQUIRE_LAYOUTS["doorkey"][0], 941,
        "extra_expert", model
    )
    assert correct["actual_native_executed_actions"] == 4
    assert correct["public_teacher_advice_queries"] == 4
    assert correct["teacher_advice_vs_executed_action_disagreements"] == 4
    assert correct["teacher_advice_not_physically_executed"] == 4
    assert extra["teacher_advice_not_physically_executed"] == 0
    assert correct["actual_native_action_histogram"][2] == 4
    assert correct["public_advised_native_action_histogram"][3] == 4
    assert correct["executed_actions_sha256"] != extra["executed_actions_sha256"]
    assert (extra["public_teacher_advice_queries"] == 4
            and extra["teacher_advice_vs_executed_action_disagreements"] == 0)
    assert np.array_equal(correct["y"], np.array([3, 3, 3, 3], dtype=np.int64))


def fake_monitor(actor_goal_count=4, expert_goal_count=6,
                 corrective_goal_count=16, distinct=True) -> dict:
    output = {}
    for task in TASKS:
        layouts = CORRECT_MONITOR_LAYOUTS[task]
        output[f"{task}:actor_only"] = {"episodes": [
            {"layout_seed": layout, "external_strict_full_physical_goal":
             i < actor_goal_count,
             "actual_executed_action_sha256": f"actor:{layout}"}
            for i, layout in enumerate(layouts)
        ]}
        for arm in ARMS:
            for seed in MODEL_SEEDS:
                count = corrective_goal_count if arm == "corrective_onpolicy" else (
                    expert_goal_count if arm == "extra_expert" else 7
                )
                output[f"{task}:{arm}:{seed}"] = {"episodes": [
                    {"layout_seed": layout,
                     "external_strict_full_physical_goal": i < count,
                     "actual_executed_action_sha256": (
                         f"{arm}:{seed}:{layout}" if distinct else f"actor:{layout}"
                     )}
                    for i, layout in enumerate(layouts)
                ]}
    return output


def fake_acquisition(n=300) -> dict:
    return {
        str(seed): {"task_data": {
            task: {"public_teacher_vs_actual_native_action_disagreements": n}
            for task in TASKS
        }} for seed in MODEL_SEEDS
    }


def test_physical_gate_uses_advisory_queries_changed_paths_actor_and_bootstrap():
    good = decide(fake_monitor(), fake_acquisition())
    assert good["registered_decision"] == "PASS"
    assert good["strict_physical_gain_corrective_minus_expert"] == 60
    assert good["training_teacher_query_nonvacuity"]
    assert good["native_evaluation_action_distinction"]
    assert good["95pct_lower_bound_strictly_positive"]
    assert decide(fake_monitor(distinct=False), fake_acquisition())[
        "registered_decision"
    ] == "INSUFFICIENT"
    assert decide(fake_monitor(), fake_acquisition(n=0))[
        "registered_decision"
    ] == "INSUFFICIENT"
    assert decide(fake_monitor(corrective_goal_count=4), fake_acquisition())[
        "registered_decision"
    ] == "FAIL"


def test_newly_opened_monitor_malformed_pairings_rejected():
    data = fake_monitor()
    data["fetch:corrective_onpolicy:941"]["episodes"][0]["layout_seed"] = 1
    with pytest.raises(ValueError, match="learned MONITOR layout pairing"):
        decide(data, fake_acquisition())
