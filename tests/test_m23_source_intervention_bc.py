"""Stage13 controls: equal *executed* native acquisition budgets, no fake teacher actions."""
from types import SimpleNamespace

import numpy as np
import pytest

from bagi.experiments import m23_source_intervention_bc as study
from bagi.experiments.m23_source_public_vision_bc import WIDTH
from bagi.experiments.m23_study_contract import (
    BC_MONITOR_LAYOUTS,
    CORRECT_MONITOR_LAYOUTS,
    FINAL_LAYOUTS,
    INTERVENE_ACQUIRE_LAYOUTS,
    INTERVENE_MONITOR_LAYOUTS,
    INTERVENE_TRAIN_LAYOUTS,
    VISION_MONITOR_LAYOUTS,
    validate_split_contract,
)


def test_fresh_source_splits_and_original_target_final_never_overlap():
    validate_split_contract()
    current = {
        n for split in (
            INTERVENE_TRAIN_LAYOUTS, INTERVENE_ACQUIRE_LAYOUTS,
            INTERVENE_MONITOR_LAYOUTS,
        ) for seeds in split.values() for n in seeds
    }
    earlier = {
        n for split in (
            BC_MONITOR_LAYOUTS, CORRECT_MONITOR_LAYOUTS,
            VISION_MONITOR_LAYOUTS, FINAL_LAYOUTS,
        ) for seeds in split.values() for n in seeds
    }
    assert len(current) == 368
    assert not current & earlier
    assert study.ACQUIRE_BUDGET == 2304
    with pytest.raises(ValueError, match="fresh registered split"):
        study.monitor_episode("unlockpickup", 453001, "actor_only", None, None)


def test_intervention_actually_executes_public_teacher_rare_actions(monkeypatch):
    def observation():
        return {
            "image": np.zeros((7, 7, 3), dtype=np.uint8),
            "mission": "use the key to open the door and then get to the goal",
            "direction": 0,
        }

    class Teacher:
        key_observed_after_pickup = False

        def __init__(self):
            self.step = 0

        def choose(self, obs, rng):
            advice = [3, 0, 5, 0, 0][self.step]
            self.step += 1
            return advice

        def update(self, before, action, after):
            assert action in range(7)

    class FakeAudit:
        def __init__(self):
            self.actions = []
            self.env = SimpleNamespace(close=lambda: None)

        def reset(self, **kwargs):
            return observation(), {}

        def step(self, action):
            self.actions.append(action)
            return observation(), 0.0, False, len(self.actions) == 5, {}

        def summary(self):
            return {
                "native_actions": len(self.actions),
                "action_counts": {
                    str(i): self.actions.count(i) for i in range(7)
                },
                "full_physical_goal": False,
                "effective_door_open_events": 1,
                "effective_pickup_events": 1,
            }

    monkeypatch.setattr(study, "make_source_actor", lambda task: Teacher())
    monkeypatch.setattr(study, "native_acquire_audit", lambda task: FakeAudit())
    monkeypatch.setattr(study, "freeze_predict", lambda *args: 2)
    frozen = SimpleNamespace(training=False)
    mix = study.acquire_one(
        "doorkey", INTERVENE_ACQUIRE_LAYOUTS["doorkey"][0], 0, 941,
        "intervention_mixed", frozen, 6,
    )
    expert = study.acquire_one(
        "doorkey", INTERVENE_ACQUIRE_LAYOUTS["doorkey"][0], 0, 941,
        "extra_expert", frozen, 6,
    )
    assert mix["actual_native_actions"] == expert["actual_native_actions"] == 5
    assert mix["actual_teacher_selected_native_steps"] == 3
    assert mix["actual_teacher_selected_rare_3_or_5"] == 2
    assert mix["queried_but_not_executed_teacher_advice"] == 2
    assert mix["actual_native_action_histogram"] == [1, 0, 2, 1, 0, 1, 0]
    assert expert["actual_native_action_histogram"] == [3, 0, 0, 1, 0, 1, 0]
    assert mix["public_teacher_label_histogram"] == expert["public_teacher_label_histogram"]
    assert mix["native_executed_action_sha256"] != expert["native_executed_action_sha256"]
    assert mix["native_episode_finished"] and not mix["budget_cut_partial"]
    assert mix["external_effective_door_open_events"] == 1


def test_exact_budget_uses_real_steps_and_marks_early_stop(monkeypatch):
    monkeypatch.setattr(study, "ACQUIRE_BUDGET", 9)

    def fake_one(task, layout, visit, seed, arm, base, remaining):
        n = min(remaining, 4)
        return {
            "x": np.zeros((n, WIDTH), dtype=np.float32),
            "y": np.zeros(n, dtype=np.int64),
            "layout_seed": layout,
            "actual_native_actions": n,
            "actual_teacher_selected_native_steps": n if arm == "intervention_mixed" else 0,
            "actual_teacher_selected_rare_3_or_5": 0,
            "queried_but_not_executed_teacher_advice": 0,
            "external_effective_door_open_events": 0,
            "external_effective_pickup_events": 0,
            "actual_native_action_histogram": [n, 0, 0, 0, 0, 0, 0],
        }

    monkeypatch.setattr(study, "acquire_one", fake_one)
    x, y, record = study.acquire_budgeted(
        "doorkey", 941, "intervention_mixed", SimpleNamespace(training=False)
    )
    assert x.shape == (9, WIDTH)
    assert len(y) == 9
    assert record["actual_new_native_actions"] == 9
    assert record["actual_new_public_teacher_queries"] == 9
    assert record["acquisition_native_episode_count"] == 3
    assert record["actual_teacher_executed_intervention_steps"] == 9
    assert record["actually_executed_native_action_histogram"] == [
        9, 0, 0, 0, 0, 0, 0
    ]


def fake_monitor(actor_goals=4, expert_goals=6, mix_goals=16, distinct=True):
    data = {}
    for task in study.TASKS:
        layouts = INTERVENE_MONITOR_LAYOUTS[task]
        data[f"{task}:actor_only"] = {"episodes": [
            {"layout_seed": layout, "external_full_physical_goal": i < actor_goals,
             "actual_executed_action_sha256": f"actor:{layout}"}
            for i, layout in enumerate(layouts)
        ]}
        for arm in study.ARMS:
            for seed in study.MODEL_SEEDS:
                goals = (
                    mix_goals if arm == "intervention_mixed"
                    else expert_goals if arm == "extra_expert" else 7
                )
                data[f"{task}:{arm}:{seed}"] = {"episodes": [
                    {"layout_seed": layout, "external_full_physical_goal": i < goals,
                     "actual_executed_action_sha256": (
                         f"{arm}:{seed}:{layout}" if distinct else f"actor:{layout}"
                     )} for i, layout in enumerate(layouts)
                ]}
    return data


def fake_acquisition(teacher_steps=600, opened=8):
    return {
        str(seed): {
            task: {
                arm: {
                    "actual_new_native_actions": 2304,
                    "actual_new_public_teacher_queries": 2304,
                    "actual_teacher_executed_intervention_steps": (
                        teacher_steps if arm == "intervention_mixed" else 0
                    ),
                    "external_effective_door_open_events": (
                        opened if arm == "intervention_mixed" and task == "doorkey"
                        else 0
                    ),
                } for arm in study.ACQUIRE_ARMS
            } for task in study.TASKS
        } for seed in study.MODEL_SEEDS
    }


def test_registered_decision_needs_physical_gain_and_real_teacher_coverage():
    good = study.decide(fake_monitor(), fake_acquisition())
    assert good["registered_decision"] == "PASS"
    assert good["strict_physical_gain_intervention_minus_expert"] == 60
    assert good["necessary_real_teacher_intervention_coverage"]
    assert good["necessary_learned_native_action_distinction"]
    assert good["bootstrap_lower_95pct_strictly_positive"]
    assert study.decide(fake_monitor(), fake_acquisition(opened=0))[
        "registered_decision"
    ] == "INSUFFICIENT"
    assert study.decide(fake_monitor(distinct=False), fake_acquisition())[
        "registered_decision"
    ] == "INSUFFICIENT"
    assert study.decide(fake_monitor(mix_goals=3), fake_acquisition())[
        "registered_decision"
    ] == "FAIL"


def test_registry_fails_closed_on_missing_or_unmatched_actual_native_budget():
    x = fake_acquisition()
    x["941"]["fetch"]["intervention_mixed"]["actual_new_native_actions"] = 2
    with pytest.raises(ValueError, match="real acquisition budget missing"):
        study.decide(fake_monitor(), x)
