"""REUSE-33 pure contract tests; mock episodes do not demonstrate learning."""
from dataclasses import dataclass

import numpy as np
import pytest

from bagi.experiments.xland_complete_demo_bc import (
    CAP,
    EVAL_LAYOUTS,
    MIN_COMPLETE_PHYSICAL,
    TRAIN_LAYOUTS,
    actual_history,
    collect_one,
    sampled_indices,
    validate_demonstrations,
    verify_registration,
)


@dataclass(frozen=True)
class TimeStep:
    n: int
    reward: float = 0.0

    def last(self):
        return self.n >= 2


def reset(layout):
    assert layout in TRAIN_LAYOUTS
    return TimeStep(0)


def step(ts, action):
    assert action == (3 if ts.n == 0 else 4)
    return TimeStep(ts.n + 1, float(ts.n == 1))


def teacher(ts):
    return ([3] if ts.n == 0 else [4]), (
        "seek_square" if ts.n == 0 else "carrying_square"
    )


def observe(ts, history):
    array = np.zeros(115, dtype=np.float32)
    array[0] = ts.n
    array[-28:] = np.eye(7, dtype=np.float32)[
        np.asarray(history, dtype=np.int64)
    ].reshape(-1)
    return array


def test_exact_predeclared_budget_and_source_disjoint():
    verify_registration()
    assert len(TRAIN_LAYOUTS) == 24
    assert len(EVAL_LAYOUTS) == 10
    assert not set(TRAIN_LAYOUTS) & set(EVAL_LAYOUTS)
    assert MIN_COMPLETE_PHYSICAL == 12 and CAP == 40
    assert actual_history((6, 6, 6, 6), 3) == (6, 6, 6, 3)
    with pytest.raises(ValueError, match="native"):
        actual_history((6, 6, 6, 6), 6)


def test_complete_two_phase_teacher_is_true_executed_history():
    rows, audit = collect_one(
        reset, step, teacher, observe,
        lambda before, after, action: (action == 3, action == 4),
        lambda ts: ts.n == 2, lambda ts: True, TRAIN_LAYOUTS[0],
    )
    assert [r["expert_label"] for r in rows] == [3, 4]
    assert rows[0]["history"] == (6, 6, 6, 6)
    assert rows[1]["history"] == (6, 6, 6, 3)
    assert rows[-1]["done"]
    assert audit["physical"]
    assert audit["native_steps"] == audit["queries"] == 2
    assert audit["blue_pickups"] == audit["blue_drops"] == 1


def test_reward_without_blue_movement_is_discarded_and_counted():
    rows, audit = collect_one(
        reset, step, teacher, observe,
        lambda before, after, action: (False, False),
        lambda ts: True, lambda ts: True, TRAIN_LAYOUTS[0],
    )
    assert not rows and not audit["physical"]
    assert audit["native_reward"] and audit["native_steps"] == 2
    assert audit["labels_used"] == 0


def test_unsupported_teacher_query_is_counted_but_never_used():
    def bad_teacher(ts):
        if ts.n == 1:
            raise ValueError("unavailable correction")
        return teacher(ts)

    rows, audit = collect_one(
        reset, step, bad_teacher, observe,
        lambda before, after, action: (action == 3, action == 4),
        lambda ts: True, lambda ts: True, TRAIN_LAYOUTS[0],
    )
    assert not rows
    assert audit["queries"] == 2 and audit["native_steps"] == 1
    assert "unavailable correction" in audit["failure"]


def test_fail_closed_when_only_navigation_source_labels():
    rows = [{"public_obs": np.zeros(115, dtype=np.float32),
             "expert_label": 0, "layout": TRAIN_LAYOUTS[0],
             "stage": "seek_square"}]
    cases = [{"layout": seed, "physical": seed in TRAIN_LAYOUTS[:12],
              "native_steps": 2, "queries": 2}
             for seed in TRAIN_LAYOUTS]
    audit = {"cases": cases, "native_teacher_actions": 48,
             "attempted_teacher_queries": 48,
             "eligible_physical_episode_count": 12, "label_rows": 1}
    with pytest.raises(ValueError, match="pickup/carry/drop"):
        validate_demonstrations(rows, audit)


def test_existing_pytorch_sampler_deterministic_equal_exposure():
    actions = np.array([0, 0, 0, 1, 3, 4], dtype=np.int64)
    a = sampled_indices(actions, 30, 17)
    b = sampled_indices(actions, 30, 17)
    assert np.array_equal(a, b)
    assert a.shape == (30,)
    assert set(a) <= set(range(len(actions)))
