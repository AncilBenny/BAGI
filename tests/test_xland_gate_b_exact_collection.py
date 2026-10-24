"""Deterministic local preregistration tests; no XLand simulation claimed."""
from dataclasses import dataclass

import numpy as np
import pytest

from bagi.experiments.xland_gate_b_exact_collection import (
    QUERY_BUDGET,
    collect_arm,
    save_numeric,
    validate_pair,
)


@dataclass(frozen=True)
class Step:
    seed: int
    n: int
    reward: float = 0.0

    def last(self):
        return self.n >= 7


def fake_reset(seed):
    return Step(seed, 0)


def fake_step(ts, action):
    assert action in range(6)
    return Step(ts.seed, ts.n + 1, float(ts.n == 6))


def fake_features(ts, history):
    arr = np.zeros(115, dtype=np.float32)
    arr[0] = ts.n
    arr[1] = ts.seed % 13
    arr[2:6] = history
    return arr


def fake_teacher(ts):
    return (2 if ts.n % 2 else 1), "seek_square"


def fake_learner(x):
    assert x.shape == (115,)
    return 0


def both():
    params = {
        "reset": fake_reset, "step": fake_step,
        "screen_nonadjacent": lambda ts: True,
        "feature_fn": fake_features, "teacher": fake_teacher,
        "learner": fake_learner, "seeds": (9701, 9702, 9708, 9710),
    }
    return (collect_arm(arm="reset_expert", **params),
            collect_arm(arm="learner_correction", **params))


def test_exact_budget_true_history_and_distinct_execution():
    a, b = both()
    assert a.complete and b.complete
    assert a.teacher_queries == b.teacher_queries == QUERY_BUDGET
    assert a.native_steps == b.native_steps == QUERY_BUDGET
    assert validate_pair(a, b)["matched"]
    assert all(r["executed_action"] == r["expert_label"] for r in a.rows)
    assert all(r["executed_action"] == 0 for r in b.rows)
    assert b.rows[1]["prior_executed_actions"] == (6, 6, 6, 0)
    assert a.rows[1]["prior_executed_actions"] == (6, 6, 6, 1)
    assert a.rows[7]["prior_executed_actions"] == (6, 6, 6, 6)
    assert a.rows[7]["state_origin"] == "reset"
    assert b.rows[1]["state_origin"] == "learner_continuation"


def test_no_cherry_pick_when_adjacent_or_teacher_unsupported():
    a = collect_arm(arm="learner_correction", reset=fake_reset, step=fake_step,
                    screen_nonadjacent=lambda ts: ts.seed != 9702,
                    feature_fn=fake_features, teacher=fake_teacher,
                    learner=fake_learner, seeds=(9701, 9702, 9708))
    assert not a.complete
    assert a.screened_seeds == (9701, 9702)
    assert a.expert_failures[0]["reason"] == "ineligible_adjacent_reset"

    def unsupported(ts):
        if ts.n == 2:
            raise ValueError("teacher recovery unavailable")
        return fake_teacher(ts)

    b = collect_arm(arm="learner_correction", reset=fake_reset, step=fake_step,
                    screen_nonadjacent=lambda ts: True,
                    feature_fn=fake_features, teacher=unsupported,
                    learner=fake_learner, seeds=(9701, 9702))
    assert not b.complete
    assert b.expert_failures[0]["reason"] == "unsupported_teacher"
    with pytest.raises(ValueError, match="Incomplete collection"):
        validate_pair(both()[0], b)


def test_rejects_bad_contract_and_out_of_scope_seed():
    common = {
        "reset": fake_reset, "step": fake_step,
        "screen_nonadjacent": lambda ts: True,
        "feature_fn": fake_features, "teacher": fake_teacher,
        "learner": fake_learner,
    }
    with pytest.raises(ValueError, match="Evaluation-layout contamination"):
        collect_arm(arm="learner_correction", seeds=(9810,), **common)
    with pytest.raises(ValueError, match="contract changed"):
        collect_arm(arm="learner_correction", budget=19, **common)
    with pytest.raises(ValueError, match="Unregistered"):
        collect_arm(arm="learner_correction", seeds=(9701, 9999), **common)
    with pytest.raises(ValueError, match="Unregistered"):
        collect_arm(arm="learner_correction", seeds=(9702, 9701), **common)
    invalid = collect_arm(arm="reset_expert", teacher=lambda ts: (6, "bad"),
        **{k: v for k, v in common.items() if k != "teacher"})
    assert invalid.teacher_queries == 1
    assert invalid.native_steps == 0
    assert invalid.expert_failures[0]["reason"] == "unsupported_teacher"
    with pytest.raises(ValueError, match="115-feature"):
        collect_arm(arm="learner_correction", feature_fn=lambda ts, h: np.zeros(54), **{k: v for k, v in common.items() if k != "feature_fn"})


def test_numeric_artifact_without_pickle(tmp_path):
    _, b = both()
    save_numeric(str(tmp_path / "corr.npz"), b)
    with np.load(tmp_path / "corr.npz", allow_pickle=False) as arrays:
        assert arrays["obs"].shape == (20, 115)
        assert arrays["prior_executed_actions"].shape == (20, 4)
        assert np.all(arrays["executed_action"] == 0)
        assert np.all(arrays["expert_label"] != arrays["executed_action"])