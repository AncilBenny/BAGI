"""No future expert-action leakage into historical learner features."""
import numpy as np
import pytest

from bagi.experiments.xland_action_provenance import (
    PAD_ACTION,
    add_preceding_histories,
    audit,
    numeric_audit_arrays,
    signature_audit,
)


def dummy_row(episode, step, action, label, *, round_num=1, done=False):
    return {
        "round": round_num, "episode": episode,
        "episode_step": step, "task": "hold",
        "obs": np.zeros(54, dtype=np.float32),
        "expert_label": label, "executed_action": action,
        "done": done, "rewarded": False,
    }


def test_prior_actions_are_executed_actions_and_reset_on_episode():
    rows = add_preceding_histories([
        dummy_row(0, 0, 1, 3),
        dummy_row(0, 1, 2, 4, done=True),
        dummy_row(1, 0, 0, 2),
    ])
    assert rows[0]["prior_executed"] == (6, 6, 6, 6)
    assert rows[1]["prior_executed"] == (6, 6, 6, 1)
    assert rows[1]["prior_expert"] == (6, 6, 6, 3)
    assert rows[2]["prior_executed"] == (6, 6, 6, 6)
    assert rows[2]["prior_expert"] == (6, 6, 6, 6)
    assert PAD_ACTION == 6
    assert rows[1]["prior_executed"] != rows[1]["prior_expert"]


def test_history_conditional_conflicts_are_measured_without_leak():
    rows = add_preceding_histories([
        dummy_row(0, 0, 0, 3, done=True),
        dummy_row(1, 0, 0, 4, done=True),
    ])
    common = signature_audit(rows, stream="executed", window=0)
    historic = signature_audit(rows, stream="executed", window=4)
    assert common["contradictory_groups"] == 1
    assert common["minority_label_errors"] == 1
    assert historic["contradictory_groups"] == 1
    assert historic["minority_label_errors"] == 1


def test_audit_requires_beta_one_action_identity():
    records = [
        dummy_row(0, 0, 3, 4, round_num=0, done=True),
        dummy_row(1, 0, 0, 3, round_num=1, done=True),
    ]
    with pytest.raises(AssertionError, match="Beta=1"):
        audit(records)


def test_numeric_saved_arrays_never_need_pickle():
    data = numeric_audit_arrays(add_preceding_histories([
        dummy_row(0, 0, 1, 3, done=True),
    ]))
    assert set(data) == {
        "obs", "expert_label", "executed_action",
        "prior_executed_actions", "round", "episode",
        "step_in_episode", "done",
    }
    assert all(array.dtype != object for array in data.values())
    assert data["prior_executed_actions"].shape == (1, 4)

def test_bytes_are_canonical_after_upstream_dtype_conversion():
    """Upstream dataset storage may decode float32 learner images as float64."""
    visible = np.array([0.0, 4.0, 255.0], dtype=np.float32)
    upstream = visible.astype(np.float64)
    assert visible.tobytes() != upstream.tobytes()
    assert visible.tobytes() == np.ascontiguousarray(
        upstream, dtype=np.float32
    ).tobytes()
