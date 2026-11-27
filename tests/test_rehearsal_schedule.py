import hashlib

import pytest

from bagi.agents.rehearsal_schedule import next_reset_seed


def row(seed, n):
    actions = [0] * n
    return {"seed": seed, "domain": "B", "learn": True, "actions": n,
            "executed_actions": actions, "action_sha256": hashlib.sha256(bytes(actions)).hexdigest()}


def test_pending_seed_continues_executed_reset_sequence():
    rows = [row(101, 100), row(102, 100), row(103, 56)]
    pending = next_reset_seed(rows, 101, 256, "B", 2)
    assert pending == 104
    assert next_reset_seed([row(pending, 100)], pending, 100, "B", 2) == 105


@pytest.mark.parametrize("key,value", [
    ("seed", 102), ("domain", "A"), ("learn", False), ("actions", 1),
    ("executed_actions", [2] * 10), ("action_sha256", "invalid"),
])
def test_invalid_trace_rejected(key, value):
    record = row(101, 10)
    record[key] = value
    with pytest.raises(ValueError, match="Invalid native rehearsal trace"):
        next_reset_seed([record], 101, 10, "B", 2)


def test_wrong_budget_rejected():
    with pytest.raises(ValueError, match="budget"):
        next_reset_seed([row(101, 10)], 101, 11, "B", 2)
