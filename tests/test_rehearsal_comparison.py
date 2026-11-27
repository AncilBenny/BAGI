import copy

import pytest

from bagi.agents.rehearsal_comparison import KEYS, check_fork_counts, compare_traces


def point():
    return {key: {"success": 1.0, "episodes": [{"seed": 100, "domain": key[0],
                                                "action_sha256": "old"}]}
            for key in KEYS}


def test_trace_delta_detects_equal_success_but_different_trajectory():
    old = point()
    new = copy.deepcopy(old)
    new["A_live"]["episodes"][0]["action_sha256"] = "new"
    result = compare_traces(new, old)
    assert result["A_greedy"]["exact"]
    assert result["A_live"] == {"exact": False, "different_episodes": 1,
                                "first_different_seed": 100, "success_delta": 0.0}


def test_trace_delta_rejects_mismatched_seeds():
    old, new = point(), point()
    new["B_greedy"]["episodes"][0]["seed"] = 101
    with pytest.raises(ValueError, match="seeds"):
        compare_traces(new, old)


def test_fork_budget_counts_prefix_once():
    assert check_fork_counts(8192, 32768, 32768) == 73728
    with pytest.raises(ValueError, match="budgets"):
        check_fork_counts(8192, 40960, 32768)
