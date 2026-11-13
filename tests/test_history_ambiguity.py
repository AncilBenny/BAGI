"""Gate B1: verify memory features, ambiguity ceilings, and graph replay."""

import pytest

from bagi.experiments.history_ambiguity import (
    ambiguity,
    enumerate_histories,
    visible_input,
)
from bagi.worlds import WorkshopEnv


def sample(history, next_tile, events=()):
    return {
        "history": history,
        "observation": [[3]],
        "action": 1,
        "next_observation": [[next_tile]],
        "reward": 0.0,
        "events": list(events),
    }


def test_history_separates_same_current_observation_with_different_outcomes():
    cases = [
        sample([[[1], 0]], 4, ("key_picked_up",)),
        sample([[[2], 0]], 5, ()),
    ]
    reactive = ambiguity(cases, 0)
    memory = ambiguity(cases, 1)
    assert reactive["minimal_empirical_whole_outcome_errors"] == 1
    assert reactive["minimal_empirical_event_errors"] == 1
    assert memory["minimal_empirical_whole_outcome_errors"] == 0
    assert memory["minimal_empirical_event_errors"] == 0


def test_history_features_exclude_privileged_information():
    case = sample([[[1], 0]], 4)
    case["info"] = {"position": [5, 4], "has_key": True, "door_open": True}
    assert visible_input(case, 0) == visible_input(
        {**case, "info": {"position": [0, 0]}}, 0
    )
    assert visible_input(case, 1) == visible_input(
        {**case, "info": {"position": [0, 0]}}, 1
    )


def test_reachable_histories_are_deterministic_and_cover_all_actions():
    env = WorkshopEnv(size=9, max_steps=100, view_size=5)
    first, meta = enumerate_histories(env)
    second, other = enumerate_histories(env)
    assert first == second
    assert meta == other
    assert len(first) == 4 * meta["reachable_nonterminal_states"]
    assert meta["reachable_nonterminal_states"] == 78
    assert all("info" not in row for row in first)
    estimates = [ambiguity(first, k)["minimal_empirical_whole_outcome_errors"]
                 for k in (0, 1, 2, 4, 8, 16)]
    assert estimates == sorted(estimates, reverse=True)


def test_negative_window_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        visible_input(sample([], 3), -1)
