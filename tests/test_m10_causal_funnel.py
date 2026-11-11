"""Dependency-free tests for M10's evaluator-only transition counter."""
import pytest

from bagi.experiments.m10_causal_funnel import EpisodeFunnel, PhysicalSnapshot


def s(key=False, visible=False, front=False, opened=False, goal=False):
    return PhysicalSnapshot(key, visible, front, opened, goal)


def test_real_sequence_and_native_action_accounting():
    meter = EpisodeFunnel()
    meter.observe(s(visible=True), 3, s(key=True), reward=0,
                  terminated=False, truncated=False)
    meter.observe(s(key=True, front=True), 5, s(key=True, opened=True),
                  reward=0, terminated=False, truncated=False)
    meter.observe(s(key=True, opened=True), 2,
                  s(key=True, opened=True, goal=True), reward=.9,
                  terminated=True, truncated=False)
    result = meter.summary()
    assert result["full_physical_goal"] is True
    assert result["valid_toggle_opportunities"] == 1
    assert result["valid_toggle_attempts"] == 1
    assert result["effective_pickup_events"] == 1
    assert result["effective_door_open_events"] == 1
    assert sum(result["action_counts"].values()) == 3
    assert [result[k] for k in (
        "first_key_pickup_at", "first_door_open_at", "first_goal_at"
    )] == [1, 2, 3]


def test_invalid_toggle_does_not_count_as_opportunity():
    meter = EpisodeFunnel()
    meter.observe(s(), 5, s(), reward=0, terminated=False, truncated=False)
    assert meter.summary()["valid_toggle_attempts"] == 0


def test_valid_but_ineffective_toggle():
    meter = EpisodeFunnel()
    meter.observe(s(key=True, front=True), 5, s(key=True, front=True),
                  reward=0, terminated=False, truncated=False)
    assert meter.summary()["ineffective_valid_toggles"] == 1
    assert meter.summary()["effective_door_open_events"] == 0


def test_reward_without_chain_is_not_physical_success():
    meter = EpisodeFunnel()
    meter.observe(s(), 2, s(goal=True), reward=.5,
                  terminated=True, truncated=False)
    assert meter.summary()["full_physical_goal"] is False


@pytest.mark.parametrize("action,reward", [
    (7, 0), (-1, 0), (True, 0), (2, float("nan")), (2, float("inf")),
])
def test_reject_invalid_transition(action, reward):
    with pytest.raises(ValueError):
        EpisodeFunnel().observe(s(), action, s(), reward=reward,
                                terminated=False, truncated=False)


def test_empty_episode_has_no_action_concentration():
    assert EpisodeFunnel().summary()["max_action_fraction"] is None
