"""REUSE-20 helper contracts do not depend on optional XLand installation."""
from types import SimpleNamespace

import numpy as np
import pytest

from bagi.experiments.xland_rule_goal_observation import (
    encoding_digest,
    same_physical_initial_state,
    task_observation_record,
)


def dummy_state():
    return SimpleNamespace(
        grid=np.zeros((5, 5, 2), dtype=np.uint8),
        agent=SimpleNamespace(
            position=np.array([2, 2]),
            direction=np.array(0),
            pocket=np.array([0, 0]),
        ),
        goal_encoding=np.array([1, 2]),
        rule_encoding=np.array([[0, 4, 5]]),
    )


def test_same_grid_and_agent_but_different_goal_same_world():
    state = dummy_state()
    other = dummy_state()
    other.goal_encoding[0] = 9
    assert same_physical_initial_state(state, other)
    other.agent.direction = np.array(3)
    assert not same_physical_initial_state(state, other)


def test_encoding_hash_changes_with_rule_or_goal():
    assert encoding_digest(np.array([1, 2])) != encoding_digest(np.array([1, 3]))


def test_wrapped_task_tokens_are_separate_from_local_features():
    state = dummy_state()
    img = np.zeros((5, 5, 2), dtype=np.uint8)
    direction = np.array([1, 0, 0, 0], dtype=np.float32)
    plain = {"img": img, "direction": direction}
    instructed = {
        **plain, "goal_encoding": state.goal_encoding,
        "rule_encoding": state.rule_encoding,
    }
    ruleset = SimpleNamespace(
        goal=state.goal_encoding, rules=state.rule_encoding,
    )
    record = task_observation_record(plain, instructed, state, ruleset)
    assert record["prior_local_image_heading_feature_count"] == 54
    assert record["wrapped_goal_is_upstream_state_and_params_goal"]
    assert record["wrapped_rules_are_upstream_state_and_params_rules"]
    assert record["visible_local_features_unchanged_by_instruction_wrapper"]


def test_missing_goal_instruction_fails_closed():
    state = dummy_state()
    with pytest.raises(ValueError, match="instruction"):
        task_observation_record(
            {"img": np.zeros((5, 5, 2)), "direction": np.ones(4)},
            {"img": np.zeros((5, 5, 2)), "direction": np.ones(4)},
            state, SimpleNamespace(goal=state.goal_encoding,
                                   rules=state.rule_encoding)
        )
