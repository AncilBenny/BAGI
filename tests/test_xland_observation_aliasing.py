"""REUSE-11: exact observable-data ambiguity bounds, no XLand learner run."""
import numpy as np
import pytest

from bagi.experiments.xland_observation_aliasing import observation_action_audit


def test_empirical_lookup_ceiling_and_cross_task_conflict():
    obs = np.array([
        [0, 0], [0, 0], [0, 0],
        [1, 0], [1, 0], [2, 0], [3, 0],
    ], dtype=np.float32)
    actions = [0, 1, 1, 3, 4, 4, 3]
    tasks = ["hold", "near", "near", "hold", "hold", "putdown", "putdown"]
    result = observation_action_audit(obs, actions, tasks)
    assert result["distinct_visible_observations"] == 4
    assert result["distinct_visible_observation_action_pairs"] == 6
    assert result["repeated_visible_observation_groups"] == 2
    assert result["ambiguous_visible_observation_groups"] == 2
    assert result["rows_in_ambiguous_groups"] == 5
    assert result["cross_task_conflict_groups"] == 1
    assert result["within_task_conflict_groups"] == 1
    assert result["conflict_groups_fully_resolved_by_task_identity"] == 1
    bound = result["empirical_deterministic_lookup"]
    assert bound["max_correct_without_task_id"] == 5
    assert bound["accuracy_without_task_id"] == pytest.approx(5 / 7)
    assert bound["max_correct_given_oracle_task_id"] == 6
    assert bound["irreducible_errors_given_task_id"] == 1
    assert result["action_support"]["4"]["distinct_visible_observations"] == 2
    assert result["action_support"]["4"]["rows_on_conflicting_observations"] == 1


def test_repeated_observation_with_identical_action_is_not_aliasing():
    obs = np.array([[0, 2], [0, 2]], dtype=np.float32)
    result = observation_action_audit(obs, [3, 3], ["hold", "near"])
    assert result["ambiguous_visible_observation_groups"] == 0
    assert result["empirical_deterministic_lookup"]["accuracy_without_task_id"] == 1.0
    assert result["distinct_visible_observation_action_pairs"] == 1


@pytest.mark.parametrize("obs,actions,tasks", [
    (np.ones((2, 2)), [1], ["hold", "near"]),
    (np.ones((1, 2)), [1], ["hold_then_near"]),
    (np.ones((1, 2)), [6], ["hold"]),
    (np.array([[np.nan, 1]]), [0], ["hold"]),
])
def test_reject_inconsistent_or_heldout_audit_inputs(obs, actions, tasks):
    with pytest.raises(ValueError):
        observation_action_audit(obs, actions, tasks)
