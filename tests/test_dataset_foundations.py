"""Gate A audit tests: duplication, split overlap, and observable aliasing."""

import pytest

from bagi.experiments.dataset_foundations import aliasing, audit, splits


def transition(episode, obs, nxt, *, key=False, event=()):
    return {
        "episode": episode, "observation": [[obs]], "action": 1,
        "next_observation": [[nxt]], "reward": 0.0,
        "info": {"has_key": key, "door_open": False, "events": list(event)},
    }


def test_guided_duplicates_are_not_independent_experience():
    rows = [
        transition(0, 0, 1, event=("switch_activated",)),
        transition(0, 1, 2),
        transition(1, 0, 1, event=("switch_activated",)),
        transition(1, 1, 2),
        transition(2, 2, 3),
        transition(3, 3, 4),
    ]
    result = audit(rows, 42, 0.5, guided=2)
    assert result["duplicate_guided_episodes"] == 1
    assert result["unique_guided_episode_signatures"] == 1
    assert result["rows_in_duplicated_supervised_transitions"] == 4
    assert result["event_positive_counts"]["switch_activated"] == 2
    assert result["splits"]["unique_episode_signatures"] == 3
    grouped = result["splits"]["trajectory_disjoint_counterfactual"]
    assert grouped["training_trajectory_groups"] + grouped["validation_trajectory_groups"] == 3


def test_hidden_flags_can_correlate_with_observable_aliasing():
    rows = [
        transition(0, 0, 0),
        transition(1, 0, 1, key=True),
        transition(2, 0, 0),
        transition(3, 2, 2),
        transition(4, 2, 2),
    ]
    result = aliasing(rows)
    assert result["repeated_observation_action_pairs"] == 2
    assert result["ambiguous_outcome_pairs"] == 1
    assert result["mixed_hidden_flag_pairs"] == 1
    assert result["ambiguous_with_mixed_hidden_flags"] == 1


def test_hidden_flags_alone_do_not_prove_ambiguous_outcome():
    result = aliasing([transition(0, 0, 0), transition(1, 0, 0, key=True)])
    assert result["mixed_hidden_flag_pairs"] == 1
    assert result["ambiguous_outcome_pairs"] == 0


def test_episode_split_can_still_leak_identical_trajectories():
    rows = [transition(i, 0, 1) for i in range(4)]
    result = splits(rows, 17, 0.25)
    assert result["actual_transition_random_split"][
        "validation_exact_duplicate_fraction"] == 1.0
    assert result["episode_disjoint_counterfactual"][
        "validation_exact_duplicate_fraction"] == 1.0
    assert result["trajectory_disjoint_counterfactual"] is None


def test_validation_guards():
    rows = [transition(0, 0, 1), transition(1, 0, 1)]
    with pytest.raises(ValueError, match="between 0 and 1"):
        splits(rows, 42, 1.0)
    with pytest.raises(ValueError, match="two episodes"):
        splits([transition(0, 0, 1), transition(0, 1, 2)], 42, 0.2)
