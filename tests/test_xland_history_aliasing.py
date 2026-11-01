"""Previous-action history must be causal, episode-reset, and deterministic."""
import numpy as np
import pytest

from bagi.experiments.xland_history_aliasing import history_lookup_audit


def test_previous_action_history_reduces_some_conflicts_but_not_all():
    # Two trajectories: a0,a3 and a0,a4, all at indistinguishable observations.
    # Current action must not enter its own history.
    obs = np.zeros((4, 3), dtype=np.float32)
    scores = history_lookup_audit(obs, [0, 3, 0, 4],
                                  [False, True, False, True])
    assert scores["0"]["max_correct_exact_lookup"] == 2
    assert scores["0"]["ambiguous_signature_groups"] == 1
    assert scores["1"]["max_correct_exact_lookup"] == 3
    assert scores["1"]["irreducible_errors_exact_lookup"] == 1
    assert scores["1"]["distinct_observation_history_signatures"] == 2
    assert scores["4"]["max_correct_exact_lookup"] == 3


def test_reset_boundaries_do_not_expose_previous_episode_actions():
    obs = np.zeros((3, 2), dtype=np.float32)
    # Episodes [3] and [4,0]: second action's context is [4], not [3,4].
    result = history_lookup_audit(obs, [3, 4, 0], [True, False, True])
    assert result["1"]["distinct_observation_history_signatures"] == 2
    assert result["4"]["distinct_observation_history_signatures"] == 2


@pytest.mark.parametrize("obs,actions,dones", [
    (np.zeros((2, 2)), [0], [False, True]),
    (np.zeros((1, 2)), [7], [False]),
    (np.array([[float("nan"), 0]]), [0], [True]),
])
def test_reject_invalid_inputs(obs, actions, dones):
    with pytest.raises(ValueError):
        history_lookup_audit(obs, actions, dones)


def test_reject_unregistered_window_changes():
    with pytest.raises(ValueError, match="preregistered"):
        history_lookup_audit(np.zeros((1, 2)), [0], [True], lookbacks=(0, 2))
