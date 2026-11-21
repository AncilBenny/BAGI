"""Synthetic public-feedback contracts; NOT native skill-learning results."""
from __future__ import annotations

import numpy as np
import pytest

from bagi.agents.public_stagnation import PublicStagnationEscape

RANKED = ((4, 0.7), (1, 0.2), (0, 0.1))


def xland_signature(obs):
    x = np.asarray(obs)
    assert x.shape == (115,)
    return tuple(x[:87].tolist())  # Strip 4x7 executed-action history only.


def public_obs(history_bit=0):
    x = np.zeros(115, dtype=np.float32)
    x[-28 + history_bit] = 1.0
    return x


def test_three_unchanged_executed_actions_trigger_one_step_escape():
    c = PublicStagnationEscape(xland_signature, enabled=True)
    for i in range(3):
        assert c.propose(public_obs(i), RANKED) == 4
        event = c.observe(4, public_obs(i + 1), 0.0)
        assert event.unchanged_public_view
    assert c.suspected_stalls == 1
    assert c.propose(public_obs(4), RANKED) == 1
    event = c.observe(1, public_obs(5), 0.0)
    assert event.escaped_last_choice and c.escape_actions == 1
    assert c.propose(public_obs(6), RANKED) == 4


def test_same_frozen_distribution_control_never_overrides():
    c = PublicStagnationEscape(xland_signature, enabled=False)
    for i in range(6):
        assert c.propose(public_obs(i % 7), RANKED) == 4
        c.observe(4, public_obs((i + 1) % 7), 0)
    assert c.escape_actions == 0 and c.suspected_stalls == 4


def test_public_progress_and_reward_reset_streak():
    c = PublicStagnationEscape(xland_signature, enabled=True)
    x = public_obs()
    for _ in range(2):
        c.propose(x, RANKED)
        c.observe(4, x, 0)
    changed = x.copy()
    changed[2] = 1
    c.propose(x, RANKED)
    c.observe(4, changed, 0)
    assert c.streak == 0
    c.propose(changed, RANKED)
    c.observe(4, changed, 1)
    assert c.streak == 0


def test_actual_executed_action_is_required():
    c = PublicStagnationEscape(xland_signature, enabled=True)
    c.propose(public_obs(), RANKED)
    with pytest.raises(ValueError, match="actual selected"):
        c.observe(3, public_obs(), 0)


def test_missing_probability_alternative_fails_closed():
    c = PublicStagnationEscape(xland_signature, enabled=True, patience=1)
    c.propose(public_obs(), ((4, 1.0), (1, 0.0)))
    c.observe(4, public_obs(1), 0)
    with pytest.raises(ValueError, match="no nonzero alternative"):
        c.propose(public_obs(2), ((4, 1.0), (1, 0.0)))


@pytest.mark.parametrize("ranked", [
    ((4, 0.7), (4, 0.3)),
    ((4, 0.2), (1, 0.8)),
    ((4, float("nan")), (1, 0.2)),
    ((4, 0.7), (1, 0.1)),
])
def test_invalid_frozen_action_distributions_rejected(ranked):
    c = PublicStagnationEscape(xland_signature, enabled=True)
    with pytest.raises(ValueError):
        c.propose(public_obs(), ranked)


def test_workshop_public_tiles_and_symbolic_actions_same_core():
    c = PublicStagnationEscape(lambda tiles: tuple(np.asarray(tiles).ravel()),
                               enabled=True, patience=2)
    before = np.zeros((5, 5), dtype=np.int8)
    ranked = (("try", 0.8), ("inspect", 0.2))
    for _ in range(2):
        assert c.propose(before, ranked) == "try"
        c.observe("try", before.copy(), 0)
    assert c.propose(before, ranked) == "inspect"
    c.observe("inspect", before, 0)
    assert c.escape_actions == 1


def test_public_unchanged_does_not_claim_native_noop():
    # Hidden key/pocket may change while the legal public view stays fixed.
    c = PublicStagnationEscape(xland_signature, enabled=True, patience=1)
    c.propose(public_obs(), RANKED)
    evidence = c.observe(4, public_obs(1), 0)
    assert evidence.suspected_stagnation
    assert not hasattr(evidence, "native_noop")


def test_reset_clears_episode_streak_without_learning_or_privileged_state():
    c = PublicStagnationEscape(xland_signature, enabled=True, patience=1)
    c.propose(public_obs(), RANKED)
    c.observe(4, public_obs(1), 0)
    c.reset()
    assert c.propose(public_obs(), RANKED) == 4
    assert c.escape_actions == c.suspected_stalls == 0
