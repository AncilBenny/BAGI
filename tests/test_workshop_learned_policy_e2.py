"""Synthetic-only G2-E2-P0 contracts; never consumes native Workshop steps."""
from __future__ import annotations

import numpy as np
import pytest

from bagi.experiments import workshop_learned_policy_e2 as pilot


class SyntheticWorkshop:
    def __init__(self, terminal_step=3):
        self.calls = 0
        self.terminal_step = terminal_step
        self.private_secret = "NEVER AN ACTOR INPUT"

    def step(self, action):
        self.calls += 1
        public = np.full((5, 5), self.calls, dtype=np.int8)
        done = self.calls == self.terminal_step
        return public, float(done), done, False, {
            "events": ["switch_activated"] if self.calls == 1 else ["goal_reached"],
            "private_secret": self.private_secret,
        }


def numeric():
    return {key: [] for key in (
        "observation", "next_observation", "action", "reward", "terminated",
        "truncated", "episode_index", "seed_index", "layout_index", "actor_index"
    )}


def test_frozen_registration_and_exact_seed_budgets(monkeypatch):
    pilot.verify_registration()
    assert pilot.SEEDS == (17, 29)
    assert pilot.EVAL_SEEDS == tuple(range(15000, 15010))
    assert pilot.TRAIN_ACTIONS * 2 + 2 * 2 * 2 * 10 * pilot.EPISODE_CAP == 7296
    monkeypatch.setattr(pilot, "TRAIN_ACTIONS", 2049)
    with pytest.raises(ValueError, match="preregistration"):
        pilot.verify_registration()


def test_only_public_observation_to_actor_and_post_action_training():
    world = SyntheticWorkshop()
    seen = []
    transitions = []
    def choose(public):
        assert public.shape == (5, 5)
        assert not hasattr(public, "private_secret")
        seen.append(public.copy())
        return 1
    def after_step(before, action, reward, after, done):
        transitions.append((before.copy(), action, reward, after.copy(), done))
    rows = numeric()
    result = pilot._episode(world, np.zeros((5, 5), dtype=np.int8), choose,
                            40, rows, (0, 0, 0, 0), after_step=after_step)
    assert world.calls == 3
    assert result["goal_success"] and result["steps"] == 3
    assert not result["partial_final"]
    assert result["events"]["switch_activated"] == 1
    assert len(seen) == len(transitions) == len(rows["action"]) == 3
    assert np.array_equal(transitions[1][0], seen[1])
    assert transitions[-1][-1]
    safe = pilot._arrays(rows, {"seed17_net_0_weight": np.ones((2, 2), dtype=np.float32)})
    assert all(array.dtype.kind != "O" for array in safe.values())
    assert safe["observation"].shape == (3, 5, 5)


def test_invalid_action_rejected_without_native_call():
    for bad in (-1, 4, "1", np.int64(1)):
        world = SyntheticWorkshop()
        with pytest.raises(ValueError, match="invalid action"):
            pilot._episode(world, np.zeros((5, 5), dtype=np.int8),
                           lambda _public, invalid=bad: invalid, 40, numeric(), (0, 0, 0, 0))
        assert world.calls == 0


def test_public_shape_and_budget_fail_closed_without_native_call():
    world = SyntheticWorkshop()
    with pytest.raises(ValueError, match="public view"):
        pilot._episode(world, np.zeros((9, 9), dtype=np.int8),
                       lambda _public: 1, 40, numeric(), (0, 0, 0, 0))
    with pytest.raises(ValueError, match="budget"):
        pilot._episode(world, np.zeros((5, 5), dtype=np.int8),
                       lambda _public: 1, 41, numeric(), (0, 0, 0, 0))
    assert world.calls == 0


def test_partial_episode_not_misreported_as_goal():
    world = SyntheticWorkshop(terminal_step=100)
    result = pilot._episode(world, np.zeros((5, 5), dtype=np.int8),
                            lambda _public: 0, 7, numeric(), (0, 0, 0, 0))
    assert world.calls == 7
    assert result["partial_final"] and not result["goal_success"]
    assert result["reward"] == 0.0
