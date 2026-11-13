"""G2-E1 Workshop native adapter contracts; no native predictor claim."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from bagi.experiments.public_effect_workshop_native import (
    BEHAVIORS,
    EPISODE_SEEDS,
    LAYOUTS,
    MAX_ACTIONS,
    _fingerprint,
    _script_actor,
    collect_workshop_native,
)


def test_workshop_registered_budget_and_seed_separation():
    assert LAYOUTS == ("default", "mirror_x")
    assert BEHAVIORS == ("uniform_random", "g1_public_script")
    assert EPISODE_SEEDS == tuple(range(12100, 12110))
    assert MAX_ACTIONS == 800


def test_script_actor_is_bounded():
    actor = _script_actor((1, 2))
    assert actor(np.zeros(25)) == 1
    assert actor(np.zeros(25)) == 2
    with pytest.raises(ValueError, match="exhausted"):
        actor(np.zeros(25))


def test_synthetic_workshop_adapter_no_real_simulator_calls(monkeypatch):
    """Mocked interface only: do NOT consume pristine Workshop TEST episodes in CI."""
    class FakeWorld:
        def __init__(self, *, size, view_size, max_steps, start_mode, layout_mode):
            self.layout_mode = layout_mode
            self.max_steps = max_steps

        def _view(self):
            view = np.zeros((5, 5), dtype=np.int8)
            view[2, 4 if self.layout_mode == "default" else 0] = 4
            view[2, 2] = 3
            view[0, 0] = self._steps % 3
            return view

        def reset(self, *, seed):
            self._steps = 0
            self._pos = np.array([1, 1])
            self._has_key = False
            self._door_open = False
            return self._view(), {"seed": seed}

        def step(self, action):
            self._steps += 1
            self._pos += np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])[action]
            self._door_open = self._steps >= 4
            terminal = self._steps == 16
            return self._view(), float(terminal), terminal, False, {}

    def fake_scripts(view):
        toward = 1 if view[2, 4] == 4 else 3
        away = 3 if toward == 1 else 1
        return {"b_good_switch": (toward, toward),
                "c_reach_goal": (away, away, 2, 2, 2, 2,
                                 toward, toward, toward, toward, toward, toward, 2, 2)}

    monkeypatch.setattr("bagi.worlds.WorkshopEnv", FakeWorld)
    monkeypatch.setattr("bagi.experiments.general_brain_native_g1.workshop_scripts",
                        fake_scripts)

    train, test, summary = collect_workshop_native()
    assert len(train) == len(test) == 20
    assert 0 < summary["native_actions"] <= MAX_ACTIONS
    assert summary["train_episodes"] == summary["test_episodes"] == 20
    assert summary["new_teacher_queries"] == 0
    assert not summary["fixed_script_is_learned"]
    for layout, episodes in zip(LAYOUTS, (train, test), strict=True):
        assert {a.episode.layout for a in episodes} == {layout}
        assert Counter(a.episode.behavior for a in episodes) == Counter(
            {behavior: 10 for behavior in BEHAVIORS}
        )
        assert all(a.episode.observations.shape[1] == 25 for a in episodes)
        assert all(a.episode.actions.min() >= 0 and a.episode.actions.max() < 4
                   for a in episodes)
        assert all(a.hidden_without_public_change <= a.evaluator_hidden_changes
                   for a in episodes)
        assert sum(a.native_actions for a in episodes) == sum(
            summary["native_actions_by_layout_behavior"][layout].values()
        )
        assert summary["unique_public_trajectories"][layout]["g1_public_script"] == 1
        assert 1 <= summary["unique_public_trajectories"][layout]["uniform_random"] <= 10
    assert {a.episode.episode_id for a in train}.isdisjoint(
        {a.episode.episode_id for a in test}
    )
    assert np.all([len(_fingerprint(a)) == 64 for a in train + test])
