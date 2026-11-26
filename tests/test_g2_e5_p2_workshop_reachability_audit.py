"""No-native-step evaluator-only Workshop reachability contracts."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from bagi.experiments import g2_e5_p2_workshop_reachability_audit as audit


class SyntheticWorkshop:
    """Small source-shaped evaluator fixture; NEVER invokes native Workshop."""

    def __init__(self, *, size, view_size, max_steps, start_mode, layout_mode):
        assert (size, view_size, max_steps, start_mode) == (9, 5, 40, "seeded_start_chamber")
        self.layout_mode = layout_mode
        self._switch = np.array(self._layout_coords(1, 3))
        self._door = np.array(self._layout_coords(3, 1))
        self._key = np.array(self._layout_coords(5, 1))
        self._lock = np.array(self._layout_coords(5, 4))
        self._goal = np.array(self._layout_coords(7, 7))
        walls = {(3, c) for c in range(2, 8)} | {(r, 4) for r in (3, 4, 6, 7)}
        self._internal_walls = {self._layout_coords(*p) for p in walls}
        self._pos = np.array(self._layout_coords(1, 1))
        self._door_open = False
        self._has_key = False

    def _layout_coords(self, row, col):
        return (row, 8 - col) if self.layout_mode == "mirror_x" else (row, col)

    def _can_enter(self, pos):
        p = tuple(map(int, pos))
        if not (0 < p[0] < 8 and 0 < p[1] < 8) or p in self._internal_walls:
            return False
        if p == tuple(self._door) and not self._door_open:
            return False
        return not (p == tuple(self._lock) and not self._has_key)

    def _observe(self):
        out = np.zeros((5, 5), dtype=np.int8)
        for row in range(5):
            for col in range(5):
                p = (int(self._pos[0]) + row - 2, int(self._pos[1]) + col - 2)
                if not (0 < p[0] < 8 and 0 < p[1] < 8) or p in self._internal_walls:
                    value = 1
                elif p == tuple(self._goal):
                    value = 2
                elif p == tuple(self._switch):
                    value = 4
                elif p == tuple(self._door):
                    value = 6 if self._door_open else 5
                elif p == tuple(self._key) and not self._has_key:
                    value = 7
                elif p == tuple(self._lock):
                    value = 8
                else:
                    value = 0
                out[row, col] = value
        out[2, 2] = 3
        return out

    def reset(self, **_kwargs):
        raise AssertionError("Forbidden: native reset during static audit")

    def step(self, _action):
        raise AssertionError("Forbidden: native step during static audit")

    def close(self):
        pass


@pytest.mark.parametrize("layout", ("default", "mirror_x"))
def test_static_reachability_and_aliasing(layout):
    world = SyntheticWorkshop(size=9, view_size=5, max_steps=40,
                              start_mode="seeded_start_chamber", layout_mode=layout)
    row = audit.audit_layout(world)
    assert row["legal_initial_starts"] == 13
    assert row["distinct_initial_public_views"] == 13
    assert row["reachable_evaluator_states"] == 79
    assert row["distinct_public_views"] == 57
    assert row["public_views_aliasing_multiple_latent_states"] == 14
    assert row["maximum_latent_states_per_public_view"] == 3
    assert row["nonterminal_public_view_action_keys"] == 224
    assert row["keys_with_ambiguous_next_public_view"] == 2
    assert row["min_shortest_actions"] == 15
    assert row["max_shortest_actions"] == 19
    assert row["all_starts_reachable_within_40"]


def test_static_invalid_action_rejected():
    world = SyntheticWorkshop(size=9, view_size=5, max_steps=40,
                              start_mode="seeded_start_chamber", layout_mode="default")
    with pytest.raises(ValueError, match="Invalid native action"):
        audit._next_state(world, (1, 1, False, False), 4)


def test_run_has_zero_native_actions_and_no_skill_promotion(monkeypatch):
    monkeypatch.setattr(audit, "verify_source", lambda: None)
    monkeypatch.setattr(audit, "_world_class", lambda: SyntheticWorkshop)
    result = audit.run()
    assert result["status"] == "complete"
    assert result["native_step_calls"] == result["native_reset_calls"] == 0
    assert result["teacher_queries"] == result["policy_updates"] == 0
    assert not result["promotion"]
    assert set(result["layouts"]) == {"default", "mirror_x"}


def test_actual_native_source_parity_without_any_step(monkeypatch):
    if importlib.util.find_spec("gymnasium") is None:
        pytest.skip("Real Workshop source integration runs in dependency-installed CI")
    world_class = audit._world_class()
    audit.verify_source()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Native step/reset is forbidden in this graph audit")

    monkeypatch.setattr(world_class, "step", forbidden)
    monkeypatch.setattr(world_class, "reset", forbidden)
    result = audit.run()
    for row in result["layouts"].values():
        assert row["reachable_evaluator_states"] == 79
        assert row["keys_with_ambiguous_next_public_view"] == 2
        assert row["max_shortest_actions"] == 19
