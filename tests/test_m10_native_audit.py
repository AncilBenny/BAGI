"""M10 DoorKey audit adapter tests using deterministic fake native transitions."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from bagi.experiments.m10_native_audit import NativeDoorKeyAudit, snapshot_doorkey


class Tile:
    def __init__(self, kind, *, locked=False, opened=False):
        self.type = kind
        self.is_locked = locked
        self.is_open = opened


class Grid:
    width = 4
    height = 3

    def __init__(self):
        self.cells = {(1, 1): Tile("door", locked=True), (3, 1): Tile("goal")}

    def get(self, x, y):
        return self.cells.get((x, y))


class FakeDoorKey:
    def __init__(self):
        self.unwrapped = self
        self.grid = Grid()
        self.agent_pos = (0, 1)
        self.front_pos = (1, 1)
        self.carrying = None
        self.n = 0

    def gen_obs(self):
        image = np.zeros((7, 7, 3), dtype=np.uint8)
        if self.n == 0:
            image[2, 2, 0] = 5
        return {"image": image}

    def reset(self, **kwargs):
        self.__init__()
        return {"image": self.gen_obs()["image"]}, {}

    def step(self, action):
        self.n += 1
        if self.n == 1 and action == 3:
            self.carrying = Tile("key")
        if self.n == 2 and action == 5 and self.carrying is not None:
            door = self.grid.get(1, 1)
            door.is_locked = False
            door.is_open = True
        if self.n == 3 and action == 2:
            self.agent_pos = (3, 1)
        success = self.n == 3 and self.agent_pos == (3, 1)
        return {}, .9 if success else 0.0, success, self.n >= 4, {}


class FakeGoalAtTimeLimit(FakeDoorKey):
    """Model a native goal on the last externally time-limited step."""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated or terminated, info


@pytest.fixture(autouse=True)
def mock_minigrid(monkeypatch):
    package = types.ModuleType("minigrid")
    package.__path__ = []
    core = types.ModuleType("minigrid.core")
    core.__path__ = []
    constants = types.ModuleType("minigrid.core.constants")
    constants.OBJECT_TO_IDX = {"key": 5}
    monkeypatch.setitem(sys.modules, "minigrid", package)
    monkeypatch.setitem(sys.modules, "minigrid.core", core)
    monkeypatch.setitem(sys.modules, "minigrid.core.constants", constants)


def test_native_key_door_goal_sequence():
    audit = NativeDoorKeyAudit(FakeDoorKey())
    audit.reset(seed=123)
    assert snapshot_doorkey(audit.env).physical.key_visible
    for action in (3, 5, 2):
        audit.step(action)
    summary = audit.summary()
    assert summary["full_physical_goal"]
    assert summary["effective_pickup_events"] == 1
    assert summary["effective_door_open_events"] == 1
    assert summary["valid_toggle_opportunities"] == 1
    assert summary["valid_toggle_attempts"] == 1
    assert summary["native_actions"] == 3
    with pytest.raises(RuntimeError):
        audit.step(0)


def test_native_adapter_preserves_goal_on_simultaneous_time_limit():
    audit = NativeDoorKeyAudit(FakeGoalAtTimeLimit())
    audit.reset(seed=432)
    for action in (3, 5, 2):
        _, reward, terminated, truncated, _ = audit.step(action)
    assert reward == .9
    assert terminated and truncated
    assert audit.summary()["native_actions"] == 3
    assert audit.summary()["full_physical_goal"] is True
    with pytest.raises(RuntimeError):
        audit.step(0)


def test_wrong_action_does_not_open_door():
    audit = NativeDoorKeyAudit(FakeDoorKey())
    audit.reset()
    audit.step(3)
    audit.step(2)
    assert audit.summary()["valid_toggle_opportunities"] == 1
    assert audit.summary()["valid_toggle_attempts"] == 0
    assert not audit.summary()["full_physical_goal"]


def test_reset_clears_counts():
    audit = NativeDoorKeyAudit(FakeDoorKey())
    audit.reset()
    audit.step(3)
    audit.reset()
    assert audit.summary()["native_actions"] == 0


def test_rejects_multiple_doors():
    env = FakeDoorKey()
    env.grid.cells[2, 1] = Tile("door")
    with pytest.raises(ValueError, match="exactly one"):
        snapshot_doorkey(env)


def test_rejects_door_identity_change():
    audit = NativeDoorKeyAudit(FakeDoorKey())
    audit.reset()
    audit.env.grid.cells[(2, 1)] = audit.env.grid.cells.pop((1, 1))
    with pytest.raises(ValueError, match="identity"):
        audit.step(0)


def test_rejects_invalid_action_before_execution():
    audit = NativeDoorKeyAudit(FakeDoorKey())
    audit.reset()
    with pytest.raises(ValueError):
        audit.step(True)
    assert audit.env.n == 0
