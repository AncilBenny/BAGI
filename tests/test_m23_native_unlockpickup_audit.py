"""M23 evaluator mock tests and maintained MiniGrid reset smoke.

The mocked event sequence is not evidence of native physical task success.
"""

import pytest

from bagi.experiments.m23_native_unlockpickup_audit import NativeUnlockPickupAudit


class Tile:
    def __init__(self, kind, color=None, is_locked=False):
        self.type = kind
        self.color = color
        self.is_locked = is_locked
        self.is_open = False


class Grid:
    width = 7
    height = 7

    def __init__(self):
        self.cells = {(3, y): Tile("wall") for y in range(7)}
        self.key = Tile("key", "yellow")
        self.door = Tile("door", "yellow", is_locked=True)
        self.box = Tile("box", "red")
        self.cells[(2, 3)] = self.key
        self.cells[(3, 3)] = self.door
        self.cells[(5, 3)] = self.box

    def get(self, x, y):
        return self.cells.get((x, y))


class NativeMockEnv:
    def __init__(self):
        self.unwrapped = self
        self.grid = Grid()
        self.agent_pos = (1, 3)
        self.front_pos = (2, 3)
        self.carrying = None
        self.events = []

    def reset(self):
        self.__init__()
        return {"image": None, "direction": 0, "mission": "pick up the red box"}, {}

    def step(self, action):
        index = len(self.events)
        self.events.append(action)
        if action == 3 and index == 0:
            self.carrying = self.grid.key
            self.grid.cells.pop((2, 3))
        elif action == 2 and index == 1:
            self.agent_pos, self.front_pos = (2, 3), (3, 3)
        elif action == 5 and index == 2:
            self.grid.door.is_locked = False
            self.grid.door.is_open = True
        elif action == 2 and index == 3:
            self.agent_pos, self.front_pos = (3, 3), (4, 3)
        elif action == 1 and index == 4:
            self.front_pos = (3, 4)
        elif action == 4 and index == 5:
            self.grid.cells[(3, 4)] = self.grid.key
            self.carrying = None
        elif action == 0 and index == 6:
            self.front_pos = (4, 3)
        elif action == 2 and index == 7:
            self.agent_pos, self.front_pos = (4, 3), (5, 3)
        elif action == 3 and index == 8:
            self.carrying = self.grid.box
            self.grid.cells.pop((5, 3))
            return {}, 0.8, True, False, {}
        return {}, 0.0, False, False, {}


def test_mock_native_chain_requires_ordered_effects():
    auditor = NativeUnlockPickupAudit(NativeMockEnv())
    auditor.reset()
    for action in (3, 2, 5, 2, 1, 4, 0, 2, 3):
        auditor.step(action)
    summary = auditor.summary()
    assert summary["full_physical_goal"]
    assert summary["native_actions"] == 9
    assert summary["key_pickup_action_index"] == 0
    assert summary["door_unlock_action_index"] == 2
    assert summary["key_drop_action_index"] == 5
    assert summary["correct_box_pickup_action_index"] == 8


def test_reward_without_causal_unlock_is_not_success():
    auditor = NativeUnlockPickupAudit(NativeMockEnv())
    auditor.reset()
    auditor.env.grid.door.is_locked = False
    for action in (3, 2, 5, 2, 1, 4, 0, 2, 3):
        auditor.step(action)
    assert auditor.summary()["reward_at_terminal"] > 0
    assert not auditor.summary()["full_physical_goal"]


def test_auditor_rejects_non_native_actions():
    auditor = NativeUnlockPickupAudit(NativeMockEnv())
    auditor.reset()
    with pytest.raises(ValueError):
        auditor.step(9)
    with pytest.raises(ValueError):
        auditor.step("pickup")


def test_maintained_unlockpickup_reset_only_on_train_layout():
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("minigrid")
    env = gym.make("MiniGrid-UnlockPickup-v0")
    try:
        evaluator = NativeUnlockPickupAudit(env)
        obs, _ = evaluator.reset(seed=451001)
        assert set(obs) == {"image", "direction", "mission"}
        assert evaluator.summary()["initially_locked"]
        assert evaluator.summary()["box_behind_locked_door"]
        assert evaluator.summary()["native_actions"] == 0
    finally:
        env.close()
