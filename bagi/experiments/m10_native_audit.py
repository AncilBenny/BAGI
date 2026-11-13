"""Evaluator-only MiniGrid DoorKey adapter for the M10 causal funnel.

Never attach this adapter to PPO training observations, rewards, or rollout data.
The audit reads private state only to measure effects after executed actions.
"""
from __future__ import annotations

from dataclasses import dataclass

from bagi.experiments.m10_causal_funnel import EpisodeFunnel, PhysicalSnapshot


@dataclass(frozen=True)
class AuditedState:
    door_cell: tuple[int, int]
    physical: PhysicalSnapshot


def snapshot_doorkey(env) -> AuditedState:
    """Audit a DoorKey world with exactly one persistent door target."""
    world = env.unwrapped
    doors = []
    for y in range(world.grid.height):
        for x in range(world.grid.width):
            tile = world.grid.get(x, y)
            if getattr(tile, "type", None) == "door":
                doors.append((x, y, tile))
    if len(doors) != 1:
        raise ValueError("M10 DoorKey audit requires exactly one door")
    x, y, door = doors[0]
    front = tuple(int(v) for v in world.front_pos)
    position = tuple(int(v) for v in world.agent_pos)
    standing_on = world.grid.get(*position)
    # The raw public image contains object indices, not private state.
    # Import lazily so pure mock-based unit tests do not need the RL extra.
    from minigrid.core.constants import OBJECT_TO_IDX

    image = world.gen_obs()["image"]
    key_visible = bool((image[:, :, 0] == OBJECT_TO_IDX["key"]).any())
    return AuditedState(
        door_cell=(x, y),
        physical=PhysicalSnapshot(
            has_key=getattr(world.carrying, "type", None) == "key",
            key_visible=key_visible,
            locked_door_front=(front == (x, y) and bool(door.is_locked)),
            door_open=bool(door.is_open),
            goal_reached=getattr(standing_on, "type", None) == "goal",
        ),
    )


class NativeDoorKeyAudit:
    """Measure a separate evaluation episode without modifying native outputs."""

    def __init__(self, env):
        self.env = env
        self.funnel = EpisodeFunnel()
        self._before: AuditedState | None = None
        self._finished = False

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.funnel = EpisodeFunnel()
        self._before = snapshot_doorkey(self.env)
        self._finished = False
        return observation, info

    def step(self, action):
        if self._before is None or self._finished:
            raise RuntimeError("reset audit before stepping or after episode end")
        if type(action) is not int or not 0 <= action <= 6:
            raise ValueError("expected native MiniGrid integer action")
        result = self.env.step(action)
        _, reward, terminated, truncated, _ = result
        after = snapshot_doorkey(self.env)
        if after.door_cell != self._before.door_cell:
            raise ValueError("target door identity changed within episode")
        self.funnel.observe(
            self._before.physical, action, after.physical,
            reward=reward, terminated=terminated, truncated=truncated,
        )
        self._before = after
        self._finished = bool(terminated or truncated)
        return result

    def summary(self) -> dict:
        return self.funnel.summary()
