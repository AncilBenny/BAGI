"""The first BAGI laboratory: a small partially observable causal world."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

EMPTY, WALL, GOAL, AGENT, SWITCH, DOOR_CLOSED, DOOR_OPEN, KEY, LOCK = range(9)


class WorkshopEnv(gym.Env):
    """A compact world with reusable causal structure.

    Rules in v0.1:
    - stepping on the switch opens the door;
    - stepping on the key picks it up;
    - carrying the key lets the agent pass the lock;
    - reaching the goal completes the episode.

    The observation is a 5x5 local view, so the complete world state is not
    always visible and later agents will need memory.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size: int = 9, max_steps: int = 100, view_size: int = 5):
        super().__init__()
        if size < 9:
            raise ValueError("Workshop size must be at least 9")
        if view_size % 2 == 0:
            raise ValueError("view_size must be odd")
        self.size = size
        self.max_steps = max_steps
        self.view_size = view_size
        self.action_space = spaces.Discrete(4)  # up, right, down, left
        self.observation_space = spaces.Box(
            0, LOCK, shape=(view_size, view_size), dtype=np.int8
        )
        self._steps = 0
        self._has_key = False
        self._door_open = False
        self._pos = np.array([1, 1], dtype=np.int64)
        self._goal = np.array([size - 2, size - 2], dtype=np.int64)
        self._switch = np.array([1, 3], dtype=np.int64)
        self._door = np.array([3, 4], dtype=np.int64)
        self._key = np.array([5, 1], dtype=np.int64)
        self._lock = np.array([5, 4], dtype=np.int64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        self._has_key = False
        self._door_open = False
        self._pos = np.array([1, 1], dtype=np.int64)
        return self._observe(), self._info(events=["reset"])

    def step(self, action):
        directions = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        candidate = self._pos + directions[int(action)]
        events: list[str] = []

        if self._can_enter(candidate):
            self._pos = candidate
        else:
            events.append("blocked")

        if np.array_equal(self._pos, self._switch) and not self._door_open:
            self._door_open = True
            events.append("switch_activated")

        if np.array_equal(self._pos, self._key) and not self._has_key:
            self._has_key = True
            events.append("key_picked_up")

        if np.array_equal(self._pos, self._lock) and self._has_key:
            events.append("lock_opened")

        self._steps += 1
        reached = bool(np.array_equal(self._pos, self._goal))
        if reached:
            events.append("goal_reached")
        reward = 1.0 if reached else 0.0
        truncated = self._steps >= self.max_steps
        return self._observe(), reward, reached, truncated, self._info(events)

    def _can_enter(self, pos: np.ndarray) -> bool:
        row, col = int(pos[0]), int(pos[1])
        if row <= 0 or col <= 0 or row >= self.size - 1 or col >= self.size - 1:
            return False
        if np.array_equal(pos, self._door) and not self._door_open:
            return False
        if np.array_equal(pos, self._lock) and not self._has_key:
            return False
        return True

    def _tile_at(self, row: int, col: int) -> int:
        if row <= 0 or col <= 0 or row >= self.size - 1 or col >= self.size - 1:
            return WALL
        pos = np.array([row, col])
        if np.array_equal(pos, self._goal):
            return GOAL
        if np.array_equal(pos, self._switch):
            return SWITCH
        if np.array_equal(pos, self._door):
            return DOOR_OPEN if self._door_open else DOOR_CLOSED
        if np.array_equal(pos, self._key) and not self._has_key:
            return KEY
        if np.array_equal(pos, self._lock):
            return LOCK
        return EMPTY

    def _observe(self):
        radius = self.view_size // 2
        view = np.zeros((self.view_size, self.view_size), dtype=np.int8)
        for vr in range(self.view_size):
            for vc in range(self.view_size):
                wr = int(self._pos[0] + vr - radius)
                wc = int(self._pos[1] + vc - radius)
                view[vr, vc] = self._tile_at(wr, wc)
        view[radius, radius] = AGENT
        return view

    def _info(self, events: list[str]):
        return {
            "position": self._pos.copy(),
            "has_key": self._has_key,
            "door_open": self._door_open,
            "events": events,
            "steps": self._steps,
        }

    def render(self):
        chars = {
            EMPTY: ".",
            WALL: "#",
            GOAL: "G",
            AGENT: "A",
            SWITCH: "S",
            DOOR_CLOSED: "D",
            DOOR_OPEN: "d",
            KEY: "K",
            LOCK: "L",
        }
        grid = np.empty((self.size, self.size), dtype="<U1")
        for row in range(self.size):
            for col in range(self.size):
                grid[row, col] = chars[self._tile_at(row, col)]
        grid[tuple(self._pos)] = "A"
        return "\n".join(" ".join(row) for row in grid)
