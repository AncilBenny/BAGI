"""The first BAGI laboratory: a small partially observable causal world."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class WorkshopEnv(gym.Env):
    """Minimal environment scaffold.

    The first version deliberately exposes a tiny local observation and a few
    deterministic rules. We will grow this into doors, switches, blocks,
    resources, hazards, and compositional tasks without changing the agent API.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size: int = 9, max_steps: int = 100):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)  # up, right, down, left
        # 5x5 egocentric occupancy view: 0 empty, 1 wall, 2 goal, 3 agent
        self.observation_space = spaces.Box(0, 3, shape=(5, 5), dtype=np.int8)
        self._pos = np.array([1, 1], dtype=np.int64)
        self._goal = np.array([size - 2, size - 2], dtype=np.int64)
        self._steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = np.array([1, 1], dtype=np.int64)
        self._goal = np.array([self.size - 2, self.size - 2], dtype=np.int64)
        self._steps = 0
        return self._observe(), {"position": self._pos.copy()}

    def step(self, action):
        directions = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        candidate = self._pos + directions[int(action)]
        if np.all(candidate > 0) and np.all(candidate < self.size - 1):
            self._pos = candidate
        self._steps += 1
        reached = bool(np.array_equal(self._pos, self._goal))
        reward = 1.0 if reached else 0.0
        truncated = self._steps >= self.max_steps
        return self._observe(), reward, reached, truncated, {"position": self._pos.copy()}

    def _observe(self):
        view = np.zeros((5, 5), dtype=np.int8)
        for vr in range(5):
            for vc in range(5):
                wr = int(self._pos[0] + vr - 2)
                wc = int(self._pos[1] + vc - 2)
                if wr <= 0 or wc <= 0 or wr >= self.size - 1 or wc >= self.size - 1:
                    view[vr, vc] = 1
                elif wr == self._goal[0] and wc == self._goal[1]:
                    view[vr, vc] = 2
        view[2, 2] = 3
        return view

    def render(self):
        grid = np.full((self.size, self.size), ".", dtype="<U1")
        grid[0, :] = grid[-1, :] = "#"
        grid[:, 0] = grid[:, -1] = "#"
        grid[tuple(self._goal)] = "G"
        grid[tuple(self._pos)] = "A"
        return "\n".join(" ".join(row) for row in grid)
