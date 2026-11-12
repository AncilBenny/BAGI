"""Public-only GoToDoor teacher; evaluator-only native physical success witness."""
from __future__ import annotations

import numpy as np
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX

from bagi.experiments.m13_public_interaction_coverage import Controller, front

DOOR = OBJECT_TO_IDX["door"]
WALL = OBJECT_TO_IDX["wall"]


def requested_color(observation: dict) -> str:
    """Parse the native public mission; no privileged target state."""
    mission = observation["mission"]
    if not isinstance(mission, str):
        raise ValueError("GoToDoor public mission must be text")
    words = mission.strip().lower().split()
    if (len(words) != 5 or words[:3] != ["go", "to", "the"]
            or words[4] != "door" or words[3] not in COLOR_TO_IDX):
        raise ValueError("Unrecognized public GoToDoor mission")
    return words[3]


class PublicGoToDoorTeacher(Controller):
    """Reactive public observation/action demonstrator, not an oracle."""

    def __init__(self):
        super().__init__("public_memory")

    def choose(self, observation: dict, rng: np.random.Generator) -> int:
        color = requested_color(observation)
        obj, _ = front(observation)
        visible_color = int(np.asarray(observation["image"])[3, 5, 1])
        if obj == DOOR:
            return 6 if visible_color == COLOR_TO_IDX[color] else int(rng.integers(2))
        if obj == WALL:
            return int(rng.integers(2))
        return 2 if rng.random() < 0.80 else int(rng.integers(2))


class NativeGoToDoorAudit:
    """External goal witness: target-colored adjacent door + executed done + move."""

    def __init__(self, env):
        self.env = env
        self.started = False
        self.finished = False
        self.target = ""
        self.native_actions = 0
        self.action_counts = {str(i): 0 for i in range(7)}
        self.forward_actions = 0
        self.effective_forward_movements = 0
        self.native_rewarded = 0
        self.target_done = 0
        self.adjacent_target_done_attempts = 0
        self.initially_adjacent = False
        self.full_physical_goal = False

    def _adjacent(self) -> bool:
        world = self.env.unwrapped
        x, y = (int(n) for n in world.agent_pos)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            u, v = x + dx, y + dy
            if 0 <= u < world.grid.width and 0 <= v < world.grid.height:
                tile = world.grid.get(u, v)
                if (getattr(tile, "type", None) == "door"
                        and getattr(tile, "color", None) == self.target):
                    return True
        return False

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.__init__(self.env)
        self.target = requested_color(observation)
        world = self.env.unwrapped
        matches = sum(
            getattr(world.grid.get(x, y), "type", None) == "door"
            and getattr(world.grid.get(x, y), "color", None) == self.target
            for y in range(world.grid.height)
            for x in range(world.grid.width)
        )
        if matches != 1:
            raise ValueError("M18 requires exactly one mission-colored physical door")
        self.initially_adjacent = self._adjacent()
        self.started = True
        return observation, info

    def step(self, action: int):
        if not self.started or self.finished:
            raise RuntimeError("M18 native evaluator needs a live reset episode")
        if type(action) is not int or action not in range(7):
            raise ValueError("M18 native action must be an integer 0..6")
        # World position and target matching are never returned to actor/teacher.
        adjacent = self._adjacent()
        before_position = tuple(int(v) for v in self.env.unwrapped.agent_pos)
        self.action_counts[str(action)] += 1
        self.native_actions += 1
        self.forward_actions += int(action == 2)
        self.adjacent_target_done_attempts += int(action == 6 and adjacent)
        outcome = self.env.step(action)
        _, reward, terminated, truncated, _ = outcome
        after_position = tuple(int(v) for v in self.env.unwrapped.agent_pos)
        self.effective_forward_movements += int(
            action == 2 and before_position != after_position
        )
        self.native_rewarded += int(terminated and reward > 0)
        if action == 6 and adjacent and terminated:
            self.target_done += 1
            self.full_physical_goal = (
                not self.initially_adjacent and self.effective_forward_movements > 0
            )
        self.finished = bool(terminated or truncated)
        return outcome

    def summary(self) -> dict:
        if not self.started:
            raise RuntimeError("M18 evaluator must be reset")
        return {
            "native_actions": self.native_actions,
            "action_counts": self.action_counts,
            "full_physical_goal": bool(self.full_physical_goal),
            "rewarded_terminations": self.native_rewarded,
            "effective_pickup_events": 0,
            "effective_door_open_events": 0,
            "correct_target_done_events": self.target_done,
            "adjacent_target_done_attempts": self.adjacent_target_done_attempts,
            "initially_adjacent_target_door": self.initially_adjacent,
            "effective_forward_movements": self.effective_forward_movements,
        }
