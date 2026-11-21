"""Evaluator-only M23 UnlockPickup native physical-effect witness.

The learned controller receives only the original public Gymnasium observation.
Privileged world state and object identity are read solely by this evaluator.
"""

from __future__ import annotations

from collections import deque

from bagi.experiments.m23_study_contract import UnlockPickupWitness


def _target_color(observation: dict) -> str:
    if set(observation) != {"image", "direction", "mission"}:
        raise ValueError("UnlockPickup must expose only standard public fields")
    mission = observation["mission"]
    if not isinstance(mission, str):
        raise TypeError("Public mission must be text")
    words = mission.strip().lower().split()
    if len(words) != 5 or words[:3] != ["pick", "up", "the"] or words[-1] != "box":
        raise ValueError("Unsupported native UnlockPickup public mission")
    return words[3]


def _is_behind_locked_door(world, box_cell: tuple[int, int]) -> bool:
    """Evaluator-only reachability test with locked doors treated as walls."""
    width, height = world.grid.width, world.grid.height
    start = tuple(int(v) for v in world.agent_pos)
    q = deque([start])
    seen = {start}
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) == box_cell:
                return False
            tile = world.grid.get(nx, ny)
            if (nx, ny) not in seen and getattr(tile, "type", None) not in {
                "wall", "door", "box", "key", "ball", "lava",
            }:
                seen.add((nx, ny))
                q.append((nx, ny))
    return True


class NativeUnlockPickupAudit:
    """Return unchanged native step data, and report separately witnessed effects."""

    def __init__(self, env):
        self.env = env
        self.started = False
        self.finished = False
        self.native_actions = 0
        self.action_counts = {str(i): 0 for i in range(7)}
        self.actual_forward_movements = 0
        self.key_pickup_index = None
        self.door_unlock_index = None
        self.key_drop_index = None
        self.correct_box_pickup_index = None
        self.terminal_reward = 0.0
        self.door = None
        self.box = None
        self.initially_locked = False
        self.box_behind_locked_door = False

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.__init__(self.env)
        target_color = _target_color(observation)
        world = self.env.unwrapped
        doors = []
        boxes = []
        for x in range(world.grid.width):
            for y in range(world.grid.height):
                tile = world.grid.get(x, y)
                if getattr(tile, "type", None) == "door":
                    doors.append(tile)
                if (getattr(tile, "type", None) == "box"
                        and getattr(tile, "color", None) == target_color):
                    boxes.append((tile, (x, y)))
        if len(doors) != 1 or len(boxes) != 1:
            raise ValueError("Evaluator requires one native door and mission-matched box")
        self.door = doors[0]
        self.box = boxes[0][0]
        self.initially_locked = bool(self.door.is_locked)
        self.box_behind_locked_door = bool(
            self.initially_locked and _is_behind_locked_door(world, boxes[0][1])
        )
        self.started = True
        return observation, info

    def step(self, action: int):
        if not self.started or self.finished:
            raise RuntimeError("Reset evaluator before taking native actions")
        if type(action) is not int or not 0 <= action < 7:
            raise ValueError("Native executed action must be an integer 0..6")
        world = self.env.unwrapped
        before_position = tuple(int(v) for v in world.agent_pos)
        before_carrying = world.carrying
        before_front = world.grid.get(*world.front_pos)
        was_locked = bool(self.door.is_locked)
        was_open = bool(self.door.is_open)
        event_index = self.native_actions
        self.native_actions += 1
        self.action_counts[str(action)] += 1
        result = self.env.step(action)
        _, reward, terminated, truncated, _ = result
        after_position = tuple(int(v) for v in world.agent_pos)
        if action == 2 and before_position != after_position:
            self.actual_forward_movements += 1
        if (action == 3 and before_carrying is None and before_front is not None
                and getattr(before_front, "type", None) == "key"
                and world.carrying is before_front and self.key_pickup_index is None):
            self.key_pickup_index = event_index
        if (action == 5 and before_front is self.door and was_locked and not was_open
                and not self.door.is_locked and self.door.is_open
                and self.door_unlock_index is None):
            self.door_unlock_index = event_index
        if (action == 4 and before_carrying is not None
                and getattr(before_carrying, "type", None) == "key"
                and world.carrying is None
                and world.grid.get(*world.front_pos) is before_carrying
                and self.key_drop_index is None):
            self.key_drop_index = event_index
        if (action == 3 and before_front is self.box and before_carrying is None
                and world.carrying is self.box and self.correct_box_pickup_index is None):
            self.correct_box_pickup_index = event_index
        if terminated and reward > 0:
            self.terminal_reward = float(reward)
        self.finished = bool(terminated or truncated)
        return result

    def summary(self) -> dict:
        if not self.started:
            raise RuntimeError("Evaluator has not been reset")
        witness = UnlockPickupWitness(
            initially_locked=self.initially_locked,
            box_initially_behind_locked_door=self.box_behind_locked_door,
            initial_goal_already_satisfied=False,
            actual_forward_movements=self.actual_forward_movements,
            key_pickup_action_index=self.key_pickup_index,
            door_unlock_action_index=self.door_unlock_index,
            key_drop_action_index=self.key_drop_index,
            correct_box_pickup_action_index=self.correct_box_pickup_index,
            reward_at_terminal=self.terminal_reward,
        )
        return {
            "native_actions": self.native_actions,
            "action_counts": dict(self.action_counts),
            "actual_forward_movements": self.actual_forward_movements,
            "key_pickup_action_index": self.key_pickup_index,
            "door_unlock_action_index": self.door_unlock_index,
            "key_drop_action_index": self.key_drop_index,
            "correct_box_pickup_action_index": self.correct_box_pickup_index,
            "reward_at_terminal": self.terminal_reward,
            "initially_locked": self.initially_locked,
            "box_behind_locked_door": self.box_behind_locked_door,
            "full_physical_goal": witness.physical_success(),
        }
