"""M20 public Fetch actor and evaluator-only physical object witness.

This is an adaptation of BAGI's already maintained MiniGrid public BC
pipeline, not a claim of a new controller or a general intelligence system.
"""
from __future__ import annotations

import numpy as np
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX

from bagi.experiments.m13_public_interaction_coverage import Controller

PICKABLE = {"key": OBJECT_TO_IDX["key"], "ball": OBJECT_TO_IDX["ball"]}
FETCH_PREFIXES = {"get a", "go get a", "fetch a", "go fetch a", "you must fetch a"}


def fetch_request(observation: dict) -> tuple[str, str]:
    """Read target color/object from native PUBLIC mission, never hidden state."""
    if set(observation) != {"image", "direction", "mission"}:
        raise ValueError("Fetch policy input must be public native observations")
    mission = observation["mission"]
    if not isinstance(mission, str):
        raise TypeError("Fetch mission must be public text")
    tokens = mission.strip().lower().split()
    if (len(tokens) < 4 or " ".join(tokens[:-2]) not in FETCH_PREFIXES
            or tokens[-2] not in COLOR_TO_IDX or tokens[-1] not in PICKABLE):
        raise ValueError("Unrecognized maintained MiniGrid Fetch mission")
    return tokens[-2], tokens[-1]


class PublicFetchTeacher(Controller):
    """Reactive public-only teacher; deliberately no hidden-map oracle."""

    def __init__(self) -> None:
        super().__init__("public_memory")

    def choose(self, observation: dict, rng: np.random.Generator) -> int:
        color, obj_type = fetch_request(observation)
        image = np.asarray(observation["image"])
        if image.shape != (7, 7, 3):
            raise ValueError("Fetch native image contract changed")
        front_obj, front_color = int(image[3, 5, 0]), int(image[3, 5, 1])
        if front_obj in PICKABLE.values():
            if front_obj == PICKABLE[obj_type] and front_color == COLOR_TO_IDX[color]:
                return 3
            return int(rng.integers(2))  # Do not deliberately pick wrong object
        if front_obj == OBJECT_TO_IDX["wall"]:
            return int(rng.integers(2))
        return 2 if rng.random() < 0.80 else int(rng.integers(2))


class NativeFetchAudit:
    """Evaluator-only witness for carrying the correct requested native object.

    The policy/teacher must NEVER access the evaluator's unwrapped environment,
    carrying field, private target locations or evaluation summary.
    """

    def __init__(self, env):
        self.env = env
        self.started = False
        self.finished = False
        self.target_color = ""
        self.target_type = ""
        self.initially_adjacent_target = False
        self.native_actions = 0
        self.action_counts = {str(i): 0 for i in range(7)}
        self.rewarded_terminations = 0
        self.effective_forward_movements = 0
        self.effective_pickup_events = 0
        self.correct_target_pickup_events = 0
        self.wrong_target_pickup_events = 0
        self.full_physical_goal = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.__init__(self.env)
        self.target_color, self.target_type = fetch_request(obs)
        world = self.env.unwrapped
        pos = tuple(int(v) for v in world.agent_pos)
        targets = []
        for x in range(world.grid.width):
            for y in range(world.grid.height):
                tile = world.grid.get(x, y)
                if (getattr(tile, "type", None) == self.target_type
                        and getattr(tile, "color", None) == self.target_color):
                    targets.append((x, y))
        if not targets:
            raise ValueError("Fetch evaluator found no mission-matched physical target")
        self.initially_adjacent_target = any(
            abs(x - pos[0]) + abs(y - pos[1]) == 1 for x, y in targets
        )
        self.started = True
        return obs, info

    def step(self, action: int):
        if not self.started or self.finished:
            raise RuntimeError("Fetch evaluator requires a live reset episode")
        if type(action) is not int or action not in range(7):
            raise ValueError("Fetch executed action must be native int 0..6")
        world = self.env.unwrapped
        before_position = tuple(int(v) for v in world.agent_pos)
        before_carrying = world.carrying
        front_tile = world.grid.get(*world.front_pos)
        self.native_actions += 1
        self.action_counts[str(action)] += 1
        result = self.env.step(action)
        _, reward, terminated, truncated, _ = result
        after_position = tuple(int(v) for v in world.agent_pos)
        self.effective_forward_movements += int(
            action == 2 and after_position != before_position
        )
        if (action == 3 and before_carrying is None and front_tile is not None
                and world.carrying is front_tile):
            self.effective_pickup_events += 1
            right = (front_tile.type == self.target_type
                     and front_tile.color == self.target_color)
            self.correct_target_pickup_events += int(right)
            self.wrong_target_pickup_events += int(not right)
            self.full_physical_goal = bool(
                right and terminated and reward > 0
                and not self.initially_adjacent_target
                and self.effective_forward_movements > 0
            )
        self.rewarded_terminations += int(terminated and reward > 0)
        self.finished = bool(terminated or truncated)
        return result

    def summary(self) -> dict:
        if not self.started:
            raise RuntimeError("Fetch evaluator must be reset")
        return {
            "native_actions": self.native_actions,
            "action_counts": self.action_counts,
            "full_physical_goal": bool(self.full_physical_goal),
            "rewarded_terminations": self.rewarded_terminations,
            "effective_pickup_events": self.effective_pickup_events,
            "effective_door_open_events": 0,
            "correct_target_pickup_events": self.correct_target_pickup_events,
            "wrong_target_pickup_events": self.wrong_target_pickup_events,
            "initially_adjacent_target": self.initially_adjacent_target,
            "effective_forward_movements": self.effective_forward_movements,
        }
