"""M10 evaluator-only physical opportunity accounting; never a policy input.

The caller supplies physical snapshots from an independent auditor after each
native action. Do not pass these snapshots, counts or private state into PPO.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PhysicalSnapshot:
    has_key: bool
    key_visible: bool
    locked_door_front: bool
    door_open: bool
    goal_reached: bool


@dataclass
class EpisodeFunnel:
    actions: int = 0
    action_counts: Counter[int] = field(default_factory=Counter)
    key_visible_steps: int = 0
    key_possession_steps: int = 0
    valid_toggle_opportunities: int = 0
    valid_toggle_attempts: int = 0
    ineffective_valid_toggles: int = 0
    effective_pickups: int = 0
    effective_door_opens: int = 0
    rewarded_terminations: int = 0
    first_key_pickup_at: int | None = None
    first_valid_toggle_at: int | None = None
    first_door_open_at: int | None = None
    first_goal_at: int | None = None

    def observe(
        self, before: PhysicalSnapshot, action: int, after: PhysicalSnapshot,
        *, reward: float, terminated: bool, truncated: bool,
    ) -> None:
        """Count pre-action opportunities separately from observed effects."""
        if type(action) is not int or not 0 <= action <= 6:
            raise ValueError("expected one executed native MiniGrid action")
        if not isinstance(before, PhysicalSnapshot) or not isinstance(
            after, PhysicalSnapshot
        ):
            raise TypeError("auditor snapshots are required")
        # A native goal on the last TimeLimit step may terminate AND truncate.
        # Preserve both signals and audit actual physical outcomes unchanged.
        if not isinstance(reward, (int, float)) or not math.isfinite(
            reward
        ) or not 0 <= reward <= 1:
            raise ValueError("native reward must be finite in [0, 1]")
        if self.first_goal_at is not None or self.rewarded_terminations:
            raise ValueError("cannot append transitions after a recorded goal")
        self.actions += 1
        self.action_counts[action] += 1
        self.key_visible_steps += int(before.key_visible)
        self.key_possession_steps += int(before.has_key)
        valid = before.has_key and before.locked_door_front and not before.door_open
        self.valid_toggle_opportunities += int(valid)
        self.valid_toggle_attempts += int(valid and action == 5)
        pickup = not before.has_key and after.has_key
        opening = not before.door_open and after.door_open
        self.effective_pickups += int(pickup)
        self.effective_door_opens += int(opening)
        self.ineffective_valid_toggles += int(valid and action == 5 and not opening)
        if pickup and self.first_key_pickup_at is None:
            self.first_key_pickup_at = self.actions
        if valid and self.first_valid_toggle_at is None:
            self.first_valid_toggle_at = self.actions
        if opening and self.first_door_open_at is None:
            self.first_door_open_at = self.actions
        if after.goal_reached and self.first_goal_at is None:
            self.first_goal_at = self.actions
        if reward > 0 and terminated:
            self.rewarded_terminations += 1

    def summary(self) -> dict:
        """Count-only output; no coordinates or private state trajectories."""
        return {
            "native_actions": self.actions,
            "action_counts": {str(k): self.action_counts[k] for k in range(7)},
            "key_visible_steps": self.key_visible_steps,
            "key_possession_steps": self.key_possession_steps,
            "valid_toggle_opportunities": self.valid_toggle_opportunities,
            "valid_toggle_attempts": self.valid_toggle_attempts,
            "ineffective_valid_toggles": self.ineffective_valid_toggles,
            "effective_pickup_events": self.effective_pickups,
            "effective_door_open_events": self.effective_door_opens,
            "rewarded_terminations": self.rewarded_terminations,
            "first_key_pickup_at": self.first_key_pickup_at,
            "first_valid_toggle_at": self.first_valid_toggle_at,
            "first_door_open_at": self.first_door_open_at,
            "first_goal_at": self.first_goal_at,
            "full_physical_goal": bool(
                self.first_key_pickup_at is not None
                and self.first_door_open_at is not None
                and self.first_goal_at is not None
                and self.rewarded_terminations
                and self.first_key_pickup_at < self.first_door_open_at
                <= self.first_goal_at
            ),
            "max_action_fraction": (
                max(self.action_counts.values()) / self.actions
                if self.actions else None
            ),
        }
