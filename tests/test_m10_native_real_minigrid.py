"""Real MiniGrid API smoke test for the evaluator-only M10 adapter.

These are native simulator actions, not trained policy or held-out evaluations.
"""
from __future__ import annotations

import gymnasium as gym
import pytest

from bagi.experiments.m10_native_audit import NativeDoorKeyAudit


@pytest.mark.parametrize("seed", [31001, 31002])
def test_native_doorkey_audit_preserves_step_contract(seed):
    """Audit actual DoorKey transitions without supplying private state to an actor."""
    env = gym.make("MiniGrid-DoorKey-8x8-v0")
    try:
        audit = NativeDoorKeyAudit(env)
        observation, info = audit.reset(seed=seed)
        assert isinstance(observation, dict)
        assert isinstance(info, dict)
        actions = (0, 1, 2, 3, 4, 5, 6)
        executed = 0
        for action in actions:
            observation, reward, terminated, truncated, info = audit.step(action)
            executed += 1
            assert isinstance(observation, dict)
            assert isinstance(info, dict)
            assert isinstance(reward, (float, int))
            if terminated or truncated:
                break
        summary = audit.summary()
        assert summary["native_actions"] == executed
        assert sum(summary["action_counts"].values()) == executed
        assert summary["valid_toggle_attempts"] <= summary["valid_toggle_opportunities"]
        assert summary["effective_door_open_events"] <= executed
    finally:
        env.close()
