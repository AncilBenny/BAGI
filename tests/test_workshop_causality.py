import numpy as np

from bagi.experiments.collect_guided import GUIDED_ACTIONS
from bagi.worlds import WorkshopEnv


def move(env, actions):
    result = None
    for action in actions:
        result = env.step(action)
    return result


def test_switch_opens_only_first_gate():
    env = WorkshopEnv()
    env.reset(seed=1)

    # The door at [3,1] blocks descent before the switch is activated.
    _, _, _, _, info = move(env, [2])
    assert np.array_equal(info["position"], np.array([2, 1]))
    _, _, _, _, info = env.step(2)
    assert np.array_equal(info["position"], np.array([2, 1]))
    assert "blocked" in info["events"]

    # Return to the start, activate the switch, then the same gate is passable.
    _, _, _, _, info = move(env, [0, 1, 1])
    assert info["door_open"] is True
    assert "switch_activated" in info["events"]
    _, _, _, _, info = move(env, [3, 3, 2, 2])
    assert np.array_equal(info["position"], np.array([3, 1]))


def test_key_allows_only_second_gate():
    env = WorkshopEnv()
    env.reset(seed=1)

    # Open the first gate and reach the key.
    _, _, _, _, info = move(env, [1, 1, 3, 3, 2, 2, 2, 2])
    assert info["has_key"] is True
    assert "key_picked_up" in info["events"]

    # The lock at [5,4] is the only opening through the second barrier.
    _, _, _, _, info = move(env, [1, 1, 1])
    assert np.array_equal(info["position"], np.array([5, 4]))
    assert "lock_opened" in info["events"]


def test_lock_blocks_agent_without_key():
    env = WorkshopEnv()
    env.reset(seed=1)

    # Open the first gate, enter the middle region, but approach the lock
    # without touching the key at [5,1].
    move(env, [1, 1, 3, 3, 2, 2])  # now at [3,1]
    move(env, [2, 1, 1, 1])  # now at [4,3], above-left of lock
    _, _, _, _, info = env.step(2)  # [5,3]
    before = info["position"].copy()
    _, _, _, _, info = env.step(1)  # try [5,4]
    assert np.array_equal(info["position"], before)
    assert "blocked" in info["events"]


def test_guided_curriculum_requires_full_chain_and_reaches_goal():
    env = WorkshopEnv()
    env.reset(seed=1)
    seen = []
    result = None
    for action in GUIDED_ACTIONS:
        result = env.step(action)
        seen.extend(result[4]["events"])
        if result[2] or result[3]:
            break

    assert result is not None
    _, reward, terminated, _, info = result
    assert reward == 1.0
    assert terminated is True
    assert np.array_equal(info["position"], np.array([7, 7]))
    assert [
        name
        for name in ("switch_activated", "key_picked_up", "lock_opened", "goal_reached")
        if name in seen
    ] == ["switch_activated", "key_picked_up", "lock_opened", "goal_reached"]
