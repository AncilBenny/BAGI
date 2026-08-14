import numpy as np

from bagi.worlds import WorkshopEnv


def move(env, actions):
    result = None
    for action in actions:
        result = env.step(action)
    return result


def test_switch_opens_door():
    env = WorkshopEnv()
    env.reset(seed=1)
    _, _, _, _, info = move(env, [1, 1])  # right, right -> switch
    assert info["door_open"] is True
    assert "switch_activated" in info["events"]


def test_key_allows_lock_passage():
    env = WorkshopEnv()
    env.reset(seed=1)
    # down x4 reaches key at [5,1]
    _, _, _, _, info = move(env, [2, 2, 2, 2])
    assert info["has_key"] is True
    assert "key_picked_up" in info["events"]

    # right x3 reaches the lock at [5,4]
    _, _, _, _, info = move(env, [1, 1, 1])
    assert np.array_equal(info["position"], np.array([5, 4]))
    assert "lock_opened" in info["events"]


def test_lock_blocks_agent_without_key():
    env = WorkshopEnv()
    env.reset(seed=1)
    # approach the lock from above without touching the key or door
    _, _, _, _, info = move(env, [2, 2, 2, 1, 1, 1])
    before = info["position"].copy()
    _, _, _, _, info = env.step(2)
    assert np.array_equal(info["position"], before)
    assert "blocked" in info["events"]
