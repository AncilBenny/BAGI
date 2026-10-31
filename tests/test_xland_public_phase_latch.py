"""Synthetic REUSE-38 public phase-latch contracts; no native skill claim."""
import numpy as np
import pytest

from bagi.experiments.xland_public_phase_latch import (
    ARMS,
    CARRY,
    NORMAL,
    SEEK,
    PublicPhasePolicy,
    add_option,
    next_phase,
    phases_from_executed,
    phases_from_teacher,
)


def test_attempt_latch_survives_navigation_and_resets_on_drop():
    assert next_phase(SEEK, 3, "putdown") == CARRY
    assert next_phase(CARRY, 0, "putdown") == CARRY
    assert next_phase(CARRY, 1, "putdown") == CARRY
    assert next_phase(CARRY, 4, "putdown") == SEEK
    assert next_phase(NORMAL, 3, "hold") == NORMAL
    with pytest.raises(ValueError):
        next_phase(SEEK, 6, "putdown")


def test_executed_actions_not_teacher_labels_and_episode_reset():
    tasks = ["putdown"] * 6 + ["hold", "hold"]
    stages = phases_from_executed(
        np.array([0, 3, 0, 1, 4, 0, 3, 4]),
        np.array([1, 1, 1, 1, 1, 2, 3, 3]), tasks,
    )
    assert stages.tolist() == [SEEK, SEEK, CARRY, CARRY, CARRY, SEEK, NORMAL, NORMAL]


def test_teacher_history_exact_reconstruction_and_done_reset():
    actions = np.array([0, 3, 1, 4, 3], dtype=np.int64)
    done = np.array([False, False, False, True, True])
    x = np.zeros((5, 115), dtype=np.float32)
    history = [6, 6, 6, 6]
    for i, action in enumerate(actions):
        x[i, -28:] = np.eye(7)[history].reshape(-1)
        history = [*history[1:], int(action)]
        if done[i]:
            history = [6, 6, 6, 6]
    expected = [SEEK, SEEK, CARRY, CARRY, SEEK]
    assert phases_from_teacher(x, actions, done).tolist() == expected
    x[2, -1] = 1.0
    with pytest.raises(ValueError, match="non-executed"):
        phases_from_teacher(x, actions, done)


def test_equal_width_sham_vs_public_phase_and_no_hidden_input():
    x = np.zeros((3, 115), dtype=np.float32)
    phases = np.array([NORMAL, SEEK, CARRY])
    sham = add_option(x, phases, ARMS[0])
    active = add_option(x, phases, ARMS[1])
    assert sham.shape == active.shape == (3, 118)
    np.testing.assert_array_equal(sham[:, -3:], np.eye(3)[[0, 1, 1]])
    np.testing.assert_array_equal(active[:, -3:], np.eye(3))
    np.testing.assert_array_equal(sham[:, :115], active[:, :115])


def test_public_policy_only_uses_observation_and_reset_state():
    class FakeBC:
        def __init__(self):
            self.observed = []
            self.actions = iter([3, 0, 4, 3])

        def predict(self, obs, deterministic=True):
            self.observed.append(obs.copy())
            return next(self.actions), None

    codes = {"hold": np.ones(33), "near": np.full(33, 2),
             "putdown": np.full(33, 3)}
    x = np.zeros(115, dtype=np.float32)
    x[54:87] = codes["putdown"]
    bc = FakeBC()
    policy = PublicPhasePolicy(bc, ARMS[1], codes)
    _, state = policy.predict(x, episode_start=np.array([True]))
    assert state == CARRY
    _, state = policy.predict(x, state=state, episode_start=np.array([False]))
    assert state == CARRY
    _, state = policy.predict(x, state=state)
    assert state == SEEK
    _, state = policy.predict(x, state=state, episode_start=np.array([True]))
    assert state == CARRY
    np.testing.assert_array_equal([row[-3:].argmax() for row in bc.observed],
                                  [SEEK, CARRY, CARRY, SEEK])
