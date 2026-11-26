"""REUSE-38 public-action phase memory for a matched imitation.BC comparison.

This is a deliberately fallible, action-triggered option-stage adapter, NOT
an inventory sensor. Only truly executed actions may change its stage.
Native pickup/drop *attempts* can fail; stage is therefore not ground truth.
"""
from __future__ import annotations

import numpy as np

WIDTH = 115
OPTION_WIDTH = 3
NORMAL, SEEK, CARRY = range(3)
ARMS = ("same_width_sham_seek", "public_action_phase_latch")
TASKS = ("hold", "near", "putdown")


def next_phase(phase: int, executed_action: int, task: str) -> int:
    """Carry an action-only memory across arbitrary navigation lengths."""
    if task not in TASKS or phase not in (NORMAL, SEEK, CARRY):
        raise ValueError("Unknown public task or option stage")
    if executed_action not in range(6):
        raise ValueError("Non-native executed action")
    if task != "putdown":
        return NORMAL
    if executed_action == 3:
        return CARRY
    if executed_action == 4:
        return SEEK
    return phase


def phases_from_executed(
    actions: np.ndarray, episodes: np.ndarray, tasks: list[str],
) -> np.ndarray:
    """Derive pre-action stages from actual TRAIN executions, never labels."""
    a = np.asarray(actions)
    e = np.asarray(episodes)
    if (a.ndim != 1 or e.shape != a.shape or len(tasks) != len(a)
            or not np.issubdtype(a.dtype, np.integer)):
        raise ValueError("Malformed executed-action episode provenance")
    stages = np.empty(len(a), dtype=np.int8)
    previous = None
    phase = NORMAL
    for i, (action, episode, task) in enumerate(zip(a, e, tasks)):
        if episode != previous:
            phase = SEEK if task == "putdown" else NORMAL
            previous = episode
        if task not in TASKS:
            raise ValueError("Unsupported public source task")
        stages[i] = phase
        phase = next_phase(phase, int(action), task)
    return stages


def phases_from_teacher(
    features: np.ndarray, executed_labels: np.ndarray, done: np.ndarray,
) -> np.ndarray:
    """Reconstruct teacher stage and verify every real four-action history.

    In these source-only full demonstrations, each label was actually
    executed; that property MUST NOT be assumed for DAgger source rows.
    """
    x = np.asarray(features)
    a = np.asarray(executed_labels)
    d = np.asarray(done)
    if (x.ndim != 2 or x.shape[1] != WIDTH
            or a.shape != (len(x),) or d.shape != (len(x),)
            or a.dtype.kind not in "iu" or d.dtype != np.bool_
            or not np.isfinite(x).all()):
        raise ValueError("Malformed numeric physical teacher trajectory")
    phase = SEEK
    history = (6, 6, 6, 6)
    stages = np.empty(len(x), dtype=np.int8)
    for i, action in enumerate(a):
        expected = np.eye(7, dtype=np.float32)[list(history)].reshape(-1)
        if not np.array_equal(x[i, -28:], expected):
            raise ValueError("Teacher row has non-executed action history")
        stages[i] = phase
        phase = next_phase(phase, int(action), "putdown")
        history = (*history[1:], int(action))
        if bool(d[i]):
            history = (6, 6, 6, 6)
            phase = SEEK
    return stages


def add_option(features: np.ndarray, phases: np.ndarray, arm: str) -> np.ndarray:
    """Same-width, same-parameter-count inputs for both BC arms."""
    x = np.asarray(features, dtype=np.float32)
    p = np.asarray(phases)
    if (arm not in ARMS or x.ndim != 2 or x.shape[1] != WIDTH
            or p.shape != (len(x),) or not np.isfinite(x).all()
            or np.any((p < NORMAL) | (p > CARRY))):
        raise ValueError("Invalid public option features")
    if arm == ARMS[0]:
        p = np.where(p == NORMAL, NORMAL, SEEK)
    out = np.concatenate((x, np.eye(OPTION_WIDTH, dtype=np.float32)[p]), axis=1)
    if out.shape != (len(x), WIDTH + OPTION_WIDTH):
        raise AssertionError("Unexpected published-BC observation width")
    return out


class PublicPhasePolicy:
    """Adapter for the existing native evaluator's episode_start/state API."""

    def __init__(self, policy, arm: str, public_task_codes: dict[str, np.ndarray]):
        if arm not in ARMS or set(public_task_codes) != set(TASKS):
            raise ValueError("Unregistered public task codes or phase arm")
        codes = {task: np.asarray(code, dtype=np.float32)
                 for task, code in public_task_codes.items()}
        if any(code.shape != (33,) for code in codes.values()):
            raise ValueError("Expected public 33-feature goal/rule instructions")
        if len({code.tobytes() for code in codes.values()}) != len(TASKS):
            raise ValueError("Public tasks cannot be distinguished")
        self.policy = policy
        self.arm = arm
        self.codes = codes

    def predict(self, observation, state=None, episode_start=None,
                deterministic=True):
        x = np.asarray(observation, dtype=np.float32)
        if x.shape != (WIDTH,) or not np.isfinite(x).all():
            raise ValueError("Expected exactly 115 public observation features")
        matches = [task for task, code in self.codes.items()
                   if np.array_equal(x[54:87], code)]
        if len(matches) != 1:
            raise ValueError("Public goal/rule encoding not recognized")
        task = matches[0]
        start = episode_start is not None and bool(np.asarray(episode_start).all())
        if start or state is None:
            phase = SEEK if task == "putdown" else NORMAL
        else:
            phase = int(state)
        if task != "putdown":
            phase = NORMAL
        selected, _ = self.policy.predict(
            add_option(x[None, :], np.asarray([phase]), self.arm)[0],
            deterministic=deterministic,
        )
        action = int(np.asarray(selected).item())
        return selected, next_phase(phase, action, task)
