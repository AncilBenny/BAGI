"""R5/R7 source-pinned public memory controls; ZERO native steps in routine CI.

The previously verified R5 native witness remains available behind its explicit
CLI execution flag; ordinary push and PR CI never call that native runner.
Evaluator-only latent fields in this file are NEVER policy/training inputs.
"""

import numpy as np
import pytest

from bagi.experiments import g2_e5_p2_workshop_reachability_audit as p2
from bagi.experiments.g2_e6_r5_memory_witness import (
    SOURCE_GIT_BLOB_SHA1,
    action_program,
)
from bagi.worlds.workshop import KEY, WorkshopEnv

LOOP_ACTIONS = (0, 2, 0, 2)  # From the approach: up, down, up, down.
HISTORY_SUFFIX = len(LOOP_ACTIONS)


def _public_from_evaluator_state(world, state):
    """Use native observation only; P2's _view helper is fixed to 5x5."""
    p2._place(world, state)
    public = np.asarray(world._observe())
    if public.shape != (world.view_size, world.view_size) or public.dtype != np.int8:
        raise ValueError("Unexpected public observation contract")
    encoded = public.tobytes()
    if world.view_size == p2.VIEW_SIZE:
        assert encoded == p2._view(world, state)
    return encoded


def _source_only_trace(world, *, collect_key):
    """Return reachable public-view/executed-action sequence without env.step."""
    frozen = action_program(world.layout_mode, collect_key)
    actions = frozen[:-1] + LOOP_ACTIONS + frozen[-1:]
    state = (*world._layout_coords(1, 1), False, False)
    trajectory = []
    for action in actions:
        before = _public_from_evaluator_state(world, state)
        trajectory.append((before, action))
        state = p2._next_state(world, state, action)
    return trajectory, state, _public_from_evaluator_state(world, state)


def test_source_pin_matches_verified_workshop():
    p2.verify_source()
    assert SOURCE_GIT_BLOB_SHA1 == p2.SOURCE_GIT_BLOB_SHA1


def test_mirror_preserves_bounded_action_budget():
    for collect_key in (False, True):
        original = action_program("default", collect_key)
        mirror = action_program("mirror_x", collect_key)
        assert len(original) == len(mirror) == 11
        assert mirror == tuple((0, 3, 2, 1)[a] for a in original)


def test_original_r5_has_one_action_history_shortcut():
    """Do not mistake registered R5 scripted action history for a GRU result."""
    for layout in ("default", "mirror_x"):
        positive = action_program(layout, True)
        negative = action_program(layout, False)
        assert positive[-1] == negative[-1]
        assert positive[-2] != negative[-2]  # One previous action suffices.


@pytest.mark.parametrize("size", (9, 11))
@pytest.mark.parametrize("layout", ("default", "mirror_x"))
@pytest.mark.parametrize("view_size", (3, 5))
def test_source_only_lock_alias_and_four_step_history_control(
    monkeypatch, size, layout, view_size
):
    """Four matched recent histories: 3x3 alias, 5x5 reactive control."""
    p2.verify_source()

    def forbid_native(*_args, **_kwargs):
        raise AssertionError("Native Workshop reset/step forbidden in CI regression")

    monkeypatch.setattr(WorkshopEnv, "reset", forbid_native)
    monkeypatch.setattr(WorkshopEnv, "step", forbid_native)
    world = WorkshopEnv(
        size=size, max_steps=40, view_size=view_size,
        start_mode="fixed", layout_mode=layout,
    )
    try:
        approach = world._layout_coords(5, 3)
        locked = (*approach, True, False)
        unlocked = (*approach, True, True)
        action = action_program(layout, True)[-1]

        # Pinned evaluator counterfactual check, never learner-only inputs.
        current_locked = _public_from_evaluator_state(world, locked)
        current_unlocked = _public_from_evaluator_state(world, unlocked)
        assert (current_locked == current_unlocked) is (view_size == 3)
        after_locked = p2._next_state(world, locked, action)
        after_unlocked = p2._next_state(world, unlocked, action)
        assert after_locked[:2] == approach
        assert after_unlocked[:2] == tuple(map(int, world._lock))
        assert after_locked[3] is False and after_unlocked[3] is True
        assert _public_from_evaluator_state(world, after_locked) == current_locked
        assert _public_from_evaluator_state(world, after_unlocked) != current_unlocked

        # New prospective source-only contrast: both routes converge and perform
        # the SAME final four actual-command suffix actions, at the same spots.
        neg_trace, neg_end, neg_post = _source_only_trace(world, collect_key=False)
        pos_trace, pos_end, pos_post = _source_only_trace(world, collect_key=True)
        # R8 negative control: merely having SEEN a public key is not evidence
        # of possessing it. Both source-legal routes observe KEY before the
        # lock; an ever-seen-KEY bit cannot separate their later effects.
        assert any(KEY in np.frombuffer(obs, dtype=np.int8) for obs, _ in neg_trace)
        assert any(KEY in np.frombuffer(obs, dtype=np.int8) for obs, _ in pos_trace)
        assert len(neg_trace) == len(pos_trace) == 11 + HISTORY_SUFFIX
        assert [action for _, action in neg_trace[-5:]] == [
            action for _, action in pos_trace[-5:]
        ]
        assert [action for _, action in pos_trace[-5:-1]] == list(LOOP_ACTIONS)
        # A fifth preceding action still reveals which scripted route was used.
        assert neg_trace[-6][1] != pos_trace[-6][1]
        assert neg_trace[-1][1] == pos_trace[-1][1] == action

        # Includes the four most recent public pre-action observation/action
        # pairs AND the current public observation and next executed action.
        # Match in 3x3, differ in 5x5; outcomes physically differ in source.
        assert (neg_trace[-5:] == pos_trace[-5:]) is (view_size == 3)
        assert neg_end[:2] == approach
        assert pos_end[:2] == tuple(map(int, world._lock))
        assert neg_post == neg_trace[-1][0]
        assert pos_post != pos_trace[-1][0]
        assert neg_end[3] is False and pos_end[3] is True
    finally:
        world.close()
