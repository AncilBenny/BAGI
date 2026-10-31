"""R5 regression: immutable-source *evaluator-only* public aliasing checks.

Do not rerun the already completed 176-native-action witness as part of
ordinary CI: repeating it on push and PR is not independent study evidence.
The native R5 entrypoint remains separately callable with its explicit flag.
"""

import numpy as np
import pytest

from bagi.experiments import g2_e5_p2_workshop_reachability_audit as p2
from bagi.experiments.g2_e6_r5_memory_witness import (
    SOURCE_GIT_BLOB_SHA1,
    action_program,
)
from bagi.worlds.workshop import WorkshopEnv


def test_source_pin_matches_verified_workshop():
    p2.verify_source()
    assert SOURCE_GIT_BLOB_SHA1 == p2.SOURCE_GIT_BLOB_SHA1


def test_mirror_preserves_bounded_action_budget():
    for collect_key in (False, True):
        original = action_program("default", collect_key)
        mirror = action_program("mirror_x", collect_key)
        assert len(original) == len(mirror) == 11
        assert mirror == tuple((0, 3, 2, 1)[a] for a in original)


@pytest.mark.parametrize("size", (9, 11))
@pytest.mark.parametrize("layout", ("default", "mirror_x"))
@pytest.mark.parametrize("view_size", (3, 5))
def test_source_only_lock_alias_negative_control_without_native_steps(
    monkeypatch, size, layout, view_size
):
    """Public views from pinned source, graph edges only; ZERO env steps/resets."""
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
        # Evaluator-only counterfactuals. Reachability of these histories was
        # already demonstrated by R5 native CI on the pinned Workshop source.
        approach = world._layout_coords(5, 3)
        locked = (*approach, True, False)  # Door opened, key not collected.
        unlocked = (*approach, True, True)  # Same position, key collected.
        action = action_program(layout, True)[-1]
        before_locked = p2._view(world, locked)
        before_unlocked = p2._view(world, unlocked)
        assert (before_locked == before_unlocked) is (view_size == 3)

        # Evaluate source-pinned collision/effect graph, NOT world.step().
        after_locked = p2._next_state(world, locked, action)
        after_unlocked = p2._next_state(world, unlocked, action)
        assert after_locked[:2] == approach
        assert after_unlocked[:2] == tuple(map(int, world._lock))
        assert after_locked[3] is False
        assert after_unlocked[3] is True
        assert p2._view(world, after_locked) == before_locked
        assert p2._view(world, after_unlocked) != before_unlocked
        assert np.frombuffer(before_locked, dtype=np.int8).shape == (
            view_size * view_size,
        )
    finally:
        world.close()
