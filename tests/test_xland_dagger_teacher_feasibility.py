"""Check REUSE-14 teacher limits before simulator execution."""
import pytest

from bagi.experiments.xland_dagger_teacher_feasibility import (
    WEAK_PREFIXES,
    correction_plan,
    initial_teacher_prefix,
    require_training_task,
)


@pytest.mark.parametrize("task", ["hold", "near", "putdown"])
def test_original_training_skills_only(task):
    require_training_task(task)


@pytest.mark.parametrize("task", ["hold_then_near", "future_skill", ""])
def test_never_query_heldout_or_unapproved_task(task):
    with pytest.raises(ValueError, match="held-out"):
        require_training_task(task)
    with pytest.raises(ValueError, match="held-out"):
        correction_plan(None, task)


def test_post_event_near_and_putdown_preserve_intermediate_stage():
    assert initial_teacher_prefix("near", [1, 0, 3]) == [1, 0, 1]
    assert initial_teacher_prefix("putdown", [2, 3, 0, 4]) == [2, 3, 1]
    with pytest.raises(ValueError, match="Hold terminates"):
        initial_teacher_prefix("hold", [0, 3])


def test_teacher_rejects_invalid_post_event_recipe():
    with pytest.raises(ValueError, match="transform"):
        initial_teacher_prefix("near", [1, 3])
    with pytest.raises(ValueError, match="pickup"):
        initial_teacher_prefix("putdown", [0, 1, 4])


def test_scripted_weak_actions_are_native_and_preregistered():
    assert set(WEAK_PREFIXES) == {"weak_turn", "weak_putdown"}
    assert all(len(xs) == 6 for xs in WEAK_PREFIXES.values())
    assert all(action in range(6) for xs in WEAK_PREFIXES.values() for action in xs)
