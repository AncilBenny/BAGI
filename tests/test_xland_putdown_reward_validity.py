"""Train-skill score requires physical manipulation, not just goal reward."""
import pytest

from bagi.experiments.xland_putdown_reward_validity import (
    parse_saved_reference,
    physically_nonvacuous,
)


def test_reward_without_physical_pickup_or_drop_not_counted_as_skill():
    assert not physically_nonvacuous(1, True, 0, 0)
    assert not physically_nonvacuous(1, False, 1, 0)
    assert not physically_nonvacuous(1, False, 0, 1)
    assert not physically_nonvacuous(0, False, 1, 1)
    assert physically_nonvacuous(1, False, 1, 1)


def test_saved_source_native_trace_requires_grounded_learned_reward():
    document = {
        "policy_seed": 29,
        "independent_reference_layout": {
            "layout_seed": 9604,
            "reward_success": 1,
            "real_steps": 2,
            "executed_actions": [0, 4],
        },
        "new_online_policy_training_steps": 0,
        "heldout_composition_teacher_queries": 0,
    }
    assert parse_saved_reference(document) == [0, 4]
    document["independent_reference_layout"]["reward_success"] = 0
    with pytest.raises(ValueError, match="source"):
        parse_saved_reference(document)
