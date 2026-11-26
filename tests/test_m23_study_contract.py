"""Pure M23 protocol checks; these are NOT native MiniGrid physics tests."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "bagi/experiments/m23_study_contract.py"
SPEC = importlib.util.spec_from_file_location("m23_study_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def valid_witness():
    return module.UnlockPickupWitness(
        initially_locked=True,
        box_initially_behind_locked_door=True,
        initial_goal_already_satisfied=False,
        actual_forward_movements=4,
        key_pickup_action_index=5,
        door_unlock_action_index=17,
        key_drop_action_index=20,
        correct_box_pickup_action_index=42,
        reward_at_terminal=0.8,
    )


def test_seed_splits_do_not_overlap():
    module.validate_split_contract()


def test_witness_requires_real_ordered_effects():
    assert valid_witness().physical_success()
    for field, value in (
        ("initially_locked", False),
        ("box_initially_behind_locked_door", False),
        ("initial_goal_already_satisfied", True),
        ("actual_forward_movements", 0),
        ("key_pickup_action_index", None),
        ("door_unlock_action_index", None),
        ("correct_box_pickup_action_index", None),
        ("key_drop_action_index", None),
        ("key_drop_action_index", 12),
        ("reward_at_terminal", 0.0),
        ("door_unlock_action_index", 4),
        ("correct_box_pickup_action_index", 12),
    ):
        assert not replace(valid_witness(), **{field: value}).physical_success(), field


def test_primary_gate_is_conjunctive():
    baseline = (3, 3, 3, 3, 3, 3, 3, 3)
    assert module.primary_zero_shot_gate((6, 6, 6, 6, 5, 5, 3, 3), baseline, baseline)
    assert not module.primary_zero_shot_gate((8, 8, 8, 8, 3, 3, 3, 3), baseline, baseline)
    assert not module.primary_zero_shot_gate((5, 5, 5, 5, 5, 5, 3, 3), baseline, baseline)
    assert not module.primary_zero_shot_gate((6, 6, 6, 6, 5, 5, 3, 3), baseline, (4,) * 8)


def test_invalid_counts_rejected():
    with pytest.raises(ValueError):
        module.primary_zero_shot_gate((1,) * 7, (0,) * 8, (0,) * 8)
    with pytest.raises(ValueError):
        module.primary_zero_shot_gate((33,) * 8, (0,) * 8, (0,) * 8)
