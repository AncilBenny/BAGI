"""R5 paired public observation witness; no fitting or teacher use."""

import numpy as np

from bagi.experiments.g2_e6_r5_memory_witness import action_program, run


def test_mirror_preserves_bounded_action_budget():
    for collect_key in (False, True):
        original = action_program("default", collect_key)
        mirror = action_program("mirror_x", collect_key)
        assert len(original) == len(mirror) == 11
        assert mirror == tuple((0, 3, 2, 1)[a] for a in original)


def test_true_native_paired_public_history_witness_and_reactive_control():
    outcome = run()
    assert outcome["native_step_returns"] == 176
    assert outcome["native_reset_calls"] == 16
    assert outcome["optimizer_updates"] == outcome["teacher_queries"] == 0
    assert len(outcome["cases"]) == 8
    for case in outcome["cases"]:
        positive, negative = case["positive"], case["negative"]
        assert case["same_last_executed_action"]
        assert positive["publicly_verified_entry"]
        assert not negative["publicly_verified_entry"]
        assert len(positive["public_action_history"]) == 11
        assert len(negative["public_action_history"]) == 11
        assert case["same_pre_lock_public"] == (case["view_size"] == 3)
        assert np.array_equal(
            positive["pre_lock_public"], negative["pre_lock_public"]
        ) == (case["view_size"] == 3)
        assert np.array_equal(
            negative["pre_lock_public"], negative["post_lock_public"]
        )
