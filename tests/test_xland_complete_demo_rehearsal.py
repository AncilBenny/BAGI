"""REUSE-35 contracts on numeric replay; tests do not show learned skill."""
import numpy as np
import pytest

from bagi.experiments import xland_complete_demo_bc as original
from bagi.experiments.xland_complete_demo_rehearsal import (
    MIXED_ARM,
    REHEARSAL_LAYOUTS,
    check_registration,
    evaluate_original,
    sample_mixed_training_rows,
)


def test_single_fixed_124_124_ratio_and_new_layouts():
    check_registration()
    assert REHEARSAL_LAYOUTS == tuple(range(10800, 10810))
    assert set(REHEARSAL_LAYOUTS).isdisjoint(original.TRAIN_LAYOUTS)
    assert set(REHEARSAL_LAYOUTS).isdisjoint(original.EVAL_LAYOUTS)
    assert MIXED_ARM == "half_original_half_complete_teacher"


def sample(n):
    return {
        "obs": np.zeros((n, 115), dtype=np.float32),
        "expert_label": np.arange(n, dtype=np.int64) % 5,
        "done": np.zeros(n, dtype=bool),
    }


def test_equal_exposure_and_deterministic_original_base_replay():
    base, demo = sample(291), sample(248)
    a = sample_mixed_training_rows(base, demo, 17)
    b = sample_mixed_training_rows(base, demo, 17)
    assert a["obs"].shape == (539, 115)
    assert a["expert_label"].shape == (539,)
    assert np.array_equal(a["expert_label"], b["expert_label"])
    original_base = original.sampled_indices(base["expert_label"], 291, 17)
    assert np.array_equal(a["expert_label"][:291],
                          base["expert_label"][original_base])
    assert len(a["expert_label"][291:415]) == 124
    assert len(a["expert_label"][415:]) == 124


def test_invalid_source_shape_or_seed_fails_closed():
    base, demo = sample(291), sample(248)
    with pytest.raises(ValueError, match="shapes or model seed"):
        sample_mixed_training_rows(base, demo, 999)
    with pytest.raises(ValueError, match="shapes or model seed"):
        sample_mixed_training_rows(sample(290), demo, 17)


def test_native_evaluator_uses_only_new_seeds_and_restores(monkeypatch):
    def native_mock(policy):
        assert original.EVAL_LAYOUTS == REHEARSAL_LAYOUTS
        return {task: {"cases": [{"layout": s} for s in original.EVAL_LAYOUTS]}
                for task in ("hold", "near", "putdown")}

    monkeypatch.setattr(original, "evaluate_policy", native_mock)
    result = evaluate_original(object())
    assert len(result["putdown"]["cases"]) == 10
    assert original.EVAL_LAYOUTS == tuple(range(10600, 10610))
