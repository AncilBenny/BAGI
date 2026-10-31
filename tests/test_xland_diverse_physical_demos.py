"""REUSE-36 pure source/budget controls; no simulated-learning claims."""

import numpy as np
import pytest

from bagi.experiments import xland_complete_demo_bc as original
from bagi.experiments.xland_diverse_physical_demos import (
    DIVERSE_EVAL_LAYOUTS,
    DIVERSE_TRAIN_LAYOUTS,
    evaluate_original,
    sample_fixed_extra,
    validate_register,
)


def test_fixed_72_source_layouts_and_new_20_test_layouts():
    validate_register()
    assert DIVERSE_TRAIN_LAYOUTS == tuple(range(10900, 10972))
    assert DIVERSE_EVAL_LAYOUTS == tuple(range(11000, 11020))
    assert set(DIVERSE_EVAL_LAYOUTS).isdisjoint(DIVERSE_TRAIN_LAYOUTS)
    assert set(DIVERSE_EVAL_LAYOUTS).isdisjoint(original.TRAIN_LAYOUTS)


def dataset(n):
    return {"obs": np.zeros((n, 115), dtype=np.float32),
            "expert_label": np.arange(n, dtype=np.int64) % 5,
            "done": np.zeros(n, dtype=bool)}


def test_balanced_source_rehearsal_exact_787_rows_and_determinism():
    original_data, complete = dataset(291), dataset(600)
    first = sample_fixed_extra(original_data, complete, 17)
    again = sample_fixed_extra(original_data, complete, 17)
    assert first["obs"].shape == (787, 115)
    assert len(first["expert_label"][291:539]) == 248
    assert len(first["expert_label"][539:]) == 248
    assert np.array_equal(first["expert_label"], again["expert_label"])
    base = original.sampled_indices(original_data["expert_label"], 291, 17)
    assert np.array_equal(first["expert_label"][:291],
                          original_data["expert_label"][base])


def test_reject_small_demo_or_unregistered_policy_seed():
    base = dataset(291)
    with pytest.raises(ValueError, match="Invalid equal exposure"):
        sample_fixed_extra(base, dataset(200), 17)
    with pytest.raises(ValueError, match="Invalid equal exposure"):
        sample_fixed_extra(base, dataset(500), 99)


def test_unchanged_original_evaluator_on_new_disjoint_seeds(monkeypatch):
    def mock(policy):
        assert original.EVAL_LAYOUTS == DIVERSE_EVAL_LAYOUTS
        return {task: {"cases": [{"layout": seed} for seed in original.EVAL_LAYOUTS]}
                for task in ("hold", "near", "putdown")}

    monkeypatch.setattr(original, "evaluate_policy", mock)
    result = evaluate_original(object())
    assert len(result["putdown"]["cases"]) == 20
    assert original.EVAL_LAYOUTS == tuple(range(10600, 10610))


def test_diverse_source_rejects_insufficient_complete_teacher_phases():
    from bagi.experiments.xland_diverse_physical_demos import (
        verify_diverse_source,
    )

    cases = [{"layout": seed, "native_steps": 1, "queries": 1,
              "physical": index < 49}
             for index, seed in enumerate(DIVERSE_TRAIN_LAYOUTS)]
    rows = [{"public_obs": np.zeros(115, dtype=np.float32),
             "expert_label": 3, "stage": "seek_square",
             "layout": DIVERSE_TRAIN_LAYOUTS[0]}]
    audit = {"predeclared_source_layouts": list(DIVERSE_TRAIN_LAYOUTS),
             "cases": cases, "eligible_physical_episode_count": 49,
             "native_teacher_actions": 72,
             "attempted_teacher_queries": 72, "label_rows": len(rows)}
    with pytest.raises(ValueError, match="genuine|complete|enough|Insufficient"):
        verify_diverse_source(rows, audit)


def test_diverse_source_rejects_unregistered_source_layout():
    from bagi.experiments.xland_diverse_physical_demos import verify_diverse_source

    cases = [{"layout": seed, "native_steps": 2, "queries": 2,
              "physical": True}
             for seed in DIVERSE_TRAIN_LAYOUTS]
    cases[0]["layout"] = 999
    rows = [{"public_obs": np.zeros(115, dtype=np.float32),
             "expert_label": 3, "stage": "seek_square",
             "layout": DIVERSE_TRAIN_LAYOUTS[0]}]
    audit = {"predeclared_source_layouts": list(DIVERSE_TRAIN_LAYOUTS),
             "cases": cases, "eligible_physical_episode_count": 72,
             "native_teacher_actions": 144,
             "attempted_teacher_queries": 144, "label_rows": len(rows)}
    with pytest.raises(ValueError, match="genuine|complete|enough|Insufficient"):
        verify_diverse_source(rows, audit)



def test_successful_real_source_phase_gate_returns_numeric_mapping(monkeypatch):
    """Catches accidental (audit, data) tuples before any BC training."""
    from bagi.experiments.xland_diverse_physical_demos import verify_diverse_source

    monkeypatch.setattr(original, "validate_demonstrations",
                        lambda rows, audit: None)
    cases = [{"layout": layout, "native_steps": 2, "queries": 2,
              "physical": i < 66}
             for i, layout in enumerate(DIVERSE_TRAIN_LAYOUTS)]
    rows = [
        {"layout": DIVERSE_TRAIN_LAYOUTS[0],
         "public_obs": np.zeros(115, dtype=np.float32),
         "expert_label": 3 if i < 66 else 4,
         "done": i == 131,
         "stage": "seek_square" if i < 66 else "carrying_square"}
        for i in range(132)
    ]
    report = {"predeclared_source_layouts": list(DIVERSE_TRAIN_LAYOUTS),
              "cases": cases, "eligible_physical_episode_count": 66,
              "native_teacher_actions": 144, "attempted_teacher_queries": 144,
              "label_rows": 132}
    saved = verify_diverse_source(rows, report)
    assert isinstance(saved, dict)
    assert set(saved) == {"obs", "expert_label", "done"}
    assert saved["obs"].shape == (132, 115)
    assert int(np.sum(saved["expert_label"] == 3)) == 66
    assert int(np.sum(saved["expert_label"] == 4)) == 66
