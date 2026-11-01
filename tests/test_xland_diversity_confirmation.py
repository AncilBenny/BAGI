"""REUSE-37 contract tests only; no simulator or learned-skill claims."""
import json

import pytest

from bagi.experiments import xland_complete_demo_bc as original
from bagi.experiments import xland_diverse_physical_demos as diversity
from bagi.experiments.xland_diversity_confirmation import (
    CONFIRM_EVAL_LAYOUTS,
    check_confirmation_registration,
    evaluate_fresh,
    require_previous_training,
)


def test_frozen_20_untouched_layouts():
    check_confirmation_registration()
    assert CONFIRM_EVAL_LAYOUTS == tuple(range(11100, 11120))
    assert set(CONFIRM_EVAL_LAYOUTS).isdisjoint(diversity.DIVERSE_TRAIN_LAYOUTS)
    assert set(CONFIRM_EVAL_LAYOUTS).isdisjoint(diversity.DIVERSE_EVAL_LAYOUTS)
    assert set(CONFIRM_EVAL_LAYOUTS).isdisjoint(original.TRAIN_LAYOUTS)


def test_json_int_label_key_normalization_and_fixed_787_budget():
    training = {
        "total_rows": 787, "epochs": 12,
        "nominal_optimization_exposures": 9444,
        "training_action_counts": {3: 250, 4: 200, 1: 337},
    }
    archived = {"original_source_sha256": "verified-sha",
                "training": json.loads(json.dumps(training))}
    require_previous_training({"results": [dict(archived,
                              model_seed=17, arm=diversity.ARMS[0])]},
                              17, diversity.ARMS[0], training, "verified-sha")
    modified = dict(training, nominal_optimization_exposures=9445)
    with pytest.raises(ValueError, match="published REUSE-36"):
        require_previous_training({"results": [dict(archived,
                                  model_seed=17, arm=diversity.ARMS[0])]},
                                  17, diversity.ARMS[0], modified, "verified-sha")
    with pytest.raises(ValueError, match="published REUSE-36"):
        require_previous_training({"results": [dict(archived,
                                  model_seed=17, arm=diversity.ARMS[0])]},
                                  17, diversity.ARMS[0], training, "WRONG-SHA")


def test_physical_evaluator_runs_only_new_layouts_and_restores(monkeypatch):
    def fake(policy):
        assert original.EVAL_LAYOUTS == CONFIRM_EVAL_LAYOUTS
        return {skill: {"cases": [{"layout": x} for x in original.EVAL_LAYOUTS]}
                for skill in ("hold", "near", "putdown")}

    monkeypatch.setattr(original, "evaluate_policy", fake)
    out = evaluate_fresh(object())
    assert len(out["putdown"]["cases"]) == 20
    assert original.EVAL_LAYOUTS == tuple(range(10600, 10610))
