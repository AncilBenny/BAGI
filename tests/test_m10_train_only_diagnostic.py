"""Fail-closed M10 diagnostic protocol and physical-action accounting tests."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from bagi.experiments.m10_train_only_diagnostic import (
    EXPECTED_UPDATES,
    MONITOR,
    SEEDS,
    TRAIN_STEPS,
    action_rng_seed,
    aggregate,
    decision_summary,
    validate_config,
)


def test_registered_m10_diagnostic_config_and_disjoint_layouts():
    config = yaml.safe_load(Path("configs/m10_train_only_diagnostic.yaml").read_text())
    assert validate_config(config) == config
    assert EXPECTED_UPDATES == 4096
    assert TRAIN_STEPS == 131072
    assert len(MONITOR) == len(set(MONITOR)) == 12
    assert not set(MONITOR).intersection(range(118001, 118013))
    assert action_rng_seed(SEEDS[0], MONITOR[0], "random") != action_rng_seed(
        SEEDS[0], MONITOR[0], "sampled"
    )
    for key, invalid in (("monitor_seeds", [118001]), ("model_seeds", [197]),
                         ("control", "oracle"), ("phase", "final_heldout")):
        altered = deepcopy(config)
        altered[key] = invalid
        with pytest.raises(ValueError):
            validate_config(altered)
    with pytest.raises(ValueError):
        action_rng_seed(SEEDS[0], 118001, "random")


def sample_row(*, opportunity=0, attempt=0, opening=0, goal=False):
    return {
        "native_actions": 8,
        "native_reward_sum": float(goal),
        "terminated": bool(goal),
        "full_physical_goal": bool(goal),
        "effective_pickup_events": 1,
        "valid_toggle_opportunities": opportunity,
        "valid_toggle_attempts": attempt,
        "ineffective_valid_toggles": int(bool(attempt and not opening)),
        "effective_door_open_events": opening,
        "max_action_fraction": 0.5,
        "action_counts": {str(i): 8 if i == 0 else 0 for i in range(7)},
    }


def test_diagnostic_action_and_episode_denominators_do_not_conflate_events():
    totals = aggregate([sample_row(opportunity=4, attempt=1, opening=1),
                        sample_row(opportunity=0)])
    assert totals["episodes"] == 2
    assert totals["native_actions"] == 16
    assert totals["episodes_with_valid_toggle_opportunity"] == 1
    assert totals["valid_toggle_opportunities"] == 4
    assert totals["valid_toggle_attempts"] == 1
    assert totals["episodes_with_effective_door_opening"] == 1
    assert totals["verified_full_physical_goal_episodes"] == 0
    with pytest.raises(ValueError):
        aggregate([])


def test_train_only_decision_is_not_physical_success_or_causal_proof():
    empty = aggregate([sample_row()])
    valid = aggregate([sample_row(opportunity=2)])
    records = [
        {"random_control": {"totals": empty},
         "monitor_after_deterministic": {"totals": valid}}
        for _ in SEEDS
    ]
    decision = decision_summary(records)
    assert decision["trained_valid_toggle_opportunities"] == 4
    assert decision["trained_valid_toggle_attempts"] == 0
    assert decision["diagnostic_priority"] == "action_selection_or_precondition_hypothesis"
    assert decision["not_independent_evaluation"] is True
    assert decision["not_causal_identification"] is True
    with pytest.raises(ValueError):
        decision_summary(records[:1])
