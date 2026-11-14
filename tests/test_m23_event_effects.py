"""M23 public effect labels vs independent evaluator-native effects."""

import numpy as np
import pytest
from minigrid.core.constants import OBJECT_TO_IDX

from bagi.experiments.m23_event_effects import (
    BATCH_SIZE,
    EPOCHS,
    PublicActionEffectPredictor,
    encode_events,
    event_metrics,
    event_weights,
    fit_event,
    public_effect_proxy,
    public_eligibility_rule,
)
from bagi.experiments.m23_source_feasibility import run_source_episode
from bagi.experiments.m23_study_contract import (
    BALANCE_MONITOR_LAYOUTS,
    EFFECT_MONITOR_LAYOUTS,
    FINAL_LAYOUTS,
    MEMORY_MONITOR_LAYOUTS,
    MONITOR_LAYOUTS,
    MODEL_SEEDS,
    TRAIN_LAYOUTS,
    validate_split_contract,
)


def image_with_front(name, state=0):
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    image[3, 5] = (OBJECT_TO_IDX[name], 0, state)
    return image


def public_rows():
    pre = [
        image_with_front("key"),
        image_with_front("empty"),
        image_with_front("door", 2),
        image_with_front("key"),
        image_with_front("empty"),
        image_with_front("door", 2),
        image_with_front("empty"),
        image_with_front("box"),
    ]
    post = [
        image_with_front("empty"),
        image_with_front("box"),
        image_with_front("door", 0),
        image_with_front("key"),
        image_with_front("empty"),
        image_with_front("door", 2),
        image_with_front("empty"),
        image_with_front("empty"),
    ]
    return {
        "observation_image": pre,
        "observation_direction": [0] * len(pre),
        "mission": ["get a blue ball"] * len(pre),
        "action": [3, 4, 5, 3, 4, 5, 4, 3],
        "next_image": post, "next_direction": [0] * len(pre),
        "reward": [0.0] * len(pre),
        "terminated": [False] * (len(pre) - 1) + [True],
        "truncated": [False] * len(pre),
    }


def test_public_interaction_proxy_recovers_only_visible_effect():
    rows = public_rows()
    expected = [1, 2, 3, 0, 0, 0, 0, 1]
    actual = [
        public_effect_proxy(before, action, after)
        for before, action, after in zip(
            rows["observation_image"], rows["action"], rows["next_image"], strict=True
        )
    ]
    assert actual == expected
    assert public_eligibility_rule(rows["observation_image"][3], 3) == 1
    assert public_eligibility_rule(rows["observation_image"][5], 5) == 3


def test_native_evaluator_truth_is_not_a_learner_input_or_proxy_label():
    rows = public_rows()
    rows["evaluator_native_effect"] = [0] * len(rows["action"])
    encoded_a = encode_events(rows)
    rows["evaluator_native_effect"] = [3] * len(rows["action"])
    encoded_b = encode_events(rows)
    np.testing.assert_array_equal(encoded_a["features"], encoded_b["features"])
    np.testing.assert_array_equal(encoded_a["proxy_labels"], encoded_b["proxy_labels"])
    assert (encoded_a["evaluator_native_effect"] !=
            encoded_b["evaluator_native_effect"]).all()
    assert "evaluator_native_effect" not in encode_events(public_rows())


def test_fresh_source_event_layouts_exclude_all_existing_closed_data():
    validate_split_contract()
    new = {seed for seeds in EFFECT_MONITOR_LAYOUTS.values() for seed in seeds}
    old = {
        seed
        for split in (
            TRAIN_LAYOUTS, MONITOR_LAYOUTS, BALANCE_MONITOR_LAYOUTS,
            MEMORY_MONITOR_LAYOUTS, FINAL_LAYOUTS,
        )
        for seeds in split.values() for seed in seeds
    }
    assert len(new) == 48
    assert new.isdisjoint(old)
    assert "unlockpickup" not in EFFECT_MONITOR_LAYOUTS
    with pytest.raises(ValueError, match="EFFECT_MONITOR"):
        run_source_episode("fetch", 443001, split="effect_monitor")
    with pytest.raises(ValueError):
        run_source_episode("unlockpickup", 453001, split="effect_monitor")


def test_native_event_scoring_counts_non_effect_and_rare_positive():
    truth = np.array([0, 0, 1, 2, 3], dtype=np.int64)
    perfect = event_metrics(truth, truth)
    assert perfect["macro_f1"] == 1.0
    assert perfect["native_positive_event_recall"] == 1.0
    never = event_metrics(truth, np.zeros_like(truth))
    assert never["native_positive_event_recall"] == 0.0
    assert never["macro_f1"] < perfect["macro_f1"]


def test_real_optimizer_updates_recorded_with_masked_action_ablation():
    rows = encode_events(public_rows())
    weights = event_weights(rows["proxy_labels"])
    assert all(0.25 <= w <= 8.0 for w in weights)
    trained = []
    for aware in (True, False):
        model, outcome = fit_event(rows, MODEL_SEEDS[0], aware=aware, weights=weights)
        assert isinstance(model, PublicActionEffectPredictor)
        assert not model.training
        assert outcome["actual_optimizer_step_calls"] == EPOCHS
        assert outcome["cached_train_sample_presentations"] == EPOCHS * len(
            rows["proxy_labels"]
        )
        trained.append(model)
    assert len(rows["proxy_labels"]) < BATCH_SIZE
    with pytest.raises(ValueError):
        event_weights(np.array([4], dtype=np.int64))
