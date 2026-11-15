"""M23 stage-7 OOD physics stress contracts: no new model fit required."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from minigrid.core.constants import OBJECT_TO_IDX

from bagi.experiments.m23_event_effects import PublicActionEffectPredictor
from bagi.experiments.m23_public_world_model import save_numeric
from bagi.experiments.m23_rule_shift import (
    PREVIOUS_SOURCE_SHA,
    numeric_checkpoint,
    public_history_effect_rule,
    scoped_metrics,
    source_checkpoint_manifest,
)
from bagi.experiments.m23_source_feasibility import (
    choose_public_rule_shift_probe,
    run_source_episode,
)
from bagi.experiments.m23_study_contract import (
    BALANCE_MONITOR_LAYOUTS,
    EFFECT_MONITOR_LAYOUTS,
    FINAL_LAYOUTS,
    MEMORY_MONITOR_LAYOUTS,
    MONITOR_LAYOUTS,
    PROBE_MONITOR_LAYOUTS,
    PROBE_TRAIN_LAYOUTS,
    RULE_SHIFT_MONITOR_LAYOUTS,
    TRAIN_LAYOUTS,
    validate_split_contract,
)


def image(name: str, state: int = 0) -> np.ndarray:
    data = np.zeros((7, 7, 3), dtype=np.uint8)
    data[3, 5] = (OBJECT_TO_IDX[name], 0, state)
    return data


def test_new_shift_monitor_disjoint_from_every_prior_split():
    validate_split_contract()
    current = {n for seeds in RULE_SHIFT_MONITOR_LAYOUTS.values() for n in seeds}
    earlier = {
        n
        for split in (
            TRAIN_LAYOUTS,
            MONITOR_LAYOUTS,
            FINAL_LAYOUTS,
            BALANCE_MONITOR_LAYOUTS,
            MEMORY_MONITOR_LAYOUTS,
            EFFECT_MONITOR_LAYOUTS,
            PROBE_TRAIN_LAYOUTS,
            PROBE_MONITOR_LAYOUTS,
        )
        for seeds in split.values()
        for n in seeds
    }
    assert len(current) == 96
    assert not current.intersection(earlier)
    assert "unlockpickup" not in RULE_SHIFT_MONITOR_LAYOUTS
    with pytest.raises(ValueError, match="RULE_SHIFT_MONITOR"):
        run_source_episode("fetch", 542001, split="rule_shift_monitor")
    with pytest.raises(ValueError):
        run_source_episode("unlockpickup", 453001, split="rule_shift_monitor")


def test_override_uses_public_front_object_only():
    rng = np.random.default_rng(197)
    base = {"image": image("box"), "direction": 0,
            "mission": "pick up the red box"}
    assert choose_public_rule_shift_probe(base, rng) == 3
    base["image"] = image("floor")
    assert choose_public_rule_shift_probe(base, rng) == 4
    base["image"] = image("door", 2)
    assert choose_public_rule_shift_probe(base, rng) == 5
    with pytest.raises(ValueError):
        choose_public_rule_shift_probe({"image": image("key")}, rng)


def test_public_history_rule_uses_only_past_public_changes():
    before = [image("empty"), image("key"), image("empty"), image("empty")]
    after = [image("empty"), image("empty"), image("key"), image("empty")]
    rows = {"observation_image": before, "next_image": after,
            "action": [4, 3, 4, 4], "evaluator_native_effect": [3, 0, 3, 1]}
    assert public_history_effect_rule(rows).tolist() == [0, 1, 2, 0]
    # Evaluator-only native labels can be changed without affecting predictions.
    rows["evaluator_native_effect"] = [0, 1, 0, 3]
    assert public_history_effect_rule(rows).tolist() == [0, 1, 2, 0]


def test_source_models_load_only_sha_verified_numeric_weights(tmp_path: Path):
    torch.manual_seed(941)
    model = PublicActionEffectPredictor().eval()
    path = tmp_path / "probe_trained-seed941.npz"
    sha = save_numeric(model, path)
    loaded = numeric_checkpoint(tmp_path, sha, "probe_trained", 941)
    with torch.no_grad():
        test = torch.zeros((1, next(model.parameters()).shape[1]))
        torch.testing.assert_close(model(test), loaded(test))
    with pytest.raises(ValueError, match="SHA256"):
        numeric_checkpoint(tmp_path, "0" * 64, "probe_trained", 941)
    with pytest.raises(ValueError, match="Unregistered"):
        numeric_checkpoint(tmp_path, sha, "unknown", 941)


def test_unverified_or_failed_study_manifest_is_not_loadable(tmp_path: Path):
    manifest = {
        "source_commit": PREVIOUS_SOURCE_SHA,
        "registered_probe_gate_pass": True,
        "new_fresh_model_fits": 16, "actual_optimizer_step_calls": 98560,
        "target_final_episodes": 0,
        "per_seed": {str(x): {} for x in (941, 947, 953, 967, 971, 977, 983, 991)},
    }
    path = tmp_path / "m23_probe_effects-summary.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert source_checkpoint_manifest(tmp_path)["registered_probe_gate_pass"]
    manifest["registered_probe_gate_pass"] = False
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        source_checkpoint_manifest(tmp_path)


def test_hard_negative_metric_is_independent_of_reward():
    native = np.array([0, 0, 1, 2, 3], dtype=np.int64)
    predicted = np.array([0, 1, 1, 0, 3], dtype=np.int64)
    result = scoped_metrics(native, predicted)
    assert result["native_no_effect_specificity"] == 0.5
    assert result["correct_native_effect_positive_recall"] == pytest.approx(2 / 3)
    assert result["per_class"]["pickup"]["support"] == 1
