"""G2-E1 synthetic-only runner contracts: no native environment interactions."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bagi.agents.public_effect_collection import PublicEpisodeAudit
from bagi.agents.public_effect_prediction import PublicEffectEpisode
from bagi.experiments.public_effect_e1_native_runner import (
    _episode_group,
    _evaluate_domain,
    _numeric_episodes,
    _safe_json,
)


def episode(layout: str, name: str, behavior: str, width: int) -> PublicEpisodeAudit:
    """Alternating synthetic labels, not real native transitions or teacher data."""
    actions = np.asarray([0, 1, 0, 1], dtype=np.int64)
    public = np.zeros((4, width), dtype=np.float32)
    public[:, 0] = [0, 1, 0, 1]
    public[:, 1] = [1, 0, 1, 0]
    labels = np.asarray([0, 1, 1, 0], dtype=np.int8)
    row = PublicEffectEpisode(layout, name, behavior, public, actions, labels)
    return PublicEpisodeAudit(row, 4, 0, 0, 0, False)


def workshop_pair():
    train = [episode("default", f"workshop-default-{behavior}-{i}",
                     behavior, 25)
             for behavior in ("uniform_random", "g1_public_script")
             for i in range(2)]
    test = [episode("mirror_x", f"workshop-mirror_x-{behavior}-{i}",
                    behavior, 25)
            for behavior in ("uniform_random", "g1_public_script")
            for i in range(2)]
    return train, test


def xland_pair():
    def rows(layout):
        return [episode(f"xland-{layout}",
                        f"xland-{layout}-{seed}-hold-{behavior}", behavior, 115)
                for seed in (17, 29)
                for behavior in ("uniform_random", "frozen_expanded_bc")]
    return rows(11900) + rows(11901), rows(12000) + rows(12001)


def test_groups_reject_wrong_source_or_task_and_do_not_conflate_behaviors():
    train, test = xland_pair()
    assert _episode_group("xland", test[0].episode) == "source_seed_17:uniform_random"
    assert _episode_group("workshop", workshop_pair()[1][0].episode) == "uniform_random"
    bad = SimpleNamespace(layout="xland-12000", episode_id="xland-12000-17-bad-uniform_random",
                          behavior="uniform_random")
    with pytest.raises(ValueError, match="Unregistered"):
        _episode_group("xland", bad)
    with pytest.raises(ValueError, match="Unknown"):
        _episode_group("unknown", train[0].episode)


def test_numeric_artifact_contains_no_pickle_object_arrays(tmp_path):
    train, _ = workshop_pair()
    numeric = {}
    _numeric_episodes("workshop", "train", train, numeric)
    path = tmp_path / "safe.npz"
    np.savez_compressed(path, **numeric)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["workshop_train_observations"].shape == (16, 25)
        assert archive["workshop_train_actions"].tolist() == [0, 1, 0, 1] * 4
        assert len(archive["workshop_train_episode_ids"]) == 4
        assert all(archive[key].dtype.kind != "O" for key in archive.files)


def test_two_seed_predictors_and_all_registered_controls_without_native_calls():
    train, test = xland_pair()
    numeric = {}
    result = _evaluate_domain("xland", train, test, numeric, epochs=2, resamples=12)
    assert result["train_rows"] == result["test_rows"] == 32
    assert result["test_public_change_positives"] == 16
    assert set(result["models"]) == {"17", "29"}
    assert len(result["test_group_episodes"]) == 4
    for seed in (17, 29):
        model = result["models"][str(seed)]
        assert model["train_episodes"] == 8
        assert set(model["slices"]) == {"all", *result["test_group_episodes"]}
        for record in model["slices"].values():
            assert set(record["metrics"]) == {
                "actual_action", "shuffled_action", "action_frequency", "constant"
            }
            assert set(record["actual_minus_baseline_brier"]) == {
                "shuffled_action", "action_frequency", "constant"
            }
        assert len(numeric[f"xland_test_seed{seed}_actual_action"]) == 32
    assert _safe_json(np.int64(3)) == 3
    with pytest.raises(TypeError, match="Unsupported"):
        _safe_json(object())


def test_train_test_overlap_fails_before_fit_or_native_calls():
    train, test = workshop_pair()
    test[0] = episode("default", "workshop-default-other-0", "uniform_random", 25)
    with pytest.raises(ValueError, match="overlap"):
        _evaluate_domain("workshop", train, test, {}, epochs=2, resamples=12)

