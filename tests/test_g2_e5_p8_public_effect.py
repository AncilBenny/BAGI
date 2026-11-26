"""Simulator-free P8 controls and leakage gate; no original EVAL data read."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from bagi.experiments import g2_e5_p8_public_effect as study


def _toy() -> study.PublicTransitions:
    observation = np.array([[1, 0], [1, 1], [0, 2], [1, 1], [1, 0], [0, 2]],
                           dtype=np.float32)
    next_obs = np.array([[1, 1], [1, 1], [0, 0], [1, 2], [1, 0], [0, 1]],
                        dtype=np.float32)
    actions = np.array([0, 1, 2, 0, 1, 2], dtype=np.int8)
    episode = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    folds = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    return study._dataset("toy", observation, actions, next_obs, episode, folds, 3)


def test_public_change_uses_only_next_public_view():
    data = _toy()
    assert data.changed.tolist() == [1, 0, 1, 1, 0, 1]
    assert data.observation.shape == (6, 2)
    assert data.fold.tolist() == [0, 0, 0, 1, 1, 1]
    assert data.source_group.tolist() == data.episode.tolist()


def test_disjoint_groups_validation_and_invalid_source_rejected():
    data = _toy()
    with pytest.raises(ValueError, match="crosses"):
        study._dataset("invalid", data.observation, data.action,
                       data.observation, data.episode,
                       np.array([0, 0, 1, 1, 1, 1], dtype=np.int8), 3)
    with pytest.raises(ValueError, match="layout crosses"):
        study._dataset("invalid", data.observation, data.action,
                       data.observation, data.episode, data.fold, 3,
                       source_group=np.zeros(6, dtype=np.int64))
    with pytest.raises(ValueError, match="Invalid"):
        study._dataset("invalid", data.observation, np.array([0, 1, 8, 0, 1, 2]),
                       data.observation, data.episode, data.fold, 3)


def test_frozen_laplace_global_and_zero_controls():
    data = _toy()
    train = np.array([0, 1, 2])
    test = np.array([3, 4, 5])
    assert study._baseline(data, train, test, "laplace_per_action").tolist() == [
        2 / 3, 1 / 3, 2 / 3,
    ]
    assert np.allclose(study._baseline(data, train, test, "global_train_rate"), 2 / 3)
    assert study._baseline(data, train, test, "always_no_change").tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match="Unregistered"):
        study._baseline(data, train, test, "oracle")


def test_matched_model_arms_are_deterministic_and_use_real_held_action():
    torch.set_num_threads(1)
    data = _toy()
    train = np.array([0, 1, 2])
    held = np.array([3, 4, 5])
    for arm in study.ARMS:
        a, steps_a = study._fit(data, train, held, arm, seed=17, fold=1)
        b, steps_b = study._fit(data, train, held, arm, seed=17, fold=1)
        assert steps_a == steps_b == study.EPOCHS
        assert a.shape == (3,)
        assert np.array_equal(a, b)
        assert np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all()
    with pytest.raises(ValueError, match="Unregistered"):
        study._fit(data, train, held, "real_future_state", seed=17, fold=0)


def test_metrics_are_finite_and_count_actions_groups_and_ece():
    p = np.array([0, 1, 0.5, 0.25], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.float32)
    actions = np.array([0, 1, 1, 0])
    episodes = np.array([0, 0, 1, 1])
    got = study._metrics(p, y, actions, episodes, 3)
    assert got["brier"] == pytest.approx((0.25 + 0.25 ** 2) / 4)
    assert got["n_rows"] == 4 and got["positive_rows"] == 2
    assert got["per_action"]["2"]["rows"] == 0
    assert got["per_action"]["2"]["brier"] is None
    assert got["n_held_groups"] == 2
    assert 0 <= got["ece10"] <= 1


def test_source_loader_isolated_e2_not_mixed_archive(monkeypatch, tmp_path):
    e1 = {}
    for domain in ("xland", "workshop"):
        pre = f"{domain}_train_"
        count = 115 if domain == "xland" else 25
        e1[pre + "observations"] = np.zeros((4, count), dtype=np.float32)
        e1[pre + "actions"] = np.array([0, 1, 1, 2], dtype=np.int64)
        e1[pre + "groups"] = np.array([0, 0, 1, 1], dtype=np.int64)
        e1[pre + "episode_ids"] = np.array([
            "xland-11900-17-hold-BC", "xland-11901-17-hold-BC",
        ] if domain == "xland" else [
            "workshop-default-uniform_random-12100",
            "workshop-default-uniform_random-12101",
        ])
        e1[pre + "behaviors"] = np.array(["uniform_random", "uniform_random"])
    called = []

    def fake_e1(_path, kind):
        called.append(kind)
        return e1

    def fake_isolated(_path):
        called.append("isolated")
        return {
            "observation": np.zeros((8, 5, 5), dtype=np.int8),
            "next_observation": np.zeros((8, 5, 5), dtype=np.int8),
            "action": np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8),
            "episode_index": np.array([0, 0, 1, 1, 92, 92, 93, 93]),
            "seed_index": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        }

    monkeypatch.setattr(study, "_safe_archive", fake_e1)
    monkeypatch.setattr(study, "load_isolated_train", fake_isolated)
    # Frozen exact row contracts intentionally reject synthetic four-row data.
    with pytest.raises(ValueError, match="counts changed"):
        study._frozen_datasets(tmp_path / "e1.zip", tmp_path / "isolated.zip")
    assert called == ["e1", "isolated"]
