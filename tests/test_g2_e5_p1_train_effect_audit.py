"""Synthetic-only tests: no simulator, no PPO fitting and no TEST arrays."""

from __future__ import annotations

import numpy as np
import pytest

from bagi.experiments import g2_e5_p1_train_effect_audit as study


def _synthetic_arrays():
    n = study.NATIVE_TRAIN_ACTIONS
    episodes = np.arange(n, dtype=np.int32) // 4
    public = np.zeros((n, study.VIEW_WIDTH), dtype=np.float32)
    public[:, 0] = (np.arange(n) % 4).astype(np.float32)
    action = (np.arange(n) % study.ACTION_COUNT).astype(np.int16)
    terminated = np.zeros(n, dtype=np.int8)
    terminated[3::4] = 1
    return {
        "public_codes": public,
        "actions": action,
        "episode_index": episodes,
        "terminated": terminated,
        "truncated": np.zeros(n, dtype=np.int8),
    }


def test_frozen_registration():
    study.verify_registration()
    assert study.FOLDS == 5
    assert study.SOURCE_SHA256.startswith("15b89768")


def test_source_digest_rejected_before_zip_read(tmp_path):
    source = tmp_path / "fake.zip"
    source.write_bytes(b"not the official archived ZIP")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        study.read_train_only(source)


def test_transition_pairing_excludes_terminal_and_reset():
    a = _synthetic_arrays()
    x, action, nxt, episode = study.transitions(a)
    assert len(action) == study.NATIVE_TRAIN_ACTIONS * 3 // 4
    assert np.all(episode == (np.arange(len(episode)) // 3))
    assert np.array_equal(x[:3, 0], [0, 1, 2])
    assert np.array_equal(nxt[:3, 0], [1, 2, 3])


def test_terminal_boundary_mismatch_rejected():
    a = _synthetic_arrays()
    a["terminated"][3] = 0
    with pytest.raises(ValueError, match="Terminal flags"):
        study.transitions(a)


def test_public_shape_and_action_boundary_fail_closed():
    a = _synthetic_arrays()
    a["public_codes"] = np.zeros((study.NATIVE_TRAIN_ACTIONS, 25), dtype=np.float32)
    with pytest.raises(ValueError, match="Malformed"):
        study.transitions(a)
    a = _synthetic_arrays()
    a["actions"][0] = 7
    with pytest.raises(ValueError, match="Malformed"):
        study.transitions(a)


def test_group_disjoint_public_action_predicts_synthetic_next_view():
    # Five TRAIN groups, 12 episodes each. Current view and action jointly
    # determine the next view; action alone is deliberately ambiguous.
    n = 60 * 3
    episode = np.repeat(np.arange(60, dtype=np.int32), 3)
    view = np.zeros((n, study.VIEW_WIDTH), dtype=np.float32)
    action = np.zeros(n, dtype=np.int16)
    nxt = view.copy()
    for i in range(n):
        view[i, 0] = (i // 3) % 2
        action[i] = i % 3
        nxt[i, 0] = view[i, 0] + action[i] + 1
    result = study.crossfit(view, action, nxt, episode)
    scores = result["scores"]
    assert scores["public_view_action"]["next_view_exact_accuracy"] == 1.0
    assert scores["action_only"]["next_view_exact_accuracy"] < 0.8
    assert scores["public_view_action"]["exact_key_covered_rows"] == n


def test_shuffled_view_control_is_deterministic():
    n = 100
    episode = np.arange(n, dtype=np.int32)
    view = np.zeros((n, study.VIEW_WIDTH), dtype=np.float32)
    view[:, 0] = np.arange(n) % 4
    nxt = view.copy()
    action = np.arange(n, dtype=np.int16) % 3
    first = study.crossfit(view, action, nxt, episode)
    second = study.crossfit(view, action, nxt, episode)
    assert first == second


def test_empty_fold_rejected():
    view = np.zeros((10, study.VIEW_WIDTH), dtype=np.float32)
    with pytest.raises(ValueError, match="five-fold"):
        study.crossfit(view, np.zeros(10, dtype=np.int16), view,
                       np.zeros(10, dtype=np.int32))


def test_cross_seed_train_transfer_has_action_only_control():
    n = 40
    source = np.zeros((n, study.VIEW_WIDTH), dtype=np.float32)
    target = np.zeros((n, study.VIEW_WIDTH), dtype=np.float32)
    source[:, 0] = np.arange(n) % 2
    target[:, 0] = (np.arange(n) + 1) % 2
    actions = np.arange(n, dtype=np.int16) % 3
    source_next = source.copy()
    target_next = target.copy()
    source_next[:, 0] = source[:, 0] + actions + 1
    target_next[:, 0] = target[:, 0] + actions + 1
    result = study.cross_seed_train(
        source, actions, source_next, target, actions, target_next
    )
    assert result["view_action_next_view_exact_accuracy"] == 1.0
    assert result["action_only_next_view_exact_accuracy"] < 0.8
    assert result["target_exact_key_covered_rows"] == n


def test_cross_seed_rejects_wrong_shapes():
    x = np.zeros((4, study.VIEW_WIDTH), dtype=np.float32)
    a = np.zeros(4, dtype=np.int16)
    with pytest.raises(ValueError, match="Malformed cross-seed"):
        study.cross_seed_train(x[:, :3], a, x, x, a, x)
