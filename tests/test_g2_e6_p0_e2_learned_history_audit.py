"""Synthetic-only Workshop history-audit contracts; never load EVAL archives."""

from __future__ import annotations

import numpy as np
import pytest

from bagi.experiments import g2_e6_p0_e2_learned_history_audit as study


def _rows() -> dict[str, np.ndarray]:
    # Four episodes, two per learned-policy seed; current view 2 repeats
    # with distinct futures 3/5 and prior public views 1/4.
    sequences = ((0, 1, 2, 3), (0, 4, 2, 5),
                 (0, 1, 2, 3), (0, 4, 2, 5))
    views = np.array([np.full((5, 5), x, dtype=np.int8)
                      for sequence in sequences for x in sequence])
    after = np.array([np.full((5, 5), x, dtype=np.int8)
                      for sequence in sequences for x in (*sequence[1:], sequence[-1])])
    return {"observation": views, "next_observation": after,
            "action": np.zeros(16, dtype=np.int8),
            "episode_index": np.repeat([0, 1, 92, 93], 4).astype(np.int16),
            "seed_index": np.repeat([0, 0, 1, 1], 4).astype(np.int8),
            "actor_index": np.zeros(16, dtype=np.int8)}


def test_counts_are_per_policy_seed_and_histories_not_fitted():
    result = study.audit_train_arrays(_rows())
    assert result["model_fits"] == result["new_native_actions"] == 0
    assert result["original_eval_arrays_read"] == 0
    for s in ("0", "1"):
        r = result["seeds"][s]
        assert r["actions"] == 8 and r["episodes"] == 2
        assert r["public_change_positive"] == 6
        assert r["public_change_negative"] == 2
        assert r["ambiguous_next_view_keys"] >= 1
        assert r["ambiguous_keys_with_distinct_prior_public_views"] >= 1
        assert r["ambiguous_keys_resolved_by_prior_views_in_sample"] >= 1
        assert r["cross_episode_ambiguous_keys"] >= 1


def test_rejects_eval_actor_and_private_field():
    rows = _rows()
    rows["actor_index"][0] = 1
    with pytest.raises(ValueError, match="Invalid public TRAIN-only"):
        study.audit_train_arrays(rows)
    rows = _rows()
    rows["private_pocket"] = np.zeros(16, dtype=np.int8)
    with pytest.raises(ValueError, match="Unexpected TRAIN fields"):
        study.audit_train_arrays(rows)


def test_rejects_broken_episode_chain_and_interleaved_seed():
    rows = _rows()
    rows["next_observation"][0, 0, 0] = 8
    with pytest.raises(ValueError, match="Broken within-episode"):
        study.audit_train_arrays(rows)
    rows = _rows()
    rows["seed_index"][5] = 1
    with pytest.raises(ValueError, match="Interleaved policy seed"):
        study.audit_train_arrays(rows)


def test_requires_strict_isolated_loader(monkeypatch, tmp_path):
    def reject(path):
        assert path.name == "train_only.zip"
        raise ValueError("Isolated TRAIN archive SHA256 mismatch")
    monkeypatch.setattr(study, "load_isolated_train", reject)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        study.run(tmp_path / "train_only.zip")
