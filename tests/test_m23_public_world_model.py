"""M23 public dynamics contracts: historical public data, new fits, no targets."""

import hashlib
import json

import numpy as np
import pytest

from bagi.experiments.m23_public_world_model import (
    MODEL_SEEDS,
    WIDTH,
    PublicNextFramePredictor,
    encode_public_rows,
    fit_one,
    load_source_artifact,
    public_features,
)


def _rows():
    zeros = np.zeros((3, 7, 7, 3), dtype=np.int64)
    return {
        "observation_image": zeros,
        "observation_direction": np.array([0, 0, 0]),
        "mission": np.array(["get a blue ball"] * 3),
        "action": np.array([3, 3, 2]),
        "next_image": zeros,
        "next_direction": np.array([0, 0, 0]),
        "reward": np.array([0.0, 0.0, 0.0]),
        "terminated": np.array([True, False, True]),
        "truncated": np.array([False, False, False]),
    }


def test_public_image_history_and_executed_action_contract():
    observation = np.zeros((7, 7, 3), dtype=np.int64)
    feat = public_features(observation, 0, "get a blue ball", 3, (7, 7, 7, 7))
    assert feat.shape == (WIDTH,)
    assert not np.isnan(feat).any()
    with pytest.raises(ValueError):
        public_features(observation, 0, "get a blue ball", 7, (7, 7, 7, 7))


def test_episode_reset_prevents_future_action_history_leakage():
    data = encode_public_rows(_rows())
    assert data["features"].shape == (3, WIDTH)
    history_start = 147 + 4 + 64
    # After terminal action 3 the next episode starts with all PAD (action 7).
    np.testing.assert_array_equal(
        data["features"][0, history_start:history_start + 32],
        data["features"][1, history_start:history_start + 32],
    )
    assert data["action"].tolist() == [3, 3, 2]


def test_small_actual_fresh_optimizer_calls():
    data = encode_public_rows(_rows())
    model, report = fit_one(data, MODEL_SEEDS[0])
    assert isinstance(model, PublicNextFramePredictor)
    assert report["optimizer_step_calls"] == 6
    assert report["training_sample_presentations"] == 18


def test_source_zip_member_marker_and_digest_validation(tmp_path):
    metadata = tmp_path / "artifacts" / "research-loop"
    metadata.mkdir(parents=True)
    (metadata / "research_branch_commit.txt").write_text(
        "594c670a0d501fbd3cafe78ef47d59bef4733c0a\n"
    )
    (metadata / "study_mode.txt").write_text("m23\n")
    out = tmp_path / "runs" / "m23_compositional_transfer"
    out.mkdir(parents=True)
    import bagi.experiments.m23_public_world_model as m23

    metadata_dict = {
        "source_commit": m23.TRAIN_SOURCE_COMMIT,
        "source_mechanism_coverage_pass": True,
        "actual_native_actions": 9,
        "source_tasks": {},
    }
    for task in ("doorkey", "fetch", "putnear"):
        path = out / f"{task}-public-train-only.npz"
        np.savez_compressed(path, **_rows())
        metadata_dict["source_tasks"][task] = {
            "native_actions": 3,
            "public_data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (metadata / "m23_compositional_transfer-summary.json").write_text(
        json.dumps(metadata_dict)
    )
    report, data = load_source_artifact(tmp_path)
    assert report["actual_native_actions"] == 9
    assert len(data["fetch"]["action"]) == 3
    (metadata / "research_branch_commit.txt").write_text("wrong-source\n")
    with pytest.raises(ValueError, match="marker mismatch"):
        load_source_artifact(tmp_path)
