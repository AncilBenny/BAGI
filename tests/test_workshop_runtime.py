"""Native integration contracts, not scientific competence or transfer tests."""
import json

import pytest

from bagi.workshop_runtime import run_workshop


def test_native_train_resume_and_frozen_evaluation(tmp_path):
    first = run_workshop(
        tmp_path / "train", mode="train", episodes=1, first_seed=101,
        model_seed=4, max_steps=4, batch_size=2,
    )
    assert first["native_actions"] == first["training_actions"] == 4
    assert first["optimizer_updates"] == 3
    assert first["physical_successes"] == 0
    checkpoint = tmp_path / "train" / "learner.pt"
    resume = run_workshop(
        tmp_path / "resume", mode="train", episodes=1, first_seed=102,
        max_steps=4, checkpoint=checkpoint,
    )
    assert resume["native_actions"] == 4 and resume["optimizer_updates"] == 4
    assert resume["lifetime_training_actions"] == 8
    assert resume["lifetime_optimizer_updates"] == 7
    assert resume["parent_checkpoint_sha256"] == first["checkpoint_sha256"]
    frozen = run_workshop(
        tmp_path / "eval", mode="evaluate", episodes=1, first_seed=201,
        max_steps=4, layout="mirror_x", checkpoint=tmp_path / "resume" / "learner.pt",
    )
    assert frozen["native_actions"] == 4
    assert frozen["training_actions"] == frozen["optimizer_updates"] == 0
    assert frozen["lifetime_optimizer_updates"] == 7
    assert frozen["physical_successes"] == 0
    assert not (tmp_path / "eval" / "learner.pt").exists()
    row = json.loads((tmp_path / "eval" / "episodes.jsonl").read_text())
    assert len(row["executed_actions"]) == frozen["native_actions"]
    with pytest.raises(FileExistsError):
        run_workshop(tmp_path / "train", mode="train", episodes=1, first_seed=101)


def test_evaluation_requires_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="requires a checkpoint"):
        run_workshop(tmp_path / "eval", mode="evaluate", episodes=1, first_seed=201)
