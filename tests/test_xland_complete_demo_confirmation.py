"""REUSE-34 locked original-recipe tests; not simulator performance evidence."""
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from bagi.experiments import xland_complete_demo_bc as old
from bagi.experiments.xland_complete_demo_confirmation import (
    CONFIRM_LAYOUTS,
    evaluate_original_on_confirm,
    load_frozen_teacher,
    validate_layouts,
    validate_original_recipe,
)


def test_new_layouts_untouched_and_fixed():
    validate_layouts()
    assert CONFIRM_LAYOUTS == tuple(range(10700, 10710))
    assert set(CONFIRM_LAYOUTS).isdisjoint(old.TRAIN_LAYOUTS)
    assert set(CONFIRM_LAYOUTS).isdisjoint(old.EVAL_LAYOUTS)


def test_original_evaluator_is_called_then_layout_reverted(monkeypatch):
    seen = []

    def original_mock(policy):
        seen.append((policy, old.EVAL_LAYOUTS))
        return {task: {"cases": [{"layout": k} for k in old.EVAL_LAYOUTS]}
                for task in ("hold", "near", "putdown")}

    monkeypatch.setattr(old, "evaluate_policy", original_mock)
    policy = object()
    report = evaluate_original_on_confirm(policy)
    assert seen == [(policy, CONFIRM_LAYOUTS)]
    assert len(report["putdown"]["cases"]) == 10
    assert old.EVAL_LAYOUTS == tuple(range(10600, 10610))


def test_original_evaluator_restored_on_native_failure(monkeypatch):
    def fail(policy):
        assert old.EVAL_LAYOUTS == CONFIRM_LAYOUTS
        raise RuntimeError("native failure")

    monkeypatch.setattr(old, "evaluate_policy", fail)
    with pytest.raises(RuntimeError, match="native failure"):
        evaluate_original_on_confirm(object())
    assert old.EVAL_LAYOUTS == tuple(range(10600, 10610))


def test_edited_source_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_frozen_teacher(tmp_path)
    (tmp_path / "summary.json").write_text("{}")
    np.savez_compressed(tmp_path / "train_source_complete_demos.npz",
                        obs=np.zeros((248, 115), dtype=np.float32),
                        expert_label=np.ones(248, dtype=np.int64),
                        done=np.zeros(248, dtype=bool))
    with pytest.raises(ValueError, match="SHA256"):
        load_frozen_teacher(tmp_path)


def test_training_recipe_must_match_original():
    training = {
        "original_source_rows": 291, "added_sampled_rows": 248,
        "total_rows": 539, "epochs": 12,
        "nominal_optimization_exposures": 6468,
        "sampling_method": "torch.WeightedRandomSampler",
    }
    previous = {"original_source_sha256": "x", "training": training}
    validate_original_recipe(training, previous, "x")
    for key, wrong in (("total_rows", 540), ("epochs", 13),
                       ("nominal_optimization_exposures", 6467)):
        with pytest.raises(ValueError, match="sampling and training"):
            validate_original_recipe({**training, key: wrong}, previous, "x")
    with pytest.raises(ValueError, match="sampling and training"):
        validate_original_recipe(training, previous, "changed")


def test_frozen_real_teacher_archive_when_present(tmp_path):
    path = Path("runs/reuse34_teacher")
    if path.is_dir():
        for item in ("summary.json", "train_source_complete_demos.npz"):
            (tmp_path / item).write_bytes((path / item).read_bytes())
    else:
        zpath = Path("/mnt/data/github-actions-artifact-12563231248.zip")
        if not zpath.is_file():
            pytest.skip("Historical SHA-verified archive not mounted")
        with ZipFile(zpath) as z:
            for item in ("summary.json", "train_source_complete_demos.npz"):
                (tmp_path / item).write_bytes(z.read(item))
    report, rows = load_frozen_teacher(tmp_path)
    assert report["source_teacher_provenance"]["eligible_physical_episode_count"] == 22
    assert rows["obs"].shape == (248, 115)
    assert np.count_nonzero(rows["expert_label"] == 4) == 22


def test_original_json_key_format_normalized_without_bypassing_label_checks():
    """Original JSON stringifies integer keys: this is not a sampling change."""
    training = {
        "original_source_rows": 291, "added_sampled_rows": 248,
        "total_rows": 539, "epochs": 12,
        "nominal_optimization_exposures": 6468,
        "label_action_counts": {0: 31, 3: 22, 4: 22},
    }
    archival = {
        "original_source_sha256": "verified",
        "training": {**training, "label_action_counts": {"0": 31, "3": 22, "4": 22}},
    }
    validate_original_recipe(training, archival, "verified")
    with pytest.raises(ValueError, match="sampling and training"):
        validate_original_recipe(
            {**training, "label_action_counts": {0: 31, 3: 22, 4: 23}},
            archival, "verified",
        )
