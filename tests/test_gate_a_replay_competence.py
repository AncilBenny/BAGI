"""Native integration smoke, separate from the registered full competence fit."""
import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torchrl") is None,
    reason="Requires pinned recurrent replay extra",
)


def test_replay_competence_native_budget_and_resume(tmp_path, monkeypatch):
    from bagi.experiments import gate_a_replay_competence as study

    monkeypatch.setattr(study, "CAPACITY", 4)
    monkeypatch.setattr(study, "source_revision",
                        lambda: {"commit": "synthetic-test", "tracked_worktree_dirty": False})
    summary = study.run(
        tmp_path / "study", seeds=(151,), budget=16, interval=8, block=8,
        updates_per_block=1, eval_count=1,
    )
    assert summary["scientific_pass"] is False
    assert summary["actual_training_actions"] == 64
    assert summary["actual_optimizer_updates"] == 8
    assert summary["actual_evaluation_episodes"] == 12
    assert len(summary["results"]) == 4
    assert all(r["midpoint_digests"] and r["final_digests"] for r in summary["results"])
    assert all(len(r["curve"]) == 3 for r in summary["results"])
