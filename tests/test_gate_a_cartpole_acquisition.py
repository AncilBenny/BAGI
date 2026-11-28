"""Small acquisition smoke: not a scientific measurement."""
import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torchrl") is None,
    reason="Requires pinned recurrent replay extra",
)


def test_acquisition_settings_and_runtime(tmp_path, monkeypatch):
    from bagi.experiments import gate_a_cartpole_acquisition as study
    from bagi.experiments import gate_a_replay_competence as base

    assert study.settings("density", 0, 16) == (0.2, 64)
    assert study.settings("exploration", 0, 16) == (0.9, 8)
    assert study.settings("both", 8, 16)[0] == pytest.approx(0.1)
    monkeypatch.setattr(base, "CAPACITY", 4)
    monkeypatch.setattr(study, "source_revision",
                        lambda: {"commit": "test", "tracked_worktree_dirty": False})
    summary = study.run(
        tmp_path / "probe", seeds=(151,), arms=("exploration",),
        budget=16, interval=8, block=8, eval_count=1, verify_baseline=False,
    )
    assert summary["actual_training_actions"] == 16
    assert summary["actual_optimizer_updates"] == 16
    assert summary["actual_evaluation_episodes"] == 3
    assert summary["results"][0]["midpoint_digests"]
    assert summary["scientific_pass"] is False
