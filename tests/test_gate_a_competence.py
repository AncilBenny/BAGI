"""Tiny native integration smoke; never scientific evidence."""
import pytest

from bagi.experiments.gate_a_competence import ARMS, BLOCK, run


def test_matched_native_competence_accounting(tmp_path):
    pytest.importorskip("sb3_contrib")
    result = run(tmp_path / "diagnostic", seeds=(81,), budget=2 * BLOCK,
                 interval=BLOCK, eval_count=1)
    assert result["scientific_pass"] is False
    assert result["actual_training_actions"] == len(ARMS) * 2 * BLOCK
    rows = {row["arm"]: row for row in result["results"]}
    assert rows["recurrent_original"]["optimizer_updates"] == 8
    assert rows["recurrent_dense"]["optimizer_updates"] == 64
    assert rows["existing_dqn"]["optimizer_updates"] == 64
    assert rows["recurrent_ppo"]["optimizer_updates"] == 32
    assert rows["recurrent_ppo"]["sample_presentations"] == 8 * BLOCK
    assert rows["recurrent_dense"]["resumed_checkpoint_sha256"]
    assert all(len(row["curve"]) == 3 for row in rows.values())
    assert all(row["evaluation_actions"] > 0 for row in rows.values())
    with pytest.raises(FileExistsError):
        run(tmp_path / "diagnostic", seeds=(81,), budget=2 * BLOCK,
            interval=BLOCK, eval_count=1)
