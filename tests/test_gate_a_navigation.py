"""Native navigation smoke with maintained RL; not scientific evidence."""
import pytest

from bagi.experiments.gate_a_competence import BLOCK, TASKS, run


def test_native_ppo_navigation_accounting(tmp_path):
    pytest.importorskip("sb3_contrib")
    result = run(tmp_path / "navigation", domain=TASKS[0], seeds=(83,),
                 budget=BLOCK, interval=BLOCK, eval_count=1, arms=("recurrent_ppo",))
    assert result["scientific_pass"] is False
    assert result["actual_training_actions"] == BLOCK
    assert result["actual_optimizer_updates"] == 16
    assert result["actual_sample_presentations"] == 4 * BLOCK
    assert result["actual_evaluation_actions"] > 0
    row = result["results"][0]
    assert len(row["curve"]) == 2
    for point in row["curve"]:
        for episode in point["episodes"]:
            assert not episode["learn"]
            assert all(0 <= action < 7 for action in episode["executed_actions"])
