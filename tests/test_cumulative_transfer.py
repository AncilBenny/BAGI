"""Native feasibility contracts: tiny runs only, never confirmatory evidence."""
import numpy as np
import pytest
import torch

from bagi.agents.recurrent_replay import Episode, RecurrentReplay
from bagi.experiments.cumulative_transfer import (
    ACTIONS, TASKS, WIDTH, evaluate, run_episode, run_pilot,
)


def test_native_replay_and_frozen_accounting():
    learner = RecurrentReplay(WIDTH, ACTIONS, seed=11)
    for task in TASKS:
        row = run_episode(learner, task, 6_900_001, learn=True, limit=4)
        assert row["actions"] == 4
        assert len(learner.replay[task]) == 1
    assert learner.training_actions == 8
    assert np.isfinite(learner.update())
    before = [p.detach().clone() for p in learner.online.parameters()]
    evaluate(learner, TASKS[1], [6_900_002])
    assert learner.updates == 1 and learner.training_actions == 8
    assert all(torch.equal(a, b) for a, b in zip(before, learner.online.parameters(), strict=True))


def test_reject_illegal_action_and_reset_preserves_shared_core():
    learner = RecurrentReplay(WIDTH, ACTIONS, seed=12)
    bad = Episode(np.zeros((2, WIDTH)), np.array([2]), np.array([0.0]),
                  np.array([False]), TASKS[1], 2)
    with pytest.raises(ValueError, match="Invalid"):
        learner.remember(bad)
    before = [p.detach().clone() for p in learner.online.gru.parameters()]
    learner.reset_head(99)
    assert all(torch.equal(a, b) for a, b in zip(before, learner.online.gru.parameters(), strict=True))


def test_pilot_matched_native_budgets(tmp_path):
    summary = run_pilot(tmp_path / "pilot", model_seeds=(13,), block=8, blocks=1, eval_count=1)
    assert summary["scientific_pass"] is False
    assert summary["actual_exposure_actions"] == 40
    assert summary["actual_training_actions"] == 32
    rows = {r["arm"]: r for r in summary["results"]}
    for name in ("fresh", "carry", "head_reset", "frozen"):
        assert rows[name]["B_exposure_actions"] == 8
        assert rows[name]["B_optimizer_updates"] == (0 if name == "frozen" else 4)
        assert len(rows[name]["B_curve"]) == 2
    assert (tmp_path / "pilot" / "native_actions.jsonl").exists()
    with pytest.raises(FileExistsError):
        run_pilot(tmp_path / "pilot", model_seeds=(13,), block=8, blocks=1, eval_count=1)
