import numpy as np
import pytest
import torch

from bagi.experiments.cumulative_transfer import TASKS
from bagi.experiments.gate_a_competence import BLOCK
from bagi.experiments.gate_a_sequential_pilot import (
    auc,
    evaluate,
    frozen_exposure,
    same_tree,
)
from bagi.experiments.gate_a_shared_interface import SharedPPO, capture_rng


def test_action_weighted_auc():
    p = [{"B_actions": i, "B_greedy": {"success": s}}
         for i, s in ((0, 0), (10, 1), (40, 1))]
    assert auc(p, "B_greedy") == 0.875


def test_cross_task_evaluation_and_frozen_exposure_preserve_state():
    pytest.importorskip("sb3_contrib")
    torch.set_num_threads(1)
    c = SharedPPO(TASKS[0], 191, 90001)
    try:
        c.train(BLOCK)
        c.env.rows = []
        rng = capture_rng()
        before = c.model._last_obs.copy()
        result = evaluate(c, TASKS[1], [91001, 91002], deterministic=False)
        assert len(result["episodes"]) == 2
        assert all(r["domain"] == TASKS[1] and not r["learn"] for r in result["episodes"])
        rows, cursor = frozen_exposure(c, 92001, 0, BLOCK)
        assert sum(r["actions"] for r in rows) == BLOCK and cursor == len(rows)
        assert c.training_actions == BLOCK and c.updates == 16
        assert same_tree(rng, capture_rng())
        assert np.array_equal(c.model._last_obs, before)
        assert c.env.domain == TASKS[0]
    finally:
        c.model_env.close()
