"""Native audit smoke, not a competence result."""
import hashlib

import pytest
import torch

from bagi.experiments.gate_a_competence import BLOCK
from bagi.experiments.gate_a_navigation_policy import run


def test_frozen_stochastic_audit_and_policy_artifact(tmp_path):
    pytest.importorskip("sb3_contrib")
    output = tmp_path / "audit"
    result = run(output, seeds=(89,), budget=2 * BLOCK, interval=BLOCK, eval_count=1)
    assert result["scientific_pass"] is False
    assert result["actual_training_actions"] == 2 * BLOCK
    assert result["actual_optimizer_updates"] == 32
    assert result["actual_sample_presentations"] == 8 * BLOCK
    row = result["results"][0]
    assert len(row["curve"]) == 3
    assert sum(row["last_train_action_counts"]) == BLOCK
    assert [r["mode"] for r in row["final_controls"]] == ["uniform", "action_prior"]
    assert row["evaluation_actions"] == result["actual_evaluation_actions"]
    path = output / "policy-89.pt"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["policy_sha256"]
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["training_actions"] == 2 * BLOCK
    assert state["optimizer_updates"] == 32
    assert state["policy"] and state["optimizer"]["state"]
