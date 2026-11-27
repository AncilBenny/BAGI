"""Exercise consecutive native switches and seed cursors without scientific claims."""
import pytest
import torch

from bagi.agents.plasticity import reset_adam_history
from bagi.experiments.cumulative_transfer import TASKS
from bagi.experiments.gate_a_competence import BLOCK
from bagi.experiments.gate_a_rehearsal import train_block
from bagi.experiments.gate_a_sequential_pilot import same_tree
from bagi.experiments.gate_a_shared_interface import SharedPPO, load_boundary, save_boundary


def test_rehearsal_returns_to_pending_seed_and_keeps_learner():
    pytest.importorskip("sb3_contrib")
    torch.set_num_threads(1)
    c = SharedPPO(TASKS[0], 201, 99001)
    try:
        c.train(BLOCK)
        c.env.rows = []
        model, policy = c.model, c.model.policy
        optimizer = policy.optimizer
        buffer = model.rollout_buffer
        cursors = {TASKS[0]: 99101, TASKS[1]: 99201}
        # Switch to B first; then cycle back to A and B twice.
        for domain in (TASKS[1], TASKS[0], TASKS[1], TASKS[0]):
            first = cursors[domain]
            rows, pending = train_block(c, domain, first, BLOCK)
            assert rows[0]["seed"] == first
            assert pending == first + len(rows)
            cursors[domain] = pending
            assert c.model is model and model.policy is policy
            assert policy.optimizer is optimizer and model.rollout_buffer is buffer
        assert c.training_actions == 5 * BLOCK
        assert c.updates == 80 and c.presentations == 20 * BLOCK
    finally:
        c.model_env.close()


def test_same_run_fork_reproduces_next_native_rollout(tmp_path):
    pytest.importorskip("sb3_contrib")
    torch.set_num_threads(1)
    c = SharedPPO(TASKS[0], 203, 99301)
    first = second = None
    try:
        c.train(BLOCK)
        c.env.rows = []
        assert reset_adam_history(c.model.policy)
        _, pending = train_block(c, TASKS[1], 99401, BLOCK)
        path = tmp_path / "common.pt"
        digest = save_boundary(c, path)
        first = load_boundary(path, digest)
        rows, cursor = train_block(first, TASKS[1], pending, BLOCK)
        second = load_boundary(path, digest)
        other_rows, other_cursor = train_block(second, TASKS[1], pending, BLOCK)
        assert rows == other_rows and cursor == other_cursor
        assert same_tree(first.model.policy.state_dict(), second.model.policy.state_dict())
        assert same_tree(first.model.policy.optimizer.state_dict(),
                         second.model.policy.optimizer.state_dict())
        assert first.training_actions == second.training_actions == 3 * BLOCK
        assert first.updates == second.updates == 48
    finally:
        c.model_env.close()
        for branch in (first, second):
            if branch is not None:
                branch.model_env.close()
