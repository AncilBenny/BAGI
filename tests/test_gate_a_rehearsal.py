"""Exercise consecutive native switches and seed cursors without scientific claims."""
import pytest
import torch

from bagi.experiments.cumulative_transfer import TASKS
from bagi.experiments.gate_a_competence import BLOCK
from bagi.experiments.gate_a_rehearsal import train_block
from bagi.experiments.gate_a_shared_interface import SharedPPO


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
