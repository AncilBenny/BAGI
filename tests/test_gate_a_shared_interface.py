"""Native continuation and switching contracts, not scientific results."""
import copy

import numpy as np
import pytest
import torch

from bagi.experiments.cumulative_transfer import TASKS
from bagi.experiments.gate_a_competence import BLOCK, NativePPOEnv
from bagi.experiments.gate_a_observation_dependence import tensor_tree_equal
from bagi.experiments.gate_a_shared_interface import (
    SharedActuators,
    SharedPPO,
    load_boundary,
    native_action,
    save_boundary,
)


def test_joint_actuators_only_execute_declared_channel():
    for index, domain in enumerate(TASKS):
        env = SharedActuators(NativePPOEnv(domain, 60001))
        try:
            obs, _ = env.reset()
            assert env.observation_space.contains(obs)
            proposal = np.array([2, 1])
            assert native_action(domain, proposal) == proposal[index]
            obs, *_ = env.step(proposal)
            assert env.observation_space.contains(obs)
            assert env.env.episode_actions == [proposal[index]] or env.env.rows
            with pytest.raises(ValueError, match="Invalid"):
                env.step([7, 0])
        finally:
            env.close()


@pytest.mark.parametrize("domain", TASKS)
def test_native_boundary_resume_matches_next_rollout(tmp_path, domain):
    pytest.importorskip("sb3_contrib")
    torch.set_num_threads(1)
    a = SharedPPO(domain, 151, 61001)
    b = None
    try:
        a.train(BLOCK)
        a.env.rows = []
        path = tmp_path / "checkpoint.pt"
        digest = save_boundary(a, path)
        with pytest.raises(FileExistsError):
            save_boundary(a, path)
        a.train(BLOCK)
        rows = copy.deepcopy(a.env.rows)
        expected_weights = copy.deepcopy(a.model.policy.state_dict())
        expected_optimizer = copy.deepcopy(a.model.policy.optimizer.state_dict())
        b = load_boundary(path, digest)
        b.train(BLOCK)
        assert b.env.rows == rows
        assert tensor_tree_equal(b.model.policy.state_dict(), expected_weights)
        assert tensor_tree_equal(b.model.policy.optimizer.state_dict(), expected_optimizer)
        assert b.training_actions == a.training_actions == 2 * BLOCK
        assert b.updates == a.updates == 32
        assert b.presentations == a.presentations == 8 * BLOCK
    finally:
        a.model_env.close()
        if b is not None:
            b.model_env.close()


def test_native_switch_keeps_policy_and_optimizer_identity():
    pytest.importorskip("sb3_contrib")
    torch.set_num_threads(1)
    c = SharedPPO(TASKS[0], 157, 62001)
    try:
        c.train(BLOCK)
        c.env.rows = []
        model, policy = c.model, c.model.policy
        optimizer, buffer = policy.optimizer, model.rollout_buffer
        weights = copy.deepcopy(policy.state_dict())
        adam = copy.deepcopy(optimizer.state_dict())
        c.switch_domain(TASKS[1], 63001)
        assert c.model is model and c.model.policy is policy
        assert policy.optimizer is optimizer and model.rollout_buffer is buffer
        assert tensor_tree_equal(weights, policy.state_dict())
        assert tensor_tree_equal(adam, optimizer.state_dict())
        c.train(BLOCK)
        assert c.training_actions == 2 * BLOCK
        assert c.updates == 32 and c.presentations == 8 * BLOCK
        assert sum(x["actions"] for x in c.env.rows) == BLOCK
        assert all(x["domain"] == TASKS[1] for x in c.env.rows)
    finally:
        c.model_env.close()
