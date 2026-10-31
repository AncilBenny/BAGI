"""M1 contracts: public executed actions, caps and safe output, no live RL fit."""
import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from bagi.experiments import m1_goal_learning as m1


def config():
    return {
        "train_task": m1.TRAIN_TASK,
        "transfer_task": m1.TRANSFER_TASK,
        "arms": list(m1.ARMS),
        "training_seeds": [17, 29],
        "train_steps_per_arm_seed": 8192,
        "rollout_steps": 128,
        "batch_size": 64,
        "epochs_per_rollout": 2,
        "evaluation_seeds": [92001, 92002, 92003, 92004, 92005, 92006],
        "transfer_seeds": [93001, 93002, 93003, 93004, 93005, 93006],
        "max_eval_steps": 120,
        "max_total_training_steps": 65536,
        "max_total_evaluation_steps": 17280,
        "output_dir": "runs/m1_goal_learning",
    }


def test_frozen_budget_and_disjoint_splits():
    assert m1.validate_config(config())["train_steps_per_arm_seed"] == 8192
    for override in (
        {"train_steps_per_arm_seed": 16384},
        {"arms": ["recurrent"]},
        {"training_seeds": [17, 17]},
        {"max_total_evaluation_steps": 30000},
        {"transfer_seeds": [92001, 93002, 93003, 93004, 93005, 93006]},
    ):
        with pytest.raises(ValueError):
            m1.validate_config({**config(), **override})


class TinyPublic(gym.Env):
    observation_space = Box(0, 8, (3,), dtype=np.uint8)
    action_space = Discrete(2)

    def __init__(self):
        super().__init__()
        self.index = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return np.array([1, 2, 3], dtype=np.uint8), {"private_mock": "not in observation"}

    def step(self, action):
        assert action in (0, 1)
        self.index += 1
        return (np.array([2, 3, 4], dtype=np.uint8),
                float(self.index == 2), self.index == 2, False,
                {"private_mock": "do not leak"})


def test_public_action_history_is_executed_and_reset():
    env = m1.CountNativeSteps(m1.PublicExecutedActions(TinyPublic(), 4))
    try:
        obs, info = env.reset(seed=17)
        assert "private_mock" in info
        assert obs.shape == (15,)
        assert obs[:3].tolist() == [1.0, 2.0, 3.0]
        assert obs[-12:].tolist() == [0, 0, 1] * 4
        after, reward, terminal, truncated, info = env.step(1)
        assert not reward and not terminal and not truncated
        assert after[-3:].tolist() == [0, 1, 0]
        assert after[-6:-3].tolist() == [0, 0, 1]
        assert env.native_steps == 1 and env.native_resets == 1
        fresh, _ = env.reset(seed=29)
        assert fresh[-12:].tolist() == [0, 0, 1] * 4
        assert env.native_steps == 1 and env.native_resets == 2
    finally:
        env.close()


class ToyPolicy:
    class Optimizer:
        def step(self):
            return 7

    optimizer = Optimizer()


class ToyAgent:
    policy = ToyPolicy()


def test_optimizer_counts_real_calls():
    agent = ToyAgent()
    counter = m1.count_optimizer_steps(agent)
    assert counter["calls"] == 0
    assert agent.policy.optimizer.step() == 7
    assert agent.policy.optimizer.step() == 7
    assert counter["calls"] == 2


def test_numeric_policy_exports_no_pickle(tmp_path):
    class FakeAgent:
        class Policy:
            def state_dict(self):
                return {"linear.weight": torch.ones((2, 3), dtype=torch.float32)}

        policy = Policy()

    dest = tmp_path / "numeric-only.npz"
    result = m1.numeric_state(FakeAgent(), dest)
    assert result["sha256"]
    assert result["members"]["tensor_000"]["name"] == "linear.weight"
    with np.load(dest, allow_pickle=False) as content:
        assert content["tensor_000"].shape == (2, 3)
    with pytest.raises(FileExistsError):
        m1.numeric_state(FakeAgent(), dest)


def test_unknown_task_is_rejected_before_env_creation():
    with pytest.raises(ValueError):
        m1.make_task("MiniGrid-HiddenOracle-v0", "recurrent")
