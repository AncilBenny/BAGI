"""M1 contracts: public executed actions, caps and safe output, no live RL fit."""
import gymnasium as gym
import numpy as np
import pytest
import torch
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


def test_memory_arms_have_distinct_public_control_inputs_without_steps():
    """Reactive and recurrent see only image; action4 and frame4 add legal history."""
    environments = {}
    try:
        for arm in m1.ARMS:
            environments[arm] = m1.make_task(m1.TRAIN_TASK, arm)
        shapes = {arm: env.observation_space.shape for arm, env in environments.items()}
        image_width = 7 * 7 * 3  # Standard MiniGrid local public image.
        n_actions = environments["action4"].action_space.n
        assert shapes["reactive"] == (image_width,)
        assert shapes["recurrent"] == (image_width,)
        assert shapes["action4"] == (image_width + 4 * (n_actions + 1),)
        assert shapes["frame4"] == (4 * image_width,)
        assert all(env.native_steps == 0 and env.native_resets == 0
                   for env in environments.values())
    finally:
        for env in environments.values():
            env.close()

class DeterministicMockPolicy:
    def __init__(self):
        self.starts = []

    def predict(self, _observation, *, state, episode_start, deterministic):
        assert deterministic
        self.starts.append((bool(episode_start[0]), state))
        return np.array(0), ("state_from_last_action",)


class PositiveTruncated(TinyPublic):
    """Positive public reward without terminated goal must NOT count."""

    def step(self, action):
        observation, reward, terminated, _, info = super().step(action)
        return observation, reward, False, terminated, info


def test_native_goal_scoring_requires_termination_and_counts_resets():
    for env_class, expected_success in ((TinyPublic, 2), (PositiveTruncated, 0)):
        policy = DeterministicMockPolicy()
        summary = m1.evaluate_goal_agent(
            policy, "mock", [201, 202],
            env_factory=lambda _task: m1.CountNativeSteps(env_class()),
        )
        assert summary["successes"] == expected_success
        assert summary["episodes"] == 2
        assert summary["evaluation_real_steps"] == 4
        assert summary["evaluation_real_resets"] == 2
        assert policy.starts == [
            (True, None), (False, ("state_from_last_action",)),
            (True, None), (False, ("state_from_last_action",)),
        ]
        assert all(row["reward"] == 1.0 for row in summary["episode_results"])


def test_training_and_transfer_observation_contracts_match_without_native_actions():
    for arm in m1.ARMS:
        train = m1.make_task(m1.TRAIN_TASK, arm)
        transfer = m1.make_task(m1.TRANSFER_TASK, arm)
        try:
            assert train.observation_space.shape == transfer.observation_space.shape
            assert train.action_space == transfer.action_space
            assert train.native_steps == train.native_resets == 0
            assert transfer.native_steps == transfer.native_resets == 0
        finally:
            train.close()
            transfer.close()


def test_command_cannot_train_outside_admitted_research_job(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", [
        "m1_goal_learning", "--execute-budgeted-research",
    ])
    monkeypatch.setenv("BAGI_M1_ADMITTED", "1")
    monkeypatch.setenv("GITHUB_WORKFLOW", "BAGI Research Loop")
    monkeypatch.setenv("GITHUB_JOB", "test")
    with pytest.raises(SystemExit, match="quota-admitted research job"):
        m1.main()

def test_frozen_goal_milestone_arithmetic_is_independent_of_weights():
    rows = []
    for arm in m1.ARMS:
        for seed in (17, 29):
            rows.append({
                "arm": arm, "model_seed": seed,
                "before": {"successes": 0},
                "after": {"successes": 3 if arm == "recurrent" else 1},
                "transfer": {"successes": 2 if arm == "recurrent" else 0},
            })
    score = m1.score_goal_milestone(rows)
    assert score["per_arm"]["recurrent"]["after_goals_of_12"] == 6
    assert score["per_arm"]["recurrent"]["tentative_learning_pass"]
    assert score["recurrent_post_minus_best_control"] == 4
    assert score["tentative_memory_advantage"]
    assert score["tentative_size_transfer"]
    weaker = [
        {**row, "after": {"successes": 0}, "transfer": {"successes": 0}}
        if row["arm"] == "recurrent" else row for row in rows
    ]
    assert not m1.score_goal_milestone(weaker)["tentative_memory_advantage"]
    with pytest.raises(ValueError, match="two independent"):
        m1.score_goal_milestone(rows[:-1])
