"""Ensure reused RL benchmark integration is information-safe and budget-matched."""
import numpy as np
import pytest

from bagi.experiments.reused_rl_baselines import TASKS, evaluate_agent, make_task, settings


@pytest.mark.parametrize("task", TASKS)
def test_standard_farama_benchmarks_provide_partial_flat_observations(task):
    env = make_task(task)
    try:
        observation, _ = env.reset(seed=41)
        assert observation.ndim == 1
        assert env.observation_space.contains(observation)
        assert env.action_space.n >= 3
    finally:
        env.close()


def test_training_budgets_are_equal_and_exact():
    good = {
        "tasks": list(TASKS), "training_seeds": [5],
        "train_steps": 512, "rollout_steps": 64,
        "evaluation_episodes": 3,
    }
    assert settings(good) == (list(TASKS), [5], 512, 3)
    with pytest.raises(ValueError, match="rollout-divisible"):
        settings({**good, "train_steps": 513})
    with pytest.raises(ValueError, match="registered"):
        settings({**good, "tasks": ["Workshop-made-up-v0"]})


class TwoStepEnvironment:
    class DummyActionSpace:
        def seed(self, seed):
            self.seed_value = seed

    action_space = DummyActionSpace()

    def __init__(self):
        self.count = 0

    def reset(self, *, seed):
        self.count = 0
        return np.array([0.0]), {}

    def step(self, action):
        assert action == 0
        self.count += 1
        return np.array([0.0]), float(self.count == 2), self.count == 2, False, {}

    def close(self):
        return None


class RecordingAgent:
    def __init__(self):
        self.states = []
        self.episode_start_flags = []

    def predict(self, obs, *, state, episode_start, deterministic):
        self.states.append(state)
        self.episode_start_flags.append(bool(episode_start[0]))
        return np.array(0), ("carry_state",)


def test_evaluation_resets_recurrent_state_on_episode_boundaries():
    agent = RecordingAgent()
    outcome = evaluate_agent(
        agent, "mock", [100, 101], env_factory=lambda _: TwoStepEnvironment()
    )
    assert outcome["successes"] == 2
    assert outcome["evaluation_real_steps"] == 4
    assert agent.episode_start_flags == [True, False, True, False]
    assert agent.states == [None, ("carry_state",), None, ("carry_state",)]
