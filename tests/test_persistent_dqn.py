"""Real optimizer updates on synthetic inputs; no claim of native task success."""
import numpy as np
import pytest
import torch

from bagi.agents.general_brain import FactChange, GeneralBrain, Goal, Skill
from bagi.agents.persistent_dqn import PersistentDQN
from bagi.runtime import AgentRuntime, StepFeedback


class Sensor:
    def initial(self, observation):
        return frozenset()

    def change(self, before, executed_action, after, public_reward):
        return FactChange(frozenset({"complete"}) if after[0, 0] == 1 else frozenset())


def make_runtime(learner, *, frozen=False):
    skill = Skill(
        "control", frozenset(), frozenset({"complete"}),
        learner.greedy if frozen else learner.policy, max_actions=2,
    )
    return AgentRuntime(GeneralBrain((skill,), Sensor()), "public-array-v1")


def episode(runtime, learner, *, learn=True):
    observation = np.zeros((1, 2), dtype=np.float32)

    def step(action):
        return StepFeedback(action, np.ones((1, 2), dtype=np.float32), 1.0, terminated=True)

    return runtime.run_episode(
        Goal("g", frozenset({"complete"})), observation, step,
        learn=learn, on_transition=learner.observe,
    )


def test_resume_weights_optimizer_replay_and_rng(tmp_path):
    learner = PersistentDQN((1, 2), 2, seed=3, batch_size=1, target_sync_steps=2)
    runtime = make_runtime(learner)
    assert episode(runtime, learner).goal_verified
    assert learner.training_actions == learner.agent.optimizer_updates == 1
    path = tmp_path / "bundle.pt"
    learner.save(path, runtime)
    restored, state = PersistentDQN.load(path, compatibility_id="public-array-v1")
    resumed = make_runtime(restored)
    resumed.restore_snapshot(state)
    assert resumed.snapshot() == runtime.snapshot()
    assert episode(runtime, learner).actions == episode(resumed, restored).actions
    assert restored.training_actions == restored.agent.optimizer_updates == 2
    assert len(restored.agent.buffer) == 2
    assert all(torch.equal(a, b) for a, b in zip(
        learner.agent.online.parameters(), restored.agent.online.parameters(), strict=True
    ))
    assert all(torch.equal(a, b) for a, b in zip(
        learner.agent.target.parameters(), restored.agent.target.parameters(), strict=True
    ))
    with pytest.raises(ValueError, match="Incompatible"):
        PersistentDQN.load(path, compatibility_id="wrong")


def test_frozen_execution_does_not_train_or_consume_exploration_rng():
    learner = PersistentDQN((1, 2), 2, seed=5, batch_size=1)
    runtime = make_runtime(learner, frozen=True)
    before = runtime.snapshot()
    rng = learner.agent.rng.getstate()
    weights = [p.detach().clone() for p in learner.agent.online.parameters()]
    assert episode(runtime, learner, learn=False).goal_verified
    assert runtime.snapshot() == before
    assert learner.training_actions == learner.agent.optimizer_updates == 0
    assert len(learner.agent.buffer) == 0
    assert learner.agent.rng.getstate() == rng
    assert all(torch.equal(a, b) for a, b in zip(
        weights, learner.agent.online.parameters(), strict=True
    ))


def test_mismatched_action_never_reaches_learner():
    learner = PersistentDQN((1, 2), 2, seed=7, batch_size=1)
    runtime = make_runtime(learner)
    obs = np.zeros((1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="provenance"):
        runtime.run_episode(
            Goal("g", frozenset({"complete"})), obs,
            lambda action: StepFeedback(1 - action, obs), on_transition=learner.observe,
        )
    assert learner.training_actions == learner.agent.optimizer_updates == 0
