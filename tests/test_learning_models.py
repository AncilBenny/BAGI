import numpy as np
import torch

from bagi.agents.dqn import DQNAgent, Transition
from bagi.models import WorldModel


def test_dqn_forward_and_learning_step():
    agent = DQNAgent(observation_shape=(5, 5), actions=4)
    observation = np.zeros((5, 5), dtype=np.int8)
    next_observation = observation.copy()
    for index in range(64):
        agent.buffer.add(
            Transition(
                observation=observation,
                action=index % 4,
                reward=0.0,
                next_observation=next_observation,
                done=False,
            )
        )
    action = agent.select_action(observation, epsilon=0.0)
    assert 0 <= action < 4
    loss = agent.learn(batch_size=64)
    assert loss is not None
    assert loss >= 0.0


def test_world_model_shapes_and_prediction():
    model = WorldModel(observation_shape=(5, 5), actions=4, cell_values=7)
    observations = torch.zeros((3, 5, 5), dtype=torch.long)
    actions = torch.tensor([0, 1, 2])
    logits, rewards = model(observations, actions)
    assert logits.shape == (3, 5, 5, 7)
    assert rewards.shape == (3,)

    predicted_observations, predicted_rewards = model.predict(observations, actions)
    assert predicted_observations.shape == (3, 5, 5)
    assert predicted_rewards.shape == (3,)
