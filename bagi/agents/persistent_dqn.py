"""Persist and reuse the existing DQN as a public-transition runtime skill.

CPU only; compatible observation/action semantics are an adapter responsibility.
This is integration of an established baseline, not a new learning algorithm.
"""
from __future__ import annotations

import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bagi.agents.dqn import DQNAgent, ReplayBuffer, Transition
from bagi.runtime import AgentRuntime, StepFeedback


class RuntimeReplay(ReplayBuffer):
    """Local sampling RNG while retaining the original replay implementation."""

    def __init__(self, rng: random.Random) -> None:
        super().__init__()
        self.rng = rng

    def sample(self, batch_size: int) -> list[Transition]:
        return self.rng.sample(self._data, batch_size)

    def transitions(self) -> tuple[Transition, ...]:
        return tuple(self._data)


class RuntimeDQN(DQNAgent):
    """Reuse the frozen DQN optimization code without modifying its source."""

    def __init__(self, observation_shape, actions, learning_rate, gamma, *, seed):
        super().__init__(observation_shape, actions, learning_rate, gamma)
        self.rng = random.Random(seed)
        self.buffer = RuntimeReplay(self.rng)
        self.optimizer_updates = 0

    def select_action(self, observation: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return self.rng.randrange(self.actions)
        with torch.no_grad():
            values = self.online(torch.as_tensor(observation).unsqueeze(0))
            return int(values.argmax(dim=1).item())

    def learn(self, batch_size: int = 64) -> float | None:
        loss = super().learn(batch_size)
        if loss is not None:
            self.optimizer_updates += 1
        return loss


class PersistentDQN:
    """One resumable learner; evaluation uses greedy action and no update hook."""

    def __init__(
        self,
        observation_shape: tuple[int, int],
        actions: int,
        *,
        seed: int,
        epsilon: float = 0.1,
        batch_size: int = 64,
        target_sync_steps: int = 100,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
    ) -> None:
        if (
            len(observation_shape) != 2
            or any(type(x) is not int or x <= 0 for x in observation_shape)
            or type(actions) is not int or actions <= 0
            or type(seed) is not int
            or type(batch_size) is not int or batch_size <= 0
            or type(target_sync_steps) is not int or target_sync_steps <= 0
            or not 0 <= epsilon <= 1 or not 0 <= gamma <= 1
            or not math.isfinite(learning_rate) or learning_rate <= 0
        ):
            raise ValueError("Invalid DQN configuration")
        self.config = {
            "observation_shape": tuple(observation_shape),
            "actions": actions,
            "seed": seed,
            "epsilon": epsilon,
            "batch_size": batch_size,
            "target_sync_steps": target_sync_steps,
            "learning_rate": learning_rate,
            "gamma": gamma,
        }
        # Keep initialization independent of unrelated training processes.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.agent = RuntimeDQN(
                observation_shape, actions, learning_rate, gamma, seed=seed
            )
        self.training_actions = 0

    def _array(self, observation: Any) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32)
        if array.shape != self.config["observation_shape"] or not np.isfinite(array).all():
            raise ValueError("Incompatible or nonfinite public observation")
        return array.copy()

    def policy(self, observation: Any, _facts: frozenset[str]) -> int:
        return self.agent.select_action(self._array(observation), self.config["epsilon"])

    def greedy(self, observation: Any, _facts: frozenset[str]) -> int:
        # Unlike epsilon-greedy selection, this consumes no exploration RNG.
        with torch.no_grad():
            values = self.agent.online(torch.as_tensor(self._array(observation)).unsqueeze(0))
            return int(values.argmax(dim=1).item())

    def observe(self, before: Any, feedback: StepFeedback) -> None:
        """Train only on feedback from the runtime's validated executed action."""
        action = feedback.executed_action
        if type(action) is not int or not 0 <= action < self.config["actions"]:
            raise ValueError("Incompatible executed action")
        reward = float(feedback.reward)
        if not math.isfinite(reward):
            raise ValueError("Nonfinite reward")
        transition = Transition(
            self._array(before), action, reward, self._array(feedback.observation),
            bool(feedback.terminated or feedback.truncated),
        )
        self.agent.buffer.add(transition)
        self.training_actions += 1
        self.agent.learn(self.config["batch_size"])
        if self.training_actions % self.config["target_sync_steps"] == 0:
            self.agent.sync_target()

    def save(self, path: Path, runtime: AgentRuntime) -> None:
        """Atomically save model, optimizer, replay, RNG and runtime outcome counts.

        The bundle is for trusted local run output. Load uses weights_only=True;
        no arbitrary serialized Python objects or NumPy pickles are needed.
        """
        replay = [
            (torch.as_tensor(t.observation.copy()), t.action, t.reward,
             torch.as_tensor(t.next_observation.copy()), t.done)
            for t in self.agent.buffer.transitions()
        ]
        state = {
            "schema": 1,
            "config": self.config.copy(),
            "runtime": runtime.snapshot(),
            "online": self.agent.online.state_dict(),
            "target": self.agent.target.state_dict(),
            "optimizer": self.agent.optimizer.state_dict(),
            "rng": self.agent.rng.getstate(),
            "training_actions": self.training_actions,
            "optimizer_updates": self.agent.optimizer_updates,
            "replay": replay,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                torch.save(state, handle)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path, *, compatibility_id: str) -> tuple[PersistentDQN, dict]:
        """Restore into a new learner; reject different adapter/policy contracts."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or state.get("schema") != 1:
            raise ValueError("Incompatible learner checkpoint")
        runtime = state["runtime"]
        if runtime.get("compatibility_id") != compatibility_id:
            raise ValueError("Incompatible runtime checkpoint")
        learner = cls(**state["config"])
        counts = (state["training_actions"], state["optimizer_updates"])
        if any(type(x) is not int or x < 0 for x in counts) or counts[1] > counts[0]:
            raise ValueError("Invalid learning counters")
        if len(state["replay"]) != min(counts[0], 20_000):
            raise ValueError("Invalid replay length")
        learner.agent.online.load_state_dict(state["online"])
        learner.agent.target.load_state_dict(state["target"])
        learner.agent.optimizer.load_state_dict(state["optimizer"])
        learner.agent.rng.setstate(state["rng"])
        for before, action, reward, after, done in state["replay"]:
            if (type(action) is not int or not 0 <= action < learner.config["actions"]
                    or type(done) is not bool or not math.isfinite(reward)):
                raise ValueError("Invalid replay transition")
            learner.agent.buffer.add(Transition(
                learner._array(before), action, float(reward), learner._array(after), done
            ))
        learner.training_actions, learner.agent.optimizer_updates = counts
        return learner, runtime
