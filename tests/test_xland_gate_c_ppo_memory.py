"""REUSE-32 contract tests; mock episodes cannot demonstrate real XLand skill."""
from dataclasses import dataclass

import numpy as np
import pytest

from bagi.experiments.xland_gate_c_ppo_memory import (
    MODEL_SEEDS,
    TEST_LAYOUTS,
    next_executed_history,
    run_closed_loop_episode,
    validate_config,
)


def config():
    return {
        "environment": "XLand-MiniGrid-R1-9x9",
        "policy_input_width": 115,
        "training_tasks": ["hold", "near", "putdown"],
        "model_seeds": list(MODEL_SEEDS),
        "training": {
            "seed_base": 9400, "native_steps_per_policy": 1024,
            "rollout_steps": 128, "batch_size": 64, "epochs_per_rollout": 2,
        },
        "evaluation": {"layout_seeds": list(TEST_LAYOUTS), "episode_cap": 40},
    }


def test_fixed_pub_input_and_action_history_contract():
    validate_config(config())
    assert next_executed_history((6, 6, 6, 6), 3) == (6, 6, 6, 3)
    with pytest.raises(ValueError, match="history"):
        next_executed_history((6, 6, 6), 1)
    with pytest.raises(ValueError, match="native"):
        next_executed_history((6, 6, 6, 6), 6)


@pytest.mark.parametrize("field,bad", [
    ("native_steps_per_policy", 1025),
    ("rollout_steps", 130),
    ("batch_size", 32),
    ("epochs_per_rollout", 4),
])
def test_no_unregistered_train_changes(field, bad):
    c = config()
    c["training"][field] = bad
    with pytest.raises(ValueError, match="Unpreregistered"):
        validate_config(c)


def test_frozen_untouched_source_only_eval_layouts():
    c = config()
    c["evaluation"]["layout_seeds"] = [9810, 9811, 9812, 9813, 9814]
    with pytest.raises(ValueError, match="Unpreregistered"):
        validate_config(c)
    c = config()
    c["training_tasks"] = ["putdown", "hold_then_near"]
    with pytest.raises(ValueError, match="Unpreregistered"):
        validate_config(c)


@dataclass(frozen=True)
class TinyStep:
    number: int
    reward: float

    def last(self):
        return self.number >= 2


class RecordingPolicy:
    def __init__(self):
        self.states = []
        self.starts = []
        self.obs = []

    def predict(self, obs, *, state, episode_start, deterministic):
        assert deterministic
        self.obs.append(obs.copy())
        self.states.append(state)
        self.starts.append(bool(episode_start[0]))
        return np.array(3 if len(self.starts) % 2 == 1 else 4), "memory"


def fake_obs(ts, history):
    public = np.zeros(115, dtype=np.float32)
    public[-28:] = np.eye(7, dtype=np.float32)[np.asarray(history, dtype=np.int64)].reshape(-1)
    return public


def fake_step(ts, action):
    assert action in (3, 4)
    return TinyStep(ts.number + 1, float(ts.number == 1))


def test_recurrent_state_reset_true_history_and_no_vacuous_reward():
    agent = RecordingPolicy()
    for _ in (1, 2):
        result = run_closed_loop_episode(
            agent, TinyStep(0, 0.0), task="putdown", native_step=fake_step,
            observe=fake_obs, eligible=True,
            on_physical_step=lambda before, after, act: (False, False),
            final_adjacent=lambda ts: True,
        )
        assert result["native_reward"]
        assert not result["physical_placement"]
        assert result["effective_blue_pickups"] == 0
        assert result["effective_blue_drops"] == 0
        assert result["native_actions"] == 2
    assert agent.starts == [True, False, True, False]
    assert agent.states == [None, "memory", None, "memory"]
    assert np.array_equal(
        agent.obs[1][-28:], np.eye(7, dtype=np.float32)[[6, 6, 6, 3]].reshape(-1)
    )


def test_ineligible_fresh_putdown_cannot_be_cherry_picked():
    agent = RecordingPolicy()
    result = run_closed_loop_episode(
        agent, TinyStep(0, 0.0), task="putdown", native_step=fake_step,
        observe=fake_obs, eligible=False,
        on_physical_step=lambda a, b, act: (True, True),
        final_adjacent=lambda x: True,
    )
    assert result["native_actions"] == 0 and not result["physical_placement"]
    assert agent.starts == []


def test_positive_physical_ordered_action_requires_native_transition():
    agent = RecordingPolicy()
    result = run_closed_loop_episode(
        agent, TinyStep(0, 0.0), task="putdown", native_step=fake_step,
        observe=fake_obs, eligible=True,
        on_physical_step=lambda before, after, a: (a == 3, a == 4),
        final_adjacent=lambda ts: True,
    )
    assert result["physical_placement"]
    assert result["effective_blue_pickups"] == 1
    assert result["effective_blue_drops"] == 1
