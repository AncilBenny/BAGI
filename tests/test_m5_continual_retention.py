"""M5 research contracts: public mission, acted actions, schedules and physical audit."""
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import yaml
from gymnasium.spaces import Box, Dict, Discrete, Text
from minigrid.core.world_object import Door, Key

from bagi.experiments import m5_continual_retention as m5
from bagi.experiments.m1_goal_learning import CountNativeSteps


class FakePublic(gym.Env):
    def __init__(self, mission="go to the matching object at the end of the hallway"):
        self.observation_space = Dict({
            "image": Box(0, 255, (7, 7, 3), dtype=np.uint8),
            "direction": Discrete(4),
            "mission": Text(max_length=200),
        })
        self.action_space = Discrete(7)
        self.mission = mission
        self.last_seed = None
        self.executed = []

    def _public(self):
        return {
            "image": np.ones((7, 7, 3), dtype=np.uint8),
            "direction": 0, "mission": self.mission,
        }

    def reset(self, *, seed=None, options=None):
        self.last_seed = seed
        self.executed = []
        return self._public(), {"private_inventory": "not a policy input"}

    def step(self, action):
        self.executed.append(action)
        return self._public(), 0.0, False, False, {"secret": 17}


def test_frozen_m5_configuration_and_independent_eval_seeds():
    config = yaml.safe_load(Path("configs/m5_continual_retention.yaml").read_text())
    assert m5.validate_config(config) == config
    for field, change in (
        ("model_seeds", [17, 29]),
        ("rollout_steps", 256),
        ("memory_heldout_seeds", list(range(104001, 104013))),
        ("output_dir", "runs/previous"),
    ):
        with pytest.raises(ValueError, match="preregistration"):
            m5.validate_config({**config, field: change})
    assert set(m5.MEMORY_TEST).isdisjoint(m5.MEMORY_MONITOR)
    assert set(m5.DOORKEY_TEST).isdisjoint(m5.DOORKEY_MONITOR)
    assert set(m5.SEEDS).isdisjoint(m5.MEMORY_TEST + m5.DOORKEY_TEST)


def test_public_mission_encoder_is_generic_repeatable_without_private_state():
    text = "use the key to open the door and then get to the goal"
    a = m5.mission_vector(text)
    assert a.shape == (64,) and a.dtype == np.float32
    assert np.array_equal(a, m5.mission_vector(text))
    assert not np.array_equal(a, m5.mission_vector("go to the matching object"))
    assert a.sum() == pytest.approx(1.0)


def test_action_history_uses_executed_commands_and_resets():
    raw = FakePublic()
    env = m5.PublicGoalActions(raw)
    try:
        first, info = env.reset(seed=19)
        assert first.shape == (243,) and "private_inventory" in info
        assert first[-32:].tolist() == ([0.0] * 7 + [1.0]) * 4
        assert np.array_equal(first[147:211], m5.mission_vector(raw.mission))
        next_obs, _, _, _, _ = env.step(3)
        assert next_obs[-8:].tolist() == [0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0]
        assert raw.executed == [3]
        fresh, _ = env.reset(seed=23)
        assert fresh[-32:].tolist() == first[-32:].tolist()
        assert raw.last_seed == 23
    finally:
        env.close()


def test_task_local_train_stream_rejects_heldout_and_is_deterministic():
    raw = FakePublic()
    env = m5.SeededTrainingNative(
        m5.PublicGoalActions(raw), seed=71, task=m5.MEMORY
    )
    try:
        env.reset(seed=102001)
        first = raw.last_seed
        assert first > 1000000 and first != 102001
        env.step(3)
        env.reset(seed=103001)
        assert raw.last_seed == first + 1
        provenance = env.provenance()
        assert provenance["native_actions"] == 1
        assert provenance["native_resets"] == 2
        assert provenance["public_reward_sum"] == 0.0
        assert provenance["reset_seed_stream_sha256"]
    finally:
        env.close()


def test_training_schedules_match_native_steps_and_fixed_phase_blocks():
    block_count = m5.STEPS_PER_TASK // m5.ROLLOUT
    expected = {
        "blocked": {m5.MEMORY: block_count, m5.DOORKEY: block_count},
        "interleaved": {m5.MEMORY: block_count, m5.DOORKEY: block_count},
        "memory_only": {m5.MEMORY: block_count, m5.DOORKEY: 0},
        "doorkey_only": {m5.MEMORY: 0, m5.DOORKEY: block_count},
    }
    for arm, totals in expected.items():
        phases = m5.schedule(arm)
        assert len(phases) == 2
        observed = {task: sum(phase.count(task) for phase in phases)
                    for task in m5.TASKS}
        assert observed == totals
        if arm == "interleaved":
            assert all(phase[:4] == m5.TASKS * 2 for phase in phases)
            assert all(phase.count(m5.MEMORY) == block_count // 2
                       for phase in phases)
    assert sum(sum(len(phase) for phase in m5.schedule(arm))
               for arm in m5.ARMS) * len(m5.SEEDS) * m5.ROLLOUT == 786432


class FakeGrid:
    def __init__(self):
        self.width = self.height = 1
        self.door = Door("yellow", is_open=False, is_locked=True)

    def get(self, x, y):
        return self.door


class FakeDoor(FakePublic):
    def __init__(self):
        super().__init__("use the key to open the door and then get to the goal")
        self.grid = FakeGrid()
        self.carrying = None

    def reset(self, *, seed=None, options=None):
        self.grid = FakeGrid()
        self.carrying = None
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self.executed.append(action)
        if action == 3:
            self.carrying = Key("yellow")
        if action == 5 and self.carrying is not None:
            self.grid.door.is_locked = False
            self.grid.door.is_open = True
        done = action == 2
        return self._public(), float(done), done, False, {"secret": "not policy input"}


class ScriptedPolicy:
    def __init__(self, commands):
        self.commands = list(commands)

    def predict(self, observation, *, deterministic):
        assert deterministic
        assert observation.shape == (243,)
        return np.array(self.commands.pop(0)), None


def test_evaluator_only_real_pickup_door_open_goal(monkeypatch):
    def factory(_task, *, audit=False):
        assert audit
        return CountNativeSteps(
            m5.DoorKeyEffectAudit(m5.PublicGoalActions(FakeDoor())),
            max_native_steps=m5.EVAL_HORIZON,
        )

    monkeypatch.setattr(m5, "make_task", factory)
    good = m5.evaluate(ScriptedPolicy([3, 5, 2]), m5.DOORKEY, (103001,))
    assert good["rewarded_goals"] == good["verified_physical_goals"] == 1
    row = good["episode_results"][0]
    assert row["verified_key_pickups"] == 1
    assert row["verified_door_openings"] == 1
    assert row["evaluation_real_actions"] == 3
    fake = m5.evaluate(ScriptedPolicy([2]), m5.DOORKEY, (103001,))
    assert fake["rewarded_goals"] == 1
    assert fake["verified_physical_goals"] == 0


def test_score_respects_both_model_seeds_and_physical_door_gate():
    records = []
    for arm in m5.ARMS:
        for seed in m5.SEEDS:
            records.append({
                "arm": arm, "model_seed": seed,
                "final_test": {
                    m5.MEMORY: {
                        "rewarded_goals": 5 if arm == "interleaved" else 2
                    },
                    m5.DOORKEY: {
                        "rewarded_goals": 4,
                        "verified_physical_goals": 2 if arm == "interleaved" else 0,
                    },
                },
            })
    passing = m5.score(records)
    assert passing["memory_retention_difference_goals_of_24"] == 6
    assert passing["primary_gate_pass"]
    failed = [
        {**row, "final_test": {
            **row["final_test"],
            m5.DOORKEY: {**row["final_test"][m5.DOORKEY], "verified_physical_goals": 0}
        }} if row["arm"] == "interleaved" and row["model_seed"] == 83 else row
        for row in records
    ]
    assert not m5.score(failed)["primary_gate_pass"]
    with pytest.raises(ValueError, match="all eight"):
        m5.score(records[:-1])
