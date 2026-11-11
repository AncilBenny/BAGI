"""M1 contracts: public executed actions, configurable training and safe output, no live RL fit."""
import json
import sys
from pathlib import Path
from types import ModuleType

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml
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
        "output_dir": "runs/m1_goal_learning",
    }


def test_configurable_training_and_disjoint_evaluation_splits():
    assert m1.validate_config(config())["train_steps_per_arm_seed"] == 8192
    changed = {**config(), "train_steps_per_arm_seed": 16384, "rollout_steps": 256,
               "batch_size": 128, "epochs_per_rollout": 4,
               "output_dir": "runs/m1_retraining/new_trial"}
    assert m1.validate_config(changed)["train_steps_per_arm_seed"] == 16384
    for override in (
        {"train_steps_per_arm_seed": 0},
        {"rollout_steps": 1},
        {"epochs_per_rollout": -1},
        {"output_dir": "../old_results"},
        {"arms": ["recurrent"]},
        {"training_seeds": [17, 17]},
        {"transfer_seeds": [92001, 93002, 93003, 93004, 93005, 93006]},
        {"evaluation_seeds": [92007, 92002, 92003, 92004, 92005, 92006]},
        {"transfer_seeds": [93007, 93002, 93003, 93004, 93005, 93006]},
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
    """Reactive/recurrent share a public action input; controls have legal history."""
    environments = {}
    try:
        for arm in m1.ARMS:
            environments[arm] = m1.make_task(m1.TRAIN_TASK, arm)
        shapes = {arm: env.observation_space.shape for arm, env in environments.items()}
        image_width = 7 * 7 * 3  # Standard MiniGrid local public image.
        n_actions = environments["action4"].action_space.n
        assert shapes["reactive"] == (image_width + n_actions + 1,)
        assert shapes["recurrent"] == shapes["reactive"]
        assert shapes["action4"] == (image_width + 4 * (n_actions + 1),)
        assert shapes["frame4"] == (4 * image_width,)
        assert all(isinstance(environments[arm].env, m1.PublicExecutedActions)
                   and environments[arm].env.n_history == 1
                   for arm in ("reactive", "recurrent"))
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
            env_factory=lambda _task, cls=env_class: m1.CountNativeSteps(cls()),
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


def test_training_can_execute_locally_without_a_quota_or_ci_identity(monkeypatch):
    import sys

    for key in ("BAGI_M1_ADMITTED", "GITHUB_WORKFLOW", "GITHUB_JOB"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(sys, "argv", ["m1_goal_learning"])
    with pytest.raises(SystemExit, match="--execute"):
        m1.main()
    calls = []
    monkeypatch.setattr(m1, "run", lambda config, output: calls.append((config, output)))
    monkeypatch.setattr(sys, "argv", ["m1_goal_learning", "--execute"])
    m1.main()
    assert len(calls) == 1


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


def test_m1_source_provenance_binds_to_checked_out_audited_head(monkeypatch):
    """Event GITHUB_SHA is not proof of the source actually used for M1."""
    actual = "a" * 40
    monkeypatch.setattr(
        m1.subprocess, "check_output",
        lambda *args, **kwargs: actual + "\n",
    )
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    monkeypatch.delenv("BAGI_RESEARCH_SOURCE_SHA", raising=False)
    assert m1.admitted_source_commit() == actual
    monkeypatch.setenv("BAGI_RESEARCH_SOURCE_SHA", "c" * 40)
    with pytest.raises(ValueError, match="admitted research HEAD"):
        m1.admitted_source_commit()
    monkeypatch.setenv("BAGI_RESEARCH_SOURCE_SHA", actual)
    assert m1.admitted_source_commit() == actual

def test_native_action_limit_blocks_extra_calls_before_simulator_and_after_reset():
    raw = TinyPublic()
    env = m1.CountNativeSteps(raw, max_native_steps=1)
    try:
        env.reset(seed=17)
        env.step(0)
        assert raw.index == 1 and env.native_steps == 1
        with pytest.raises(RuntimeError, match="before env.step"):
            env.step(0)
        assert raw.index == 1 and env.native_steps == 1
        env.reset(seed=29)
        with pytest.raises(RuntimeError, match="before env.step"):
            env.step(1)
        assert raw.index == 0 and env.native_resets == 2
        assert env.native_steps == 1
    finally:
        env.close()


def test_native_action_limit_is_configured_per_environment_and_starts_unspent():
    for arm in m1.ARMS:
        env = m1.make_task(
            m1.TRAIN_TASK, arm, max_native_steps=8192
        )
        evaluation = m1.make_task(
            m1.TRANSFER_TASK, arm,
            max_native_steps=m1.MAX_EVAL_EPISODE_STEPS
        )
        try:
            assert env.max_native_steps == 8192
            assert evaluation.max_native_steps == 120
            assert env.native_steps == evaluation.native_steps == 0
            assert env.native_resets == evaluation.native_resets == 0
        finally:
            env.close()
            evaluation.close()


@pytest.mark.parametrize("invalid", (0, -1, True, 2.5, "8192"))
def test_native_action_limit_rejects_invalid_caps(invalid):
    env = TinyPublic()
    with pytest.raises(ValueError, match="positive integer"):
        m1.CountNativeSteps(env, max_native_steps=invalid)
    env.close()


def test_training_reward_telemetry_counts_public_rewards_not_diagnostic_info():
    # Both mocks return the same positive reward. A truncation is not a
    # rewarded terminal goal, even though positive native reward was observed.
    for kind, expected_goals in ((TinyPublic, 1), (PositiveTruncated, 0)):
        env = m1.CountNativeSteps(kind(), max_native_steps=3)
        try:
            env.reset(seed=17)
            env.step(0)
            _, reward, terminated, truncated, _ = env.step(0)
            assert reward == 1.0
            assert terminated != truncated
            assert env.native_steps == 2
            assert env.native_positive_reward_steps == 1
            assert env.native_rewarded_goal_terminations == expected_goals
            assert env.native_public_reward_sum == 1.0
            # A new episode must not erase actual training counters.
            env.reset(seed=29)
            env.step(0)
            assert env.native_steps == 3
            assert env.native_positive_reward_steps == 1
            assert env.native_rewarded_goal_terminations == expected_goals
            assert env.native_resets == 2
            with pytest.raises(RuntimeError, match="before env.step"):
                env.step(0)
            assert env.native_steps == 3
        finally:
            env.close()


def test_single_previous_executed_action_and_episode_reset_are_public_only():
    raw = TinyPublic()
    env = m1.CountNativeSteps(m1.PublicExecutedActions(raw, 1))
    try:
        first, info = env.reset(seed=17)
        assert info["private_mock"] == "not in observation"
        assert first.shape == (6,)
        assert first[-3:].tolist() == [0, 0, 1]
        subsequent, _, _, _, _ = env.step(1)
        assert subsequent[-3:].tolist() == [0, 1, 0]
        assert subsequent.shape == first.shape
        assert env.native_steps == 1
        fresh, _ = env.reset(seed=29)
        assert fresh[-3:].tolist() == [0, 0, 1]
        assert env.native_steps == 1 and env.native_resets == 2
    finally:
        env.close()

def test_interrupted_model_measurement_counts_real_steps_and_resets_only():
    training = m1.CountNativeSteps(TinyPublic())
    evaluation = m1.CountNativeSteps(TinyPublic())
    try:
        training.reset(seed=17)
        training.step(0)
        evaluation.reset(seed=92001)
        evaluation.step(0)
        count = {"calls": 3}
        snapshot = m1.interrupted_model_measurement(
            training, [evaluation], count
        )
        assert snapshot["status"] == "interrupted_not_completed"
        assert snapshot["training_real_actions"] == 1
        assert snapshot["training_resets"] == 1
        assert snapshot["evaluation_real_actions"] == 1
        assert snapshot["evaluation_resets"] == 1
        assert snapshot["optimizer_step_calls"] == 3
    finally:
        training.close()
        evaluation.close()


def test_interrupted_training_writes_provenance_bound_partial_ledger(
        monkeypatch, tmp_path):
    """Failure during learn preserves performed work but never scores a fit."""
    class InterruptedAgent:
        def __init__(self, _policy, env, **_kwargs):
            self.env = env
            self.policy = ToyPolicy()

        def learn(self, total_timesteps):
            assert total_timesteps == 8192
            self.env.reset(seed=17)
            self.env.step(0)
            self.policy.optimizer.step()
            raise RuntimeError("synthetic training interruption")

    def evaluated(_agent, task, _seeds, env_factory):
        env = env_factory(task)
        env.reset(seed=92001)
        env.step(0)
        env.close()
        return {"evaluation_real_steps": 1}

    sb3 = ModuleType("stable_baselines3")
    sb3.PPO = InterruptedAgent
    contrib = ModuleType("sb3_contrib")
    contrib.RecurrentPPO = InterruptedAgent
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
    monkeypatch.setitem(sys.modules, "sb3_contrib", contrib)
    monkeypatch.setattr(
        m1, "make_task",
        lambda _task, _arm, max_native_steps=None: m1.CountNativeSteps(
            TinyPublic(), max_native_steps=max_native_steps
        ),
    )
    monkeypatch.setattr(m1, "evaluate_goal_agent", evaluated)
    monkeypatch.setattr(m1, "admitted_source_commit", lambda: "a" * 40)
    monkeypatch.chdir(tmp_path)
    conf = tmp_path / "frozen.yaml"
    conf.write_text(yaml.safe_dump(config()), encoding="utf-8")
    output = Path("runs/m1_goal_learning/summary.json")
    with pytest.raises(RuntimeError, match="synthetic training interruption"):
        m1.run(conf, output)
    progress = json.loads(Path("runs/m1_goal_learning/progress.json").read_text())
    assert progress["status"] == "interrupted_not_a_completed_study"
    assert progress["source_commit"] == "a" * 40
    assert progress["failure_phase"] == "training"
    assert progress["failure_type"] == "RuntimeError"
    assert progress["completed_models"] == 0
    assert progress["records"] == []
    assert progress["observed_training_real_actions"] == 1
    assert progress["observed_evaluation_real_actions"] == 1
    assert progress["observed_optimizer_step_calls"] == 1
    assert progress["in_flight_model"]["training_resets"] == 1
    assert progress["in_flight_model"]["evaluation_resets"] == 1
    assert not output.exists()
    assert not Path("runs/m1_goal_learning/progress.json.pending").exists()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        m1.run(conf, output)


def test_m2_extended_study_preserves_independent_frozen_seed_contract():
    longer = {
        **config(),
        "registered_study": "m2",
        "train_steps_per_arm_seed": 65536,
        "evaluation_seeds": list(m1.M2_EVAL_SEEDS),
        "transfer_seeds": list(m1.M2_TRANSFER_SEEDS),
        "output_dir": "runs/m2_goal_learning",
    }
    assert m1.validate_config(longer)["registered_study"] == "m2"
    for change in (
        {"registered_study": "m3"},
        {"train_steps_per_arm_seed": 8192},
        {"output_dir": "runs/m1_goal_learning"},
        {"epochs_per_rollout": 3},
        {"evaluation_seeds": list(m1.FROZEN_EVAL_SEEDS)},
        {"transfer_seeds": list(m1.FROZEN_TRANSFER_SEEDS)},
        {"evaluation_seeds": [94000, 94002, 94003, 94004, 94005, 94006]},
        {"transfer_seeds": [95000, 95002, 95003, 95004, 95005, 95006]},
    ):
        with pytest.raises(ValueError):
            m1.validate_config({**longer, **change})


def test_sampled_policy_evaluation_is_reproducible_public_goal_scoring():
    class SampledPolicy:
        def __init__(self):
            self.modes = []
            self.starts = []

        def predict(self, _observation, *, state, episode_start, deterministic):
            self.modes.append(deterministic)
            self.starts.append(bool(episode_start[0]))
            return np.array(0), state

    agent = SampledPolicy()
    stats = m1.evaluate_goal_agent(
        agent, "mock", [94001, 94002],
        env_factory=lambda _task: m1.CountNativeSteps(TinyPublic()),
        deterministic=False,
    )
    assert stats["successes"] == 2
    assert stats["evaluation_real_steps"] == 4
    assert agent.modes == [False] * 4
    assert agent.starts == [True, False, True, False]
    assert all(not row["deterministic"] for row in stats["episode_results"])
