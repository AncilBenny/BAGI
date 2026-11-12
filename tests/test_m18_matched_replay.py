"""M18 public new-goal learner, exact disjoint streams and native done audit."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import pytest
import torch
import yaml
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX

from bagi.experiments.m15_public_action_bc import BCPolicy, WIDTH
from bagi.experiments.m18_matched_replay import (
    ARMS,
    FINAL,
    MONITOR,
    SEEDS,
    TASKS,
    TRAIN,
    collect_episode,
    evaluate_episode,
    fit_phase,
    registered_gates,
    validate_config,
)
from bagi.experiments.m18_public_door_selection import (
    NativeGoToDoorAudit,
    PublicGoToDoorTeacher,
    requested_color,
)


def test_frozen_splits_disjoint_from_m15_to_m17_closed_finals():
    config = yaml.safe_load(Path("configs/m18_matched_replay.yaml").read_text())
    assert validate_config(config) == config
    seeds = sum(TRAIN + MONITOR + FINAL, ())
    assert len(seeds) == len(set(seeds))
    assert len(TRAIN[0]) == len(TRAIN[1]) == 48
    assert len(MONITOR[0]) == len(MONITOR[1]) == 12
    assert len(FINAL[0]) == len(FINAL[1]) == 32
    assert SEEDS == (641, 647, 653, 659)
    assert len(ARMS) == 5
    assert TASKS == ("MiniGrid-DoorKey-6x6-v0", "MiniGrid-GoToDoor-8x8-v0")
    for a, b in ((213001, 213025), (223001, 223025),
                 (233001, 233025), (243001, 243025),
                 (253001, 253033), (263001, 263033)):
        assert not set(seeds).intersection(range(a, b))
    wrong = dict(config)
    wrong["primary_target_min_goal_gain"] = 0
    with pytest.raises(ValueError, match="preregistered"):
        validate_config(wrong)


def test_teacher_reacts_to_public_goal_color_without_privileged_state():
    obs = {"mission": "go to the red door", "direction": 0,
           "image": np.zeros((7, 7, 3), dtype=np.uint8)}
    obs["image"][3, 5, 0] = OBJECT_TO_IDX["door"]
    obs["image"][3, 5, 1] = COLOR_TO_IDX["red"]
    teacher = PublicGoToDoorTeacher()
    assert requested_color(obs) == "red"
    assert teacher.choose(obs, np.random.default_rng(12)) == 6
    obs["image"][3, 5, 1] = COLOR_TO_IDX["blue"]
    assert teacher.choose(obs, np.random.default_rng(12)) != 6
    with pytest.raises(ValueError):
        requested_color({**obs, "mission": "hidden private coordinates"})


def test_real_native_audit_does_not_count_unexecuted_goal_or_reward():
    env = gym.wrappers.TimeLimit(gym.make(TASKS[1]), max_episode_steps=384)
    audit = NativeGoToDoorAudit(env)
    try:
        obs, _ = audit.reset(seed=MONITOR[1][0])
        assert requested_color(obs)
        _, _, terminated, truncated, _ = audit.step(0)
        assert not terminated and not truncated
        summary = audit.summary()
        assert summary["native_actions"] == 1
        assert sum(summary["action_counts"].values()) == 1
        assert summary["rewarded_terminations"] == 0
        assert not summary["full_physical_goal"]
    finally:
        env.close()


@pytest.mark.parametrize("task,seed", [
    (TASKS[0], TRAIN[0][0]), (TASKS[1], TRAIN[1][0])
])
def test_actual_training_observations_and_executed_actions(task, seed):
    episode = collect_episode(task, seed)
    assert episode["actual_native_actions"] == len(episode["actions"])
    assert episode["features"].shape == (len(episode["actions"]), WIDTH)
    assert sum(episode["action_histogram"].values()) == len(episode["actions"])
    assert len(episode["action_sha256"]) == 64
    with pytest.raises(ValueError):
        collect_episode(task, MONITOR[TASKS.index(task)][0])


def test_real_monitor_and_independent_final_guard():
    result = evaluate_episode(TASKS[1], MONITOR[1][0], "uniform", None, "monitor")
    assert 0 < result["native_actions"] <= 384
    assert "correct_target_done_events" in result
    with pytest.raises(ValueError):
        evaluate_episode(TASKS[1], FINAL[1][0], "uniform", None, "monitor")


def test_fresh_optimizer_and_cached_exposure_accounting():
    rng = np.random.default_rng(18)
    xs = rng.standard_normal((64, WIDTH)).astype(np.float32)
    ys = rng.integers(0, 7, size=64, dtype=np.int64)
    torch.manual_seed(SEEDS[0])
    model = BCPolicy()
    stat = fit_phase(model, xs, ys, SEEDS[0], (xs, ys))
    assert stat["optimizer_step_calls"] == 10
    assert stat["target_cached_label_presentations"] == 640
    assert stat["additional_cached_label_presentations"] == 640


def test_primary_gate_checks_native_physical_goals_and_model_seeds():
    final = {}
    for index, task in enumerate(TASKS):
        for mode in ("teacher", "uniform", *(
            f"{arm}_{seed}" for seed in SEEDS for arm in ARMS
        )):
            success = (0 if mode in ("teacher", "uniform") else
                       24 if index == 0 else 19 if mode.startswith("naive_") else 16)
            final[f"{task}:{mode}"] = {
                "totals": {"episodes": 32, "full_physical_goal_episodes": success}
            }
    with pytest.raises(ValueError):
        registered_gates({})
    gate = registered_gates(final)
    assert gate["target_naive_minus_scratch"] == 12
    assert gate["positive_model_seed_contrasts"] == 4
    assert gate["primary_cross_objective_gate_pass"]
    for seed in SEEDS[2:]:
        final[f"{TASKS[1]}:naive_{seed}"]["totals"][
            "full_physical_goal_episodes"
        ] = 16
    assert not registered_gates(final)["primary_cross_objective_gate_pass"]
