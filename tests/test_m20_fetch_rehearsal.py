"""M20 reverse native-task-order, matched source rehearsal and physical goals."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from bagi.experiments.m10_native_audit import NativeDoorKeyAudit
from bagi.experiments.m14_public_open_door import TraversableOpenDoorController
from bagi.experiments.m15_public_action_bc import WIDTH, BCPolicy
from bagi.experiments.m20_fetch_physics import (
    NativeFetchAudit,
    PublicFetchTeacher,
    fetch_request,
)
from bagi.experiments.m20_fetch_rehearsal import (
    ARMS,
    FINAL,
    MONITOR,
    SEEDS,
    TASKS,
    TRAIN,
    collect_episode,
    evaluate_episode,
    fit_phase,
    make_audit,
    make_teacher,
    registered_gates,
    validate_config,
)


def test_reversed_native_task_order_and_sealed_layout_streams():
    config = yaml.safe_load(Path("configs/m20_fetch_rehearsal.yaml").read_text())
    assert validate_config(config) == config
    seeds = sum(TRAIN + MONITOR + FINAL, ())
    assert len(seeds) == len(set(seeds))
    assert TASKS == ("MiniGrid-DoorKey-6x6-v0", "MiniGrid-Fetch-5x5-N2-v0")
    assert SEEDS == (751, 757, 761, 769)
    assert len(TRAIN[0]) == len(TRAIN[1]) == 48
    assert len(MONITOR[0]) == len(MONITOR[1]) == 12
    assert len(FINAL[0]) == len(FINAL[1]) == 32
    assert ARMS == ("source", "scratch", "naive", "extra_target", "source_replay")
    for first, last in ((213001, 213025), (223001, 223025),
                        (233001, 233025), (243001, 243025),
                        (253001, 253033), (263001, 263033),
                        (273001, 273033), (283001, 283033)):
        assert set(seeds).isdisjoint(range(first, last))
    altered = dict(config)
    altered["primary_old_task_min_replay_gain"] = 1
    with pytest.raises(ValueError, match="preregistered"):
        validate_config(altered)


def test_public_actor_mapping_and_evaluator_only_native_audits():
    assert isinstance(make_teacher(TASKS[0]), TraversableOpenDoorController)
    assert isinstance(make_teacher(TASKS[1]), PublicFetchTeacher)
    source_audit = make_audit(TASKS[0])
    target_audit = make_audit(TASKS[1])
    try:
        assert isinstance(source_audit, NativeDoorKeyAudit)
        assert isinstance(target_audit, NativeFetchAudit)
    finally:
        source_audit.env.close()
        target_audit.env.close()


@pytest.mark.parametrize("task,seed", [
    (TASKS[0], TRAIN[0][0]), (TASKS[1], TRAIN[1][0])
])
def test_new_public_teacher_actions_and_actual_trace_hash(task, seed):
    episode = collect_episode(task, seed)
    assert episode["layout_seed"] == seed
    assert episode["features"].shape == (episode["actual_native_actions"], WIDTH)
    assert len(episode["actions"]) == episode["actual_native_actions"]
    assert sum(episode["action_histogram"].values()) == episode["actual_native_actions"]
    assert len(episode["action_sha256"]) == 64
    assert np.isfinite(episode["features"]).all()
    with pytest.raises(ValueError):
        collect_episode(task, MONITOR[TASKS.index(task)][0])


def test_separate_final_and_train_monitor_streams():
    row = evaluate_episode(TASKS[0], MONITOR[0][0], "uniform", None, "monitor")
    assert 0 < row["native_actions"] <= 384
    assert "effective_pickup_events" in row
    assert sum(row["action_counts"].values()) == row["native_actions"]
    with pytest.raises(ValueError):
        evaluate_episode(TASKS[0], FINAL[0][0], "uniform", None, "monitor")


def test_equal_extra_cached_label_exposure_and_optimizer_steps():
    rng = np.random.default_rng(19)
    x = rng.standard_normal((64, WIDTH)).astype(np.float32)
    y = rng.integers(0, 7, 64, dtype=np.int64)
    torch.manual_seed(SEEDS[0])
    model = BCPolicy()
    result = fit_phase(model, x, y, SEEDS[0], (x, y))
    assert result["optimizer_step_calls"] == 10
    assert result["target_cached_label_presentations"] == 640
    assert result["additional_cached_label_presentations"] == 640


def test_preregistered_retention_requires_native_physical_goals_and_target_ability():
    scores = {}
    for index, task in enumerate(TASKS):
        for mode in ("teacher", "uniform", *(
            f"{arm}_{seed}" for seed in SEEDS for arm in ARMS
        )):
            successes = 0
            if mode not in ("teacher", "uniform"):
                if index == 0:
                    successes = (24 if mode.startswith("source_replay_") else 19)
                else:
                    successes = (
                        21 if mode.startswith("source_replay_")
                        else 14 if mode.startswith("scratch_") else 22
                    )
            scores[f"{task}:{mode}"] = {"totals": {
                "episodes": 32, "full_physical_goal_episodes": successes,
            }}
    with pytest.raises(ValueError):
        registered_gates({})
    gate = registered_gates(scores)
    assert gate["old_replay_minus_exposure_matched_target"] == 20
    assert gate["old_positive_model_seed_contrasts"] == 4
    assert gate["target_naive_minus_replay"] == 4
    assert gate["target_naive_minus_scratch"] == 32
    assert gate["target_positive_model_seed_contrasts"] == 4
    assert gate["primary_cross_mechanics_gate_pass"]
    for seed in SEEDS[2:]:
        scores[f"{TASKS[0]}:source_replay_{seed}"]["totals"][
            "full_physical_goal_episodes"
        ] = 19
    assert not registered_gates(scores)["primary_cross_mechanics_gate_pass"]


def test_fetch_mission_and_wrong_object_rejection_are_public_only():
    from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX

    obs = {
        "image": np.zeros((7, 7, 3), dtype=np.uint8),
        "direction": 0,
        "mission": "go fetch a blue ball",
    }
    assert fetch_request(obs) == ("blue", "ball")
    actor = PublicFetchTeacher()
    rng = np.random.default_rng(20)
    obs["image"][3, 5, 0] = OBJECT_TO_IDX["ball"]
    obs["image"][3, 5, 1] = COLOR_TO_IDX["blue"]
    assert actor.choose(obs, rng) == 3
    obs["image"][3, 5, 1] = COLOR_TO_IDX["red"]
    assert actor.choose(obs, rng) in (0, 1)
    obs["mission"] = "unknown private target"
    with pytest.raises(ValueError):
        actor.choose(obs, rng)


def test_train_only_native_fetch_witness_separates_reward_and_pickup():
    audit = make_audit(TASKS[1])
    teacher = make_teacher(TASKS[1])
    rng = np.random.default_rng(20_000)
    try:
        obs, _ = audit.reset(seed=TRAIN[1][1])
        for _ in range(384):
            action = teacher.choose(obs, rng)
            following, _, terminated, truncated, _ = audit.step(action)
            teacher.update(obs, action, following)
            obs = following
            if terminated or truncated:
                break
        summary = audit.summary()
        assert 0 < summary["native_actions"] <= 384
        assert sum(summary["action_counts"].values()) == summary["native_actions"]
        assert summary["correct_target_pickup_events"] in (0, 1)
        assert summary["wrong_target_pickup_events"] in (0, 1)
        if summary["full_physical_goal"]:
            assert summary["correct_target_pickup_events"] == 1
            assert summary["rewarded_terminations"] == 1
            assert summary["effective_forward_movements"] > 0
            assert not summary["initially_adjacent_target"]
    finally:
        audit.env.close()
