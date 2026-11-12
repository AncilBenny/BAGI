"""M16 source-to-target learning and retention preregistration contracts."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from bagi.experiments.m15_public_action_bc import BCPolicy, WIDTH
from bagi.experiments.m16_sequential_public_bc import (
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


def test_frozen_new_sequential_source_target_and_heldout_layouts():
    config = yaml.safe_load(Path("configs/m16_sequential_public_bc.yaml").read_text())
    assert validate_config(config) == config
    seeds = sum(TRAIN + MONITOR + FINAL, ())
    assert len(seeds) == len(set(seeds))
    assert len(TRAIN[0]) == len(TRAIN[1]) == 32
    assert len(MONITOR[0]) == len(MONITOR[1]) == 8
    assert len(FINAL[0]) == len(FINAL[1]) == 24
    assert len(SEEDS) == 2
    assert ARMS == ("source", "scratch", "naive", "replay")
    assert not set(seeds).intersection(range(213001, 213025))
    assert not set(seeds).intersection(range(223001, 223025))
    wrong = deepcopy(config)
    wrong["source_replay_ratio"] = 0
    with pytest.raises(ValueError, match="preregistered"):
        validate_config(wrong)


def test_replay_uses_extra_cached_source_labels_but_no_extra_optimizer_calls():
    rng = np.random.default_rng(16)
    x_source = rng.normal(size=(32, WIDTH)).astype(np.float32)
    y_source = rng.integers(0, 7, size=32, dtype=np.int64)
    x_target = rng.normal(size=(32, WIDTH)).astype(np.float32)
    y_target = rng.integers(0, 7, size=32, dtype=np.int64)
    torch.manual_seed(SEEDS[0])
    standard = BCPolicy()
    replay = deepcopy(standard)
    normal = fit_phase(standard, x_target, y_target, SEEDS[0])
    with_memory = fit_phase(
        replay, x_target, y_target, SEEDS[0], (x_source, y_source)
    )
    assert normal["optimizer_step_calls"] == with_memory["optimizer_step_calls"] == 10
    assert normal["target_cached_label_presentations"] == 320
    assert with_memory["target_cached_label_presentations"] == 320
    assert normal["source_replay_label_presentations"] == 0
    assert with_memory["source_replay_label_presentations"] == 320
    assert normal["parameter_count"] == with_memory["parameter_count"]
    assert np.isfinite(with_memory["mean_training_loss"])


@pytest.mark.parametrize("task,seed", [(TASKS[0], TRAIN[0][0]), (TASKS[1], TRAIN[1][0])])
def test_real_native_public_teacher_provenance(task, seed):
    row = collect_episode(task, seed)
    assert row["task"] == task and row["layout_seed"] == seed
    assert row["actual_native_actions"] == len(row["actions"]) == len(row["features"])
    assert row["features"].shape[1] == WIDTH
    assert sum(row["action_histogram"].values()) == len(row["actions"])
    assert len(row["action_sha256"]) == 64
    assert np.isfinite(row["features"]).all()
    assert not set(row).intersection({"carrying", "private_grid", "agent_pos"})
    with pytest.raises(ValueError):
        collect_episode(task, MONITOR[TASKS.index(task)][0])


def test_real_monitor_only_uniform_baseline_and_locked_final():
    row = evaluate_episode(TASKS[0], MONITOR[0][0], "uniform", None, "monitor")
    assert 0 < row["native_actions"] <= 256
    assert sum(row["action_counts"].values()) == row["native_actions"]
    assert row["split"] == "monitor"
    assert "full_physical_goal" in row
    with pytest.raises(ValueError):
        evaluate_episode(TASKS[0], FINAL[0][0], "uniform", None, "monitor")


def test_registered_transfer_and_retention_gate_resists_incomplete_results():
    data = {}
    for task_i, task in enumerate(TASKS):
        for mode in ("teacher", "uniform", *(
            f"{arm}_{seed}" for seed in SEEDS for arm in ARMS
        )):
            arm = mode.rsplit("_", 1)[0]
            if mode == "teacher":
                value = 20
            elif mode == "uniform":
                value = 0
            elif task_i == 0:
                value = {"source": 10, "scratch": 6, "naive": 6, "replay": 10}[arm]
            else:
                value = {"source": 5, "scratch": 7, "naive": 10, "replay": 9}[arm]
            data[f"{task}:{mode}"] = {
                "totals": {"full_physical_goal_episodes": value}
            }
    with pytest.raises(ValueError):
        registered_gates({})
    verdict = registered_gates(data)
    assert verdict["naive_target_minus_scratch_target"] == 6
    assert verdict["replay_source_minus_naive_source"] == 8
    assert verdict["both_gates_pass"] is True
    data[f"{TASKS[1]}:replay_{SEEDS[0]}"]["totals"]["full_physical_goal_episodes"] = 1
    assert not registered_gates(data)["rehearsal_retention_gate_pass"]
