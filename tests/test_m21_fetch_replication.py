"""Independent M21 literal replication; no M20 closed final layout exposure."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from bagi.experiments.m15_public_action_bc import WIDTH, BCPolicy
from bagi.experiments.m20_fetch_physics import NativeFetchAudit, PublicFetchTeacher
from bagi.experiments.m21_fetch_replication import (
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


def test_m21_registered_source_and_disjoint_unopened_finals():
    cfg = yaml.safe_load(Path("configs/m21_fetch_replication.yaml").read_text())
    assert validate_config(cfg) == cfg
    assert cfg["output_dir"] == "runs/m21_fetch_replication"
    assert TASKS == ("MiniGrid-DoorKey-6x6-v0", "MiniGrid-Fetch-5x5-N2-v0")
    assert SEEDS == (811, 821, 823, 827)
    assert cfg["train_seeds"] == [
        list(range(331001, 331049)), list(range(341001, 341049))
    ]
    assert cfg["monitor_seeds"] == [
        list(range(332001, 332013)), list(range(342001, 342013))
    ]
    assert cfg["final_seeds"] == [
        list(range(333001, 333033)), list(range(343001, 343033))
    ]
    assert ARMS == ("source", "scratch", "naive", "extra_target", "source_replay")
    m21 = set(sum(TRAIN + MONITOR + FINAL, ()))
    m20 = set()
    for start, n in (
        (311001, 48), (321001, 48), (312001, 12), (322001, 12),
        (313001, 32), (323001, 32)
    ):
        m20.update(range(start, start + n))
    assert len(m21) == 184
    assert m21.isdisjoint(m20)
    assert len(m21) == len(sum(TRAIN + MONITOR + FINAL, ()))
    changed = dict(cfg)
    changed["primary_target_min_transfer_gain"] = 1
    with pytest.raises(ValueError, match="preregistered"):
        validate_config(changed)


def test_m21_same_public_fetch_teacher_and_external_audit():
    assert isinstance(make_teacher(TASKS[1]), PublicFetchTeacher)
    auditor = make_audit(TASKS[1])
    try:
        assert isinstance(auditor, NativeFetchAudit)
    finally:
        auditor.env.close()


def test_m21_seals_final_layouts_from_train_and_monitor():
    ep = collect_episode(TASKS[1], TRAIN[1][0])
    assert ep["actual_native_actions"] == len(ep["actions"]) == len(ep["features"])
    assert sum(ep["action_histogram"].values()) == ep["actual_native_actions"]
    with pytest.raises(ValueError):
        collect_episode(TASKS[1], MONITOR[1][0])
    monitor = evaluate_episode(TASKS[1], MONITOR[1][0], "uniform", None, "monitor")
    assert "correct_target_pickup_events" in monitor
    with pytest.raises(ValueError):
        evaluate_episode(TASKS[1], FINAL[1][0], "uniform", None, "monitor")


def test_m21_equal_extra_cached_label_exposure():
    rng = np.random.default_rng(21)
    x = rng.standard_normal((64, WIDTH)).astype(np.float32)
    y = rng.integers(0, 7, size=64, dtype=np.int64)
    torch.manual_seed(SEEDS[0])
    model = BCPolicy()
    result = fit_phase(model, x, y, SEEDS[0], (x, y))
    assert result["optimizer_step_calls"] == 10
    assert result["target_cached_label_presentations"] == 640
    assert result["additional_cached_label_presentations"] == 640


def test_m21_identical_gate_thresholds_are_conjunctive():
    scores = {}
    for task_idx, task in enumerate(TASKS):
        for mode in ("teacher", "uniform", *(
            f"{arm}_{seed}" for seed in SEEDS for arm in ARMS
        )):
            goals = 0
            if mode not in ("teacher", "uniform"):
                if task_idx == 0:
                    goals = 24 if mode.startswith("source_replay_") else 19
                else:
                    goals = (
                        21 if mode.startswith("source_replay_")
                        else 14 if mode.startswith("scratch_") else 22
                    )
            scores[f"{task}:{mode}"] = {"totals": {
                "episodes": 32, "full_physical_goal_episodes": goals,
            }}
    assert registered_gates(scores)["primary_cross_mechanics_gate_pass"]
    scores[f"{TASKS[1]}:naive_{SEEDS[0]}"]["totals"][
        "full_physical_goal_episodes"
    ] = 14
    assert not registered_gates(scores)["primary_cross_mechanics_gate_pass"]
