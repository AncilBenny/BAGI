"""M17 controlled old-source replay versus equal extra target presentations."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from bagi.experiments.m15_public_action_bc import WIDTH, BCPolicy
from bagi.experiments.m17_matched_replay import (
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


def test_m17_exact_preregistered_splits_and_arms():
    config = yaml.safe_load(Path("configs/m17_matched_replay.yaml").read_text())
    assert validate_config(config) == config
    all_seeds = sum(TRAIN + MONITOR + FINAL, ())
    assert len(all_seeds) == len(set(all_seeds))
    assert len(TRAIN[0]) == len(TRAIN[1]) == 48
    assert len(MONITOR[0]) == len(MONITOR[1]) == 12
    assert len(FINAL[0]) == len(FINAL[1]) == 32
    assert len(SEEDS) == 4
    assert ARMS == ("source", "scratch", "naive", "extra_target", "source_replay")
    assert not set(all_seeds).intersection(range(233001, 233025))
    assert not set(all_seeds).intersection(range(243001, 243025))
    damaged = copy.deepcopy(config)
    damaged["primary_target_min_goal_gain"] = 1
    with pytest.raises(ValueError, match="preregistered"):
        validate_config(damaged)


def test_equal_cached_exposure_and_optimizer_calls_across_replay_arms():
    rng = np.random.default_rng(17)
    old_x = rng.standard_normal((64, WIDTH)).astype(np.float32)
    old_y = rng.integers(0, 7, size=64, dtype=np.int64)
    new_x = rng.standard_normal((64, WIDTH)).astype(np.float32)
    new_y = rng.integers(0, 7, size=64, dtype=np.int64)
    torch.manual_seed(SEEDS[0])
    init = BCPolicy()
    naive = copy.deepcopy(init)
    target_extra = copy.deepcopy(init)
    source_extra = copy.deepcopy(init)
    naive_result = fit_phase(naive, new_x, new_y, SEEDS[0])
    target_result = fit_phase(
        target_extra, new_x, new_y, SEEDS[0], (new_x, new_y)
    )
    source_result = fit_phase(
        source_extra, new_x, new_y, SEEDS[0], (old_x, old_y)
    )
    assert all(
        x["optimizer_step_calls"] == 10
        and x["target_cached_label_presentations"] == 640
        for x in (naive_result, target_result, source_result)
    )
    assert naive_result["additional_cached_label_presentations"] == 0
    assert target_result["additional_cached_label_presentations"] == 640
    assert source_result["additional_cached_label_presentations"] == 640
    assert not all(torch.equal(a, b) for a, b in zip(
        target_extra.parameters(), source_extra.parameters()
    ))


@pytest.mark.parametrize("task,seed", [
    (TASKS[0], TRAIN[0][0]), (TASKS[1], TRAIN[1][0])
])
def test_genuine_native_public_actions_only_on_train_layouts(task, seed):
    episode = collect_episode(task, seed)
    assert episode["layout_seed"] == seed
    assert episode["actual_native_actions"] == len(episode["actions"])
    assert len(episode["actions"]) == len(episode["features"])
    assert episode["features"].shape[1] == WIDTH
    assert sum(episode["action_histogram"].values()) == len(episode["actions"])
    assert len(episode["action_sha256"]) == 64
    assert np.isfinite(episode["features"]).all()
    assert not set(episode).intersection({"carrying", "agent_pos", "private_state"})
    with pytest.raises(ValueError):
        collect_episode(task, MONITOR[TASKS.index(task)][0])


def test_real_native_monitor_and_no_accidental_final_opening():
    row = evaluate_episode(TASKS[1], MONITOR[1][0], "uniform", None, "monitor")
    assert row["native_actions"] > 0
    assert row["native_actions"] <= 256
    assert sum(row["action_counts"].values()) == row["native_actions"]
    with pytest.raises(ValueError):
        evaluate_episode(TASKS[1], FINAL[1][0], "uniform", None, "monitor")


def test_seed_pair_common_action_rng_across_replay_arms_on_monitor():
    # A zero-logit network yields the same native categorical draws in each arm,
    # because same-layout, same-seed streams are intentionally coupled.
    torch.manual_seed(15)
    model = BCPolicy()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    a = evaluate_episode(TASKS[0], MONITOR[0][0], f"source_replay_{SEEDS[0]}",
                         model, "monitor")
    b = evaluate_episode(TASKS[0], MONITOR[0][0], f"extra_target_{SEEDS[0]}",
                         model, "monitor")
    assert a["rng_seed"] == b["rng_seed"]
    assert a["native_actions"] == b["native_actions"]
    assert a["action_counts"] == b["action_counts"]
    assert a["full_physical_goal"] == b["full_physical_goal"]


def test_full_physical_gate_requires_all_arms_and_multiple_seeds():
    final = {}
    for task_idx, task in enumerate(TASKS):
        for mode in ("teacher", "uniform", *(
            f"{arm}_{seed}" for seed in SEEDS for arm in ARMS
        )):
            if mode in ("teacher", "uniform"):
                successes = 0
            elif task_idx == 0:
                successes = 24
            elif mode.startswith("source_replay"):
                successes = 26
            elif mode.startswith("extra_target"):
                successes = 21
            else:
                successes = 20
            final[f"{task}:{mode}"] = {
                "totals": {"episodes": 32, "full_physical_goal_episodes": successes}
            }
    with pytest.raises(ValueError):
        registered_gates({})
    gate = registered_gates(final)
    assert gate["target_source_replay_minus_exposure_matched_target"] == 20
    assert gate["positive_seed_contrasts"] == 4
    assert gate["primary_exposure_matched_gate_pass"]
    for seed in SEEDS[2:]:
        final[f"{TASKS[1]}:source_replay_{seed}"]["totals"][
            "full_physical_goal_episodes"
        ] = 21
    assert not registered_gates(final)["primary_exposure_matched_gate_pass"]
    broken = final[f"{TASKS[1]}:source_replay_{SEEDS[0]}"]["totals"]
    broken["episodes"] = 31
    with pytest.raises(ValueError, match="missing physical"):
        registered_gates(final)
