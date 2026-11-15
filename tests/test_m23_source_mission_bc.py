"""Stage10 source-only BC public-mission mask and physical evidence gates."""
import numpy as np
import pytest

from bagi.experiments.m15_public_action_bc import WIDTH
from bagi.experiments.m23_source_mission_bc import (
    LEARNED_ARMS,
    MISSION_START,
    MODEL_SEEDS,
    TASKS,
    decide,
    evaluate_episode,
    fit_source_policy,
    mask_public_mission,
)
from bagi.experiments.m23_study_contract import (
    BC_MONITOR_LAYOUTS,
    BC_TRAIN_LAYOUTS,
    FINAL_LAYOUTS,
    SKILL_MONITOR_LAYOUTS,
    SKILL_TRAIN_LAYOUTS,
    validate_split_contract,
)


def test_mission_ablation_masks_only_public_mission_coordinates():
    x = np.arange(WIDTH, dtype=np.float32)
    original = x.copy()
    blind = mask_public_mission(x)
    assert np.array_equal(x, original)
    assert not blind[MISSION_START:MISION_START + 64].any()
    assert np.array_equal(blind[:MISSION_START], x[:MISSION_START])
    assert np.array_equal(blind[MISSION_START + 64:], x[MISSION_START + 64:])
    with pytest.raises(ValueError):
        mask_public_mission(np.zeros((8, 12), dtype=np.float32))


def test_frozen_source_layouts_are_disjoint_and_target_is_forbidden():
    validate_split_contract()
    old = {seed for split in (FINAL_LAYOUTS, SKILL_TRAIN_LAYOUTS,
                              SKILL_MONITOR_LAYOUTS)
           for layouts in split.values() for seed in layouts}
    new = {seed for split in (BC_TRAIN_LAYOUTS, BC_MONITOR_LAYOUTS)
           for layouts in split.values() for seed in layouts}
    assert len(new) == 288
    assert not old.intersection(new)
    assert len(MODEL_SEEDS) == 3
    with pytest.raises(ValueError, match="source MONITOR"):
        evaluate_episode("unlockpickup", 453001, "actor_only", None, None)


def test_source_mission_and_blind_train_same_label_presentations():
    x = np.zeros((8, WIDTH), dtype=np.float32)
    x[:, MISSION_START] = 1.0
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    model, mission = fit_source_policy(x, y, LEARNED_ARMS[0], MODEL_SEEDS[0])
    control, blind = fit_source_policy(x, y, LEARNED_ARMS[1], MODEL_SEEDS[0])
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in control.parameters()
    )
    assert mission["actual_optimizer_step_calls"] == blind["actual_optimizer_step_calls"]
    assert mission["cached_native_action_label_presentations"] == 12 * len(y)
    assert blind["cached_native_action_label_presentations"] == 12 * len(y)
    assert mission["source_train_x_sha256"] != blind["source_train_x_sha256"]


def fake_monitor(goal_counts: dict, different: bool = True) -> dict:
    results = {}
    for task in TASKS:
        layouts = BC_MONITOR_LAYOUTS[task]
        actor = [
            {"layout_seed": layout,
             "external_physical_full_goal": i < goal_counts[(task, "actor_only")],
             "actual_executed_action_sha256": f"actor:{layout}"}
            for i, layout in enumerate(layouts)
        ]
        results[f"{task}:actor_only"] = {"episodes": actor}
        for seed in MODEL_SEEDS:
            for arm in LEARNED_ARMS:
                count = goal_counts[(task, arm)]
                results[f"{task}:{arm}:{seed}"] = {"episodes": [
                    {"layout_seed": layout,
                     "external_physical_full_goal": i < count,
                     "actual_executed_action_sha256": (
                         f"{arm}:{seed}:{layout}" if different
                         else f"actor:{layout}"
                     )}
                    for i, layout in enumerate(layouts)
                ]}
    return results


def test_physical_gate_positive_requires_real_change_and_actor_noninferiority():
    counts = {
        (task, arm): value for task in TASKS
        for arm, value in {
            "actor_only": 2, "mission_bc": 15, "mission_blind_bc": 10
        }.items()
    }
    decision = decide(fake_monitor(counts))
    assert decision["registered_decision"] == "PASS"
    assert decision["total_paired_mission_minus_blind_physical_goal_gain"] == 30
    assert decision["necessary_native_action_distinction"]
    unchanged = decide(fake_monitor(counts, different=False))
    assert unchanged["registered_decision"] == "INSUFFICIENT"


def test_positive_reward_cannot_replace_native_physical_goal():
    counts = {
        (task, arm): value for task in TASKS
        for arm, value in {
            "actor_only": 20, "mission_bc": 1, "mission_blind_bc": 0
        }.items()
    }
    outcome = decide(fake_monitor(counts))
    assert outcome["registered_decision"] == "FAIL"
    assert not outcome["physical_mission_noninferior_to_source_actor"]


def test_malformed_layout_pairing_and_unregistered_seed_rejected():
    counts = {
        (task, arm): 3 for task in TASKS
        for arm in ("actor_only", *LEARNED_ARMS)
    }
    data = fake_monitor(counts)
    data["fetch:mission_bc:941"]["episodes"][0]["layout_seed"] = 1
    with pytest.raises(ValueError, match="paired mission/blind"):
        decide(data)
    with pytest.raises(ValueError):
        fit_source_policy(np.ones((8, WIDTH), dtype=np.float32),
                          np.zeros(8, dtype=np.int64), "mission_bc", 999)
