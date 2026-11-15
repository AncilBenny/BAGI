"""M23 source-task physical-goal control and public-memory action gates."""

import numpy as np
import pytest
from minigrid.core.constants import OBJECT_TO_IDX

from bagi.experiments.m23_effect_action_selection import (
    MODES,
    TASKS,
    accept_public_override,
    decide,
    run_source_goal,
)
from bagi.experiments.m23_rule_shift import PublicEffectMemory
from bagi.experiments.m23_study_contract import (
    ACTION_MONITOR_LAYOUTS,
    ACTION_TRAIN_LAYOUTS,
    BALANCE_MONITOR_LAYOUTS,
    EFFECT_MONITOR_LAYOUTS,
    FINAL_LAYOUTS,
    MEMORY_MONITOR_LAYOUTS,
    MONITOR_LAYOUTS,
    PROBE_MONITOR_LAYOUTS,
    PROBE_TRAIN_LAYOUTS,
    RULE_SHIFT_MONITOR_LAYOUTS,
    TRAIN_LAYOUTS,
    validate_split_contract,
)


def image(obj: str, state: int = 0) -> np.ndarray:
    value = np.zeros((7, 7, 3), dtype=np.uint8)
    value[3, 5] = (OBJECT_TO_IDX[obj], 0, state)
    return value


def obs(obj: str, state: int = 0) -> dict:
    return {"image": image(obj, state), "mission": "get a blue key",
            "direction": 0}


def test_public_history_gates_observably_ineffective_attempts():
    memory = PublicEffectMemory()
    empty = obs("empty")
    key = obs("key")
    assert accept_public_override("unfiltered", empty, 4, memory)
    assert accept_public_override("front_rule_gate", empty, 4, memory)
    assert not accept_public_override("history_rule_gate", empty, 4, memory)
    assert accept_public_override("history_rule_gate", key, 3, memory)
    memory.update_after(key["image"], 3, empty["image"])
    assert accept_public_override("history_rule_gate", empty, 4, memory)
    assert accept_public_override("history_rule_gate", obs("door", 2), 5, memory)
    memory.update_after(empty["image"], 4, key["image"])
    assert not accept_public_override("history_rule_gate", empty, 4, memory)
    assert not accept_public_override("history_rule_gate", obs("door", 2), 5, memory)


def test_private_nonpublic_observations_and_actions_rejected():
    memory = PublicEffectMemory()
    with pytest.raises(ValueError):
        accept_public_override("history_rule_gate", {
            **obs("key"), "native_carrying": "key",
        }, 3, memory)
    with pytest.raises(ValueError):
        accept_public_override("unfiltered", obs("key"), 2, memory)
    with pytest.raises(ValueError):
        accept_public_override("unregistered", obs("key"), 3, memory)


def test_stage8_source_layouts_never_use_earlier_or_target_final():
    validate_split_contract()
    new = {
        x for split in (ACTION_TRAIN_LAYOUTS, ACTION_MONITOR_LAYOUTS)
        for rows in split.values() for x in rows
    }
    prior = {
        x
        for split in (
            TRAIN_LAYOUTS, MONITOR_LAYOUTS, FINAL_LAYOUTS,
            BALANCE_MONITOR_LAYOUTS, MEMORY_MONITOR_LAYOUTS,
            EFFECT_MONITOR_LAYOUTS, PROBE_TRAIN_LAYOUTS,
            PROBE_MONITOR_LAYOUTS, RULE_SHIFT_MONITOR_LAYOUTS,
        )
        for rows in split.values()
        for x in rows
    }
    assert len(new) == 192
    assert not new.intersection(prior)
    assert set(TASKS) == {"doorkey", "fetch"}
    assert len(MODES) == 3
    with pytest.raises(ValueError, match="unregistered"):
        run_source_goal("unlockpickup", 453001, "history_rule_gate",
                        "action_monitor")
    with pytest.raises(ValueError, match="outside frozen"):
        run_source_goal("fetch", 612001, "history_rule_gate", "action_monitor")


def test_physical_goal_gate_requires_paired_task_gains_and_disagreement():
    registered = {}
    for task in TASKS:
        for mode in MODES:
            registered[f"{task}:{mode}"] = {
                "episodes": [{
                    "seed": seed,
                    "external_physical_full_goal": (
                        (i < 10) if mode == "history_rule_gate" else (i < 4)
                    ),
                    "front_history_rule_disagreement_attempts": 1,
                } for i, seed in enumerate(ACTION_MONITOR_LAYOUTS[task])]
            }
    passed = decide(registered)
    assert passed["registered_source_action_selection_gate_pass"]
    assert passed["total_paired_goal_gain_memory_minus_front"] == 12
    registered["fetch:history_rule_gate"]["episodes"][8][
        "external_physical_full_goal"
    ] = False
    # This remains a registered gain and should not fail from one loss.
    assert decide(registered)["total_paired_goal_gain_memory_minus_front"] == 11


def test_no_physical_goal_not_replaced_with_reward_only_label():
    registered = {}
    for task in TASKS:
        for mode in MODES:
            registered[f"{task}:{mode}"] = {
                "episodes": [{
                    "seed": seed,
                    "external_physical_full_goal": False,
                    "front_history_rule_disagreement_attempts": 1,
                } for seed in ACTION_MONITOR_LAYOUTS[task]]
            }
    result = decide(registered)
    assert result["total_paired_goal_gain_memory_minus_front"] == 0
    assert not result["registered_source_action_selection_gate_pass"]
