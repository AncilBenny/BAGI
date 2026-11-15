"""Stage9 native source mission-relevance policy and preregistered gate."""
import numpy as np
import pytest
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX

from bagi.experiments.m23_mission_skill_selection import (
    MODES,
    TASKS,
    accept_override,
    decide,
    mission_kind,
    mission_relevant_override,
    run_source_goal,
)
from bagi.experiments.m23_rule_shift import PublicEffectMemory
from bagi.experiments.m23_study_contract import (
    ACTION_MONITOR_LAYOUTS,
    ACTION_TRAIN_LAYOUTS,
    FINAL_LAYOUTS,
    SKILL_MONITOR_LAYOUTS,
    SKILL_TRAIN_LAYOUTS,
    validate_split_contract,
)


def obs(mission="get a blue key", item="key", color="blue"):
    image = np.zeros((7, 7, 3), dtype=np.uint8)
    image[3, 5] = (
        OBJECT_TO_IDX[item], COLOR_TO_IDX[color], 0
    )
    return {"image": image, "mission": mission, "direction": 0}


def test_mission_input_is_public_only_and_unknown_mission_rejected():
    assert mission_kind(obs()) == ("fetch", ("blue", "key"))
    keydoor = obs("use the key to open the door and then get to the goal")
    assert mission_kind(keydoor) == ("doorkey", None)
    with pytest.raises(ValueError):
        mission_kind({**obs(), "carrying": "key"})
    with pytest.raises(ValueError):
        mission_kind(obs("unlock pickup the blue box"))
    with pytest.raises(ValueError):
        mission_kind({**obs(), "mission": 17})


def test_mission_relevance_rejects_wrong_fetch_object_and_irrelevant_drop():
    memory = PublicEffectMemory()
    assert mission_relevant_override(obs(), 3, memory)
    assert not mission_relevant_override(obs(color="red"), 3, memory)
    assert not mission_relevant_override(obs(item="ball"), 3, memory)
    assert not mission_relevant_override(obs(), 4, memory)
    assert not mission_relevant_override(obs(), 5, memory)
    for mode in MODES:
        assert type(accept_override(mode, obs(), 3, memory)) is bool
    with pytest.raises(ValueError):
        accept_override("mission_skill_gate", obs(), 6, memory)


def test_doorkey_public_mission_only_considers_key_pickup():
    memory = PublicEffectMemory()
    mission = "use the key to open the door and then get to the goal"
    assert mission_relevant_override(obs(mission), 3, memory)
    assert not mission_relevant_override(obs(mission, item="ball"), 3, memory)
    assert not mission_relevant_override(obs(mission), 4, memory)


def test_new_frozen_layouts_disjoint_and_target_not_dispatchable():
    validate_split_contract()
    prior = {
        x for split in (ACTION_TRAIN_LAYOUTS, ACTION_MONITOR_LAYOUTS, FINAL_LAYOUTS)
        for seeds in split.values() for x in seeds
    }
    fresh = {
        x for split in (SKILL_TRAIN_LAYOUTS, SKILL_MONITOR_LAYOUTS)
        for seeds in split.values() for x in seeds
    }
    assert len(fresh) == 256
    assert not prior & fresh
    assert len(TASKS) == 2 and len(MODES) == 4
    with pytest.raises(ValueError, match="unregistered"):
        run_source_goal("unlockpickup", 453001, "mission_skill_gate",
                        "skill_monitor")
    with pytest.raises(ValueError, match="outside registered"):
        run_source_goal("fetch", 662001, "mission_skill_gate",
                        "skill_monitor")


def test_numerical_gate_requires_real_physical_goals_not_reward():
    monitor = {}
    for task in TASKS:
        for mode in MODES:
            goals = {
                "actor_only": 18, "front_rule_gate": 12,
                "history_rule_gate": 12, "mission_skill_gate": 20,
            }[mode]
            monitor[f"{task}:{mode}"] = {"episodes": [
                {"seed": seed,
                 "external_physical_full_goal": i < goals,
                 "external_rewarded_terminations": 1,
                 "history_mission_rule_disagreement_attempts": 1}
                for i, seed in enumerate(SKILL_MONITOR_LAYOUTS[task])
            ]}
    result = decide(monitor)
    assert result["registered_numerical_gate_pass"]
    assert result["pooled_paired_full_goal_gain"] == 16
    assert result["registered_decision"] == "PASS"
    for item in monitor["fetch:mission_skill_gate"]["episodes"]:
        item["external_physical_full_goal"] = False
    failed = decide(monitor)
    assert failed["registered_decision"] == "FAIL"
    assert failed["mission_minus_history_paired_goal_gains"]["fetch"] == -12
    with pytest.raises(ValueError, match="paired source monitor"):
        monitor["fetch:actor_only"]["episodes"][0]["seed"] = 1
        decide(monitor)


def test_nonvacuity_cannot_be_replaced_with_reward():
    monitor = {}
    for task in TASKS:
        for mode in MODES:
            monitor[f"{task}:{mode}"] = {"episodes": [
                {"seed": seed,
                 "external_physical_full_goal": False,
                 "external_rewarded_terminations": 1,
                 "history_mission_rule_disagreement_attempts": 0}
                for seed in SKILL_MONITOR_LAYOUTS[task]
            ]}
    result = decide(monitor)
    assert result["registered_decision"] == "INSUFFICIENT"
    assert not result["registered_numerical_gate_pass"]
