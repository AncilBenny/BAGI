"""G1 evidence and native adapter contracts; mock success is NOT learned transfer."""
from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

import numpy as np
import pytest

from bagi.agents.general_brain import GeneralBrain, Goal, Skill
from bagi.agents.public_evidence import PublicEvidenceSensor
from bagi.experiments.general_brain_native_g1 import (
    MODEL_SEEDS,
    OPTION_SPLIT,
    WORKSHOP_MODES,
    XLAND_LAYOUTS,
    FixedScript,
    verify_registration,
    workshop_public_direction,
    workshop_scripts,
    xland_brain,
    xland_reward_sensor,
)


def test_public_evidence_requires_observations_not_action_attempts():
    sensor = PublicEvidenceSensor({"door_open": lambda x: x.get("door")})
    assert sensor.initial({"door": None}) == frozenset()
    attempted = sensor.change({"door": None}, "open", {"door": None}, 0.0)
    assert attempted.gained == frozenset()
    observed = sensor.change({"door": False}, "open", {"door": True}, 0.0)
    assert observed.gained == frozenset({"door_open"})
    offscreen = sensor.change({"door": True}, "walk", {"door": None}, 0.0)
    assert offscreen.lost == frozenset()
    contradicted = sensor.change({"door": None}, "walk", {"door": False}, 0.0)
    assert contradicted.lost == frozenset({"door_open"})


def test_native_reward_is_not_a_physical_success_claim():
    sensor = xland_reward_sensor()
    gained = sensor.change(np.zeros(115), 4, np.zeros(115), 0.7)
    assert gained.gained == frozenset({"native_reward_observed"})
    assert "physical_placement" not in gained.gained
    assert sensor.change(np.zeros(115), 3, np.zeros(115), 0.0).gained == frozenset()


def test_workshop_uses_public_direction_and_mirrors_scripts():
    right = np.zeros((5, 5), dtype=np.int8)
    left = right.copy()
    right[2, 4] = 4
    left[2, 0] = 4
    assert workshop_public_direction(right) == (1, 3)
    assert workshop_public_direction(left) == (3, 1)
    r = workshop_scripts(right)
    l = workshop_scripts(left)
    assert r["a_bad_switch"] == (3, 3)
    assert r["b_good_switch"] == (1, 1)
    assert len(r["c_reach_goal"]) == 14
    assert l["b_good_switch"] == (3, 3)
    assert l["c_reach_goal"] == (1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2)
    with pytest.raises(ValueError, match="switch"):
        workshop_public_direction(np.zeros((5, 5), dtype=np.int8))


def test_verified_fallback_uses_two_actual_failed_actions_before_alternative():
    bad = FixedScript((0, 0))
    good = FixedScript((1, 1))
    sensor = PublicEvidenceSensor({"ok": lambda obs: obs.get("ok")})
    brain = GeneralBrain((
        Skill("a_bad", frozenset(), frozenset({"ok"}), bad, max_actions=2),
        Skill("b_good", frozenset(), frozenset({"ok"}), good, max_actions=2),
    ), sensor, max_attempts_per_skill=1)
    brain.reset(Goal("test", frozenset({"ok"})), {"ok": None})
    actions = []
    for i in range(4):
        cmd = brain.decide()
        actions.append(cmd.action)
        result = brain.observe(cmd.action, {"ok": True if i == 3 else None})
    assert actions == [0, 0, 1, 1]
    assert result.complete and brain.actual_actions == 4
    assert brain.experience["a_bad"].failures == 1
    assert brain.experience["b_good"].successes == 1


def test_action_provenance_failure_does_not_invent_physical_progress():
    brain = GeneralBrain((Skill("one", frozenset(), frozenset({"done"}),
                                lambda obs, facts: 3),),
                         PublicEvidenceSensor({"done": lambda obs: obs.get("done")}))
    brain.reset(Goal("test", frozenset({"done"})), {"done": None})
    brain.decide()
    with pytest.raises(ValueError, match="provenance"):
        brain.observe(4, {"done": True}, 1.0)
    assert not brain.complete and brain.actual_actions == 0


def test_preregistered_independent_xland_layouts_and_finite_budgets():
    verify_registration()
    assert XLAND_LAYOUTS == tuple(range(11300, 11305))
    assert MODEL_SEEDS == (17, 29)
    assert OPTION_SPLIT == 20
    assert all(i not in range(10900, 10972) for i in XLAND_LAYOUTS)
    assert set(WORKSHOP_MODES) == {"default", "mirror_x"}


@dataclass
class DummyPolicy:
    action: int

    def predict(self, observation, deterministic=False):
        return np.asarray(self.action), None


def test_xland_brain_has_capped_same_observation_policies():
    a = xland_brain(DummyPolicy(2), DummyPolicy(3), "verified_option_fallback")
    b = xland_brain(DummyPolicy(2), DummyPolicy(3), "direct_expanded_bc")
    assert set(a.experience) == {"a_expanded", "b_original_fallback"}
    assert a.max_episode_actions == b.max_episode_actions == 40
    assert a._skills["a_expanded"].max_actions == 20
    assert a._skills["b_original_fallback"].max_actions == 20
    assert b._skills["direct_expanded"].max_actions == 40


@pytest.mark.skipif(find_spec("gymnasium") is None, reason="Gymnasium lives on CI runner")
@pytest.mark.parametrize("layout_mode", WORKSHOP_MODES)
def test_real_workshop_native_public_only(layout_mode):
    from bagi.experiments.general_brain_native_g1 import workshop_native_one

    direct = workshop_native_one(layout_mode, "direct_script")
    assumed = workshop_native_one(layout_mode, "assumed_effect")
    verified = workshop_native_one(layout_mode, "verified_replan")
    assert direct["goal_rewarded"] and direct["native_actions"] == 16
    assert not assumed["goal_rewarded"] and assumed["native_actions"] == 16
    assert verified["goal_rewarded"] and verified["native_actions"] == 18
    assert verified["failed_options"] == 1
    assert verified["evaluator_only_causal_chain"] == {"door_open": True, "has_key": True}


def test_no_full_simulator_state_or_info_in_brain_sensor_interface():
    sensor = PublicEvidenceSensor({"goal": lambda public: public.get("goal")})
    assert "info" not in sensor.change.__code__.co_varnames
    assert sensor.change({"goal": None}, "act", {"goal": False}, 0.0).gained == frozenset()
