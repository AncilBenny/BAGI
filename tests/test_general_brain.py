"""Contract tests for a generic brain, not evidence of trained intelligence."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from bagi.agents.general_brain import (
    FactChange,
    GeneralBrain,
    Goal,
    NoPlanError,
    PublicFactSensor,
    Skill,
)


@dataclass
class PublicWorld:
    """Mock dynamics: hidden implementation, observable fact-based feedback."""

    facts: frozenset[str]
    actions: dict[int | str, tuple[frozenset[str], frozenset[str]]]
    steps: int = 0

    def step(self, action: int | str) -> frozenset[str]:
        self.steps += 1
        needed, gained = self.actions[action]
        if needed <= self.facts:
            self.facts |= gained
        return self.facts


def option(name, action, *, requires=(), produces=(), cost=1.0, cap=1):
    return Skill(
        name, frozenset(requires), frozenset(produces),
        lambda _observation, _facts: action, cap, cost,
    )


def execute(brain: GeneralBrain, world: PublicWorld, goal: Goal):
    brain.reset(goal, world.facts)
    decisions = []
    while not brain.complete and not brain.exhausted:
        try:
            decision = brain.decide()
        except NoPlanError:
            break
        decisions.append(decision.skill_id)
        next_observation = world.step(decision.action)
        brain.observe(decision.action, next_observation)
    return decisions


def test_same_brain_architecture_on_physical_and_digital_workflows():
    physical = PublicWorld(frozenset(), {
        0: (frozenset(), frozenset({"power_on"})),
        1: (frozenset({"power_on"}), frozenset({"item_secured"})),
        2: (frozenset({"item_secured"}), frozenset({"delivered"})),
    })
    brain_a = GeneralBrain((
        option("power", 0, produces={"power_on"}),
        option("secure", 1, requires={"power_on"}, produces={"item_secured"}),
        option("deliver", 2, requires={"item_secured"}, produces={"delivered"}),
    ), PublicFactSensor())
    path_a = execute(brain_a, physical, Goal("delivery", frozenset({"delivered"})))
    assert path_a == ["power", "secure", "deliver"]
    assert brain_a.complete and physical.steps == 3

    digital = PublicWorld(frozenset(), {
        "draft": (frozenset(), frozenset({"draft_ready"})),
        "review": (frozenset({"draft_ready"}), frozenset({"reviewed"})),
        "publish": (frozenset({"reviewed"}), frozenset({"published"})),
    })
    brain_b = GeneralBrain((
        option("write", "draft", produces={"draft_ready"}),
        option("review", "review", requires={"draft_ready"}, produces={"reviewed"}),
        option("publish", "publish", requires={"reviewed"}, produces={"published"}),
    ), PublicFactSensor())
    path_b = execute(brain_b, digital, Goal("publication", frozenset({"published"})))
    assert path_b == ["write", "review", "publish"]
    assert brain_b.complete and digital.steps == 3


def test_attempted_action_and_reward_do_not_imply_success():
    world = PublicWorld(frozenset(), {0: (frozenset({"unmet_guard"}),
                                      frozenset({"confirmed"}))})
    brain = GeneralBrain((option("attempt", 0, produces={"confirmed"}),),
                         PublicFactSensor(), max_attempts_per_skill=1)
    brain.reset(Goal("test", frozenset({"confirmed"})), world.facts)
    decision = brain.decide()
    result = brain.observe(decision.action, world.step(decision.action), 1.0)
    assert not result.complete and result.skill_succeeded is False
    assert "confirmed" not in result.confirmed_facts
    assert brain.experience["attempt"].failures == 1
    with pytest.raises(NoPlanError):
        brain.decide()


def test_failure_replans_to_alternative_and_learns_for_next_goal():
    world = PublicWorld(frozenset(), {
        0: (frozenset({"never"}), frozenset({"checked"})),
        1: (frozenset(), frozenset({"checked"})),
        2: (frozenset({"checked"}), frozenset({"submitted"})),
    })
    brain = GeneralBrain((
        option("a_unreliable_check", 0, produces={"checked"}),
        option("b_reliable_check", 1, produces={"checked"}),
        option("submit", 2, requires={"checked"}, produces={"submitted"}),
    ), PublicFactSensor(), max_attempts_per_skill=1)
    first = execute(brain, world, Goal("first", frozenset({"submitted"})))
    assert first == ["a_unreliable_check", "b_reliable_check", "submit"]
    assert brain.complete
    assert brain.experience["a_unreliable_check"].failures == 1
    assert brain.experience["b_reliable_check"].successes == 1
    # The learned probabilities survive the next task reset.
    second_world = PublicWorld(frozenset(), world.actions)
    next_path = execute(brain, second_world, Goal("second", frozenset({"checked"})))
    assert next_path == ["b_reliable_check"]
    assert brain.complete


def test_facts_retracted_by_public_evidence_are_not_assumed_persistent():
    world = PublicWorld(frozenset({"registered"}), {
        0: (frozenset({"registered"}), frozenset({"authorized"})),
    })
    brain = GeneralBrain((option("authorize", 0, requires={"registered"},
                                  produces={"authorized"}),), PublicFactSensor())
    brain.reset(Goal("g", frozenset({"authorized"})), world.facts)
    decision = brain.decide()
    brain.observe(decision.action, frozenset({"authorized"}))
    assert "registered" not in brain.facts
    assert brain.complete


def test_per_episode_observation_and_action_provenance_guards():
    brain = GeneralBrain((option("issue", 1, produces={"ack"}),), PublicFactSensor())
    with pytest.raises(RuntimeError, match="reset"):
        brain.decide()
    brain.reset(Goal("ack", frozenset({"ack"})), frozenset())
    brain.decide()
    with pytest.raises(RuntimeError, match="Observe"):
        brain.decide()
    with pytest.raises(ValueError, match="provenance"):
        brain.observe(2, frozenset({"ack"}))
    assert brain.actual_actions == 0
    result = brain.observe(1, frozenset({"ack"}))
    assert result.complete
    with pytest.raises(RuntimeError, match="already confirmed"):
        brain.decide()


def test_bounded_missing_plan_and_planner_does_not_mutate_belief():
    brain = GeneralBrain((option("step", 0, requires={"unknown"},
                                  produces={"result"}),), PublicFactSensor())
    brain.reset(Goal("unreachable", frozenset({"result"})), frozenset())
    before = brain.facts
    with pytest.raises(NoPlanError):
        brain.plan()
    assert brain.facts == before
    with pytest.raises(NoPlanError):
        brain.decide()
    assert brain.exhausted


def test_sensor_delta_is_validated_and_model_never_sees_hidden_info():
    with pytest.raises(ValueError, match="both"):
        FactChange(frozenset({"x"}), frozenset({"x"}))
    assert set(PublicFactSensor().initial(frozenset({"visible"}))) == {"visible"}


def test_bounded_multi_action_skill_does_not_hallucinate_progress():
    brain = GeneralBrain((option("long", 4, produces={"complete"}, cap=3),),
                         PublicFactSensor(), max_episode_actions=2)
    brain.reset(Goal("g", frozenset({"complete"})), frozenset())
    for n in range(2):
        decision = brain.decide()
        result = brain.observe(decision.action, frozenset())
        assert result.skill_succeeded is None and not result.complete
        assert brain.actual_actions == n + 1
    assert brain.exhausted
    with pytest.raises(RuntimeError, match="budget"):
        brain.decide()
    assert brain.experience["long"].successes == 0


def test_goal_validation_and_skill_validation():
    with pytest.raises(ValueError):
        Goal("", frozenset({"ok"}))
    with pytest.raises(ValueError):
        Goal("g", frozenset())
    with pytest.raises(ValueError):
        option("x", 0, produces={"y"}, cost=0)
    with pytest.raises(ValueError):
        GeneralBrain((option("dup", 0, produces={"x"}),
                      option("dup", 0, produces={"x"})), PublicFactSensor())


def test_synthetic_multidomain_smoke_has_real_action_provenance():
    from bagi.experiments.general_brain_smoke import run

    report = run()
    assert not report["is_real_environment_or_learned_general_intelligence"]
    assert all(x["goal_verified"] for x in report["tasks"])
    assert [x["real_mock_actions"] for x in report["tasks"]] == [3, 3, 3, 1]
    assert report["tasks"][-1]["skill_sequence"] == ["b_good_check"]