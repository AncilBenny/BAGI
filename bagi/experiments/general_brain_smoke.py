"""REUSE-39: dependency-free, SYNTHETIC multi-task architecture smoke test.

This is a runnable API/recovery check, not autonomous competence in XLand,
Workshop, robotics, computer-use applications, or the real world.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from bagi.agents.general_brain import (
    GeneralBrain,
    Goal,
    NoPlanError,
    PublicFactSensor,
    Skill,
)


@dataclass
class ObservableTask:
    """Minimal laboratory: policy receives ONLY the public fact observation."""

    facts: frozenset[str]
    transition_rules: dict[int | str, tuple[frozenset[str], frozenset[str]]]
    real_actions: int = 0

    def step(self, action: int | str) -> frozenset[str]:
        self.real_actions += 1
        needs, effects = self.transition_rules[action]
        if needs <= self.facts:
            self.facts |= effects
        return self.facts


def option(
    name: str,
    action: int | str,
    *,
    needs: tuple[str, ...] = (),
    establishes: tuple[str, ...],
) -> Skill:
    return Skill(
        skill_id=name, requires=frozenset(needs),
        establishes=frozenset(establishes),
        policy=lambda observation, memory: action,
        max_actions=1,
    )


def episode(brain: GeneralBrain, task: ObservableTask, goal: Goal) -> dict:
    brain.reset(goal, task.facts)
    executed = []
    while not brain.complete and not brain.exhausted:
        try:
            decision = brain.decide()
        except NoPlanError:
            break
        after = task.step(decision.action)
        brain.observe(decision.action, after)
        executed.append(decision.skill_id)
    return {
        "goal": goal.goal_id,
        "goal_verified": brain.complete,
        "real_mock_actions": task.real_actions,
        "skill_sequence": executed,
        "confirmed_public_facts": sorted(brain.facts),
    }


def run() -> dict:
    # Task 1: supply chain. The core knows none of these names/semantics.
    logistics = ObservableTask(frozenset(), {
        0: (frozenset(), frozenset({"inventory_loaded"})),
        1: (frozenset({"inventory_loaded"}), frozenset({"route_ready"})),
        2: (frozenset({"route_ready"}), frozenset({"shipment_arrived"})),
    })
    logistics_brain = GeneralBrain((
        option("load", 0, establishes=("inventory_loaded",)),
        option("route", 1, needs=("inventory_loaded",), establishes=("route_ready",)),
        option("send", 2, needs=("route_ready",), establishes=("shipment_arrived",)),
    ), PublicFactSensor())
    supply = episode(
        logistics_brain, logistics, Goal("ship_supply", frozenset({"shipment_arrived"}))
    )

    # Task 2: software publishing. Same core, different actions, predicates.
    publishing = ObservableTask(frozenset(), {
        "draft": (frozenset(), frozenset({"draft_saved"})),
        "test": (frozenset({"draft_saved"}), frozenset({"tests_passed"})),
        "publish": (frozenset({"tests_passed"}), frozenset({"release_published"})),
    })
    publishing_brain = GeneralBrain((
        option("write", "draft", establishes=("draft_saved",)),
        option("test", "test", needs=("draft_saved",), establishes=("tests_passed",)),
        option("publish", "publish", needs=("tests_passed",),
               establishes=("release_published",)),
    ), PublicFactSensor())
    document = episode(
        publishing_brain, publishing, Goal("publish_release", frozenset({"release_published"}))
    )

    # Task 3: same reusable check skill in two goals. Learn which option fails,
    # then prefer a better option under another goal, using only public feedback.
    rules = {
        3: (frozenset({"unavailable"}), frozenset({"checked"})),
        4: (frozenset(), frozenset({"checked"})),
        5: (frozenset({"checked"}), frozenset({"submitted"})),
    }
    recover_brain = GeneralBrain((
        option("a_bad_check", 3, establishes=("checked",)),
        option("b_good_check", 4, establishes=("checked",)),
        option("submit", 5, needs=("checked",), establishes=("submitted",)),
    ), PublicFactSensor(), max_attempts_per_skill=1)
    recovery = episode(
        recover_brain, ObservableTask(frozenset(), rules),
        Goal("submit_first", frozenset({"submitted"})),
    )
    reused = episode(
        recover_brain, ObservableTask(frozenset(), rules),
        Goal("verify_second", frozenset({"checked"})),
    )
    result = {
        "experiment": "general_brain_synthetic_multitask_contract_v1",
        "is_real_environment_or_learned_general_intelligence": False,
        "tasks": [supply, document, recovery, reused],
        "learned_cross_goal_skill_reliability": {
            key: {
                "successes": stat.successes,
                "failures": stat.failures,
                "posterior_success": stat.success_probability,
            }
            for key, stat in recover_brain.experience.items()
        },
    }
    assert all(x["goal_verified"] for x in result["tasks"])
    assert recovery["skill_sequence"][:2] == ["a_bad_check", "b_good_check"]
    assert reused["skill_sequence"] == ["b_good_check"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    outcome = run()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print(json.dumps(outcome, indent=2))