"""Task-independent public-outcome sensing for the General Brain.

Observation predicates belong to environment adapters. A real action attempt,
positive reward or an expected option effect does not automatically establish a
physical effect. Reward is only a separate *public game outcome* fact.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from bagi.agents.general_brain import Action, FactChange


@dataclass
class PublicEvidenceSensor:
    """Accumulate observable positive facts; retract only observed negatives.

    Predicates must read ONLY the learner-visible public observation. A value
    of None means unobserved, not false; e.g. a door moving out of view does
    not erase a previously witnessed state change. Privileged evaluator
    state and Gymnasium ``info`` never enter this interface.
    """

    predicates: Mapping[str, Callable[[Any], bool | None]] = field(default_factory=dict)
    reward_fact: str | None = "native_reward_observed"

    def __post_init__(self) -> None:
        if self.reward_fact is not None and not self.reward_fact:
            raise ValueError("Reward fact must have a name")
        if any(not name or not callable(check) for name, check in self.predicates.items()):
            raise ValueError("Evidence predicates need nonempty names and functions")

    def initial(self, observation: Any) -> frozenset[str]:
        return frozenset(
            name for name, test in self.predicates.items() if test(observation) is True
        )

    def change(
        self, before: Any, executed_action: Action,
        after: Any, public_reward: float,
    ) -> FactChange:
        gained: set[str] = set()
        lost: set[str] = set()
        for name, test in self.predicates.items():
            # No inference from attempted actions or hidden game metadata.
            evidence = test(after)
            if evidence is True:
                gained.add(name)
            elif evidence is False:
                lost.add(name)
        if self.reward_fact is not None and float(public_reward) > 0:
            gained.add(self.reward_fact)
        lost.difference_update(gained)
        return FactChange(frozenset(gained), frozenset(lost))
