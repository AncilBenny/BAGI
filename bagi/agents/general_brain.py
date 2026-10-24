"""BAGI experimental task-independent goal/skill control kernel.

The core plans over abstract observable facts, executes reusable closed-loop
skills, verifies their effects, and updates cross-task skill reliability.
It does NOT hard-code any environment entities, actions, task phases or goals.

Only a PublicSensor can produce evidence for beliefs. The sensor interface
accepts public observations, the actual executed action, and public reward;
it has no environment ``info``, simulator-state, or teacher-state channel.

This is a small testable architecture foundation, not a learned world model,
a universal planner, or evidence of general intelligence.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

Action = int | str  # Simulator actions OR symbolic commands for app/tool adapters.


class PublicSensor(Protocol):
    """Derive facts only from policy-legal observations and public feedback."""

    def initial(self, observation: Any) -> frozenset[str]: ...

    def change(
        self, before: Any, executed_action: Action, after: Any, public_reward: float
    ) -> FactChange: ...


@dataclass(frozen=True)
class FactChange:
    gained: frozenset[str] = frozenset()
    lost: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.gained & self.lost:
            raise ValueError("A fact cannot be both confirmed and retracted")


@dataclass(frozen=True)
class Goal:
    """Task-independent objective described by *observable* success facts."""

    goal_id: str
    required: frozenset[str]

    def __post_init__(self) -> None:
        if not self.goal_id or not self.required or not all(self.required):
            raise ValueError("A goal needs an ID and nonempty observable conditions")


@dataclass(frozen=True)
class Skill:
    """Reusable option: initiation facts, expected effects, closed-loop policy.

    ``policy`` is supplied by an adapter or learned model. Preconditions and
    expected effects are *hypotheses* for planning, not proven facts.
    """

    skill_id: str
    requires: frozenset[str]
    establishes: frozenset[str]
    policy: Callable[[Any, frozenset[str]], Action] = field(repr=False, compare=False)
    max_actions: int = 4
    estimated_cost: float = 1.0

    def __post_init__(self) -> None:
        if not self.skill_id or not self.establishes or self.max_actions < 1:
            raise ValueError("A skill needs an ID, effect and positive action cap")
        if not math.isfinite(self.estimated_cost) or self.estimated_cost <= 0:
            raise ValueError("Expected cost must be positive and finite")
        if not callable(self.policy):
            raise ValueError("Skill policy must be callable")


@dataclass
class SkillExperience:
    successes: int = 0
    failures: int = 0

    @property
    def success_probability(self) -> float:
        # One shared Beta(1, 1) prior across tasks for a named reusable skill.
        return (self.successes + 1) / (self.successes + self.failures + 2)

    def record(self, succeeded: bool) -> None:
        if succeeded:
            self.successes += 1
        else:
            self.failures += 1


@dataclass(frozen=True)
class Decision:
    skill_id: str
    action: Action


@dataclass(frozen=True)
class ObservationResult:
    confirmed_facts: frozenset[str]
    complete: bool
    skill_finished: bool
    skill_succeeded: bool | None
    action_count: int


class NoPlanError(RuntimeError):
    """No path is presently supported by the available skill descriptions."""


class GeneralBrain:
    """Goal-conditioned option planning + observable belief + online outcomes.

    Skills and their policies are replaceable. This layer has no task names,
    object identities, reward shaping rules, or privileged state access.
    Learned outcome statistics survive ``reset`` and can transfer to a new
    goal using the same skill; transient belief and retry counts never do.
    """

    def __init__(
        self,
        skills: tuple[Skill, ...],
        sensor: PublicSensor,
        *,
        max_episode_actions: int = 40,
        max_attempts_per_skill: int = 2,
        max_planning_states: int = 256,
    ) -> None:
        if not skills or len({s.skill_id for s in skills}) != len(skills):
            raise ValueError("Skill IDs must be nonempty and unique")
        if min(max_episode_actions, max_attempts_per_skill, max_planning_states) < 1:
            raise ValueError("Episode, attempts, and planning limits must be positive")
        self._skills = {skill.skill_id: skill for skill in skills}
        self.sensor = sensor
        self.max_episode_actions = max_episode_actions
        self.max_attempts_per_skill = max_attempts_per_skill
        self.max_planning_states = max_planning_states
        self.experience = {skill.skill_id: SkillExperience() for skill in skills}
        self.goal: Goal | None = None
        self.facts: frozenset[str] = frozenset()
        self._observation: Any = None
        self._active: str | None = None
        self._active_steps = 0
        self._attempts: dict[str, int] = {}
        self._pending: Decision | None = None
        self.actual_actions = 0
        self._exhausted = False

    @property
    def complete(self) -> bool:
        return self.goal is not None and self.goal.required <= self.facts

    @property
    def exhausted(self) -> bool:
        return self._exhausted or self.actual_actions >= self.max_episode_actions

    def reset(self, goal: Goal, public_observation: Any) -> None:
        self.goal = goal
        self.facts = frozenset(self.sensor.initial(public_observation))
        self._observation = public_observation
        self._active = None
        self._active_steps = 0
        self._attempts = {}
        self._pending = None
        self.actual_actions = 0
        self._exhausted = False

    def plan(self) -> tuple[str, ...]:
        """Bounded uniform-cost planning over *hypothesized* option effects.

        Assumed effects are used only during imagined search. Real persistent
        belief is updated exclusively through the sensor's public evidence.
        """
        if self.goal is None:
            raise RuntimeError("Call reset(goal, public_observation) first")
        start = self.facts
        if self.goal.required <= start:
            return ()
        queue: list[tuple[float, tuple[str, ...], frozenset[str]]] = [
            (0.0, (), start)
        ]
        best = {start: 0.0}
        expansions = 0
        while queue and expansions < self.max_planning_states:
            cost, path, facts = heapq.heappop(queue)
            if cost > best.get(facts, math.inf) + 1e-9:
                continue
            if self.goal.required <= facts:
                return path
            expansions += 1
            for name in sorted(self._skills):
                skill = self._skills[name]
                if self._attempts.get(name, 0) >= self.max_attempts_per_skill:
                    continue
                if not skill.requires <= facts or skill.establishes <= facts:
                    continue
                candidate = facts | skill.establishes
                likelihood = self.experience[name].success_probability
                candidate_cost = cost + skill.estimated_cost / max(likelihood, 0.01)
                if candidate_cost + 1e-9 < best.get(candidate, math.inf):
                    best[candidate] = candidate_cost
                    heapq.heappush(queue, (candidate_cost, (*path, name), candidate))
        raise NoPlanError("No valid bounded skill composition for this goal")

    def decide(self) -> Decision:
        if self.goal is None:
            raise RuntimeError("Call reset(goal, public_observation) first")
        if self._pending is not None:
            raise RuntimeError("Observe the executed action before deciding again")
        if self.complete:
            raise RuntimeError("Goal already confirmed from public evidence")
        if self.exhausted:
            raise RuntimeError("Episode action budget is exhausted")
        if self._active is None:
            try:
                self._active = self.plan()[0]
            except NoPlanError:
                self._exhausted = True
                raise
            self._active_steps = 0
        chosen = self._skills[self._active]
        action = chosen.policy(self._observation, self.facts)
        if (isinstance(action, bool)
                or not isinstance(action, (int, str))
                or (isinstance(action, int) and action < 0)
                or (isinstance(action, str) and not action)):
            raise ValueError("A skill must propose a valid integer or string action")
        self._pending = Decision(self._active, action)
        return self._pending

    def observe(
        self,
        executed_action: Action,
        new_public_observation: Any,
        public_reward: float = 0.0,
    ) -> ObservationResult:
        if self._pending is None or self._active is None:
            raise RuntimeError("No issued action awaiting a real observation")
        if executed_action != self._pending.action:
            raise ValueError("Executed action differs from issued action; provenance error")
        if not math.isfinite(public_reward):
            raise ValueError("Public reward must be finite")
        delta = self.sensor.change(
            self._observation, executed_action, new_public_observation, public_reward
        )
        if not isinstance(delta, FactChange):
            raise TypeError("Public sensor must return FactChange")
        self.facts = (self.facts - delta.lost) | delta.gained
        self._observation = new_public_observation
        self.actual_actions += 1
        self._active_steps += 1
        skill = self._skills[self._active]
        succeeded = skill.establishes <= self.facts
        finished = succeeded or self._active_steps >= skill.max_actions
        skill_result = succeeded if finished else None
        if finished:
            self.experience[skill.skill_id].record(succeeded)
            if not succeeded:
                self._attempts[skill.skill_id] = (
                    self._attempts.get(skill.skill_id, 0) + 1
                )
            self._active = None
            self._active_steps = 0
        self._pending = None
        return ObservationResult(
            self.facts,
            self.complete,
            finished,
            skill_result,
            self.actual_actions,
        )


class PublicFactSensor:
    """Reference adapter for worlds exposing facts in *public* observations.

    This is intentionally generic and appropriate for tests/instrumented apps.
    A vision-based world must supply a separately tested visible-only encoder.
    Hidden simulator state and privileged ``info`` must never be used here.
    """

    def initial(self, observation: frozenset[str]) -> frozenset[str]:
        return frozenset(observation)

    def change(
        self,
        before: frozenset[str], executed_action: Action,
        after: frozenset[str], public_reward: float,
    ) -> FactChange:
        return FactChange(frozenset(after - before), frozenset(before - after))