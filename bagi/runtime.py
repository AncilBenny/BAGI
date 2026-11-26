"""One public-observation episode lifecycle for the existing GeneralBrain.

Version 1 checkpoints contain option outcome counts, not policy weights, world
models or transient environment state. Adapters must provide a compatibility ID
that changes when skill semantics, policy weights or observation contracts change.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bagi.agents.general_brain import Action, GeneralBrain, Goal, NoPlanError, SkillExperience


@dataclass(frozen=True)
class StepFeedback:
    executed_action: Action
    observation: Any
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class EpisodeResult:
    goal_verified: bool
    reason: str
    actions: tuple[Action, ...]
    skills: tuple[str, ...]
    public_facts: frozenset[str]


class AgentRuntime:
    """Reuse the same brain across goals; persist compatible outcome learning."""

    def __init__(self, brain: GeneralBrain, compatibility_id: str) -> None:
        if not isinstance(compatibility_id, str) or not compatibility_id.strip():
            raise ValueError("A nonempty adapter/policy compatibility ID is required")
        self.brain = brain
        self.compatibility_id = compatibility_id

    def run_episode(
        self,
        goal: Goal,
        initial_observation: Any,
        step: Callable[[Action], StepFeedback],
        *,
        learn: bool = True,
        on_transition: Callable[[Any, StepFeedback], None] | None = None,
    ) -> EpisodeResult:
        """Execute only through the public adapter; never substitute teacher labels.

        The optional learner sees only public before/after and actual action.
        learn=False disables that hook and freezes outcome counters. Evaluation
        must also freeze policy behavior and use separate adapter/data splits.
        """
        self.brain.reset(goal, initial_observation)
        observation = initial_observation
        actions: list[Action] = []
        skills: list[str] = []
        reason = "exhausted"
        while not self.brain.complete and not self.brain.exhausted:
            try:
                decision = self.brain.decide()
            except NoPlanError:
                reason = "no_plan"
                break
            before = copy.deepcopy(observation) if learn and on_transition else None
            feedback = step(decision.action)
            if not isinstance(feedback, StepFeedback):
                raise TypeError("Adapter must return StepFeedback")
            self.brain.observe(
                feedback.executed_action, feedback.observation, feedback.reward, learn=learn
            )
            if learn and on_transition is not None:
                on_transition(before, feedback)
            observation = feedback.observation
            actions.append(feedback.executed_action)
            skills.append(decision.skill_id)
            if feedback.terminated or feedback.truncated:
                reason = "terminated" if feedback.terminated else "truncated"
                break
        if self.brain.complete:
            reason = "goal_verified"
        return EpisodeResult(
            self.brain.complete, reason, tuple(actions), tuple(skills), self.brain.facts
        )

    def snapshot(self) -> dict[str, Any]:
        """Return plain JSON-compatible, versioned outcome state."""
        return {
            "schema": 1,
            "compatibility_id": self.compatibility_id,
            "experience": {
                key: {"successes": stat.successes, "failures": stat.failures}
                for key, stat in sorted(self.brain.experience.items())
            },
        }

    def save(self, path: Path) -> None:
        """Atomically replace a checkpoint; no executable serialization."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                json.dump(self.snapshot(), handle, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load(self, path: Path) -> None:
        """Validate the entire checkpoint before changing any learned counts."""
        self.restore_snapshot(json.loads(Path(path).read_text(encoding="utf-8")))

    def restore_snapshot(self, data: Any) -> None:
        """Restore validated outcome counts from a compatible local bundle."""
        if (
            not isinstance(data, dict)
            or type(data.get("schema")) is not int
            or data["schema"] != 1
            or data.get("compatibility_id") != self.compatibility_id
            or not isinstance(data.get("experience"), dict)
        ):
            raise ValueError("Incompatible outcome checkpoint")
        experience = data["experience"]
        if set(experience) != set(self.brain.experience):
            raise ValueError("Checkpoint skill IDs differ")
        restored = {}
        for key, counts in experience.items():
            if not isinstance(counts, dict) or set(counts) != {"successes", "failures"}:
                raise ValueError("Invalid checkpoint counts")
            if any(type(value) is not int or value < 0 for value in counts.values()):
                raise ValueError("Invalid checkpoint counts")
            restored[key] = SkillExperience(**counts)
        self.brain.experience.update(restored)
