"""Task-independent public-feedback stagnation diagnostic and one-step escape.

A repeated command and unchanged public view are *suspected* stagnation, not
proof of a native no-op: hidden progress can occur without visual evidence.
No simulator state, evaluator labels, task names or action identities enter.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PublicStepEvidence:
    unchanged_public_view: bool
    suspected_stagnation: bool
    escaped_last_choice: bool


class PublicStagnationEscape:
    """Compare a frozen ranked policy with a one-step public-feedback override.

    The signature strips adapter-specific action-history encodings, but must
    use only information already available to the acting policy. Propose
    requires ranked probabilities from the SAME frozen policy; it never asks
    an oracle, changes model weights or assumes a failed physical effect.
    """

    def __init__(self, signature: Callable[[Any], Hashable], *,
                 enabled: bool, patience: int = 3) -> None:
        if not callable(signature) or type(patience) is not int or patience < 1:
            raise ValueError("A public signature and positive patience are required")
        self.signature = signature
        self.enabled = bool(enabled)
        self.patience = patience
        self.reset()

    def reset(self) -> None:
        self.last_action: int | str | None = None
        self.streak = 0
        self.pending: tuple[int | str, Hashable, bool] | None = None
        self.suspected_stalls = 0
        self.escape_actions = 0
        self.public_unchanged_steps = 0

    def propose(self, public_observation: Any,
                ranked: Sequence[tuple[int | str, float]]) -> int | str:
        if self.pending is not None:
            raise RuntimeError("Must observe the previous actual executed action")
        if len(ranked) < 2 or len({a for a, _ in ranked}) != len(ranked):
            raise ValueError("Need at least two unique actions from one frozen policy")
        if any(type(a) not in (int, str) or not math.isfinite(float(p))
               or float(p) < 0 for a, p in ranked):
            raise ValueError("Invalid frozen policy action probabilities")
        scores = [float(p) for _, p in ranked]
        if abs(sum(scores) - 1.0) > 1e-4 or any(
            scores[i] < scores[i + 1] for i in range(len(scores) - 1)
        ):
            raise ValueError("Actions must be ranked by a normalized policy")
        chosen = ranked[0][0]
        escaped = False
        if self.enabled and self.streak >= self.patience and chosen == self.last_action:
            alternative = next(((a, p) for a, p in ranked[1:]
                                if a != chosen and p > 0), None)
            if alternative is None:
                raise ValueError("Frozen policy has no nonzero alternative")
            chosen = alternative[0]
            escaped = True
        self.pending = (chosen, self.signature(public_observation), escaped)
        return chosen

    def observe(self, executed_action: int | str, next_public_observation: Any,
                public_reward: float) -> PublicStepEvidence:
        if self.pending is None:
            raise RuntimeError("Must propose before recording public evidence")
        chosen, before, escaped = self.pending
        if executed_action != chosen:
            raise ValueError("Action history must reflect the actual selected command")
        if not math.isfinite(float(public_reward)):
            raise ValueError("Public reward must be finite")
        unchanged = before == self.signature(next_public_observation)
        self.public_unchanged_steps += int(unchanged)
        if unchanged and public_reward <= 0:
            self.streak = self.streak + 1 if chosen == self.last_action else 1
        else:
            self.streak = 0
        self.last_action = chosen
        self.pending = None
        suspected = unchanged and self.streak >= self.patience
        self.suspected_stalls += int(suspected)
        self.escape_actions += int(escaped)
        return PublicStepEvidence(bool(unchanged), bool(suspected), bool(escaped))
