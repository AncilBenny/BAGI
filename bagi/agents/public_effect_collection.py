"""G2-E1: domain-neutral collection of REAL public transitions.

Native adapters own reset, step, public feature extraction and evaluator-only
hidden-state comparison. The acting callback sees only the public observation;
no teacher, private simulator state or counterfactual step enters its input.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from bagi.agents.public_effect_prediction import (
    PublicEffectEpisode,
    checked_episodes,
    disjoint_layouts,
    public_change,
)


@dataclass(frozen=True)
class PublicEpisodeAudit:
    episode: PublicEffectEpisode
    native_actions: int
    native_rewarded_steps: int
    evaluator_hidden_changes: int
    hidden_without_public_change: int
    terminal: bool


def _checked_public(observation: object, width: int) -> np.ndarray:
    x = np.asarray(observation)
    if (x.shape != (width,) or x.dtype.kind not in "biuf"
            or not np.isfinite(x).all()):
        raise ValueError("Invalid finite public observation width/dtype")
    return np.array(x, dtype=np.float32, copy=True)


def collect_public_episode(
    *, layout: str, episode_id: str, behavior: str,
    initial_public: object,
    choose_action: Callable[[np.ndarray], object],
    native_step: Callable[[int], tuple[object, float, bool, bool]],
    width: int, n_actions: int, cap: int, history_bits: int = 0,
) -> PublicEpisodeAudit:
    """Execute at most cap REAL actions; native_step is called exactly once each.

    native_step returns (next_public, public_reward, terminal,
    evaluator_only_hidden_change). Hidden change is recorded only AFTER acting.
    The adapter must construct next_public with the ACTUALLY executed action.
    """
    if (not layout or not episode_id or not behavior or type(width) is not int
            or width < 1 or type(n_actions) is not int or n_actions < 2
            or type(cap) is not int or cap < 1 or type(history_bits) is not int
            or history_bits < 0 or history_bits >= width):
        raise ValueError("Invalid preregistered collection metadata")
    current = _checked_public(initial_public, width)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    changes: list[int] = []
    rewarded = 0
    hidden = 0
    hidden_invisible = 0
    terminal = False
    for _ in range(cap):
        policy_view = current.copy()
        policy_view.setflags(write=False)
        proposal = np.asarray(choose_action(policy_view))
        if (proposal.shape not in ((), (1,)) or proposal.dtype.kind not in "iu"):
            raise ValueError("Actor must propose one native integer action")
        action = int(proposal.reshape(-1)[0])
        if action < 0 or action >= n_actions:
            raise ValueError("Actor proposed out-of-range native action")
        after, reward, finished, changed_hidden = native_step(action)
        if (not np.isfinite(reward) or type(finished) is not bool
                or type(changed_hidden) is not bool):
            raise ValueError("Invalid native public reward/terminal/evaluator flags")
        next_public = _checked_public(after, width)
        change = public_change(current, next_public, history_bits=history_bits)
        observations.append(current)
        actions.append(action)
        changes.append(change)
        rewarded += int(float(reward) > 0)
        hidden += int(changed_hidden)
        hidden_invisible += int(changed_hidden and not change)
        current = next_public
        terminal = finished
        if terminal:
            break
    episode = PublicEffectEpisode(
        layout=layout, episode_id=episode_id, behavior=behavior,
        observations=np.stack(observations),
        actions=np.asarray(actions, dtype=np.int64),
        visible_changes=np.asarray(changes, dtype=np.int8),
    )
    checked_episodes([episode], width=width, n_actions=n_actions)
    return PublicEpisodeAudit(episode, len(actions), rewarded, hidden,
                              hidden_invisible, terminal)


def verify_fixed_collection(
    train: list[PublicEpisodeAudit], test: list[PublicEpisodeAudit], *,
    train_layouts: tuple[str, ...], test_layouts: tuple[str, ...],
    episodes_per_layout: int, width: int, n_actions: int,
    cap: int, max_native_actions: int,
) -> dict[str, int]:
    """Reject missing/substituted layouts and unaccounted native actions."""
    if (not train_layouts or not test_layouts or len(set(train_layouts)) != len(train_layouts)
            or len(set(test_layouts)) != len(test_layouts)
            or set(train_layouts) & set(test_layouts)
            or type(episodes_per_layout) is not int or episodes_per_layout < 1):
        raise ValueError("Invalid fixed layout registration")
    all_audits = train + test
    if (not all_audits or any(a.native_actions != len(a.episode.actions)
            or a.native_actions < 1 or a.native_actions > cap
            or a.hidden_without_public_change > a.evaluator_hidden_changes
            or a.evaluator_hidden_changes > a.native_actions
            or a.native_rewarded_steps > a.native_actions for a in all_audits)):
        raise ValueError("Invalid native interaction accounting")
    for rows, layouts in ((train, train_layouts), (test, test_layouts)):
        counts = Counter(a.episode.layout for a in rows)
        if counts != Counter({name: episodes_per_layout for name in layouts}):
            raise ValueError("Missing, substituted or duplicated registered layout")
        checked_episodes([a.episode for a in rows], width=width, n_actions=n_actions)
    disjoint_layouts([a.episode for a in train], [a.episode for a in test])
    actions = sum(a.native_actions for a in all_audits)
    if actions > max_native_actions:
        raise ValueError("Exceeded registered native interaction budget")
    return {
        "native_actions": actions,
        "train_episodes": len(train),
        "test_episodes": len(test),
        "hidden_without_public_change": sum(a.hidden_without_public_change
                                             for a in all_audits),
        "native_rewarded_steps": sum(a.native_rewarded_steps for a in all_audits),
    }
