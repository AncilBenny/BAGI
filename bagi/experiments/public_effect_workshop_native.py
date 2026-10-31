"""G2-E1: fixed-budget Workshop adapter for public-action effect collection.

The collector and actor see only the 5x5 observation. Private state is read
inside the post-action evaluator callback solely for disagreement accounting.
G1's public-switch script is a fixture, not a learned Workshop controller.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

from bagi.agents.public_effect_collection import (
    PublicEpisodeAudit,
    collect_public_episode,
    verify_fixed_collection,
)

LAYOUTS = ("default", "mirror_x")
BEHAVIORS = ("uniform_random", "g1_public_script")
EPISODE_SEEDS = tuple(range(12100, 12110))
CAP = 20
WIDTH = 25
N_ACTIONS = 4
MAX_ACTIONS = 800


def _fingerprint(audit: PublicEpisodeAudit) -> str:
    """Hash full executed public trajectory prefix, not episode ID or seed."""
    episode = audit.episode
    digest = hashlib.sha256()
    for array in (episode.observations.astype("<f4"),
                  episode.actions.astype("<i8"),
                  episode.visible_changes.astype("i1")):
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _script_actor(script: tuple[int, ...]):
    position = 0

    def choose(_public):
        nonlocal position
        if position >= len(script):
            raise ValueError("G1 script exhausted before terminal")
        action = script[position]
        position += 1
        return action

    return choose


def _random_actor(seed: int):
    rng = np.random.default_rng(seed)

    def choose(_public):
        return int(rng.integers(N_ACTIONS))

    return choose


def _native_step(world):
    def step(action: int):
        # Evaluator-only private snapshots; never enter the acting callback.
        before = (tuple(map(int, world._pos)), bool(world._has_key),
                  bool(world._door_open))
        next_view, reward, terminated, truncated, _info = world.step(action)
        after = (tuple(map(int, world._pos)), bool(world._has_key),
                 bool(world._door_open))
        return (np.asarray(next_view).reshape(-1), float(reward),
                bool(terminated or truncated), before != after)

    return step


def collect_workshop_native() -> tuple[list[PublicEpisodeAudit],
                                       list[PublicEpisodeAudit], dict]:
    """Collect exactly 40 native episodes, <=800 steps; no oracle action data."""
    from bagi.experiments.general_brain_native_g1 import workshop_scripts
    from bagi.worlds import WorkshopEnv

    if (LAYOUTS != ("default", "mirror_x") or len(EPISODE_SEEDS) != 10
            or CAP != 20 or MAX_ACTIONS != 800 or N_ACTIONS != 4):
        raise ValueError("Workshop preregistered collection changed")
    split: dict[str, list[PublicEpisodeAudit]] = {name: [] for name in LAYOUTS}
    trajectories: dict[str, dict[str, list[str]]] = {
        layout: {behavior: [] for behavior in BEHAVIORS} for layout in LAYOUTS
    }
    for layout in LAYOUTS:
        for behavior in BEHAVIORS:
            for index, seed in enumerate(EPISODE_SEEDS):
                world = WorkshopEnv(size=9, view_size=5, max_steps=CAP,
                                    start_mode="fixed", layout_mode=layout)
                initial, _evaluator_info = world.reset(seed=seed)
                if behavior == "g1_public_script":
                    fixture = workshop_scripts(initial)
                    script = tuple(fixture["b_good_switch"] + fixture["c_reach_goal"])
                    if len(script) != 16:
                        raise ValueError("G1 public script fixture changed")
                    choose_action = _script_actor(script)
                else:
                    # Independent per-episode random stream, paired across geometries.
                    choose_action = _random_actor(32100 + index)

                audit = collect_public_episode(
                    layout=layout, episode_id=f"workshop-{layout}-{behavior}-{seed}",
                    behavior=behavior, initial_public=np.asarray(initial).reshape(-1),
                    choose_action=choose_action, native_step=_native_step(world),
                    width=WIDTH, n_actions=N_ACTIONS, cap=CAP,
                )
                split[layout].append(audit)
                trajectories[layout][behavior].append(_fingerprint(audit))
    train, test = split["default"], split["mirror_x"]
    counts = verify_fixed_collection(
        train, test, train_layouts=("default",), test_layouts=("mirror_x",),
        episodes_per_layout=20, width=WIDTH, n_actions=N_ACTIONS,
        cap=CAP, max_native_actions=MAX_ACTIONS,
    )
    for layout in LAYOUTS:
        if Counter(a.episode.behavior for a in split[layout]) != Counter(
            {behavior: 10 for behavior in BEHAVIORS}
        ):
            raise ValueError("Workshop behavior balance changed")
    counts["unique_public_trajectories"] = {
        layout: {behavior: len(set(trajectories[layout][behavior]))
                 for behavior in BEHAVIORS} for layout in LAYOUTS
    }
    counts["native_actions_by_layout_behavior"] = {
        layout: {behavior: sum(a.native_actions for a in split[layout]
                               if a.episode.behavior == behavior)
                 for behavior in BEHAVIORS} for layout in LAYOUTS
    }
    counts["fixed_script_is_learned"] = False
    counts["new_teacher_queries"] = 0
    return train, test, counts
