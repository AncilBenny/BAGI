"""G2-E1: frozen-policy/random XLand PUBLIC transition collection adapter.

All actions are learner-executed; the native-state difference is evaluator-only.
No new teacher calls, heldout labels, or policy updates occur during collection.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np

from bagi.agents.public_effect_collection import (
    PublicEpisodeAudit,
    collect_public_episode,
    verify_fixed_collection,
)

TRAIN_LAYOUTS = tuple(range(11900, 11915))
TEST_LAYOUTS = tuple(range(12000, 12010))
MODEL_SEEDS = (17, 29)
TASKS = ("hold", "near", "putdown")
BEHAVIORS = ("frozen_expanded_bc", "uniform_random")
CAP = 20
WIDTH = 115
HISTORY_BITS = 28
N_ACTIONS = 6
MAX_ACTIONS = 6000


def verify_registration() -> None:
    """Fail closed on altered preregistration or duplicate layout families."""
    if (TRAIN_LAYOUTS != tuple(range(11900, 11915))
            or TEST_LAYOUTS != tuple(range(12000, 12010))
            or set(TRAIN_LAYOUTS) & set(TEST_LAYOUTS)
            or MODEL_SEEDS != (17, 29)
            or TASKS != ("hold", "near", "putdown")
            or BEHAVIORS != ("frozen_expanded_bc", "uniform_random")
            or (CAP, WIDTH, HISTORY_BITS, N_ACTIONS, MAX_ACTIONS)
            != (20, 115, 28, 6, 6000)):
        raise ValueError("G2-E1 XLand preregistration changed")


def _history_signature(history: tuple[int, ...]) -> np.ndarray:
    if len(history) != 4 or any(a not in range(7) for a in history):
        raise ValueError("Invalid executed-action history")
    return np.eye(7, dtype=np.float32)[np.asarray(history)].reshape(-1)


def _native_changed(before, after) -> bool:
    """Evaluator-only native-state delta; not an input or a visible-change label."""
    a, b = before.state, after.state
    for x, y in (
        (a.grid, b.grid),
        (a.agent.position, b.agent.position),
        (a.agent.direction, b.agent.direction),
        (a.agent.pocket, b.agent.pocket),
    ):
        if not np.array_equal(np.asarray(x), np.asarray(y)):
            return True
    return False


def collect_xland_one(
    *, timestep, policy, behavior: str, layout: int, model_seed: int,
    task: str, native_step: Callable, observe: Callable,
) -> PublicEpisodeAudit:
    """Run one legal public policy, remembering only *executed* actions.

    native_step(before_timestep, integer_action) is called exactly once per
    collected transition. Hidden native state is compared AFTER stepping and
    never reaches choose_action or policy.predict.
    """
    verify_registration()
    if (layout not in TRAIN_LAYOUTS + TEST_LAYOUTS or model_seed not in MODEL_SEEDS
            or task not in TASKS or behavior not in BEHAVIORS
            or (behavior == "frozen_expanded_bc" and policy is None)):
        raise ValueError("Unregistered G2-E1 native episode")
    if bool(timestep.last()):
        raise ValueError("Native XLand episode terminal at reset")
    current = timestep
    history = (6, 6, 6, 6)
    rng = np.random.default_rng(
        32100 + (model_seed - 17) * 100000 + TASKS.index(task) * 10000 + layout
    )

    def public(ts):
        view = np.asarray(observe(ts, history), dtype=np.float32)
        if (view.shape != (WIDTH,) or not np.isfinite(view).all()
                or not np.array_equal(view[-HISTORY_BITS:], _history_signature(history))):
            raise ValueError("Non-public or incorrect executed-action history")
        return view

    def choose_action(view):
        if behavior == "uniform_random":
            return int(rng.integers(N_ACTIONS))
        predicted, _state = policy.predict(view, deterministic=True)
        action = np.asarray(predicted)
        if action.shape not in ((), (1,)) or action.dtype.kind not in "iu":
            raise ValueError("Frozen BC proposed invalid native action")
        return int(action.reshape(-1)[0])

    def execute(action):
        nonlocal current, history
        before = current
        after = native_step(before, action)
        native_delta = _native_changed(before, after)  # evaluator only
        history = (*history[1:], action)
        current = after
        return public(after), float(after.reward), bool(after.last()), native_delta

    return collect_public_episode(
        layout=f"xland-{layout}",
        episode_id=f"xland-{layout}-{model_seed}-{task}-{behavior}",
        behavior=behavior, initial_public=public(current),
        choose_action=choose_action, native_step=execute,
        width=WIDTH, n_actions=N_ACTIONS, cap=CAP, history_bits=HISTORY_BITS,
    )


def collect_xland_native(
    source_dir: Path, old_teacher_dir: Path, diverse_dir: Path,
) -> tuple[list[PublicEpisodeAudit], list[PublicEpisodeAudit], dict]:
    """300 fixed native episodes <=6000 actions; no new expert labels."""
    import jax
    import xminigrid
    from xminigrid.wrappers import (
        DirectionObservationWrapper,
        RulesAndGoalsObservationWrapper,
    )

    from bagi.experiments import xland_diverse_physical_demos as diverse
    from bagi.experiments.general_brain_native_g1 import source_policies
    from bagi.experiments.xland_existing_bc import get_rules
    from bagi.experiments.xland_gate_b_exact_collection import public_features

    verify_registration()
    models, training = source_policies(source_dir, old_teacher_dir, diverse_dir)
    room, defaults = xminigrid.make("XLand-MiniGrid-R1-9x9")
    wrapped = RulesAndGoalsObservationWrapper(DirectionObservationWrapper(room))
    reset, step = jax.jit(wrapped.reset), jax.jit(wrapped.step)
    train: list[PublicEpisodeAudit] = []
    test: list[PublicEpisodeAudit] = []
    for layouts, destination in ((TRAIN_LAYOUTS, train), (TEST_LAYOUTS, test)):
        for seed in MODEL_SEEDS:
            policy = models[seed][diverse.ARMS[1]]
            for task in TASKS:
                params = defaults.replace(ruleset=get_rules(task), max_steps=CAP)
                for behavior in BEHAVIORS:
                    for layout in layouts:
                        timestep = reset(params, jax.random.key(layout))

                        def step_one(before, action, *, params=params):
                            return step(params, before, action)

                        destination.append(collect_xland_one(
                            timestep=timestep, policy=policy, behavior=behavior,
                            layout=layout, model_seed=seed, task=task,
                            native_step=step_one, observe=public_features,
                        ))
    counts = verify_fixed_collection(
        train, test,
        train_layouts=tuple(f"xland-{x}" for x in TRAIN_LAYOUTS),
        test_layouts=tuple(f"xland-{x}" for x in TEST_LAYOUTS),
        episodes_per_layout=len(MODEL_SEEDS) * len(TASKS) * len(BEHAVIORS),
        width=WIDTH, n_actions=N_ACTIONS, cap=CAP, max_native_actions=MAX_ACTIONS,
    )
    for rows, layouts in ((train, TRAIN_LAYOUTS), (test, TEST_LAYOUTS)):
        actual = Counter((a.episode.layout, a.episode.behavior) for a in rows)
        expected = Counter({(f"xland-{layout}", behavior):
                            len(MODEL_SEEDS) * len(TASKS)
                            for layout in layouts for behavior in BEHAVIORS})
        if actual != expected:
            raise ValueError("XLand preregistered behavior/seed/task balance changed")
    counts["source_training"] = {str(seed): training[seed][diverse.ARMS[1]]
                                 for seed in MODEL_SEEDS}
    counts["actions_by_split_behavior"] = {
        name: {behavior: sum(a.native_actions for a in rows
                             if a.episode.behavior == behavior)
               for behavior in BEHAVIORS}
        for name, rows in (("train", train), ("test", test))
    }
    counts["new_teacher_queries"] = 0
    counts["new_online_policy_training_steps"] = 0
    counts["discarded_native_branches"] = 0
    return train, test, counts
