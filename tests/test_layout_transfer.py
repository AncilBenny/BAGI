"""Mirror layout should preserve causal mechanics while changing observations."""

import numpy as np
import pytest

from bagi.experiments.reachable_state_holdout import enumerate_reachable
from bagi.worlds import WorkshopEnv

ACTIONS = [1, 1, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2]
MIRROR_ACTION = {0: 0, 1: 3, 2: 2, 3: 1}


def test_mirror_preserves_every_guided_causal_event():
    original = WorkshopEnv(size=9)
    reflected = WorkshopEnv(size=9, layout_mode="mirror_x")
    _, start_a = original.reset(seed=2)
    _, start_b = reflected.reset(seed=2)
    assert start_a["position"].tolist() == [1, 1]
    assert start_b["position"].tolist() == [1, 7]
    seen = []
    for action in ACTIONS:
        _, reward_a, done_a, _, info_a = original.step(action)
        _, reward_b, done_b, _, info_b = reflected.step(MIRROR_ACTION[action])
        assert reward_a == reward_b
        assert done_a == done_b
        assert info_a["events"] == info_b["events"]
        assert info_b["position"].tolist() == [
            int(info_a["position"][0]), 8 - int(info_a["position"][1])
        ]
        seen.extend(info_b["events"])
    assert {"switch_activated", "key_picked_up", "lock_opened", "goal_reached"} <= set(seen)


def test_reflection_has_equal_reachable_graph_size_and_new_visible_inputs():
    original = WorkshopEnv(size=9)
    reflected = WorkshopEnv(size=9, layout_mode="mirror_x")
    default_rows, default_meta = enumerate_reachable(original)
    mirror_rows, mirror_meta = enumerate_reachable(reflected)
    assert default_meta["reachable_nonterminal_states"] == 78
    assert mirror_meta["reachable_nonterminal_states"] == 78
    assert len(default_rows) == len(mirror_rows) == 312
    original_views = {(tuple(np.array(r["observation"]).flatten()), r["action"])
                      for r in default_rows}
    mirror_views = {(tuple(np.array(r["observation"]).flatten()), r["action"])
                    for r in mirror_rows}
    assert len(mirror_views - original_views) > 0


def test_seeded_start_is_symmetry_equivariant():
    env_default = WorkshopEnv(size=9, start_mode="seeded_start_chamber")
    env_mirror = WorkshopEnv(size=9, start_mode="seeded_start_chamber",
                             layout_mode="mirror_x")
    for seed in [1, 2, 3, 99]:
        _, a = env_default.reset(seed=seed)
        _, b = env_mirror.reset(seed=seed)
        assert b["position"].tolist() == [int(a["position"][0]),
                                          8 - int(a["position"][1])]


def test_layout_mode_validation():
    with pytest.raises(ValueError, match="layout_mode"):
        WorkshopEnv(layout_mode="rotate_arbitrary")
