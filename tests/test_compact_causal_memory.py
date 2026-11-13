"""Tests for the nonprivileged, rule-based compact causal memory reference."""

from bagi.experiments.compact_causal_memory import (
    compact_feature,
    lookup_with_compact_memory,
    sensed_causal_memory,
)
from bagi.worlds import WorkshopEnv
from bagi.worlds.workshop import KEY, SWITCH


def observed(tile, direction=1):
    view = [[0] * 5 for _ in range(5)]
    offset = ((-1, 0), (0, 1), (1, 0), (0, -1))[direction]
    view[2 + offset[0]][2 + offset[1]] = tile
    return view


def example(tile, after_tile, label):
    return {
        "history": [[observed(tile), 1]],
        "observation": observed(0),
        "action": 1,
        "next_observation": observed(after_tile),
        "reward": 0.0,
        "events": [label],
    }


def test_two_bits_from_visible_moves_only():
    history = [
        [observed(SWITCH), 1],
        [observed(0), 1],
        [observed(KEY, direction=2), 2],
    ]
    assert sensed_causal_memory(history[:1]) == (True, False)
    assert sensed_causal_memory(history) == (True, True)
    assert sensed_causal_memory([]) == (False, False)


def test_compact_memory_is_independent_of_privileged_metadata():
    row = example(KEY, 0, "key_picked_up")
    alternate = {**row, "info": {"position": [0, 0], "has_key": False}}
    assert compact_feature(row) == compact_feature(alternate)


def test_compact_memory_can_disambiguate_identical_current_views():
    train = [
        example(SWITCH, 4, "switch_activated"),
        example(KEY, 7, "key_picked_up"),
    ]
    heldout = [
        example(SWITCH, 4, "switch_activated"),
        example(KEY, 7, "key_picked_up"),
    ]
    result = lookup_with_compact_memory(train, heldout)
    assert result["heldout_key_coverage"] == 1.0
    assert result["heldout_whole_accuracy_with_unknown_as_error"] == 1.0


def test_inferred_memory_matches_actual_workshop_flags_on_guided_path():
    env = WorkshopEnv(size=9, max_steps=100, view_size=5)
    actions = [1, 1, 3, 3, 2, 2, 2, 2, 1, 1]
    observation, _ = env.reset(seed=12)
    history = []
    for action in actions:
        history.append([observation.tolist(), action])
        observation, _, _, _, info = env.step(action)
        assert sensed_causal_memory(history) == (
            bool(info["door_open"]), bool(info["has_key"])
        )
