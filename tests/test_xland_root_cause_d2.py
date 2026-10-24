"""G2-D2 contract tests, not evidence of learned spatial competence."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from bagi.experiments.xland_root_cause_d2 import (
    NEW_LAYOUTS,
    aggregate_cases,
    diagnostic_episode,
    evaluator_geometry,
    failure_stage,
    next_history,
    observation_for_policy,
    verify_registration,
)

DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass(frozen=True)
class Step:
    pos: tuple[int, int]
    heading: int = 1
    target: tuple[int, int] | None = (2, 4)
    pocket: int = 0
    n: int = 0
    reward: float = 0.0

    def last(self):
        return self.n >= 40 or self.reward > 0


class Policy:
    def __init__(self, actions):
        self.actions = list(actions)
        self.observations = []

    def predict(self, x, *, deterministic):
        assert deterministic
        self.observations.append(x.copy())
        return np.array(self.actions.pop(0)), None


def observe(_ts, history):
    a = np.zeros(115, dtype=np.float32)
    a[-28:] = np.eye(7, dtype=np.float32)[list(history)].reshape(-1)
    return a


def inspect(ts):
    target = ts.target
    return {
        "target": target,
        "geometry": evaluator_geometry(
            ts.pos, ts.heading, target if target is not None else (-99, -99),
            DIRECTIONS,
        ),
        "pocket_empty": ts.pocket == 0,
    }


def step(ts, action):
    if action == 3 and ts.pos == (2, 3) and ts.heading == 1 and ts.pocket == 0:
        return Step(ts.pos, ts.heading, None, 1, ts.n + 1)
    if action == 0 and ts.pos == (2, 2):
        return Step((2, 3), ts.heading, ts.target, ts.pocket, ts.n + 1)
    if action == 1:
        return Step(ts.pos, (ts.heading + 1) % 4, ts.target, ts.pocket, ts.n + 1)
    return Step(ts.pos, ts.heading, ts.target, ts.pocket, ts.n + 1)


def transition(before, after, action):
    return (action == 3 and before.pocket == 0 and after.pocket == 1,
            False, False)


def run(actions, start=Step((2, 2)), layout=NEW_LAYOUTS[0]):
    # Stop early via a native terminal after exactly the actions in this test.
    limit = len(actions)
    policy = Policy(actions)

    def bounded(ts, action):
        out = step(ts, action)
        if out.n >= limit:
            return Step(out.pos, out.heading, out.target, out.pocket, 40)
        return out

    case = diagnostic_episode(
        policy, start, layout=layout, eligible=True,
        observe=observe, native_step=bounded, inspect_state=inspect,
        visible_target=lambda ts: ts.target is not None,
        effective_transition=transition,
        final_adjacent=lambda _ts: False,
        shortest_pickup_actions=3,
    )
    assert len(policy.observations) == limit
    return case, policy


def test_registration_disjoint_and_budget():
    verify_registration()
    assert NEW_LAYOUTS == tuple(range(11600, 11610))


@pytest.mark.parametrize(("position", "heading", "expected"), [
    ((2, 3), 1, (1, True, True)),
    ((2, 3), 0, (1, True, False)),
    ((2, 2), 1, (2, False, False)),
    ((2, 5), 3, (1, True, True)),
])
def test_native_heading_geometry(position, heading, expected):
    s = evaluator_geometry(position, heading, (2, 4), DIRECTIONS)
    assert (s["distance"], s["adjacent"], s["facing"]) == expected


def test_navigation_funnel_separates_far_away_from_rotation():
    distant, _ = run([1, 1])
    assert distant["adjacent_steps"] == 0
    assert distant["failure_stage"] == "saw_target_never_approached_adjacent"
    close, _ = run([2, 2], start=Step((2, 3), heading=0))
    assert close["adjacent_steps"] > 0
    assert close["facing_steps"] == 0
    assert close["failure_stage"] == "approached_adjacent_never_faced"


def test_discarded_probe_is_counted_not_executed_and_policy_input_is_public():
    case, policy = run([0, 1, 0], start=Step((2, 3), heading=1))
    assert case["native_actions"] == 3
    assert case["discarded_pickup_probes"] >= 1
    assert case["pickup_ready_steps"] > 0
    assert case["missed_pickup_ready_steps"] > 0
    assert case["effective_blue_pickups"] == 0
    assert policy.observations[0].shape == (115,)
    assert np.all(policy.observations[0][:87] == 0)
    assert np.array_equal(policy.observations[1][-28:],
                          np.eye(7, dtype=np.float32)[[6, 6, 6, 0]].reshape(-1))


def test_recovery_of_correct_pickup_does_not_assume_drop_success():
    case, _ = run([3], start=Step((2, 3), heading=1))
    assert case["effective_blue_pickups"] == 1
    assert case["effective_blue_drops"] == 0
    assert case["failure_stage"] == "acquired_but_never_effectively_dropped"


def test_actual_executed_history_provenance_and_input_guard():
    x = observe(None, (6, 6, 6, 6))
    assert observation_for_policy(x, (6, 6, 6, 6)).shape == (115,)
    assert next_history((6, 6, 6, 6), 3) == (6, 6, 6, 3)
    with pytest.raises(ValueError, match="truly executed"):
        observation_for_policy(x, (6, 6, 6, 3))
    with pytest.raises(ValueError, match="Invalid actual"):
        next_history((6, 6, 6, 6), 7)


def test_ineligible_layout_fails_closed_without_simulator_calls():
    case = diagnostic_episode(
        Policy([]), Step((2, 2)), layout=NEW_LAYOUTS[0], eligible=False,
        observe=observe, native_step=lambda *_: None, inspect_state=inspect,
        visible_target=lambda *_: True, effective_transition=transition,
        final_adjacent=lambda *_: False, shortest_pickup_actions=None,
    )
    assert case["native_actions"] == case["discarded_pickup_probes"] == 0
    assert failure_stage(case) == "ineligible_nonadjacent_gate"


def test_aggregator_never_substitutes_layouts():
    cases = []
    for layout in NEW_LAYOUTS:
        item, _ = run([1], layout=layout)
        cases.append(item)
    data = aggregate_cases(cases)
    assert data["eligible"] == 10 and data["native_actions"] == 10
    with pytest.raises(ValueError, match="replaced or skipped"):
        aggregate_cases(cases[:-1])


def test_real_step_geometry_never_changes_predicted_action_sequence():
    normal, _ = run([0, 1], start=Step((2, 2)))
    rotated, _ = run([0, 1], start=Step((2, 3), heading=1))
    assert normal["actions"] == rotated["actions"] == [0, 1]
    assert normal["native_actions"] == rotated["native_actions"] == 2
    assert rotated["discarded_pickup_probes"] >= normal["discarded_pickup_probes"]
