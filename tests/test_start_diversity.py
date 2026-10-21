"""Opt-in seeded starting positions and matched data collection checks."""

import pytest

from bagi.experiments.start_diversity import collect, expert_actions_from_start
from bagi.worlds import WorkshopEnv


GUIDED = [1, 1, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2]


def test_default_fixed_start_unaffected_by_seed():
    env = WorkshopEnv(size=9)
    for seed in [1, 5, 501, 9021]:
        _, info = env.reset(seed=seed)
        assert info["position"].tolist() == [1, 1]


def test_seeded_start_is_reproducible_and_multiple_reachable_cells():
    env = WorkshopEnv(size=9, start_mode="seeded_start_chamber")
    positions = []
    for seed in range(40):
        _, info = env.reset(seed=seed)
        pos = tuple(info["position"].tolist())
        assert pos[0] in (1, 2)
        assert 1 <= pos[1] <= 7
        assert pos != (1, 3)
        assert not info["has_key"] and not info["door_open"]
        positions.append(pos)
        _, repeated = env.reset(seed=seed)
        assert tuple(repeated["position"]) == pos
    assert len(set(positions)) > 3


def test_rejects_unknown_start_mode():
    with pytest.raises(ValueError, match="start_mode"):
        WorkshopEnv(start_mode="random_everywhere")


def test_guided_oracle_can_reach_goal_from_every_candidate_start():
    env = WorkshopEnv(size=9, start_mode="seeded_start_chamber")
    for seed in range(30):
        obs, info = env.reset(seed=seed)
        assert obs.shape == (5, 5)
        route = expert_actions_from_start(info["position"].tolist(), GUIDED)
        goal = False
        for action in route:
            _, _, terminated, _, step_info = env.step(action)
            goal |= "goal_reached" in step_info["events"]
            if terminated:
                break
        assert goal


def test_equal_total_budget_and_event_support():
    for mode in ("fixed", "seeded_start_chamber"):
        env = WorkshopEnv(size=9, start_mode=mode)
        rows, summary = collect(env, budget=300, expert_episodes=5, seed=131, guided_actions=GUIDED)
        assert len(rows) == 300
        assert summary["real_interactions"] == 300
        assert summary["expert_goal_episodes"] == 5
        assert summary["event_positive_rows"]["goal_reached"] >= 5
