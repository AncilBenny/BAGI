"""Verify matched budgets and no privileged observations in Gate A3."""
import random
from collections import Counter

import numpy as np
import pytest

from bagi.experiments.coverage_collection import collect, coverage, select_action
from bagi.experiments.dataset_foundations import model_input


class OneStepEnv:
    max_steps = 4

    class Actions:
        n = 4

    action_space = Actions()

    def reset(self, *, seed=None):
        self.calls = 0
        self.reset_seed = seed
        return np.array([[0]]), {}

    def step(self, action):
        self.calls += 1
        return np.array([[int(action)]]), 0.0, False, False, {"events": []}


def test_novelty_prioritizes_unseen_actions_from_current_observation():
    obs = [[1]]
    seen = Counter({
        model_input({"observation": obs, "action": 0}): 5,
        model_input({"observation": obs, "action": 1}): 2,
        model_input({"observation": obs, "action": 2}): 1,
    })
    assert select_action(obs, random.Random(11), seen, 4,
                         policy="novelty", epsilon=0.0) == 3


def test_both_policies_account_for_exact_equal_real_interactions():
    for policy in ("random", "novelty"):
        env = OneStepEnv()
        rows = collect(env, budget=19, guided_episodes=1,
                       policy=policy, seed=7, epsilon=0.2)
        assert len(rows) == 19
        assert len({row["episode"] for row in rows}) >= 2
        assert all("position" not in row["info"] for row in rows)
        assert all("has_key" not in row["info"] for row in rows)


def test_coverage_uses_observable_action_keys_and_deduplicates():
    rows = [{
        "episode": 0, "observation": [[0]], "action": 0,
        "next_observation": [[1]], "reward": 0,
        "info": {"events": ["switch_activated"]},
    }] * 4
    key = model_input(rows[0])
    m = coverage(rows, {key}, {key})
    assert m["real_transitions"] == 4
    assert m["unique_observation_action_keys"] == 1
    assert m["unique_supervised_transitions"] == 1
    assert m["reachable_input_fraction_covered"] == 1.0
    assert m["positive_event_rows"]["switch_activated"] == 4
    assert m["unique_event_positive_input_keys"]["switch_activated"] == 1


def test_collection_rejects_bad_budgets_and_epsilon():
    with pytest.raises(ValueError, match="Positive budget"):
        collect(OneStepEnv(), budget=0, guided_episodes=1,
                policy="random", seed=0, epsilon=0.1)
    with pytest.raises(ValueError, match="Positive budget"):
        collect(OneStepEnv(), budget=5, guided_episodes=0,
                policy="random", seed=0, epsilon=1.1)
