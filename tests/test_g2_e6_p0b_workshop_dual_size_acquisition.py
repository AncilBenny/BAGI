"""G2-E6-P0B synthetic contracts ONLY; no native Workshop reset or step."""

from __future__ import annotations

from collections import Counter
from typing import ClassVar

import numpy as np
import pytest

from bagi.experiments import g2_e6_p0b_workshop_dual_size_acquisition as p0b


class FakeWorld:
    calls: ClassVar[list[tuple[int, str, int]]] = []

    def __init__(self, *, size, view_size, max_steps, start_mode, layout_mode):
        assert size in (9, 11) and view_size == 5 and max_steps > 0
        assert start_mode == "seeded_start_chamber"
        self.size, self.layout_mode = size, layout_mode
        self.steps = 0

    def reset(self, *, seed):
        self.steps = 0
        self.calls.append((self.size, self.layout_mode, seed))
        return np.zeros((5, 5), dtype=np.int8), {"secret": "evaluator-only"}

    def step(self, action):
        assert 0 <= action < 4
        self.steps += 1
        return (np.full((5, 5), self.steps % 9, dtype=np.int8), 0.0,
                False, False, {"secret": "evaluator-only", "events": ["switch_activated"]})

    def close(self):
        pass


class FakeEpisode:
    def __init__(self, cap=3):
        self.calls = 0
        self.cap = cap

    def step(self, action):
        self.calls += 1
        done = self.calls == self.cap
        return (np.full((5, 5), self.calls, dtype=np.int8), float(done), done, False,
                {"secret": "PRIVATE", "events": ["goal_reached"] if done else []})


def test_frozen_constants_and_disjoint_reset_seed_contract(monkeypatch):
    p0b.verify_registration()
    assert p0b.MAX_NATIVE == 8192 + 11520
    assert p0b.EXPECTED_UPDATES == 2 * (4096 - 63)
    assert p0b.TRAIN_RESET_BASE == 20000
    assert set(p0b.RESET_SEEDS).isdisjoint(
        {p0b.TRAIN_RESET_BASE + 1000 * i + j for i in range(2) for j in range(300)}
    )
    monkeypatch.setattr(p0b, "TRAIN_RESET_BASE", 18000)
    with pytest.raises(ValueError, match="preregistration"):
        p0b.verify_registration()


def test_public_only_actions_and_chronological_numeric_trace():
    world = FakeEpisode()
    seen, updates = [], []

    def choose(public):
        assert isinstance(public, np.ndarray) and public.shape == (5, 5)
        seen.append(public.copy())
        return 2

    def after_step(before, action, reward, after, done):
        updates.append((before.copy(), action, after.copy(), done))

    numeric = p0b._empty_numeric()
    result = p0b._episode(world, np.zeros((5, 5), dtype=np.int8), choose, 40,
                          numeric, (0, 20000, 17, 9, 0, 0, 0), after_step)
    assert world.calls == 3
    assert result["goal_success"] and result["steps"] == 3
    assert len(seen) == len(updates) == len(numeric["executed_action"]) == 3
    assert all(np.array_equal(a, b) for a, b in zip(seen, numeric["public_pre_action"]))
    assert all(np.array_equal(a, b) for a, b in zip(numeric["public_post_action"][:-1],
                                                    numeric["public_pre_action"][1:]))
    assert numeric["step_in_episode"] == [0, 1, 2]
    assert numeric["terminated"] == [0, 0, 1]
    assert numeric["native_reward"] == [0.0, 0.0, 1.0]
    assert result["events"]["goal_reached"] == 1
    assert len(result["public_trajectory_sha256"]) == 64
    safe = p0b._arrays(numeric, 0)
    assert safe["public_pre_action"].shape == (3, 5, 5)
    assert all(v.dtype.kind != "O" for v in safe.values())


def test_invalid_action_observation_and_budget_fail_before_native_call():
    for bad in (-1, 4, "2", np.int64(2)):
        world = FakeEpisode()
        with pytest.raises(ValueError, match="actor action"):
            p0b._episode(world, np.zeros((5, 5), dtype=np.int8),
                          lambda _view, a=bad: a, 40, p0b._empty_numeric(),
                          (0, 20000, 17, 9, 0, 0, 0))
        assert world.calls == 0
    world = FakeEpisode()
    with pytest.raises(ValueError, match="public observation"):
        p0b._episode(world, np.zeros((9, 9), dtype=np.int8),
                      lambda _: 1, 40, p0b._empty_numeric(),
                      (0, 20000, 17, 9, 0, 0, 0))
    with pytest.raises(ValueError, match="budget"):
        p0b._episode(world, np.zeros((5, 5), dtype=np.int8),
                      lambda _: 1, 41, p0b._empty_numeric(),
                      (0, 20000, 17, 9, 0, 0, 0))
    assert world.calls == 0


def test_logged_native_action_survives_optimizer_exception():
    world = FakeEpisode()
    numeric = p0b._empty_numeric()

    def broken_update(*_args):
        raise RuntimeError("synthetic optimizer failure")

    with pytest.raises(RuntimeError, match="optimizer failure"):
        p0b._episode(world, np.zeros((5, 5), dtype=np.int8), lambda _: 1,
                      40, numeric, (0, 20000, 17, 9, 0, 0, 0), broken_update)
    assert world.calls == len(numeric["executed_action"]) == 1


def test_small_synthetic_train_budget_and_size_allocation(monkeypatch):
    monkeypatch.setattr(p0b, "WorkshopEnv", FakeWorld)
    monkeypatch.setattr(p0b, "TRAIN_PER_SIZE", 32)
    monkeypatch.setattr(p0b, "TRAIN_PER_SEED", 64)
    monkeypatch.setattr(p0b, "EPISODE_CAP", 8)
    FakeWorld.calls.clear()
    numeric = p0b._empty_numeric()
    ledger = {"native_resets": 0, "optimizer_updates": 0,
              "native_step_returns": 0, "native_train_step_returns": 0,
              "native_diagnostic_step_returns": 0, "native_step_exceptions": 0}
    training, untrained, trained, weights, next_id = p0b._train_seed(
        0, numeric, 0, ledger=ledger
    )
    assert ledger == {
        "native_resets": 8, "optimizer_updates": 1,
        "native_step_returns": 64, "native_train_step_returns": 64,
        "native_diagnostic_step_returns": 0, "native_step_exceptions": 0,
    }
    assert training["train_actions_by_size"] == {9: 32, 11: 32}
    assert training["optimizer_updates"] == 1
    assert next_id == 8
    assert len(numeric["executed_action"]) == 64
    assert Counter(numeric["size"]) == {9: 32, 11: 32}
    assert len({r["reset_seed"] for r in training["training"]}) == 8
    assert set(numeric["reset_seed"]).isdisjoint(p0b.RESET_SEEDS)
    assert all(v == 0 for v in numeric["phase_index"])
    assert len(weights) == 12
    assert len(p0b._weight_digest(weights, 17, "trained")) == 64
    assert isinstance(p0b._greedy(untrained, np.zeros((5, 5), dtype=np.int8)), int)
    assert isinstance(p0b._greedy(trained, np.zeros((5, 5), dtype=np.int8)), int)


def test_small_synthetic_matched_diagnostics_no_size13(monkeypatch):
    monkeypatch.setattr(p0b, "WorkshopEnv", FakeWorld)
    monkeypatch.setattr(p0b, "RESET_SEEDS", (19000, 19001))
    monkeypatch.setattr(p0b, "EPISODE_CAP", 2)
    FakeWorld.calls.clear()
    numeric = p0b._empty_numeric()
    net = p0b.DQNAgent(observation_shape=(5, 5), actions=4).online
    ledger = {"native_resets": 0, "optimizer_updates": 0,
              "native_step_returns": 0, "native_train_step_returns": 0,
              "native_diagnostic_step_returns": 0, "native_step_exceptions": 0}
    result, next_id = p0b._diagnose(0, net, net, numeric, 0, ledger=ledger)
    assert ledger == {
        "native_resets": 24, "optimizer_updates": 0,
        "native_step_returns": 48, "native_train_step_returns": 0,
        "native_diagnostic_step_returns": 48, "native_step_exceptions": 0,
    }
    assert next_id == 2 * 2 * 3 * 2
    assert len(numeric["executed_action"]) == next_id * 2
    assert set(numeric["size"]) == {9, 11}
    assert set(numeric["actor_index"]) == {1, 2, 3}
    assert set(numeric["phase_index"]) == {1}
    assert len(result["9"]["default"]["trained_greedy"]) == 2
    counts = Counter(FakeWorld.calls)
    assert len(counts) == 2 * 2 * 2
    assert set(counts.values()) == {3}



def test_failed_train_preserves_completed_resets_and_updates(monkeypatch):
    """An aborted native run must not erase earlier successful optimizer updates."""
    monkeypatch.setattr(p0b, "WorkshopEnv", FakeWorld)
    monkeypatch.setattr(p0b, "TRAIN_PER_SIZE", 40)
    monkeypatch.setattr(p0b, "TRAIN_PER_SEED", 80)
    monkeypatch.setattr(p0b, "EPISODE_CAP", 8)
    FakeWorld.calls.clear()

    def synthetic_learn(self, batch_size=64):
        assert batch_size == 64
        if len(self.buffer) == 65:
            raise RuntimeError("synthetic learn crash")
        return 0.5 if len(self.buffer) >= 64 else None

    monkeypatch.setattr(p0b.DQNAgent, "learn", synthetic_learn)
    numeric = p0b._empty_numeric()
    ledger = {"native_resets": 0, "optimizer_updates": 0,
              "native_step_returns": 0, "native_train_step_returns": 0,
              "native_diagnostic_step_returns": 0, "native_step_exceptions": 0}
    with pytest.raises(RuntimeError, match="synthetic learn crash"):
        p0b._train_seed(0, numeric, 0, ledger=ledger)
    assert len(FakeWorld.calls) == 9
    assert ledger == {
        "native_resets": 9, "optimizer_updates": 1,
        "native_step_returns": 65, "native_train_step_returns": 65,
        "native_diagnostic_step_returns": 0, "native_step_exceptions": 0,
    }
    assert len(numeric["executed_action"]) == 65
    assert all(phase == 0 for phase in numeric["phase_index"])

def test_training_reset_collision_aborts_before_native_reset(monkeypatch):
    monkeypatch.setattr(p0b, "WorkshopEnv", FakeWorld)
    monkeypatch.setattr(p0b, "TRAIN_RESET_BASE", 19000)
    monkeypatch.setattr(p0b, "TRAIN_PER_SEED", 64)
    monkeypatch.setattr(p0b, "TRAIN_PER_SIZE", 32)
    monkeypatch.setattr(p0b, "EPISODE_CAP", 8)
    FakeWorld.calls.clear()
    with pytest.raises(ValueError, match="reset collision"):
        p0b._train_seed(0, p0b._empty_numeric(), 0)
    assert not FakeWorld.calls


def test_numeric_phase_isolation_and_strict_weight_shapes():
    numeric = p0b._empty_numeric()
    for phase in (0, 1):
        values = (np.zeros((5, 5), dtype=np.int8),
                  np.ones((5, 5), dtype=np.int8), 1, 0.0, 0, 0,
                  phase, 0, 20000 if phase == 0 else 19000,
                  17, 9, 0, phase, phase)
        for name, value in zip(p0b.FIELDS, values, strict=True):
            numeric[name].append(value)
    train = p0b._arrays(numeric, 0)
    diagnostic = p0b._arrays(numeric, 1)
    assert train["executed_action"].shape == (1,)
    assert diagnostic["executed_action"].shape == (1,)
    assert train["phase_index"].tolist() == [0]
    assert diagnostic["phase_index"].tolist() == [1]
    assert train["reset_seed"].tolist() == [20000]
    assert diagnostic["reset_seed"].tolist() == [19000]
    with pytest.raises(ValueError, match="phase"):
        p0b._arrays(numeric, 2)
    net = p0b.DQNAgent(observation_shape=(5, 5), actions=4).online
    weights = p0b._weights(net, 17, "trained")
    bad = {**weights, "seed17_trained_net_0_weight": np.zeros((127, 25), dtype=np.float32)}
    with pytest.raises(ValueError, match="names or shapes"):
        p0b._weight_digest(bad, 17, "trained")

def test_returned_native_step_is_counted_before_invalid_public_post_view():
    """A failed post-step validator must not erase an executed native action."""
    class MalformedAfter:
        def __init__(self):
            self.calls = 0

        def step(self, action):
            self.calls += 1
            return np.zeros((4, 4), dtype=np.int8), 0.0, False, False, {"events": []}

    world = MalformedAfter()
    numeric = p0b._empty_numeric()
    ledger = {"native_step_returns": 0, "native_train_step_returns": 0,
              "native_diagnostic_step_returns": 0, "native_step_exceptions": 0}
    with pytest.raises(ValueError, match="public observation"):
        p0b._episode(world, np.zeros((5, 5), dtype=np.int8), lambda _: 1,
                     40, numeric, (0, 20000, 17, 9, 0, 0, 0), ledger=ledger)
    assert world.calls == ledger["native_step_returns"] == 1
    assert ledger["native_train_step_returns"] == 1
    assert ledger["native_diagnostic_step_returns"] == 0
    assert ledger["native_step_exceptions"] == 0
    assert not numeric["executed_action"]  # returned action is not a valid trace row


def test_native_step_exception_marks_count_uncertain():
    """An env.step exception may occur after internal mutation; never infer zero."""
    class RaisingStep:
        def step(self, action):
            raise RuntimeError("synthetic step exception")

    ledger = {"native_step_returns": 0, "native_train_step_returns": 0,
              "native_diagnostic_step_returns": 0, "native_step_exceptions": 0}
    with pytest.raises(RuntimeError, match="synthetic step exception"):
        p0b._episode(RaisingStep(), np.zeros((5, 5), dtype=np.int8), lambda _: 2,
                     40, p0b._empty_numeric(),
                     (0, 20000, 17, 9, 0, 0, 0), ledger=ledger)
    assert ledger["native_step_returns"] == 0
    assert ledger["native_step_exceptions"] == 1
