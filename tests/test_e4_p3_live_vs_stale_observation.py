"""Simulator-free G2-E4-P3 single-variable contract tests."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from bagi.experiments import e4_p3_live_vs_stale_observation as e4


class FakeSpace:
    n: ClassVar[int] = 7

    def seed(self, seed):
        self.seeded = seed

    def contains(self, action):
        return isinstance(action, (int, np.integer)) and 0 <= action < self.n


class FakeWorld:
    def __init__(self):
        self.action_space = FakeSpace()
        self.count = 0
        self.last_seed = None

    def reset(self, *, seed):
        self.last_seed = seed
        self.count = 0
        return np.array([0, 5, 9], dtype=np.uint8), {"privileged": True}

    def step(self, action):
        self.count += 1
        done = self.count == 3
        return np.array([self.count, action, 9], dtype=np.uint8), 0.0, done, False, {"secret": 1}


class FakePolicy:
    def __init__(self):
        self.inputs = []

    def predict(self, obs, *, deterministic):
        assert deterministic is False
        self.inputs.append(np.asarray(obs).copy())
        return np.array(2), None


def test_frozen_resources_and_source():
    e4.verify_registration()
    assert e4.MAX_TEST_ACTIONS == 9216
    assert e4.TASK == "MiniGrid-Empty-Random-6x6-v0"
    assert e4.SOURCE_ARTIFACT == 12606845219
    assert e4.TEST_SEEDS == tuple(range(28000, 28016))
    assert len(set(e4.TEST_SEEDS) & set(e4.p2.TEST_SEEDS)) == 0


def test_guard_detects_budget_mutation(monkeypatch):
    monkeypatch.setattr(e4, "MAX_TEST_ACTIONS", 9217)
    with pytest.raises(ValueError, match="prospective"):
        e4.verify_registration()


def test_frozen_seed_and_coupled_rng():
    assert e4.paired_seed(17, 28000) == 28000000
    assert e4.paired_seed(29, 28015) == 28100015
    for model, seed in ((3, 28000), (17, 26000), (17, 27000), (29, 28100)):
        with pytest.raises(ValueError):
            e4.paired_seed(model, seed)


def test_only_live_arm_sees_new_public_frames(monkeypatch):
    monkeypatch.setattr(e4.p1, "_layout_digest", lambda _env: "a" * 64)
    monkeypatch.setattr(e4.p1, "_public_digest", lambda obs: np.array(obs).tobytes())
    ledger = {"train_actions": 0, "test_actions": 0}
    live_actor = FakePolicy()
    stale_actor = FakePolicy()
    live, live_trace = e4.evaluate_episode(
        FakeWorld(), live_actor, 17, 28000, "live_public", ledger
    )
    stale, stale_trace = e4.evaluate_episode(
        FakeWorld(), stale_actor, 17, 28000, "stale_initial_public", ledger
    )
    assert live["sampling_seed"] == stale["sampling_seed"]
    assert ledger == {"train_actions": 0, "test_actions": 6}
    assert live["native_actions"] == stale["native_actions"] == 3
    assert [int(x[0]) for x in live_actor.inputs] == [0, 1, 2]
    assert [int(x[0]) for x in stale_actor.inputs] == [0, 0, 0]
    assert [int(x[0]) for x in live_trace["public_pre_action"]] == [0, 1, 2]
    assert [int(x[0]) for x in stale_trace["public_pre_action"]] == [0, 1, 2]
    assert np.array_equal(stale_trace["actor_public_input"], np.tile([0, 5, 9], (3, 1)))
    assert live["evaluator_layout_sha256"] == stale["evaluator_layout_sha256"]
    assert live["native_goal_success"] == stale["native_goal_success"] == 0


def _row(seed, arm, success=0, *, world="same"):
    return {
        "model_seed": 17,
        "reset_seed": seed,
        "arm": arm,
        "sampling_seed": e4.paired_seed(17, seed),
        "evaluator_layout_sha256": f"{world}{seed}",
        "initial_public_sha256": f"initial{seed}",
        "native_actions": 1,
        "native_reward": float(success),
        "native_terminated": bool(success),
        "native_truncated": not bool(success),
        "native_goal_success": success,
        "action_histogram": [0, 0, 1, 0, 0, 0, 0],
    }


def test_score_paired_evidence_and_feedback_gate():
    rows = {
        arm: [_row(s, arm, success=int(arm == "live_public" and i < 3))
              for i, s in enumerate(e4.TEST_SEEDS)]
        for arm in e4.ARMS
    }
    res = e4.score(rows)
    assert res["native_goals"] == {"live_public": 3, "stale_initial_public": 0}
    assert res["paired_live_wins"] == 3 and res["paired_live_losses"] == 0
    assert res["strong_online_feedback_signal"] is True
    rows["stale_initial_public"][0]["evaluator_layout_sha256"] = "forged"
    with pytest.raises(ValueError, match="not paired"):
        e4.score(rows)


def test_score_rejects_reordered_test_seeds():
    rows = {arm: [_row(seed, arm) for seed in e4.TEST_SEEDS] for arm in e4.ARMS}
    rows["stale_initial_public"].reverse()
    with pytest.raises(ValueError, match="paired TEST"):
        e4.score(rows)


def test_numeric_evidence_no_pickle(tmp_path):
    trace = {
        "public_pre_action": np.array([[0, 1], [1, 1]], dtype=np.uint8),
        "actor_public_input": np.array([[0, 1], [0, 1]], dtype=np.uint8),
        "actions": np.array([2, 1], dtype=np.int16),
        "rewards": np.array([0.0, 1.0]),
        "terminated": np.array([False, True]),
        "truncated": np.array([False, False]),
    }
    path = tmp_path / "trace.npz"
    digest = e4.save_numeric(path, [trace])
    assert len(digest) == 64
    with np.load(path, allow_pickle=False) as out:
        assert out["episode_offsets"].tolist() == [0, 2]
        np.testing.assert_array_equal(out["actor_public_input"], trace["actor_public_input"])
        assert not any(out[name].dtype.kind == "O" for name in out.files)


def test_invalid_actor_public_contract(monkeypatch):
    monkeypatch.setattr(e4.p1, "_layout_digest", lambda _env: "a" * 64)
    monkeypatch.setattr(e4.p1, "_public_digest", lambda obs: np.array(obs).tobytes())
    with pytest.raises(ValueError, match="observation condition"):
        e4.evaluate_episode(
            FakeWorld(), FakePolicy(), 17, 28000, "privileged_grid", {"test_actions": 0}
        )
