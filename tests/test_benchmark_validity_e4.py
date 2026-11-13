"""Synthetic-only tests of G2-E4-P0 diversity and exact-step preflight."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from bagi.experiments import benchmark_validity_e4 as audit


class FakeDiscrete:
    def __init__(self, n: int = 7):
        self.n = n

    def contains(self, action):
        return type(action) is int and 0 <= action < self.n


class FakeGrid:
    def __init__(self, marker: int):
        self.marker = marker

    def encode(self):
        return np.full((3, 3, 3), self.marker, dtype=np.uint8)


class FakeWorld:
    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self, marker: int, early_stop: bool = False):
        self.action_space = FakeDiscrete()
        self.unwrapped = SimpleNamespace(grid=FakeGrid(marker), agent_pos=(0, 1), agent_dir=0)
        self.marker = marker
        self.early_stop = early_stop
        self.calls = 0
        self.resets = []
        self.private = {"secret": "DO NOT ACT ON ME"}

    def reset(self, *, seed):
        self.resets.append(seed)
        self.calls = 0
        return np.array([self.marker, 0], dtype=np.uint8), self.private

    def step(self, action):
        self.calls += 1
        done = self.early_stop and self.calls == 3
        return np.array([self.marker, self.calls], dtype=np.uint8), 0.0, done, False, self.private

    def close(self):
        return None


def test_frozen_registry_budget_and_fixed_script():
    audit.verify_registration()
    assert audit.MAX_RESETS == 64
    assert audit.MAX_NATIVE_ACTIONS == 1536
    actions = audit.fixed_actions()
    assert len(actions) == 24
    assert actions == audit.fixed_actions()
    assert all(0 <= action < 7 for action in actions)
    assert audit.FUTURE_TEST_SEED_FLOOR > max(audit.AUDIT_SEEDS)


def test_paired_common_actions_not_chosen_by_evaluator_state():
    left = FakeWorld(marker=1)
    right = FakeWorld(marker=2)
    script = audit.fixed_actions()
    a = audit.inspect_one(left, audit.AUDIT_SEEDS[0], script)
    b = audit.inspect_one(right, audit.AUDIT_SEEDS[0], script)
    assert a["native_actions"] == b["native_actions"] == 24
    assert a["evaluator_layout_sha256"] != b["evaluator_layout_sha256"]
    assert a["public_sequence_sha256"] != b["public_sequence_sha256"]
    assert a["initial_public_sha256"] != b["initial_public_sha256"]
    assert left.resets == right.resets == [21000]


def test_early_termination_reduces_actual_action_accounting():
    env = FakeWorld(marker=4, early_stop=True)
    row = audit.inspect_one(env, audit.AUDIT_SEEDS[0], audit.fixed_actions())
    assert row["native_actions"] == env.calls == 3
    assert row["native_terminated"]
    assert not row["native_truncated"]


def test_rejects_unregistered_seeds_and_action_scripts_before_reset():
    env = FakeWorld(marker=1)
    with pytest.raises(ValueError, match="Unregistered"):
        audit.inspect_one(env, 25000, audit.fixed_actions())
    with pytest.raises(ValueError, match="Unregistered"):
        audit.inspect_one(env, 21000, tuple([0] * 24))
    assert env.calls == 0
    assert not env.resets


def test_rejects_nonpublic_image_and_bad_native_action_space():
    with pytest.raises(ValueError, match="PUBLIC"):
        audit._hash_array(np.full((3, 3, 3), 0))
    with pytest.raises(ValueError, match="PUBLIC"):
        audit._hash_array(np.array([np.nan]))
    env = FakeWorld(marker=1)
    env.action_space = FakeDiscrete(n=4)
    with pytest.raises(ValueError, match="action space"):
        audit.inspect_one(env, audit.AUDIT_SEEDS[0], audit.fixed_actions())
    assert not env.resets


def test_no_benchmark_promotion_without_both_layout_and_public_diversity():
    rows = []
    for i, seed in enumerate(audit.AUDIT_SEEDS):
        rows.append({
            "reset_seed": seed,
            "evaluator_layout_sha256": f"layout{i}",
            "initial_public_sha256": "same-start",
            "public_sequence_sha256": "same-public-sequence",
            "native_actions": 24,
        })
    summary = audit.summarize_task(rows)
    assert summary["distinct_evaluator_reset_layouts"] == 16
    assert summary["distinct_initial_public_views"] == 1
    assert not summary["qualified_diversity_only"]
    for i, row in enumerate(rows):
        row["public_sequence_sha256"] = f"trace{i % 8}"
    assert audit.summarize_task(rows)["qualified_diversity_only"]
    with pytest.raises(ValueError, match="Missing"):
        audit.summarize_task(rows[:-1])


def test_registration_change_fails_closed(monkeypatch):
    monkeypatch.setattr(audit, "PROBE_ACTIONS", 25)
    with pytest.raises(ValueError, match="preregistration"):
        audit.verify_registration()


def test_numeric_digest_depends_on_shape_and_values_not_pointer():
    obs = np.array([3, 1, 0], dtype=np.int8)
    assert audit._hash_array(obs) == audit._hash_array(obs.copy())
    assert audit._hash_array(obs) != audit._hash_array(np.array([3, 1, 0], dtype=np.int16))
    obs[0] = 4
    assert audit._hash_array(obs) != audit._hash_array(np.array([3, 1, 0], dtype=np.int8))
