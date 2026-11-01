"""No learner training or native reset/step; evaluator-only source contract."""

from __future__ import annotations

import pytest

from bagi.experiments import g2_e6_p0_workshop_size_native_parity as audit


def test_registration_frozen(monkeypatch):
    audit.verify_registration()
    monkeypatch.setattr(audit, "SIZES", (9, 13))
    with pytest.raises(ValueError, match="registration changed"):
        audit.verify_registration()


def test_source_hash_change_fails_closed(monkeypatch):
    def changed_source():
        raise ValueError("Workshop native source changed")

    monkeypatch.setattr(audit.p2, "verify_source", changed_source)
    with pytest.raises(ValueError, match="native source changed"):
        audit.run()


def test_native_source_parity_without_reset_or_step(monkeypatch):
    pytest.importorskip("gymnasium")
    world_class = audit.p2._world_class()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native reset/step forbidden in evaluator graph")

    monkeypatch.setattr(world_class, "reset", forbidden)
    monkeypatch.setattr(world_class, "step", forbidden)
    result = audit.run()
    assert result["status"] == "complete_evaluator_only"
    assert result["native_step_calls"] == result["native_reset_calls"] == 0
    assert len(result["cases"]) == 6
    for row in result["cases"]:
        assert row["all_starts_reachable_within_40"]
        assert row["min_shortest_actions"] <= row["max_shortest_actions"] <= 40
        assert row["keys_with_ambiguous_next_public_view"] > 0
