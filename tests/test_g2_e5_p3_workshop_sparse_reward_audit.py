"""Simulator-free probability invariants and guarded real-source parity."""

from __future__ import annotations

import importlib.util

import pytest
from test_g2_e5_p2_workshop_reachability_audit import SyntheticWorkshop

from bagi.experiments import g2_e5_p2_workshop_reachability_audit as graph
from bagi.experiments import g2_e5_p3_workshop_sparse_reward_audit as audit


class ContractSyntheticWorkshop(SyntheticWorkshop):
    def __init__(self, *, max_steps, **kwargs):
        super().__init__(max_steps=max_steps, **kwargs)
        self.max_steps = max_steps


def synthetic(layout="default"):
    return ContractSyntheticWorkshop(size=9, view_size=5, max_steps=40,
                             start_mode="seeded_start_chamber", layout_mode=layout)


@pytest.mark.parametrize("layout", ("default", "mirror_x"))
def test_exact_random_sparse_reward_probability(layout):
    result = audit.audit_random_policy(synthetic(layout))
    assert result["registered_starts"] == 13
    assert result["reachable_evaluator_states"] == 79
    assert result["uniform_random_goal_probability_40"] == pytest.approx(
        0.0002487796882890816, abs=1e-13
    )
    assert result["minimum_start_probability"] == pytest.approx(
        0.00003891015236460202, abs=1e-13
    )
    assert result["maximum_start_probability"] == pytest.approx(
        0.000517205074650073, abs=1e-13
    )
    final = result["checkpoint_probabilities"]["40"]
    assert final["switch_activated"] == pytest.approx(0.746203535245184, abs=1e-12)
    assert final["key_acquired"] == pytest.approx(0.062034535411611365, abs=1e-12)
    assert final["native_goal"] == pytest.approx(0.0002487796882890816, abs=1e-12)
    assert result["hypothetical_410_episode_no_goal_probability"] > 0.90


def test_rejects_changed_horizon():
    world = synthetic()
    world.max_steps = 41
    with pytest.raises(ValueError, match="horizon"):
        audit.audit_random_policy(world)


def test_run_is_evaluator_only(monkeypatch):
    monkeypatch.setattr(graph, "verify_source", lambda: None)
    monkeypatch.setattr(graph, "_world_class", lambda: ContractSyntheticWorkshop)
    result = audit.run()
    assert result["status"] == "complete"
    assert result["native_step_calls"] == result["native_reset_calls"] == 0
    assert result["teacher_queries"] == result["policy_updates"] == 0
    assert result["heldout_reads"] == 0
    assert not result["promotion"]
    assert set(result["layouts"]) == {"default", "mirror_x"}


def test_native_source_parity_never_steps_or_resets(monkeypatch):
    if importlib.util.find_spec("gymnasium") is None:
        pytest.skip("Real Workshop source parity executes in dependency-installed CI")
    graph.verify_source()
    world_class = graph._world_class()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Native reset/step forbidden in evaluator probability audit")

    monkeypatch.setattr(world_class, "step", forbidden)
    monkeypatch.setattr(world_class, "reset", forbidden)
    result = audit.run()
    assert result["layouts"]["default"]["registered_starts"] == 13
    assert result["layouts"]["mirror_x"]["checkpoint_probabilities"]["40"][
        "native_goal"
    ] == pytest.approx(0.0002487796882890816, abs=1e-12)
