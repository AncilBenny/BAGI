"""G2-D3 integration contracts. Synthetic behavior is not a native result."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from bagi.experiments.general_brain_stagnation_g2 import (
    CAP,
    CONDITIONS,
    FRESH_LAYOUTS,
    MAX_NATIVE_LEARNER_ACTIONS,
    execute_episode,
    public_signature,
    ranked_frozen_policy,
    summarize_cases,
    validate_registration,
    validated_public_features,
)


def test_frozen_contract_and_correct_40_step_budget():
    validate_registration()
    assert FRESH_LAYOUTS == tuple(range(11700, 11710))
    assert CONDITIONS == ("direct", "stagnation_escape")
    assert MAX_NATIVE_LEARNER_ACTIONS == 2 * 2 * 3 * 10 * CAP == 4800


def public_observation(stage, history):
    result = np.zeros(115, dtype=np.float32)
    result[0] = stage
    result[-28:] = np.eye(7, dtype=np.float32)[list(history)].reshape(-1)
    return result


def test_public_signature_excludes_only_actual_history_bits():
    a = public_observation(0, (6, 6, 6, 6))
    b = public_observation(0, (6, 6, 6, 1))
    c = public_observation(1, (6, 6, 6, 6))
    assert public_signature(a) == public_signature(b)
    assert public_signature(a) != public_signature(c)


def test_actual_executed_action_history_provenance():
    original = public_observation(0, (6, 6, 6, 6))
    assert np.array_equal(validated_public_features(original, (6, 6, 6, 6)), original)
    with pytest.raises(ValueError, match="actual executed"):
        validated_public_features(original, (6, 6, 6, 4))
    with pytest.raises(ValueError, match="115"):
        validated_public_features(original[:114], (6, 6, 6, 6))


class FrozenPolicy:
    device = torch.device("cpu")

    def __init__(self, probabilities=(0.75, 0.25, 0, 0, 0, 0)):
        self.probabilities = torch.tensor([probabilities], dtype=torch.float32)
        self.queries = []

    def predict(self, x, *, deterministic):
        assert deterministic
        self.queries.append(np.asarray(x).copy())
        return np.asarray(int(torch.argmax(self.probabilities))), None

    def get_distribution(self, x):
        assert x.shape == (1, 115)
        return SimpleNamespace(distribution=SimpleNamespace(probs=self.probabilities))


def test_native_distribution_requires_argmax_parity():
    p = FrozenPolicy()
    x = public_observation(0, (6, 6, 6, 6))
    ranked = ranked_frozen_policy(p, x)
    assert ranked[0][0] == 0 and ranked[1][0] == 1
    assert pytest.approx(sum(v for _, v in ranked)) == 1.0


def test_mismatched_upstream_argmax_fails_closed():
    p = FrozenPolicy()
    p.predict = lambda *_args, **_kwargs: (np.asarray(1), None)
    with pytest.raises(ValueError, match="argmax"):
        ranked_frozen_policy(p, public_observation(0, (6, 6, 6, 6)))


@dataclass(frozen=True)
class Step:
    n: int = 0
    stage: int = 0
    reward: float = 0.0

    def last(self):
        return self.n >= 6 or self.reward > 0


def run_one(monkeypatch, *, condition="direct", fake_hidden_change=False, reward_only=False):
    monkeypatch.setitem(sys.modules, "bagi.experiments.xland_gate_b_exact_collection",
                        SimpleNamespace(native_action=lambda a: int(np.asarray(a).item())))
    p = FrozenPolicy()
    states = []

    def take(ts, action):
        states.append((ts.n, action))
        next_stage = 1 if action == 1 else ts.stage
        return Step(ts.n + 1, next_stage, 1.0 if reward_only else 0.0)

    def observe(ts, history):
        return public_observation(ts.stage, history)

    case = execute_episode(
        p, Step(), layout=FRESH_LAYOUTS[0], seed=17, task="putdown",
        condition=condition, eligible=True, observe=observe, native_step=take,
        public_target_seen=lambda ts: ts.stage == 1,
        evaluator_transition=lambda *_args: (False, False),
        final_adjacent=lambda _ts: False,
        physical_changed=lambda *_args: fake_hidden_change,
    )
    return case, p


def test_direct_baseline_does_not_modify_frozen_actions(monkeypatch):
    row, model = run_one(monkeypatch)
    assert row["native_actions"] == 6
    assert row["actual_executed_actions"] == [0] * 6
    assert row["escaped_actions"] == 0
    assert row["trailing_drop_action_run"] == 0
    assert len(model.queries) == 6


def test_public_stagnation_escape_changes_single_action_from_original_policy(monkeypatch):
    row, model = run_one(monkeypatch, condition="stagnation_escape")
    assert row["actual_executed_actions"][:3] == [0, 0, 0]
    assert row["actual_executed_actions"][3] == 1
    assert row["escaped_actions"] > 0
    assert not row["physical_placement"]
    assert np.array_equal(model.queries[4][-28:],
                          np.eye(7, dtype=np.float32)[[0, 0, 0, 1]].reshape(-1))


def test_public_stagnation_flag_not_equivalent_native_noop(monkeypatch):
    row, _model = run_one(monkeypatch, fake_hidden_change=True)
    assert row["suspected_stagnation_steps"] > 0
    assert row["false_public_stagnation_alerts"] == row["suspected_stagnation_steps"]


def test_ineligible_layout_no_steps_or_teacher(monkeypatch):
    monkeypatch.setitem(sys.modules, "bagi.experiments.xland_gate_b_exact_collection",
                        SimpleNamespace(native_action=lambda a: int(np.asarray(a).item())))
    row = execute_episode(
        FrozenPolicy(), Step(), layout=FRESH_LAYOUTS[0], seed=17,
        task="putdown", condition="stagnation_escape", eligible=False,
        observe=lambda *_: None, native_step=lambda *_: None,
        public_target_seen=lambda *_: False,
        evaluator_transition=lambda *_: (False, False),
        final_adjacent=lambda *_: False, physical_changed=lambda *_: False,
    )
    assert row["native_actions"] == row["escaped_actions"] == 0
    assert not row["physical_placement"]


def test_fixed_layout_aggregation_refuses_substitution():
    cases = [{"layout": i, "task": "putdown", "eligible": False,
              "native_actions": 0, "native_reward": False,
              "physical_placement": False, "target_publicly_seen": False,
              "effective_blue_pickups": 0, "effective_blue_drops": 0,
              "native_drop_action_attempts": 0,
              "suspected_stagnation_steps": 0, "escaped_actions": 0,
              "false_public_stagnation_alerts": 0,
              "trailing_drop_action_run": 0} for i in FRESH_LAYOUTS]
    assert summarize_cases(cases, "putdown")["eligible"] == 0
    with pytest.raises(ValueError, match="Missing/replaced"):
        summarize_cases(cases[:-1], "putdown")


def test_workflow_is_pinned_cpu_and_branch_safe():
    path = Path(__file__).parents[1] / ".github/workflows/bagi-general-stagnation-g2.yml"
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert data["on"]["push"]["branches"] == ["setup/initial-platform"]
    assert data["jobs"]["native-stagnation-trial"]["runs-on"] == "ubuntu-latest"
    assert int(data["jobs"]["native-stagnation-trial"]["timeout-minutes"]) <= 25


def test_reward_only_does_not_count_as_physical_placement(monkeypatch):
    row, _ = run_one(monkeypatch, reward_only=True)
    assert row["native_reward"]
    assert not row["physical_placement"]
    assert row["effective_blue_pickups"] == row["effective_blue_drops"] == 0


def test_invalid_policy_distribution_is_rejected():
    p = FrozenPolicy(probabilities=(0.5, 0.5, 0.5, 0, 0, 0))
    with pytest.raises(ValueError, match="probabilities"):
        ranked_frozen_policy(p, public_observation(0, (6, 6, 6, 6)))
