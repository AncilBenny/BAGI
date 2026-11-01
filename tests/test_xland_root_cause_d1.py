"""No native dependencies needed to test root-cause causal isolation contracts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from bagi.experiments.xland_root_cause_d1 import (
    CAP,
    CONDITIONS,
    FRESH_LAYOUTS,
    INPUT_WIDTH,
    PAD_HISTORY,
    TASKS,
    execute_diagnostic_episode,
    failure_stage,
    next_actual_history,
    policy_features,
    summarize_cases,
    validate_paired_runs,
    validate_registration,
    visible_target,
)

BLUE = np.array([8, 2], dtype=np.int64)


@dataclass(frozen=True)
class FakeStep:
    n: int = 0
    carrying: bool = False
    placed: bool = False
    reward: float = 0.0

    @property
    def observation(self):
        image = np.zeros((5, 5, 2), dtype=np.int64)
        image[1, 3] = BLUE
        return {"img": image}

    def last(self):
        return bool(self.placed or self.n >= 4)


class FakePolicy:
    def __init__(self, succeed=True):
        self.succeed = succeed
        self.inputs = []

    def predict(self, obs, deterministic=True):
        assert deterministic
        self.inputs.append(np.asarray(obs).copy())
        if not self.succeed:
            return np.asarray(0, dtype=np.int64), None
        # Real-policy action based ONLY on legal public float32 features.
        n = int(obs[0])
        action = 3 if n == 1 else 4 if n == 2 else 0
        return np.asarray(action, dtype=np.int64), None


def step_native(before, taken):
    next_n = before.n + 1
    picked = taken == 3 and before.n == 1 and not before.carrying
    carrying = before.carrying or picked
    dropped = taken == 4 and carrying
    return FakeStep(
        next_n, carrying and not dropped, dropped,
        1.0 if dropped else 0.0,
    )


def transition(before, after, taken):
    pickup = taken == 3 and not before.carrying and after.carrying
    drop = taken == 4 and before.carrying and after.placed
    return pickup, drop, False


def public_features(ts, history):
    features = np.zeros(INPUT_WIDTH, dtype=np.float32)
    features[0] = ts.n
    features[-28:] = np.eye(7, dtype=np.float32)[list(history)].reshape(-1)
    return features


def episode(policy, *, task="putdown", eligible=True, condition=CONDITIONS[0],
            step=step_native):
    return execute_diagnostic_episode(
        policy, FakeStep(), layout=FRESH_LAYOUTS[0], task=task,
        condition=condition, native_step=step, observe=public_features,
        eligible=eligible, target_tile=BLUE,
        effective_transition=transition, final_adjacent=lambda ts: ts.placed,
    )


def test_registration_and_real_executed_history_vs_masked_copy():
    validate_registration()
    hist = (6, 6, 3, 1)
    source = public_features(FakeStep(), hist)
    live = policy_features(source, hist, CONDITIONS[0])
    masked = policy_features(source, hist, CONDITIONS[1])
    assert np.array_equal(live, source)
    assert np.array_equal(masked[:-28], source[:-28])
    assert np.array_equal(masked[-28:], PAD_HISTORY)
    assert np.array_equal(source[-28:], live[-28:])
    assert next_actual_history(hist, 4) == (6, 3, 1, 4)
    with pytest.raises(ValueError, match="history mismatch"):
        policy_features(source, (6, 6, 6, 6), CONDITIONS[0])
    with pytest.raises(ValueError, match="Malformed"):
        next_actual_history((6, 6, 6, 9), 1)


def test_real_native_physical_case_separates_actor_from_probe():
    calls = []

    def traced(before, taken):
        calls.append((before.n, taken))
        return step_native(before, taken)

    model = FakePolicy()
    case = episode(model, step=traced)
    assert case["eligible"]
    assert case["physical_placement"] and case["native_reward"]
    assert case["native_actions"] == 3
    assert case["effective_blue_pickups"] == 1
    assert case["effective_blue_drops"] == 1
    assert case["ordered_blue_drop"]
    assert case["first_effective_pickup_action"] == 1
    assert case["first_effective_drop_action"] == 2
    assert case["counterfactual_native_probes"] == 1
    assert len(calls) == case["native_actions"] + case["counterfactual_native_probes"]
    assert calls == [(0, 0), (0, 3), (1, 3), (2, 4)]
    assert case["native_action_histogram"] == {0: 1, 3: 1, 4: 1}
    assert all(inp.shape == (115,) for inp in model.inputs)
    assert case["descriptive_failure_stage"] == "complete_physical_sequence"


def test_ignored_feasible_pickup_reports_action_selection_bottleneck():
    policy = FakePolicy(succeed=False)
    case = episode(policy)
    assert not case["physical_placement"]
    assert case["native_actions"] == CAP // 10
    assert case["counterfactual_native_probes"] == case["native_actions"]
    assert case["pickup_opportunity_steps"] == 1
    assert case["missed_feasible_pickup_steps"] == 1
    assert case["effective_blue_pickups"] == 0
    assert case["descriptive_failure_stage"] == (
        "feasible_pickup_not_executed_successfully"
    )


def test_ablation_never_changes_true_recorded_action_history():
    live = episode(FakePolicy(), condition=CONDITIONS[0])
    erased_policy = FakePolicy()
    erased = episode(erased_policy, condition=CONDITIONS[1])
    assert live["physical_placement"] == erased["physical_placement"]
    assert live["native_actions"] == erased["native_actions"]
    assert all(np.array_equal(v[-28:], PAD_HISTORY) for v in erased_policy.inputs)
    assert erased["counterfactual_native_probes"] == 1


def test_no_learner_or_counterfactual_actions_on_ineligible_layout():
    events = []

    def disallowed(*args):
        events.append(args)
        raise AssertionError("Should never step an ineligible episode")

    case = episode(FakePolicy(), eligible=False, step=disallowed)
    assert not case["eligible"]
    assert case["native_actions"] == case["counterfactual_native_probes"] == 0
    assert case["descriptive_failure_stage"] == "initially_ineligible"
    assert not events


def test_public_visibility_detection_and_validation():
    assert visible_target(FakeStep().observation["img"], BLUE)
    assert not visible_target(np.zeros((5, 5, 2), dtype=int), BLUE)
    with pytest.raises(ValueError, match="tile shape"):
        visible_target(np.zeros((5, 5)), BLUE)


def test_stage_funnel_does_not_conflate_native_reward_and_physical_success():
    case = episode(FakePolicy())
    case["physical_placement"] = False
    case["native_reward"] = True
    case["effective_blue_drops"] = 0
    assert failure_stage(case) == "pickup_succeeded_but_no_effective_drop"
    case["effective_blue_drops"] = 1
    assert failure_stage(case) == "drop_occurred_but_no_valid_final_goal"
    case["target_visible_steps"] = 0
    assert failure_stage(case) == "target_never_publicly_seen"
    case["target_visible_steps"] = 2
    case["pickup_opportunity_steps"] = 0
    assert failure_stage(case) == "target_seen_but_no_pickup_precondition"


def test_aggregated_step_and_stage_accounting():
    ok = episode(FakePolicy())
    fail = episode(FakePolicy(succeed=False))
    stats = summarize_cases([ok, fail], "putdown")
    assert stats["eligible"] == 2
    assert stats["physical_placements"] == 1
    assert stats["effective_pickup_episodes"] == 1
    assert stats["pickup_opportunity_episodes"] == 2
    assert stats["native_learner_actions"] == 7
    assert stats["discarded_counterfactual_native_probes"] == 5
    assert sum(stats["failure_funnel"].values()) == 2


def test_strict_paired_provenance_and_eligibility():
    pair = []
    for seed in (17, 29):
        for condition in CONDITIONS:
            tasks = {}
            for task in TASKS:
                cases = [
                    {"layout": layout, "eligible": True, "native_actions": 1}
                    for layout in FRESH_LAYOUTS
                ]
                tasks[task] = {
                    "cases": cases, "native_rewarded": 0,
                    "physical_placements": 0, "pickup_opportunity_episodes": 0,
                }
            pair.append({"model_seed": seed, "condition": condition, "tasks": tasks})
    paired = validate_paired_runs(pair)
    assert sorted(paired) == ["17", "29"]
    pair[1]["tasks"]["putdown"]["cases"][0]["eligible"] = False
    with pytest.raises(ValueError, match="eligibility"):
        validate_paired_runs(pair)
    pair[1]["tasks"]["putdown"]["cases"][0]["eligible"] = True
    pair[2]["tasks"]["near"]["cases"][0]["layout"] = 11304
    with pytest.raises(ValueError, match="replaced"):
        validate_paired_runs(pair)


def test_native_workflow_yaml_is_parseable_and_bounded():
    from pathlib import Path

    import yaml

    workflow = Path(__file__).resolve().parents[1] / (
        ".github/workflows/bagi-root-cause-d1.yml"
    )
    spec = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    assert spec["name"].startswith("BAGI G2-D1 Root Cause:")
    job = spec["jobs"]["bounded-root-cause-diagnosis"]
    assert job["timeout-minutes"] == 25
    assert job["runs-on"] == "ubuntu-latest"
    assert any("summary.json" in str(s) for s in job["steps"])
