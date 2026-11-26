"""Deterministic local preregistration tests; no XLand simulation claimed."""
from dataclasses import dataclass, replace

import numpy as np
import pytest

from bagi.experiments.xland_gate_b_exact_collection import (
    BALANCED_LAYOUT_QUOTAS,
    QUERY_BUDGET,
    SOURCE_LAYOUTS,
    collect_arm,
    load_verified_collection_pair,
    save_numeric,
    validate_pair,
)


@dataclass(frozen=True)
class Step:
    seed: int
    n: int
    reward: float = 0.0

    def last(self):
        return self.n >= 7


def fake_reset(seed):
    return Step(seed, 0)


def fake_step(ts, action):
    assert action in range(6)
    return Step(ts.seed, ts.n + 1, float(ts.n == 6))


def fake_features(ts, history):
    arr = np.zeros(115, dtype=np.float32)
    arr[0] = ts.n
    arr[1] = ts.seed % 13
    arr[-28:] = np.eye(7, dtype=np.float32)[
        np.asarray(history, dtype=np.int64)
    ].reshape(-1)
    return arr


def fake_teacher(ts):
    return (2 if ts.n % 2 else 1), "seek_square"


def fake_learner(x):
    assert x.shape == (115,)
    return 0


def both():
    params = {
        "reset": fake_reset, "step": fake_step,
        "screen_nonadjacent": lambda ts: True,
        "feature_fn": fake_features, "teacher": fake_teacher,
        "learner": fake_learner, "seeds": (9701, 9702, 9708, 9710),
    }
    return (collect_arm(arm="reset_expert", **params),
            collect_arm(arm="learner_correction", **params))


def test_exact_budget_true_history_and_distinct_execution():
    a, b = both()
    assert a.complete and b.complete
    assert a.teacher_queries == b.teacher_queries == QUERY_BUDGET
    assert a.native_steps == b.native_steps == QUERY_BUDGET
    assert validate_pair(a, b)["matched"]
    assert all(r["executed_action"] == r["expert_label"] for r in a.rows)
    assert all(r["executed_action"] == 0 for r in b.rows)
    assert b.rows[1]["prior_executed_actions"] == (6, 6, 6, 0)
    assert a.rows[1]["prior_executed_actions"] == (6, 6, 6, 1)
    assert a.rows[7]["prior_executed_actions"] == (6, 6, 6, 6)
    assert a.rows[7]["state_origin"] == "reset"
    assert b.rows[1]["state_origin"] == "learner_continuation"
    assert b.summary()["accepted_teacher_labels"] == QUERY_BUDGET
    assert b.summary()["rejected_teacher_queries"] == 0
    assert b.summary()["expert_stage_label_counts"] == {"seek_square": 20}


def test_equal_20_step_budgets_do_not_hide_source_layout_drift(tmp_path):
    def shortened_expert_episodes(ts, action):
        # Reset expert ends each source episode at step 3; the learner does
        # not. Total query/step budgets remain identical, but layouts differ.
        if ts.n == 2 and action != 0:
            return Step(ts.seed, 7)
        return fake_step(ts, action)

    params = {
        "reset": fake_reset,
        "step": shortened_expert_episodes,
        "screen_nonadjacent": lambda ts: True,
        "feature_fn": fake_features,
        "teacher": fake_teacher,
        "learner": fake_learner,
        "seeds": SOURCE_LAYOUTS,
    }
    reset = collect_arm(arm="reset_expert", **params)
    correction = collect_arm(arm="learner_correction", **params)
    assert reset.complete and correction.complete
    assert reset.teacher_queries == correction.teacher_queries == QUERY_BUDGET
    assert reset.native_steps == correction.native_steps == QUERY_BUDGET
    assert (reset.summary()["source_layout_label_counts"]
            != correction.summary()["source_layout_label_counts"])
    with pytest.raises(ValueError, match="Mismatched source-layout"):
        validate_pair(reset, correction)

    # Forged paired-accounting cannot bypass the same check after save/reload.
    for arm in (reset, correction):
        save_numeric(str(tmp_path / f"seed17_{arm.arm}.npz"), arm)
    record = {
        "model_seed": 17, "numeric_source_sha256": "trusted-sha",
        "matched": True,
        "arms": {c.arm: c.summary() for c in (reset, correction)},
        "pair_accounting": {
            "matched": True,
            "labels_per_arm": QUERY_BUDGET,
            "native_steps_per_arm": QUERY_BUDGET,
            "heldout_teacher_queries": 0,
            "reset_arm": reset.summary(),
            "correction_arm": correction.summary(),
        },
    }
    with pytest.raises(ValueError, match="Mismatched source-layout"):
        load_verified_collection_pair(
            tmp_path, model_seed=17, source_checksum="trusted-sha",
            model_record=record,
        )


def test_no_cherry_pick_when_adjacent_or_teacher_unsupported():
    a = collect_arm(arm="learner_correction", reset=fake_reset, step=fake_step,
                    screen_nonadjacent=lambda ts: ts.seed != 9702,
                    feature_fn=fake_features, teacher=fake_teacher,
                    learner=fake_learner, seeds=(9701, 9702, 9708))
    assert not a.complete
    assert a.screened_seeds == (9701, 9702)
    assert a.expert_failures[0]["reason"] == "ineligible_adjacent_reset"

    def unsupported(ts):
        if ts.n == 2:
            raise ValueError("teacher recovery unavailable")
        return fake_teacher(ts)

    b = collect_arm(arm="learner_correction", reset=fake_reset, step=fake_step,
                    screen_nonadjacent=lambda ts: True,
                    feature_fn=fake_features, teacher=unsupported,
                    learner=fake_learner, seeds=(9701, 9702))
    assert not b.complete
    assert b.expert_failures[0]["reason"] == "unsupported_teacher"
    with pytest.raises(ValueError, match="Incomplete collection"):
        validate_pair(both()[0], b)


def test_rejects_bad_contract_and_out_of_scope_seed():
    common = {
        "reset": fake_reset, "step": fake_step,
        "screen_nonadjacent": lambda ts: True,
        "feature_fn": fake_features, "teacher": fake_teacher,
        "learner": fake_learner,
    }
    with pytest.raises(ValueError, match="Evaluation-layout contamination"):
        collect_arm(arm="learner_correction", seeds=(9810,), **common)
    with pytest.raises(ValueError, match="contract changed"):
        collect_arm(arm="learner_correction", budget=19, **common)
    with pytest.raises(ValueError, match="Unregistered"):
        collect_arm(arm="learner_correction", seeds=(9701, 9999), **common)
    with pytest.raises(ValueError, match="Unregistered"):
        collect_arm(arm="learner_correction", seeds=(9702, 9701), **common)
    invalid = collect_arm(arm="reset_expert", teacher=lambda ts: (6, "bad"),
        **{k: v for k, v in common.items() if k != "teacher"})
    assert invalid.teacher_queries == 1
    assert invalid.native_steps == 0
    assert invalid.expert_failures[0]["reason"] == "unsupported_teacher"
    unknown = collect_arm(arm="learner_correction",
        teacher=lambda ts: (1, "unregistered_stage"),
        **{k: v for k, v in common.items() if k != "teacher"})
    assert not unknown.complete
    assert unknown.teacher_queries == 1
    assert unknown.summary()["accepted_teacher_labels"] == 0
    assert unknown.summary()["rejected_teacher_queries"] == 1
    with pytest.raises(ValueError, match="115-feature"):
        collect_arm(arm="learner_correction", feature_fn=lambda ts, h: np.zeros(54), **{k: v for k, v in common.items() if k != "feature_fn"})


def test_pair_validator_rejects_corrupted_policy_history_bits():
    expert, correction = both()
    changed = list(correction.rows)
    row = dict(changed[1])
    obs = row["public_obs"].copy()
    obs[-28:] = 0
    row["public_obs"] = obs
    changed[1] = row
    with pytest.raises(ValueError, match="Encoded action history"):
        validate_pair(expert, replace(correction, rows=tuple(changed)))


def test_numeric_artifact_without_pickle(tmp_path):
    _, b = both()
    save_numeric(str(tmp_path / "corr.npz"), b)
    with np.load(tmp_path / "corr.npz", allow_pickle=False) as arrays:
        assert arrays["obs"].shape == (20, 115)
        assert arrays["prior_executed_actions"].shape == (20, 4)
        assert np.all(arrays["executed_action"] == 0)
        assert np.all(arrays["expert_label"] != arrays["executed_action"])
        assert arrays["expert_stage_code"].dtype == np.int8
        assert arrays["state_origin_code"].dtype == np.int8
        assert np.array_equal(arrays["expert_stage_code"], np.zeros(20, dtype=np.int8))
        assert arrays["state_origin_code"][0] == 0
        assert arrays["state_origin_code"][1] == 2
def test_persisted_pair_contract_rechecks_histories_stages_and_origin(tmp_path):
    expert, correction = both()
    for collection in (expert, correction):
        save_numeric(
            str(tmp_path / f"seed17_{collection.arm}.npz"), collection
        )
    record = {
        "model_seed": 17, "numeric_source_sha256": "trusted-sha",
        "matched": True, "pair_accounting": validate_pair(expert, correction),
        "arms": {c.arm: c.summary() for c in (expert, correction)},
    }
    result = load_verified_collection_pair(
        tmp_path, model_seed=17, source_checksum="trusted-sha",
        model_record=record,
    )
    assert all(result[arm]["obs"].shape == (20, 115) for arm in result)
    with pytest.raises(ValueError, match="provenance"):
        load_verified_collection_pair(
            tmp_path, model_seed=17, source_checksum="wrong",
            model_record=record,
        )
    # Preserve agreement between arm metadata and paired accounting so the
    # test isolates a corrupted histogram, not a stale duplicate summary.
    bad_summary = {
        **record["arms"]["learner_correction"],
        "expert_stage_label_counts": {"carrying_square": 20},
    }
    bad = {
        **record,
        "arms": {**record["arms"], "learner_correction": bad_summary},
        "pair_accounting": {
            **record["pair_accounting"], "correction_arm": bad_summary,
        },
    }
    with pytest.raises(ValueError, match="histograms"):
        load_verified_collection_pair(
            tmp_path, model_seed=17, source_checksum="trusted-sha",
            model_record=bad,
        )


def test_persisted_pair_rejects_corrupt_numeric_provenance(tmp_path):
    expert, correction = both()
    for collection in (expert, correction):
        save_numeric(
            str(tmp_path / f"seed17_{collection.arm}.npz"), collection
        )
    record = {
        "model_seed": 17, "numeric_source_sha256": "trusted-sha",
        "matched": True, "pair_accounting": validate_pair(expert, correction),
        "arms": {c.arm: c.summary() for c in (expert, correction)},
    }
    target = tmp_path / "seed17_learner_correction.npz"
    with np.load(target, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True)
                  for key in archive.files}
    for name, row, value, error in (
        ("expert_stage_code", 0, 9, "teacher-stage"),
        ("state_origin_code", 1, 1, "state-origin"),
        ("episode_step", 1, 39, "Noncontiguous"),
    ):
        modified = {key: np.array(arr, copy=True)
                    for key, arr in arrays.items()}
        modified[name][row] = value
        np.savez_compressed(target, **modified)
        with pytest.raises(ValueError, match=error):
            load_verified_collection_pair(
                tmp_path, model_seed=17, source_checksum="trusted-sha",
                model_record=record,
            )

def test_balanced_10_each_collects_actual_steps_and_saves_quota_boundaries(tmp_path):
    # These fake episodes last 12 actions; the cutoff at ten is preregistered.
    @dataclass(frozen=True)
    class LongStep:
        seed: int
        n: int
        reward: float = 0.0

        def last(self):
            return self.n >= 12

    def reset(seed):
        return LongStep(seed, 0)

    def step(ts, action):
        assert action in range(6)
        return LongStep(ts.seed, ts.n + 1)

    kwargs = {
        "reset": reset, "step": step,
        "screen_nonadjacent": lambda ts: True,
        "feature_fn": fake_features, "teacher": fake_teacher,
        "learner": fake_learner,
        "seeds": tuple(seed for seed, _ in BALANCED_LAYOUT_QUOTAS),
        "per_layout_quota": tuple(count for _, count in BALANCED_LAYOUT_QUOTAS),
    }
    reset_arm = collect_arm(arm="reset_expert", **kwargs)
    correction_arm = collect_arm(arm="learner_correction", **kwargs)
    assert reset_arm.complete and correction_arm.complete
    assert all(not row["done"] for row in reset_arm.rows[:10])
    assert reset_arm.summary()["source_layout_label_counts"] == {9701: 10, 9702: 10}
    assert correction_arm.summary()["source_layout_label_counts"] == {9701: 10, 9702: 10}
    evidence = validate_pair(reset_arm, correction_arm, per_layout_quota=(10, 10))
    assert evidence["fixed_source_layout_quotas"] == {"9701": 10, "9702": 10}
    for arm in (reset_arm, correction_arm):
        save_numeric(str(tmp_path / f"seed17_{arm.arm}.npz"), arm)
    record = {
        "model_seed": 17, "numeric_source_sha256": "trusted-sha",
        "matched": True, "pair_accounting": evidence,
        "arms": {arm.arm: arm.summary() for arm in (reset_arm, correction_arm)},
    }
    loaded = load_verified_collection_pair(
        tmp_path, model_seed=17, source_checksum="trusted-sha",
        model_record=record,
    )
    assert all(loaded[arm]["obs"].shape == (20, 115) for arm in loaded)


def test_quota_unmet_fails_closed_and_never_substitutes_source_layout():
    args = {
        "reset": fake_reset, "step": fake_step,
        "screen_nonadjacent": lambda ts: True,
        "feature_fn": fake_features, "teacher": fake_teacher,
        "learner": fake_learner,
        "seeds": tuple(seed for seed, _ in BALANCED_LAYOUT_QUOTAS),
        "per_layout_quota": (10, 10),
    }
    stopped = collect_arm(arm="reset_expert", **args)
    other = collect_arm(arm="learner_correction", **args)
    assert not stopped.complete
    assert stopped.native_steps == stopped.teacher_queries == 7
    assert stopped.screened_seeds == (9701,)
    assert stopped.expert_failures[0]["reason"] == "source_layout_quota_unmet"
    with pytest.raises(ValueError, match="Incomplete collection"):
        validate_pair(stopped, other, per_layout_quota=(10, 10))
    with pytest.raises(ValueError, match="Unregistered fixed"):
        collect_arm(
            arm="reset_expert", **{**args, "per_layout_quota": (11, 9)}
        )
