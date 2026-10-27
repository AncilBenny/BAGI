"""Synthetic-only P10a scoring and anti-leakage gates: zero real fits."""
from dataclasses import replace

import numpy as np
import pytest

from bagi.experiments import g2_e5_p8_public_effect as p8
from bagi.experiments import g2_e5_p10a_offline_scoring as score
from bagi.experiments import g2_e5_p10a_terminal_selection as select


def _archive(domain):
    episodes = 4
    width = 115 if domain == "xland" else 25
    obs = np.zeros((episodes * 4, width), dtype=np.int8)
    actions = np.tile(np.array([0, 1, 2, 1], dtype=np.int8), episodes)
    groups = np.repeat(np.arange(episodes), 4)
    labels = np.zeros(len(actions), dtype=np.int8)
    for e in range(episodes):
        obs[e * 4 + 1, 0] = 1
        labels[e * 4] = 1
        labels[e * 4 + 1] = 1
        labels[e * 4 + 3] = e % 2
    ids = ([f"xland-{11900 + e}-17-hold-BC" for e in range(episodes)]
           if domain == "xland" else
           [f"workshop-default-uniform_random-{12100 + e}" for e in range(episodes)])
    p = domain + "_train_"
    return {p + "observations": obs, p + "actions": actions,
            p + "groups": groups, p + "labels": labels,
            p + "episode_ids": np.asarray(ids),
            p + "behaviors": np.asarray(["uniform_random"] * episodes)}


def test_held_subsets_and_exact_public_metrics():
    data = select._build_dataset(_archive("xland"), "xland", strict=False)
    held = np.flatnonzero(data.fold == 0)
    pred = np.full(len(held), 0.25)
    metrics = score._score(pred, data, held)
    assert metrics["all"]["n_rows"] == 8
    assert metrics["tail"]["n_rows"] == 2
    assert metrics["non_tail"]["n_rows"] == 6
    assert metrics["all"]["metrics"]["positive_rows"] == int(data.changed[held].sum())
    assert sum(x["positives"] + x["negatives"] for x in
               metrics["all"]["metrics"]["per_action"].values()) == 8
    assert len(metrics["all"]["group_brier"]) == 2
    with pytest.raises(ValueError, match="Invalid held"):
        score._score(np.array([np.nan] * len(held)), data, held)


def test_group_weighting_not_row_weighting_and_duplicate_guard():
    records = []
    for seed in p8.SEEDS:
        for fold in (0, 1):
            group = "100" if fold == 0 else "101"
            records.append({"dataset": "toy", "contract": "shifted", "arm": "reactive",
                            "model_seed": seed, "fold": fold,
                            "subsets": {"tail": {"group_brier": {group: float(fold)}}}})
    pooled = score._group_mean(records, "toy", "shifted", "reactive", "tail")
    assert pooled["brier"] == 0.5 and pooled["n_groups"] == 2
    records[1]["subsets"]["tail"]["group_brier"] = {"100": 1.0}
    with pytest.raises(ValueError, match="Duplicate held source"):
        score._group_mean(records, "toy", "shifted", "reactive", "tail")


def test_gate_boundaries_and_negative_controls():
    domains = ("e1_xland_train", "e1_workshop_train")
    summaries = {}
    eces = {}
    for domain in domains:
        summaries[domain] = {}
        eces[domain] = {}
        for contract in score.CONTRACTS:
            summaries[domain][contract] = {}
            eces[domain][contract] = {}
            for arm in (*p8.ARMS, *p8.BASELINES):
                summaries[domain][contract][arm] = {
                    subset: {"brier": 0.5 if arm == "laplace_per_action" else 0.2}
                    for subset in score.SUBSETS}
                eces[domain][contract][arm] = 0.1
    summaries[domains[0]]["tail_swap"]["reactive"]["tail"]["brier"] = 0.18
    assert score._gate(summaries, eces)["decision"] == "PASS_OFFLINE_LABEL_COVERAGE_ONLY"
    eces[domains[1]]["tail_swap"]["reactive"] = 0.121
    assert score._gate(summaries, eces)["decision"] == "STOP"
    eces[domains[1]]["tail_swap"]["reactive"] = 0.1
    summaries[domains[0]]["tail_swap"]["reactive"]["tail"]["brier"] = 0.181
    assert score._gate(summaries, eces)["decision"] == "STOP"
    assert score._relative_worsening(0.0, 0.0) is None


def test_registered_counts_without_real_fits():
    archive = {**_archive("xland"), **_archive("workshop")}
    calls = []

    def fake_fit(data, train, held, arm, seed, fold):
        calls.append((data.name, len(train), len(held), arm, seed, fold))
        nsteps = p8.EPOCHS * ((len(train) + p8.BATCH - 1) // p8.BATCH)
        return np.full(len(held), 0.25, dtype=np.float64), nsteps

    result = score.score_archive(archive, strict=False, fit_fn=fake_fit)
    assert result["completed_fits"] == 48
    assert len(result["records"]) == 96
    assert len(calls) == 48
    assert result["native_actions"] == result["policy_updates"] == 0
    assert result["gate"]["decision"] == "STOP"
    assert all(len(r["removed_source_index_sha256"]) == 64 for r in result["records"])


def test_wrong_optimizer_step_budget_rejected():
    archive = {**_archive("xland"), **_archive("workshop")}

    def wrong_fit(data, train, held, arm, seed, fold):
        return np.full(len(held), 0.5), 0

    with pytest.raises(ValueError, match="optimizer budget"):
        score.score_archive(archive, strict=False, fit_fn=wrong_fit)


def test_selection_is_label_independent():
    data = select._build_dataset(_archive("xland"), "xland", strict=False)
    changed = 1 - data.changed
    a = select._selection(data, 0)
    b = select._selection(replace(data, changed=changed), 0)
    assert np.array_equal(a["removed"], b["removed"])
