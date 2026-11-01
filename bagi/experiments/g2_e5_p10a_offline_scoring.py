"""P10a registered E1 TRAIN-only scoring, with no native interactions.

This module is intentionally never imported by a simulator. Model fitting is
only possible via the explicit --execute-registered-fits CLI flag after CI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bagi.experiments import g2_e5_p8_public_effect as p8
from bagi.experiments import g2_e5_p10a_terminal_selection as p10a

CONTRACTS = ("shifted", "tail_swap")
SUBSETS = ("all", "non_tail", "tail", "early_terminal", "action4_tail")


def _subsets(data: p10a.TerminalDataset, held: np.ndarray) -> dict[str, np.ndarray]:
    return {"all": held,
            "non_tail": held[~data.tail[held]],
            "tail": held[data.tail[held]],
            "early_terminal": held[data.early_terminal[held]],
            "action4_tail": held[data.tail[held] & (data.action[held] == 4)]
            if data.name == "e1_xland_train" else np.empty(0, dtype=np.int64)}


def _score(prob: np.ndarray, data: p10a.TerminalDataset,
           held: np.ndarray) -> dict:
    """Score exactly the same held source rows for both training contracts."""
    prob = np.asarray(prob, dtype=np.float64)
    if (prob.shape != (len(held),) or not np.isfinite(prob).all()
            or np.any(prob < 0) or np.any(prob > 1)
            or len(np.unique(held)) != len(held)
            or np.any(held < 0) or np.any(held >= len(data.action))):
        raise ValueError("Invalid held-fold probabilities or row identities")
    position = np.full(len(data.action), -1, dtype=np.int64)
    position[held] = np.arange(len(held))
    results = {}
    for name, rows in _subsets(data, held).items():
        if not len(rows):
            results[name] = {"n_rows": 0, "metrics": None, "group_brier": {}}
            continue
        p = prob[position[rows]]
        y = data.changed[rows].astype(np.float64)
        action = data.action[rows]
        group = data.source_group[rows]
        metrics = p8._metrics(p, y, action, group, data.n_actions)
        for a in range(data.n_actions):
            a_rows = action == a
            metrics["per_action"][str(a)]["positives"] = int(y[a_rows].sum())
            metrics["per_action"][str(a)]["negatives"] = int(a_rows.sum() - y[a_rows].sum())
        by_group = {str(int(g)): float(np.mean((p[group == g] - y[group == g]) ** 2))
                    for g in np.unique(group)}
        results[name] = {"n_rows": len(rows), "metrics": metrics,
                         "group_brier": by_group}
    return results


def _group_mean(rows: list[dict], dataset: str, contract: str,
                arm: str, subset: str) -> dict:
    """Each source group contributes ONCE per initializer, not once per row."""
    seed_means = {}
    seed_groups = {}
    for seed in p8.SEEDS:
        values = {}
        selected = [r for r in rows if (r["dataset"] == dataset
                    and r["contract"] == contract and r["arm"] == arm
                    and r["model_seed"] == seed)]
        if len(selected) != 2 or {r["fold"] for r in selected} != {0, 1}:
            raise ValueError("Missing P10a fold or initializer")
        for row in selected:
            for group, brier in row["subsets"][subset]["group_brier"].items():
                if group in values:
                    raise ValueError("Duplicate held source group")
                values[group] = brier
        if not values:
            return {"brier": None, "n_groups": 0, "seed_brier": {}}
        seed_means[str(seed)] = float(np.mean(list(values.values())))
        seed_groups[str(seed)] = set(values)
    if seed_groups[str(p8.SEEDS[0])] != seed_groups[str(p8.SEEDS[1])]:
        raise ValueError("Initializer source-group support differs")
    return {"brier": float(np.mean(list(seed_means.values()))),
            "n_groups": len(seed_groups[str(p8.SEEDS[0])]),
            "seed_brier": seed_means}


def _relative_worsening(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old <= 0:
        return None
    return new / old - 1.0


def _gate(group_summary: dict, pooled_ece: dict) -> dict:
    """Conservative preregistered STOP/PASS, never an AGI promotion gate."""
    def get(domain: str, contract: str, arm: str, subset: str) -> float | None:
        return group_summary[domain][contract][arm][subset]["brier"]

    xland, workshop = "e1_xland_train", "e1_workshop_train"
    checks = {
        "xland_tail_10pct": _relative_worsening(
            get(xland, "tail_swap", "reactive", "tail"),
            get(xland, "shifted", "reactive", "tail")),
        "xland_nontail_2pct": _relative_worsening(
            get(xland, "tail_swap", "reactive", "non_tail"),
            get(xland, "shifted", "reactive", "non_tail")),
        "workshop_all_2pct": _relative_worsening(
            get(workshop, "tail_swap", "reactive", "all"),
            get(workshop, "shifted", "reactive", "all")),
        "workshop_tail_2pct": _relative_worsening(
            get(workshop, "tail_swap", "reactive", "tail"),
            get(workshop, "shifted", "reactive", "tail")),
    }
    tests = [checks["xland_tail_10pct"] is not None
             and checks["xland_tail_10pct"] <= -0.10]
    tests += [checks[k] is not None and checks[k] <= 0.02
              for k in ("xland_nontail_2pct", "workshop_all_2pct", "workshop_tail_2pct")]
    for domain in (xland, workshop):
        old_ece = pooled_ece[domain]["shifted"]["reactive"]
        new_ece = pooled_ece[domain]["tail_swap"]["reactive"]
        tests.append(new_ece - old_ece <= 0.02)
        tests.append(get(domain, "tail_swap", "reactive", "all")
                     < get(domain, "tail_swap", "laplace_per_action", "all"))
    return {"decision": "PASS_OFFLINE_LABEL_COVERAGE_ONLY" if all(tests) else "STOP",
            "relative_brier_changes": checks, "checks": [bool(t) for t in tests]}


def score_archive(archive: dict, *, strict: bool = True,
                  fit_fn=None) -> dict:
    """One registered run. Tests inject a fake fitter; production uses P8."""
    if fit_fn is None:
        fit_fn = p8._fit
    p8.torch.set_num_threads(1)
    records = []
    all_predictions = {}
    steps = 0
    fits = 0
    for domain in ("xland", "workshop"):
        data = p10a._build_dataset(archive, domain, strict=strict)
        proxy = p8.PublicTransitions(data.name, data.observation, data.action,
                                     data.changed.astype(np.float32), data.episode,
                                     data.fold, data.n_actions, data.source_group)
        for fold in p8.FOLDS:
            selection = p10a._selection(data, fold)
            held = selection["held"]
            for seed in p8.SEEDS:
                for contract in CONTRACTS:
                    train = selection[contract]
                    for arm in (*p8.ARMS, *p8.BASELINES):
                        if arm in p8.ARMS:
                            prob, nsteps = fit_fn(proxy, train, held, arm, seed, fold)
                            expected = p8.EPOCHS * ((len(train) + p8.BATCH - 1) // p8.BATCH)
                            if nsteps != expected:
                                raise ValueError("P10a optimizer budget differs from P8")
                            steps += nsteps
                            fits += 1
                        else:
                            prob = p8._baseline(proxy, train, held, arm)
                        scores = _score(prob, data, held)
                        records.append({"dataset": data.name, "fold": fold,
                                        "model_seed": seed, "contract": contract,
                                        "arm": arm, "train_rows": len(train),
                                        "held_rows": len(held),
                                        "removed_source_index_sha256":
                                            selection["removed_source_index_sha256"],
                                        "subsets": scores})
                        all_predictions[(data.name, contract, arm, seed, fold)] = (
                            np.asarray(prob, dtype=np.float64), data.changed[held])
    if fits != 48 or len(records) != 96:
        raise ValueError("P10a registered fit/score count mismatch")
    group_summary = {}
    pooled_ece = {}
    for domain in ("e1_xland_train", "e1_workshop_train"):
        group_summary[domain] = {}
        pooled_ece[domain] = {}
        for contract in CONTRACTS:
            group_summary[domain][contract] = {}
            pooled_ece[domain][contract] = {}
            for arm in (*p8.ARMS, *p8.BASELINES):
                group_summary[domain][contract][arm] = {
                    subset: _group_mean(records, domain, contract, arm, subset)
                    for subset in SUBSETS}
                eces = []
                for seed in p8.SEEDS:
                    p = np.concatenate([all_predictions[(domain, contract, arm, seed, fold)][0]
                                        for fold in p8.FOLDS])
                    y = np.concatenate([all_predictions[(domain, contract, arm, seed, fold)][1]
                                        for fold in p8.FOLDS])
                    bins = np.minimum((p * 10).astype(np.int64), 9)
                    ece = sum(float(np.mean(bins == b)
                                    * abs(p[bins == b].mean() - y[bins == b].mean()))
                              for b in range(10) if np.any(bins == b))
                    eces.append(ece)
                pooled_ece[domain][contract][arm] = float(np.mean(eces))
    return {"study": "G2-E5-P10a", "status": "offline_train_only_scored",
            "registered_fits": 48, "completed_fits": fits,
            "supervised_optimizer_steps": steps, "native_actions": 0,
            "native_resets": 0, "teacher_queries": 0, "policy_updates": 0,
            "original_test_eval_array_reads": 0,
            "gate": _gate(group_summary, pooled_ece),
            "group_summary": group_summary, "pooled_ece": pooled_ece,
            "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-registered-fits", action="store_true")
    args = parser.parse_args()
    if not args.execute_registered_fits:
        raise SystemExit("STOP: P10a fits require --execute-registered-fits after green CI")
    from bagi.experiments.g2_e5_train_history_audit import _safe_archive
    result = score_archive(_safe_archive(args.e1_zip, "e1"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"study": result["study"], "decision": result["gate"]["decision"],
                      "fits": result["completed_fits"], "native_actions": 0}))


if __name__ == "__main__":
    main()
