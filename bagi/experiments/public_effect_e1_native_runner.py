"""G2-E1: preregistered two-domain, public-only native effect-prediction gate.

Native actors and collectors are frozen. This driver only trains standard
logistic effect heads on TRAIN public transitions, scores heldout TEST, and
writes JSON/NPZ numeric evidence (never torch/pickle checkpoints).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

PREDICTOR_SEEDS = (17, 29)
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
BOOTSTRAP_RESAMPLES = 1000
TOTAL_MAX_NATIVE_ACTIONS = 6800
METHODS = ("actual_action", "shuffled_action", "action_frequency", "constant")


def _episode_group(domain: str, episode) -> str:
    """Group heldout results by original behavior and source-policy seed."""
    if domain == "workshop":
        if episode.layout not in ("default", "mirror_x") or episode.behavior not in (
            "uniform_random", "g1_public_script"
        ):
            raise ValueError("Unregistered Workshop group")
        return episode.behavior
    if domain != "xland":
        raise ValueError("Unknown effect-prediction domain")
    tokens = episode.episode_id.split("-")
    if (len(tokens) != 5 or tokens[0] != "xland"
            or tokens[1] != episode.layout.removeprefix("xland-")
            or tokens[2] not in ("17", "29")
            or tokens[3] not in ("hold", "near", "putdown")
            or tokens[4] != episode.behavior
            or episode.behavior not in ("frozen_expanded_bc", "uniform_random")):
        raise ValueError("Unregistered XLand source-seed/task/behavior group")
    return f"source_seed_{tokens[2]}:{episode.behavior}"


def _numeric_episodes(domain: str, split: str, audits: list, numeric: dict) -> None:
    """Preserve complete legal transition rows with numeric, non-object arrays."""
    from bagi.agents.public_effect_prediction import checked_episodes

    if not audits:
        raise ValueError("Empty preregistered native collection")
    episodes = [audit.episode for audit in audits]
    width, n_actions = (115, 6) if domain == "xland" else (25, 4)
    observations, actions, labels, groups = checked_episodes(
        episodes, width=width, n_actions=n_actions
    )
    prefix = f"{domain}_{split}"
    numeric[f"{prefix}_observations"] = observations
    numeric[f"{prefix}_actions"] = actions
    numeric[f"{prefix}_labels"] = labels.astype(np.int8)
    numeric[f"{prefix}_groups"] = groups
    numeric[f"{prefix}_episode_ids"] = np.asarray(
        [e.episode_id for e in episodes], dtype=np.str_
    )
    numeric[f"{prefix}_layouts"] = np.asarray(
        [e.layout for e in episodes], dtype=np.str_
    )
    numeric[f"{prefix}_behaviors"] = np.asarray(
        [e.behavior for e in episodes], dtype=np.str_
    )
    numeric[f"{prefix}_native_actions"] = np.asarray(
        [a.native_actions for a in audits], dtype=np.int32
    )
    numeric[f"{prefix}_hidden_without_public_change"] = np.asarray(
        [a.hidden_without_public_change for a in audits], dtype=np.int32
    )


def _evaluate_domain(
    domain: str, train_audits: list, test_audits: list,
    numeric: dict, *, epochs: int = EPOCHS,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    """Fit only TRAIN and score complete heldout episodes, with paired controls."""
    from bagi.agents.public_effect_prediction import (
        binary_metrics,
        checked_episodes,
        disjoint_layouts,
        duplicate_trajectory_groups,
        effect_predictions,
        episode_bootstrap_brier_difference,
        fit_public_effects,
    )

    width, n_actions = (115, 6) if domain == "xland" else (25, 4)
    train = [a.episode for a in train_audits]
    test = [a.episode for a in test_audits]
    disjoint_layouts(train, test)
    _numeric_episodes(domain, "train", train_audits, numeric)
    _numeric_episodes(domain, "test", test_audits, numeric)
    _, _, train_y, _ = checked_episodes(train, width=width, n_actions=n_actions)
    _, _, test_y, groups = checked_episodes(test, width=width, n_actions=n_actions)
    if not np.any(train_y == 0) or not np.any(train_y == 1):
        raise ValueError("TRAIN missing a public-change class; fail closed")

    episode_groups = [_episode_group(domain, e) for e in test]
    slices = {"all": np.ones(len(test_y), dtype=bool)}
    for name in sorted(set(episode_groups)):
        indices = np.asarray([i for i, group in enumerate(episode_groups)
                              if group == name], dtype=np.int64)
        slices[name] = np.isin(groups, indices)
    result = {
        "train_rows": int(len(train_y)), "test_rows": int(len(test_y)),
        "train_public_change_positives": int(train_y.sum()),
        "test_public_change_positives": int(test_y.sum()),
        "train_duplicates": duplicate_trajectory_groups(train),
        "test_duplicates": duplicate_trajectory_groups(test),
        "test_group_episodes": dict(sorted(Counter(episode_groups).items())),
        "models": {},
    }
    for seed in PREDICTOR_SEEDS:
        fit = fit_public_effects(
            train, width=width, n_actions=n_actions, seed=seed,
            epochs=epochs, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        )
        predictions = effect_predictions(fit, test, width=width, n_actions=n_actions)
        if set(predictions) != set(METHODS):
            raise ValueError("Missing preregistered effect baseline")
        for method in METHODS:
            numeric[f"{domain}_test_seed{seed}_{method}"] = np.asarray(
                predictions[method], dtype=np.float32
            )
        model_result = {
            "train_rows": fit.train_rows,
            "train_episodes": fit.train_episodes,
            "train_action_counts": list(fit.train_action_counts),
            "slices": {},
        }
        for name, mask in slices.items():
            if not np.any(mask) or len(np.unique(groups[mask])) < 2:
                raise ValueError("Insufficient heldout episodes for grouped bootstrap")
            y = test_y[mask]
            scored = {method: binary_metrics(y, predictions[method][mask])
                      for method in METHODS}
            comparisons = {
                baseline: episode_bootstrap_brier_difference(
                    y, predictions["actual_action"][mask],
                    predictions[baseline][mask], groups[mask],
                    seed=seed + (0 if baseline == "shuffled_action" else 100),
                    resamples=resamples,
                )
                for baseline in ("shuffled_action", "action_frequency", "constant")
            }
            model_result["slices"][name] = {
                "episodes": int(len(np.unique(groups[mask]))),
                "metrics": scored,
                "actual_minus_baseline_brier": comparisons,
            }
        result["models"][str(seed)] = model_result
    return result


def _safe_json(value):
    """Permit numeric metadata only, never silently stringify arbitrary objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported summary type: {type(value).__name__}")


def _write_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True,
                               default=_safe_json) + "\n", encoding="utf-8")


def run(source_dir: Path, old_teacher_dir: Path, diverse_dir: Path,
        output_dir: Path) -> dict:
    """One bounded preregistered real native experiment, no policy updates."""
    import torch

    from bagi.experiments.public_effect_workshop_native import collect_workshop_native
    from bagi.experiments.public_effect_xland_native import collect_xland_native

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    numeric_path = output_dir / "public_effect_numeric.npz"
    numeric: dict[str, np.ndarray] = {}
    summary = {
        "status": "started", "study": "G2-E1 public-only action-effect prediction",
        "registered_predictor_seeds": list(PREDICTOR_SEEDS),
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "bootstrap_episode_resamples": BOOTSTRAP_RESAMPLES,
        "max_native_actions": TOTAL_MAX_NATIVE_ACTIONS,
        "new_teacher_queries": 0,
        "new_behavior_policy_training_actions": 0,
        "new_native_counterfactual_calls": 0,
        "workshop_fixed_script_is_learned": False,
        "learned_cross_domain_weights_shared": False,
    }
    _write_summary(summary_path, summary)
    torch.set_num_threads(1)
    try:
        workshop_train, workshop_test, workshop_counts = collect_workshop_native()
        summary["workshop_collection"] = workshop_counts
        _numeric_episodes("workshop", "train", workshop_train, numeric)
        _numeric_episodes("workshop", "test", workshop_test, numeric)
        _write_summary(summary_path, summary)
        xland_train, xland_test, xland_counts = collect_xland_native(
            source_dir, old_teacher_dir, diverse_dir
        )
        summary["xland_collection"] = xland_counts
        _numeric_episodes("xland", "train", xland_train, numeric)
        _numeric_episodes("xland", "test", xland_test, numeric)
        _write_summary(summary_path, summary)
        actual_actions = int(workshop_counts["native_actions"]
                             + xland_counts["native_actions"])
        if actual_actions > TOTAL_MAX_NATIVE_ACTIONS:
            raise ValueError("Exceeded combined preregistered native action budget")
        summary["actual_native_actions"] = actual_actions
        summary["workshop_prediction"] = _evaluate_domain(
            "workshop", workshop_train, workshop_test, numeric
        )
        summary["xland_prediction"] = _evaluate_domain(
            "xland", xland_train, xland_test, numeric
        )
        summary["status"] = "complete"
        _write_summary(summary_path, summary)
        return summary
    except Exception as exc:
        summary["status"] = "failed"
        summary["failure_type"] = type(exc).__name__
        summary["failure_message"] = str(exc)[:500]
        _write_summary(summary_path, summary)
        raise
    finally:
        if numeric:
            np.savez_compressed(numeric_path, **numeric)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--old-teacher-dir", type=Path, required=True)
    parser.add_argument("--diverse-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.source_dir, args.old_teacher_dir, args.diverse_dir, args.output_dir)


if __name__ == "__main__":
    main()

