"""G2-E5-P5: descriptive TRAIN-only public transition aliasing audit.

No simulator, policy, oracle, learned model, or held-out TEST input. Exact
public signatures are a diagnostic, not evidence of a Markov representation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HISTORY_WIDTHS = (0, 4)
XLAND_HISTORY_BITS = 28


def _check(observation: np.ndarray, action: np.ndarray, next_observation: np.ndarray,
           episode: np.ndarray, n_actions: int) -> None:
    if (
        observation.ndim != 2 or observation.shape != next_observation.shape
        or len(observation) == 0 or action.shape != (len(observation),)
        or episode.shape != (len(observation),) or n_actions < 2
        or observation.dtype.kind not in "biuf" or next_observation.dtype.kind not in "biuf"
        or action.dtype.kind not in "iu" or episode.dtype.kind not in "iu"
        or not np.isfinite(observation).all() or not np.isfinite(next_observation).all()
        or np.any(action < 0) or np.any(action >= n_actions)
    ):
        raise ValueError("Invalid public TRAIN transitions")
    for ident in np.unique(episode):
        indices = np.flatnonzero(episode == ident)
        if indices[-1] - indices[0] + 1 != len(indices):
            raise ValueError("TRAIN episode rows are not contiguous")


def _shift_within_episodes(observation: np.ndarray, action: np.ndarray,
                           episode: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                           np.ndarray, np.ndarray]:
    """Infer next public view from the next executed row; discard episode tails."""
    valid = np.flatnonzero(episode[:-1] == episode[1:])
    if len(valid) == 0:
        raise ValueError("No within-episode public transitions")
    return observation[valid], action[valid], observation[valid + 1], episode[valid]


def audit_transitions(observation: np.ndarray, action: np.ndarray,
                      next_observation: np.ndarray, episode: np.ndarray,
                      *, n_actions: int, target_history_bits: int = 0) -> dict:
    """Report repeat support and conflicting next views, never infer causality.

    Keys use ONLY the public current observation, actual executed action and
    optionally the four earlier executed commands. The next public view is a
    diagnostic label, never a policy input. No evaluator/private state.
    """
    obs = np.asarray(observation)
    act = np.asarray(action)
    nxt = np.asarray(next_observation)
    group = np.asarray(episode)
    _check(obs, act, nxt, group, n_actions)
    if (type(target_history_bits) is not int or target_history_bits < 0
            or target_history_bits >= obs.shape[1]):
        raise ValueError("Invalid excluded executed-history suffix")
    outcomes = nxt[:, :nxt.shape[1] - target_history_bits] if target_history_bits else nxt
    histories = np.full((len(act), max(HISTORY_WIDTHS)), -1, dtype=np.int16)
    for ident in np.unique(group):
        indices = np.flatnonzero(group == ident)
        for t, index in enumerate(indices):
            for lag in range(1, min(max(HISTORY_WIDTHS), t) + 1):
                histories[index, lag - 1] = int(act[indices[t - lag]])
    result = {"train_rows": len(act), "train_episodes": len(np.unique(group)),
              "next_public_history_bits_excluded": target_history_bits, "variants": {}}
    for width in HISTORY_WIDTHS:
        keys: dict[bytes, list[int]] = defaultdict(list)
        for i in range(len(act)):
            key = (np.ascontiguousarray(obs[i]).tobytes()
                   + int(act[i]).to_bytes(2, "little")
                   + histories[i, :width].tobytes())
            keys[key].append(i)
        repeated = [rows for rows in keys.values() if len(rows) > 1]
        ambiguous = [rows for rows in repeated if len({
            np.ascontiguousarray(outcomes[i]).tobytes() for i in rows
        }) > 1]
        cross_episode = [rows for rows in ambiguous if len(set(group[rows])) > 1]
        result["variants"][f"executed_history_{width}"] = {
            "unique_public_action_keys": len(keys),
            "repeated_key_rows": sum(map(len, repeated)),
            "ambiguous_keys": len(ambiguous),
            "ambiguous_key_rows": sum(map(len, ambiguous)),
            "cross_episode_ambiguous_keys": len(cross_episode),
            "cross_episode_ambiguous_rows": sum(map(len, cross_episode)),
        }
    return result


def run(e1: Path, e2: Path) -> dict:
    """Read only hash-verified archived TRAIN arrays through the existing gate."""
    from bagi.experiments.g2_e5_train_history_audit import _safe_archive

    first = _safe_archive(e1, "e1")
    second = _safe_archive(e2, "e2")
    datasets = {}
    for domain, n_actions in (("xland", 6), ("workshop", 4)):
        prefix = f"{domain}_train_"
        obs, action, group = (first[prefix + name]
                              for name in ("observations", "actions", "groups"))
        current, executed, after, episodes = _shift_within_episodes(obs, action, group)
        datasets[f"e1_{domain}_train"] = audit_transitions(
            current, executed, after, episodes, n_actions=n_actions,
            target_history_bits=XLAND_HISTORY_BITS if domain == "xland" else 0)
    for seed_index, seed in enumerate((17, 29)):
        mask = ((second["seed_index"] == seed_index)
                & (second["actor_index"] == 0))
        datasets[f"e2_workshop_dqn_train_seed{seed}"] = audit_transitions(
            second["observation"][mask].reshape(-1, 25),
            second["action"][mask],
            second["next_observation"][mask].reshape(-1, 25),
            second["episode_index"][mask], n_actions=4)
    return {"study": "G2-E5-P5", "status": "descriptive_train_only",
            "native_actions": 0, "native_resets": 0, "teacher_queries": 0,
            "model_updates": 0, "test_reads": 0, "datasets": datasets,
            "limitation": "Observed ambiguity requires repeated public-action keys; "
                          "absence of conflicts cannot establish full observability, "
                          "learned beliefs, policy improvement or transfer."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-zip", type=Path, required=True)
    parser.add_argument("--e2-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.e1_zip, args.e2_zip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"study": result["study"], "status": result["status"],
                      "native_actions": 0, "test_reads": 0}))


if __name__ == "__main__":
    main()
