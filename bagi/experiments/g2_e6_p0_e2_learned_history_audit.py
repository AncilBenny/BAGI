"""Read-only G2-E6-P0 Workshop learned-DQN public-history source audit.

Accepts ONLY the hash-pinned E2 actor0-only TRAIN ZIP through the existing P7
consumer. No mixed E2 archive, EVAL rows, simulator, model, or teacher.
This is an identifiability diagnostic, not a heldout prediction score.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bagi.experiments.g2_e5_e2_train_isolation import load_isolated_train


def audit_train_arrays(data: dict[str, np.ndarray]) -> dict:
    """Summarize executed TRAIN transitions without fitting or hidden state."""
    required = {"observation", "next_observation", "action", "episode_index",
                "seed_index", "actor_index"}
    if set(data) != required:
        raise ValueError("Unexpected TRAIN fields")
    obs, nxt = data["observation"], data["next_observation"]
    act, ep = data["action"], data["episode_index"]
    seed, actor = data["seed_index"], data["actor_index"]
    n = len(act)
    if (not n or obs.shape != (n, 5, 5) or nxt.shape != obs.shape
            or any(x.shape != (n,) for x in (act, ep, seed, actor))
            or any(x.dtype.kind not in "iu" for x in data.values())
            or np.any(actor != 0) or np.any((act < 0) | (act > 3))
            or not np.isin(seed, (0, 1)).all()):
        raise ValueError("Invalid public TRAIN-only source")
    result = {"status": "source_identifiability_only", "policy": "learned_workshop_dqn",
              "new_native_actions": 0, "new_optimizer_steps": 0,
              "original_eval_arrays_read": 0, "model_fits": 0, "seeds": {}}
    for s in (0, 1):
        indices = np.flatnonzero(seed == s)
        if not len(indices):
            raise ValueError("Missing learned policy seed")
        if np.any(np.diff(indices) != 1):
            raise ValueError("Interleaved policy seed rows")
        episode = ep[indices]
        starts = np.r_[0, np.flatnonzero(episode[1:] != episode[:-1]) + 1]
        ends = np.r_[starts[1:], len(indices)]
        if len(set(map(int, episode[starts]))) != len(starts):
            raise ValueError("Repeated disjoint episode identifier")
        groups = defaultdict(list)
        positive = Counter()
        action_counts = Counter()
        changed_cells = 0
        for start, end in zip(starts, ends):
            for t in range(start, end):
                i = int(indices[t])
                if t + 1 < end and not np.array_equal(nxt[i], obs[indices[t + 1]]):
                    raise ValueError("Broken within-episode public chain")
                y = int(np.any(obs[i] != nxt[i]))
                positive[y] += 1
                action_counts[int(act[i])] += 1
                changed_cells += int(np.count_nonzero(obs[i] != nxt[i]))
                prior = []
                for lag in (1, 2):
                    prior.append((b"V" + obs[indices[t - lag]].tobytes())
                                 if t - lag >= start else b"P")
                key = (obs[i].tobytes(), int(act[i]))
                groups[key].append((nxt[i].tobytes(), y, int(episode[t]),
                                    tuple(prior)))
        repeated = [rows for rows in groups.values() if len(rows) > 1]
        ambiguous = [rows for rows in repeated if len({r[0] for r in rows}) > 1]
        binary = [rows for rows in repeated if len({r[1] for r in rows}) > 1]
        resolved = 0
        distinct = 0
        for rows in ambiguous:
            by_history = defaultdict(set)
            for future, _, _, hist in rows:
                by_history[hist].add(future)
            if len(by_history) > 1:
                distinct += 1
                if all(len(futures) == 1 for futures in by_history.values()):
                    resolved += 1
        result["seeds"][str(s)] = {
            "actions": len(indices), "episodes": len(starts),
            "public_change_positive": positive[1],
            "public_change_negative": positive[0],
            "changed_public_cells": changed_cells,
            "action_counts": dict(sorted(action_counts.items())),
            "unique_current_view_action_keys": len(groups),
            "repeated_keys": len(repeated),
            "repeated_rows": sum(map(len, repeated)),
            "ambiguous_next_view_keys": len(ambiguous),
            "ambiguous_next_view_rows": sum(map(len, ambiguous)),
            "ambiguous_binary_change_keys": len(binary),
            "ambiguous_binary_change_rows": sum(map(len, binary)),
            "cross_episode_ambiguous_keys": sum(
                len({r[2] for r in rows}) > 1 for rows in ambiguous),
            "ambiguous_keys_with_distinct_prior_public_views": distinct,
            "ambiguous_keys_resolved_by_prior_views_in_sample": resolved,
            "ambiguous_keys_still_ambiguous_after_prior_views":
                len(ambiguous) - resolved,
        }
    result["limitations"] = [
        "Both DQN seeds share one Workshop geometry: no layout transfer evidence.",
        "Prior-view resolution is in-sample key counting, not heldout prediction.",
        "Last action has a recorded next view but no next pre-action chain.",
        "Public changes are not hidden physical effects or native rewarded goals.",
        "Consumer requires exact isolated TRAIN ZIP bytes, not just its manifest.",
    ]
    return result


def run(source: Path) -> dict:
    # Never override the pinned production digest; P7 validates ZIP/manifest,
    # field whitelist, actor0-only partition, episode order, and full chain.
    return audit_train_arrays(load_isolated_train(source))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isolated-train-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing audit")
    result = run(args.isolated_train_zip)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "seeds": result["seeds"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
