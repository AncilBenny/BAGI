"""Explicit paired trace comparison and unique-action fork accounting."""
from __future__ import annotations

KEYS = ("A_greedy", "A_live", "B_greedy", "B_live")


def compare_traces(point, reference):
    """Record cross-run divergence without confusing it with task failure."""
    result = {}
    for key in KEYS:
        live, old = point[key], reference[key]
        episodes, expected = live["episodes"], old["episodes"]
        if not episodes or len(episodes) != len(expected):
            raise ValueError("Unmatched monitor episodes")
        if any((a["seed"], a["domain"]) != (b["seed"], b["domain"])
               for a, b in zip(episodes, expected, strict=True)):
            raise ValueError("Unmatched monitor seeds")
        differences = [a["seed"] for a, b in zip(episodes, expected, strict=True) if a != b]
        result[key] = {
            "exact": live == old, "different_episodes": len(differences),
            "first_different_seed": differences[0] if differences else None,
            "success_delta": live["success"] - old["success"],
        }
    return result


def check_fork_counts(common, rehearsal, control):
    """Count a shared training prefix once, with equal branch lifetime budgets."""
    if (common, rehearsal, control) != (8192, 32768, 32768):
        raise ValueError("Unexpected fork budgets")
    return common + rehearsal + control
