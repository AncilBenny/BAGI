"""Trace accounting for native rehearsal at completed rollout boundaries."""
from __future__ import annotations

import hashlib


def next_reset_seed(rows, first, budget, domain, legal_actions):
    """Return the pending reset seed after validating executed episode fragments."""
    if not rows or budget <= 0 or legal_actions <= 0:
        raise ValueError("Empty or invalid rehearsal block")
    total = 0
    for offset, row in enumerate(rows):
        actions = row["executed_actions"]
        if (row["domain"] != domain or row["seed"] != first + offset
                or row["learn"] is not True or not 1 <= len(actions) <= 100
                or row["actions"] != len(actions)
                or any(type(a) is not int or not 0 <= a < legal_actions for a in actions)
                or row["action_sha256"] != hashlib.sha256(bytes(actions)).hexdigest()):
            raise ValueError("Invalid native rehearsal trace")
        total += len(actions)
    if total != budget:
        raise ValueError("Native rehearsal budget mismatch")
    return first + len(rows)
