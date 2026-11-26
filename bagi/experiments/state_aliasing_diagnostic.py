"""Measure non-Markov aliasing in Workshop observations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

CAUSAL_EVENTS = {"switch_activated", "key_picked_up", "lock_opened", "goal_reached"}


def signature(row: dict) -> tuple:
    info = row.get("info", {})
    events = tuple(sorted(event for event in info.get("events", []) if event in CAUSAL_EVENTS))
    return (
        json.dumps(row["next_observation"], separators=(",", ":")),
        float(row["reward"]),
        events,
    )


def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    transitions = Path(config["transitions"])
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    with transitions.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (json.dumps(row["observation"], separators=(",", ":")), int(row["action"]))
            groups[key].append(row)

    repeated_groups = 0
    ambiguous_groups = 0
    repeated_transitions = 0
    ambiguous_transitions = 0
    hidden_state_ambiguous_groups = 0
    examples = []

    for rows in groups.values():
        if len(rows) < 2:
            continue
        repeated_groups += 1
        repeated_transitions += len(rows)
        outcomes = {signature(row) for row in rows}
        if len(outcomes) <= 1:
            continue
        ambiguous_groups += 1
        ambiguous_transitions += len(rows)

        hidden_states = {
            (
                bool(row.get("info", {}).get("has_key", False)),
                bool(row.get("info", {}).get("door_open", False)),
            )
            for row in rows
        }
        if len(hidden_states) > 1:
            hidden_state_ambiguous_groups += 1

        if len(examples) < 5:
            examples.append(
                {
                    "count": len(rows),
                    "outcomes": len(outcomes),
                    "hidden_states": sorted([list(value) for value in hidden_states]),
                }
            )

    result = {
        "transitions": sum(len(rows) for rows in groups.values()),
        "unique_observation_action_pairs": len(groups),
        "repeated_groups": repeated_groups,
        "ambiguous_groups": ambiguous_groups,
        "ambiguous_group_fraction": ambiguous_groups / max(1, repeated_groups),
        "repeated_transitions": repeated_transitions,
        "ambiguous_transitions": ambiguous_transitions,
        "ambiguous_transition_fraction": ambiguous_transitions / max(1, repeated_transitions),
        "hidden_state_ambiguous_groups": hidden_state_ambiguous_groups,
        "examples": examples,
    }

    print("Workshop observation aliasing diagnostic")
    print(f"transitions={result['transitions']}")
    print(f"repeated_obs_action_groups={repeated_groups}")
    print(
        f"ambiguous_groups={ambiguous_groups}/{repeated_groups} "
        f"({result['ambiguous_group_fraction']:.3f})"
    )
    print(
        f"ambiguous_repeated_transitions={ambiguous_transitions}/{repeated_transitions} "
        f"({result['ambiguous_transition_fraction']:.3f})"
    )
    print(f"ambiguities_with_different_hidden_state={hidden_state_ambiguous_groups}")

    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved diagnostic to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/state_aliasing_causal_v2.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
