"""Gate B1: empirical ambiguity ceiling for reactive vs observation-history inputs.

Full simulator state is only used to enumerate a representative shortest path
to each reachable state, NEVER supplied as a predictor feature. This is not
a trained model, and in-sample majority-outcome accuracy is NOT a test score.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from itertools import pairwise
from pathlib import Path

import yaml

from bagi.worlds import WorkshopEnv

EVENT_NAMES = ("switch_activated", "key_picked_up", "lock_opened", "goal_reached")


def encode(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def state_key(info: dict) -> tuple:
    """Privileged state identifier for dataset construction only."""
    return (tuple(int(x) for x in info["position"]),
            bool(info["has_key"]), bool(info["door_open"]))


def enumerate_histories(env: WorkshopEnv, max_states: int = 512) -> tuple[list[dict], dict]:
    """One shortest executable observation/action history per nonterminal state."""
    _, initial = env.reset(seed=0)
    known = {state_key(initial)}
    queue = deque([()])
    samples = []
    real_replay_steps = 0
    while queue:
        path = queue.popleft()
        for action in range(env.action_space.n):
            observation, _ = env.reset(seed=0)
            history = []
            for previous_action in path:
                old_observation = observation.tolist()
                observation, _, terminal, truncated, _ = env.step(previous_action)
                real_replay_steps += 1
                if terminal or truncated:
                    raise AssertionError("Shortest path unexpectedly terminated")
                history.append([old_observation, int(previous_action)])
            next_observation, reward, terminal, truncated, info = env.step(action)
            real_replay_steps += 1
            samples.append({
                "history": history,
                "observation": observation.tolist(),
                "action": int(action),
                "next_observation": next_observation.tolist(),
                "reward": float(reward),
                "events": [name for name in EVENT_NAMES if name in info["events"]],
            })
            identifier = state_key(info)
            if identifier not in known and not (terminal or truncated):
                known.add(identifier)
                if len(known) > max_states:
                    raise ValueError("Too many reachable states for configured bound")
                queue.append((*path, action))
    return samples, {
        "reachable_nonterminal_states": len(known),
        "state_action_samples": len(samples),
        "simulator_replay_steps": real_replay_steps,
        "history_selection": "one BFS-shortest executed path per reachable hidden state",
    }


def visible_input(sample: dict, history_length: int) -> str:
    if history_length < 0:
        raise ValueError("history_length must be nonnegative")
    # Previous observation/action pairs plus the current observation and
    # proposed action. The previous action is always what was actually taken.
    history = sample["history"][-history_length:] if history_length else []
    return encode([history, sample["observation"], sample["action"]])


def outcome(sample: dict, *, events_only: bool = False) -> str:
    parts = [sample["reward"], sample["events"]]
    if not events_only:
        parts.insert(0, sample["next_observation"])
    return encode(parts)


def ambiguity(samples: list[dict], history_length: int) -> dict:
    groups = defaultdict(list)
    for sample in samples:
        groups[visible_input(sample, history_length)].append(sample)
    impossible_errors = 0
    event_errors = 0
    conflicting_groups = 0
    conflicting_rows = 0
    event_conflicting_groups = 0
    for group in groups.values():
        frequencies = Counter(outcome(sample) for sample in group)
        events = Counter(outcome(sample, events_only=True) for sample in group)
        impossible_errors += len(group) - max(frequencies.values())
        event_errors += len(group) - max(events.values())
        if len(frequencies) > 1:
            conflicting_groups += 1
            conflicting_rows += len(group)
        if len(events) > 1:
            event_conflicting_groups += 1
    n = len(samples)
    return {
        "history_length": history_length,
        "samples": n,
        "unique_visible_input_groups": len(groups),
        "conflicting_outcome_groups": conflicting_groups,
        "conflicting_outcome_rows": conflicting_rows,
        "conflicting_event_groups": event_conflicting_groups,
        "minimal_empirical_whole_outcome_errors": impossible_errors,
        "minimal_empirical_event_errors": event_errors,
        "empirical_majority_outcome_ceiling": 1 - impossible_errors / n if n else None,
        "empirical_majority_event_ceiling": 1 - event_errors / n if n else None,
    }


def run(config_path: str) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    env = WorkshopEnv(**cfg["environment"])
    samples, meta = enumerate_histories(env, int(cfg["evaluation"]["max_states"]))
    lengths = [int(k) for k in cfg["evaluation"]["history_lengths"]]
    if lengths != sorted(set(lengths)) or not lengths or lengths[0] != 0:
        raise ValueError("History lengths must be distinct, sorted and begin at zero")
    rows = [ambiguity(samples, length) for length in lengths]
    # More observable history refines, never removes, information from a group.
    for old, new in pairwise(rows):
        if new["minimal_empirical_whole_outcome_errors"] > (
            old["minimal_empirical_whole_outcome_errors"]
        ):
            raise AssertionError("Longer observable histories worsened ambiguity ceiling")
    output = {
        "question": "Can a history of visible observations/actions disambiguate reachable Workshop one-step outcomes that a reactive observation cannot?",
        "evaluation": meta,
        "source_of_features": "Only executed visible observations and actions; simulator hidden flags enumerate reachable states but never form feature keys.",
        "results": rows,
        "limitations": [
            "These are in-sample empirical majority-label ceilings, NOT learned-model accuracy.",
            "One shortest-path history per hidden state under one fixed layout; trajectory distribution and history selection influence estimates.",
            "A longer history creates more distinct groups and can overfit; no held-out evaluation or transfer is claimed.",
            "This measures one-step ambiguity, not whether GRU training will solve planning.",
            "All simulator replay steps are diagnostic interactions, not training experience.",
        ],
    }
    path = Path(cfg["evaluation"]["output_dir"]) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    for row in rows:
        print(f"k={row['history_length']} ambiguous_groups="
              f"{row['conflicting_outcome_groups']} "
              f"minimal_errors={row['minimal_empirical_whole_outcome_errors']} "
              f"event_errors={row['minimal_empirical_event_errors']}")
    print(f"Saved information-matched memory diagnostic: {path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/history_ambiguity.yaml")
    run(parser.parse_args().config)
