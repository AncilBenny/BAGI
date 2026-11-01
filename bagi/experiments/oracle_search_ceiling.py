"""Oracle diagnostic for Workshop v0.2 random-shooting search."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

STAGES = ("switch", "door", "key", "lock", "goal")
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass(frozen=True)
class State:
    row: int
    col: int
    has_key: bool = False
    door_open: bool = False


def geometry(size: int) -> dict:
    return {
        "switch": (1, 3),
        "door": (3, 1),
        "key": (5, 1),
        "lock": (5, 4),
        "goal": (size - 2, size - 2),
        "walls": (
            {(3, col) for col in range(1, size - 1) if col != 1}
            | {(row, 4) for row in range(3, size - 1) if row != 5}
        ),
    }


def transition(state: State, action: int, geom: dict, size: int) -> tuple[State, set[str]]:
    dr, dc = DIRECTIONS[action]
    candidate = (state.row + dr, state.col + dc)
    row, col = candidate
    can_enter = 0 < row < size - 1 and 0 < col < size - 1
    can_enter &= candidate not in geom["walls"]
    can_enter &= candidate != geom["door"] or state.door_open
    can_enter &= candidate != geom["lock"] or state.has_key
    pos = candidate if can_enter else (state.row, state.col)

    door_open = state.door_open
    has_key = state.has_key
    events: set[str] = set()
    if pos == geom["switch"] and not door_open:
        door_open = True
        events.add("switch")
    if pos == geom["key"] and not has_key:
        has_key = True
        events.add("key")
    if pos == geom["lock"] and has_key:
        events.add("lock")
    if pos == geom["goal"]:
        events.add("goal")
    return State(pos[0], pos[1], has_key, door_open), events


def select_action(
    state: State,
    geom: dict,
    size: int,
    horizon: int,
    candidates: int,
    gamma: float,
    event_weights: dict[str, float],
    generator: torch.Generator,
) -> int:
    sequences = torch.randint(0, len(DIRECTIONS), (candidates, horizon), generator=generator)
    sequences[: len(DIRECTIONS), 0] = torch.arange(len(DIRECTIONS))
    states = [state] * candidates
    scores = torch.zeros(candidates, dtype=torch.float32)
    discount = 1.0
    for step in range(horizon):
        next_states = []
        values = []
        for index in range(candidates):
            next_state, events = transition(
                states[index], int(sequences[index, step]), geom, size
            )
            next_states.append(next_state)
            values.append(sum(event_weights[event] for event in events))
        states = next_states
        scores += discount * torch.tensor(values, dtype=torch.float32)
        discount *= gamma
    return int(sequences[int(torch.argmax(scores).item()), 0].item())


def evaluate(config: dict, candidates: int, seed: int) -> dict[str, float]:
    env = config["environment"]
    plan = config["planning"]
    size = int(env["size"])
    episodes = int(config["episodes"])
    geom = geometry(size)
    generator = torch.Generator().manual_seed(seed)
    reached_counts = {stage: 0 for stage in STAGES}
    event_weights = dict(zip(STAGES[:1] + STAGES[2:], plan["event_weights"], strict=True))

    for _ in range(episodes):
        state = State(1, 1)
        reached = {stage: False for stage in STAGES}
        for _step in range(int(env["max_steps"])):
            action = select_action(
                state,
                geom,
                size,
                int(plan["horizon"]),
                candidates,
                float(plan["gamma"]),
                event_weights,
                generator,
            )
            state, events = transition(state, action, geom, size)
            pos = (state.row, state.col)
            reached["switch"] |= "switch" in events
            reached["door"] |= pos == geom["door"]
            reached["key"] |= "key" in events
            reached["lock"] |= "lock" in events
            reached["goal"] |= "goal" in events
            if reached["goal"]:
                break
        for stage in STAGES:
            reached_counts[stage] += int(reached[stage])
    return {stage: reached_counts[stage] / episodes for stage in STAGES}


def run(config_path: str) -> None:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    rows = []
    for candidates in [int(value) for value in config["candidate_counts"]]:
        per_seed = [evaluate(config, candidates, int(seed)) for seed in config["planner_seeds"]]
        means = {
            stage: sum(row[stage] for row in per_seed) / len(per_seed) for stage in STAGES
        }
        rows.append({"candidates": candidates, **means})
        print(
            f"candidates={candidates:4d} "
            + " ".join(f"{stage}={means[stage]:.3f}" for stage in STAGES)
        )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "question": "Does random-shooting MPC succeed when imagination is exact?",
                "results": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/oracle_search_ceiling.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
