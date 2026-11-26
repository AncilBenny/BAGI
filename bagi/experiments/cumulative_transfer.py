"""Gate A feasibility pilot: cross-domain native interactions, not confirmation.

The same recurrent learner is carried or reset across MiniGrid and CartPole.
This runner never consumes final held-out data. Pilot scores are exploratory.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401 - registers maintained native tasks
import numpy as np
import torch

from bagi.agents.general_brain import FactChange, GeneralBrain, Goal, Skill
from bagi.agents.recurrent_replay import Episode, RecurrentReplay
from bagi.runtime import AgentRuntime, StepFeedback
from bagi.workshop_runtime import source_revision

TASKS = ("MiniGrid-Empty-Random-6x6-v0", "CartPole-v1")
WIDTH, ACTIONS, HORIZON = 161, 7, 100
ARMS = ("fresh", "frozen", "carry", "head_reset")
GOAL = Goal("native_episode", frozenset({"episode_end"}))


class EndSensor:
    """Physical outcome remains evaluator-only; public end flag is not success."""

    def initial(self, observation):
        return frozenset()

    def change(self, before, executed_action, after, public_reward):
        return FactChange(frozenset({"episode_end"}) if after[-1] else frozenset())


def make_env(domain):
    if domain == TASKS[0]:
        return gym.make(domain, max_steps=HORIZON)
    if domain == TASKS[1]:
        return gym.make(domain, max_episode_steps=HORIZON)
    raise ValueError("Unregistered pilot domain")


def encode(domain, obs, action=None, reward=0.0, done=False):
    """Separate modality slots; public interface ID, previous action and reward."""
    x = np.zeros(WIDTH, dtype=np.float32)
    if domain == TASKS[0]:
        x[:147] = np.asarray(obs["image"], dtype=np.float32).reshape(147) / 10
        x[151] = 1
    elif domain == TASKS[1]:
        x[147:151] = np.tanh(np.asarray(obs, dtype=np.float32))
        x[152] = 1
    else:
        raise ValueError("Unknown observation interface")
    if action is not None:
        x[153 + action] = 1
    # Reward is logged in replay, not embedded here: 153..159 are action slots.
    x[-1] = float(done)
    return x


def run_episode(learner, domain, seed, *, learn, limit=HORIZON):
    if not 1 <= limit <= HORIZON:
        raise ValueError("Invalid native episode limit")
    env = make_env(domain)
    learner.begin_episode()
    actions, rewards, terminals = [], [], []
    physical = False
    try:
        raw, _ = env.reset(seed=seed)
        initial = encode(domain, raw)
        observations = [initial]
        legal = int(env.action_space.n)
        skill = Skill(
            "control", frozenset(), GOAL.required,
            lambda obs, facts: learner.act(obs, legal, epsilon=0.2 if learn else 0.0),
            max_actions=limit,
        )
        runtime = AgentRuntime(
            GeneralBrain((skill,), EndSensor(), max_episode_actions=limit),
            "gate-a-public-interface-v1",
        )

        def step(action):
            nonlocal physical
            if not env.action_space.contains(action):
                raise ValueError("Illegal native action")
            raw_after, reward, terminated, truncated, _ = env.step(action)
            actions.append(action)
            rewards.append(float(reward))
            done = bool(terminated or truncated)
            terminals.append(done)
            after = encode(domain, raw_after, action, reward, done)
            observations.append(after)
            if domain == TASKS[0]:
                native = env.unwrapped
                tile = native.grid.get(*native.agent_pos)
                physical = bool(terminated and tile is not None and tile.type == "goal")
            else:
                # Native pole failure is termination; success requires surviving horizon.
                physical = bool(truncated and not terminated and len(actions) == HORIZON)
            return StepFeedback(action, after, float(reward), bool(terminated), bool(truncated))

        result = runtime.run_episode(GOAL, initial, step, learn=learn)
        if tuple(actions) != result.actions or not actions:
            raise RuntimeError("Native action provenance mismatch")
        if learn:
            learner.remember(Episode(
                np.stack(observations), np.asarray(actions, dtype=np.int64),
                np.asarray(rewards, dtype=np.float32), np.asarray(terminals, dtype=np.bool_),
                domain, legal,
            ))
        return {
            "seed": seed, "domain": domain, "actions": len(actions),
            "action_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
            "reward": sum(rewards), "physical_success": physical,
            "native_end": terminals[-1], "learn": learn,
        }
    finally:
        env.close()
        learner.begin_episode()


def train_block(learner, domain, steps, seed_base, episode_offset, *, learn):
    rows = []
    remaining = steps
    while remaining:
        row = run_episode(
            learner, domain, seed_base + episode_offset + len(rows),
            learn=learn, limit=min(remaining, HORIZON),
        )
        rows.append(row)
        remaining -= row["actions"]
    before = learner.updates
    if learn:
        for _ in range(4):
            learner.update()
    if learner.updates - before != (4 if learn else 0):
        raise RuntimeError("Optimizer accounting mismatch")
    return rows


def evaluate(learner, domain, seeds):
    before = (learner.updates, learner.training_actions, learner.presentations, learner.rng.getstate())
    rows = [run_episode(learner, domain, seed, learn=False) for seed in seeds]
    after = (learner.updates, learner.training_actions, learner.presentations, learner.rng.getstate())
    if before != after:
        raise RuntimeError("Frozen evaluation changed learning state")
    return {
        "success": sum(row["physical_success"] for row in rows) / len(rows),
        "actions": sum(row["actions"] for row in rows), "episodes": rows,
    }


def run_pilot(output: Path, *, model_seeds=(41, 43, 47), block=256, blocks=16, eval_count=12):
    """Matched native budgets per arm; source A shared once per model seed."""
    if not model_seeds or len(set(model_seeds)) != len(model_seeds):
        raise ValueError("Require distinct model seeds")
    if min(block, blocks, eval_count) < 1 or block * blocks >= 100_000 or eval_count >= 100_000:
        raise ValueError("Pilot seed namespace would overlap; allocate a new manifest")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    start = time.monotonic()
    results = []
    traces = []
    for index, seed in enumerate(model_seeds):
        # Pilot-only namespace; never final held-out seeds. Distinct family/reset streams.
        base = 7_000_000 + index * 1_000_000
        eval_a = range(base + 400_000, base + 400_000 + eval_count)
        eval_b = range(base + 500_000, base + 500_000 + eval_count)
        source = RecurrentReplay(WIDTH, ACTIONS, seed=seed)
        offset = 0
        for _ in range(blocks):
            rows = train_block(source, TASKS[0], block, base, offset, learn=True)
            offset += len(rows)
            traces.extend(dict(row, model_seed=seed, arm="shared_A") for row in rows)
        a_before = evaluate(source, TASKS[0], eval_a)
        results.append({"seed": seed, "arm": "shared_A", "A": a_before,
                        "train_actions": source.training_actions, "updates": source.updates,
                        "replay_presentations": source.presentations})
        for arm in ARMS:
            learner = (RecurrentReplay(WIDTH, ACTIONS, seed=seed)
                       if arm == "fresh" else copy.deepcopy(source))
            if arm == "head_reset":
                learner.reset_head(seed + 10_000)
            offset = 0
            initial = (learner.training_actions, learner.updates, learner.presentations)
            curve = []
            for i in range(blocks + 1):
                curve.append({"B_actions": i * block, "B": evaluate(learner, TASKS[1], eval_b)})
                if i == blocks:
                    break
                rows = train_block(
                    learner, TASKS[1], block, base + 200_000, offset, learn=arm != "frozen"
                )
                offset += len(rows)
                traces.extend(dict(row, model_seed=seed, arm=arm) for row in rows)
            after = evaluate(learner, TASKS[0], eval_a)
            values = [point["B"]["success"] for point in curve]
            auc = sum((a + b) / 2 for a, b in zip(values, values[1:])) / blocks
            results.append({
                "seed": seed, "arm": arm, "B_curve": curve, "B_success_auc": auc,
                "A_before_shared": a_before["success"], "A_after": after,
                "A_change": after["success"] - a_before["success"],
                "B_exposure_actions": blocks * block,
                "B_training_actions": learner.training_actions - initial[0],
                "B_optimizer_updates": learner.updates - initial[1],
                "B_replay_presentations": learner.presentations - initial[2],
            })
    summary = {
        "kind": "gate_a_feasibility_pilot", "scientific_pass": False,
        "source": source_revision(), "versions": {p: version(p) for p in ("gymnasium", "minigrid", "torch", "numpy")},
        "config": {"tasks": TASKS, "horizon": HORIZON, "seeds": model_seeds,
                   "block": block, "blocks": blocks, "eval_count": eval_count},
        "actual_exposure_actions": sum(row["actions"] for row in traces),
        "actual_training_actions": sum(row["actions"] for row in traces if row["learn"]),
        "seconds": time.monotonic() - start, "results": results,
        "limitations": "Exploratory seeds; MiniGrid layouts can recur; no confirmatory inference.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "native_actions.jsonl").open("x") as handle:
        for row in traces:
            handle.write(json.dumps(row) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--eval-count", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(run_pilot(args.output, block=args.block, blocks=args.blocks,
                               eval_count=args.eval_count), indent=2))
