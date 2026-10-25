"""G2-E2-P0: preregistered public-only Workshop DQN feasibility pilot.

This is an integration of the existing DQN, not a new learning method.
The evaluator may inspect info.events only after real actions; the actor cannot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from bagi.agents.dqn import DQNAgent, Transition
from bagi.worlds import WorkshopEnv

SEEDS = (17, 29)
LAYOUTS = ("default", "mirror_x")
EVAL_SEEDS = tuple(range(15000, 15010))
TRAIN_ACTIONS = 2048
EPISODE_CAP = 40
MAX_ACTIONS = 7296
ACTIONS = 4
OBS_SHAPE = (5, 5)
EVENTS = ("switch_activated", "key_picked_up", "lock_opened", "goal_reached")


def verify_registration() -> None:
    if (SEEDS != (17, 29) or LAYOUTS != ("default", "mirror_x")
            or EVAL_SEEDS != tuple(range(15000, 15010))
            or TRAIN_ACTIONS != 2048 or EPISODE_CAP != 40
            or MAX_ACTIONS != 7296 or ACTIONS != 4 or OBS_SHAPE != (5, 5)):
        raise ValueError("G2-E2-P0 preregistration changed")


def _public(observation: np.ndarray) -> np.ndarray:
    value = np.asarray(observation)
    if value.shape != OBS_SHAPE or not np.issubdtype(value.dtype, np.integer):
        raise ValueError("Workshop actor requires only native integer 5x5 public view")
    if np.any(value < 0) or np.any(value > 8):
        raise ValueError("Unregistered Workshop observation tile")
    return value.copy()


def _episode(env, initial, choose, cap: int, numeric: dict[str, list],
             metadata: tuple[int, int, int, int], after_step=None) -> dict:
    """Only choose(public) can select an action; info is evaluator-only."""
    if not 1 <= cap <= EPISODE_CAP or len(numeric["action"]) + cap > MAX_ACTIONS:
        raise ValueError("Invalid real-step episode cap or total native budget")
    observation = _public(initial)
    trace = hashlib.sha256()
    event_counts: Counter[str] = Counter()
    total_reward = 0.0
    terminated = truncated = False
    for step in range(cap):
        action = choose(observation.copy())
        if type(action) is not int or not 0 <= action < ACTIONS:
            raise ValueError("Actor produced an invalid action")
        next_view, reward, terminated, truncated, info = env.step(action)
        next_observation = _public(next_view)
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError("Nonfinite native reward")
        done = bool(terminated or truncated)
        if after_step is not None:
            after_step(observation.copy(), action, reward, next_observation.copy(), done)
        trace.update(observation.tobytes())
        trace.update(bytes([action]))
        trace.update(np.asarray([reward], dtype="<f8").tobytes())
        trace.update(next_observation.tobytes())
        for event in info.get("events", ()):
            if event in EVENTS:
                event_counts[event] += 1
        numeric["observation"].append(observation)
        numeric["next_observation"].append(next_observation)
        numeric["action"].append(action)
        numeric["reward"].append(reward)
        numeric["terminated"].append(int(terminated))
        numeric["truncated"].append(int(truncated))
        numeric["episode_index"].append(metadata[0])
        numeric["seed_index"].append(metadata[1])
        numeric["layout_index"].append(metadata[2])
        numeric["actor_index"].append(metadata[3])
        observation = next_observation
        total_reward += reward
        if done:
            break
    if terminated and total_reward != 1.0:
        raise ValueError("Native goal termination must carry reward one")
    return {
        "steps": step + 1,
        "reward": total_reward,
        "goal_success": bool(terminated and total_reward == 1.0),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "partial_final": not (terminated or truncated),
        "events": {name: event_counts[name] for name in EVENTS},
        "public_trajectory_sha256": trace.hexdigest(),
    }


def _arrays(numeric: dict[str, list], weights: dict[str, np.ndarray]) -> dict:
    result = {
        "observation": np.asarray(numeric["observation"], dtype=np.int8).reshape(-1, 5, 5),
        "next_observation": np.asarray(numeric["next_observation"], dtype=np.int8).reshape(-1, 5, 5),
        "action": np.asarray(numeric["action"], dtype=np.int8),
        "reward": np.asarray(numeric["reward"], dtype=np.float32),
        "terminated": np.asarray(numeric["terminated"], dtype=np.int8),
        "truncated": np.asarray(numeric["truncated"], dtype=np.int8),
        "episode_index": np.asarray(numeric["episode_index"], dtype=np.int32),
        "seed_index": np.asarray(numeric["seed_index"], dtype=np.int8),
        "layout_index": np.asarray(numeric["layout_index"], dtype=np.int8),
        "actor_index": np.asarray(numeric["actor_index"], dtype=np.int8),
        **weights,
    }
    if any(value.dtype.kind == "O" for value in result.values()):
        raise ValueError("Object/pickle arrays forbidden")
    return result


def _greedy_actor(agent):
    def choose(public):
        return agent.select_action(public, epsilon=0.0)
    return choose


def _random_actor(rng):
    def choose(_public):
        return int(rng.integers(ACTIONS))
    return choose


def run(output_dir: Path) -> dict:
    verify_registration()
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric: dict[str, list] = {name: [] for name in (
        "observation", "next_observation", "action", "reward", "terminated",
        "truncated", "episode_index", "seed_index", "layout_index", "actor_index"
    )}
    weights: dict[str, np.ndarray] = {}
    result = {
        "status": "running", "source_sha": os.environ.get("GITHUB_SHA", "unavailable"),
        "study": "G2-E2-P0", "preregistered_max_actions": MAX_ACTIONS,
        "versions": {"numpy": np.__version__, "torch": torch.__version__},
        "native_actions": 0, "teacher_queries": 0, "seed_results": {},
    }
    episode_index = 0
    try:
        torch.set_num_threads(1)
        for seed_index, seed in enumerate(SEEDS):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            world = WorkshopEnv(size=9, view_size=5, max_steps=EPISODE_CAP,
                                start_mode="seeded_start_chamber", layout_mode="default")
            agent = DQNAgent(observation_shape=OBS_SHAPE, actions=ACTIONS,
                             learning_rate=0.0003, gamma=0.99, device="cpu")
            trained = 0
            updates = 0
            training = []
            while trained < TRAIN_ACTIONS:
                reset_seed = 14000 + 100 * seed_index + len(training)
                initial, _ = world.reset(seed=reset_seed)

                def choose(public):
                    epsilon = 1.0 - 0.9 * trained / TRAIN_ACTIONS
                    return agent.select_action(public, epsilon)

                def update(before, action, reward, after, done):
                    nonlocal trained, updates
                    agent.buffer.add(Transition(before, action, reward, after, done))
                    if agent.learn(batch_size=64) is not None:
                        updates += 1
                    trained += 1
                    if trained % 128 == 0:
                        agent.sync_target()

                row = _episode(world, initial, choose, min(EPISODE_CAP, TRAIN_ACTIONS - trained),
                               numeric, (episode_index, seed_index, 0, 0), after_step=update)
                row["reset_seed"] = reset_seed
                training.append(row)
                result["native_actions"] += row["steps"]
                episode_index += 1
            if trained != TRAIN_ACTIONS or updates != TRAIN_ACTIONS - 63:
                raise ValueError("DQN real-action/update budget mismatch")
            for name, tensor in agent.online.state_dict().items():
                weights[f"seed{seed}_{name.replace('.', '_')}"] = (
                    tensor.detach().cpu().numpy().astype(np.float32).copy()
                )
            evaluation: dict[str, dict[str, list]] = {}
            for layout_index, layout in enumerate(LAYOUTS):
                evaluation[layout] = {}
                for actor_index, actor in enumerate(("learned_greedy", "uniform_random"), 1):
                    episodes = []
                    for index, reset_seed in enumerate(EVAL_SEEDS):
                        eval_world = WorkshopEnv(size=9, view_size=5, max_steps=EPISODE_CAP,
                                                 start_mode="seeded_start_chamber",
                                                 layout_mode=layout)
                        initial, _ = eval_world.reset(seed=reset_seed)
                        rng = np.random.default_rng(
                            25000 + 1000 * seed_index + 100 * layout_index + index
                        )
                        select = (_greedy_actor(agent) if actor == "learned_greedy"
                                  else _random_actor(rng))
                        row = _episode(eval_world, initial, select, EPISODE_CAP, numeric,
                                       (episode_index, seed_index, layout_index, actor_index))
                        row["reset_seed"] = reset_seed
                        episodes.append(row)
                        result["native_actions"] += row["steps"]
                        episode_index += 1
                    evaluation[layout][actor] = episodes
            def successes(layout, actor):
                return sum(int(row["goal_success"]) for row in evaluation[layout][actor])
            paired = [
                (int(a["goal_success"]), int(b["goal_success"]))
                for a, b in zip(evaluation["mirror_x"]["learned_greedy"],
                                evaluation["mirror_x"]["uniform_random"], strict=True)
            ]
            passed = (any(row["goal_success"] for row in training)
                      and successes("default", "learned_greedy") >= 3
                      and successes("mirror_x", "learned_greedy") >= 2
                      and successes("mirror_x", "learned_greedy")
                      - successes("mirror_x", "uniform_random") >= 2)
            result["seed_results"][str(seed)] = {
                "training": training, "training_actions": trained, "replay_updates": updates,
                "training_unique_public_trajectories": len({
                    r["public_trajectory_sha256"] for r in training
                }),
                "evaluation_actions_by_layout_actor": {
                    layout: {actor: sum(r["steps"] for r in episodes)
                             for actor, episodes in arms.items()}
                    for layout, arms in evaluation.items()
                },
                "evaluation": evaluation, "feasibility_pass": bool(passed),
                "mirror_paired_wins": sum(a > b for a, b in paired),
                "mirror_paired_losses": sum(a < b for a, b in paired),
                "mirror_paired_ties": sum(a == b for a, b in paired),
                "unique_public_trajectories": {
                    layout: {actor: len({r["public_trajectory_sha256"] for r in episodes})
                             for actor, episodes in arms.items()}
                    for layout, arms in evaluation.items()
                },
            }
            if result["native_actions"] > MAX_ACTIONS:
                raise ValueError("Exceeded preregistered native action budget")
        result["feasibility_pass"] = all(
            row["feasibility_pass"] for row in result["seed_results"].values()
        )
        result["status"] = "complete"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["native_actions"] = len(numeric["action"])
        np.savez_compressed(output_dir / "public_policy_numeric.npz", **_arrays(numeric, weights))
        (output_dir / "summary.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/g2_e2_p0_workshop_policy")
    args = parser.parse_args()
    result = run(Path(args.output_dir))
    print(json.dumps({"status": result["status"], "native_actions": result["native_actions"],
                      "feasibility_pass": result["feasibility_pass"]}))


if __name__ == "__main__":
    main()
