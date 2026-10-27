"""G2-E6-P0B: frozen public-only Workshop DQN source acquisition.

Source-contract runner ONLY: no execution from import or CI tests. One bounded
native run may be authorized separately after CI. Not a belief learner.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from bagi.agents.dqn import DQNAgent, Transition
from bagi.experiments import g2_e5_p2_workshop_reachability_audit as p2
from bagi.worlds import WorkshopEnv

SEEDS = (17, 29)
SIZES = (9, 11)
LAYOUTS = ("default", "mirror_x")
ARMS = ("trained_greedy", "untrained_greedy", "uniform_random")
RESET_SEEDS = tuple(range(19000, 19012))
TRAIN_PER_SIZE = 2048
TRAIN_RESET_BASE = 20000  # prospective erratum: avoids diagnostic 19000..19011
TRAIN_PER_SEED = 4096
EPISODE_CAP = 40
MAX_DIAGNOSTIC = 11520
MAX_NATIVE = 19712
EXPECTED_UPDATES = 8066
OBS_SHAPE = (5, 5)
ACTIONS = 4
DQN_SOURCE_GIT_BLOB_SHA1 = "4c5fac098e53e9c81884d36e14b894679b40513d"
DQN_WEIGHT_SHAPES = {
    "net.0.weight": (128, 25), "net.0.bias": (128,),
    "net.2.weight": (128, 128), "net.2.bias": (128,),
    "net.4.weight": (4, 128), "net.4.bias": (4,),
}
EVENTS = ("switch_activated", "key_picked_up", "lock_opened", "goal_reached")
FIELDS = (
    "public_pre_action", "public_post_action", "executed_action", "native_reward",
    "terminated", "truncated", "episode_id", "step_in_episode", "reset_seed",
    "policy_seed", "size", "layout_index", "actor_index", "phase_index",
)


def verify_registration() -> None:
    if (
        SEEDS != (17, 29) or SIZES != (9, 11)
        or LAYOUTS != ("default", "mirror_x")
        or ARMS != ("trained_greedy", "untrained_greedy", "uniform_random")
        or RESET_SEEDS != tuple(range(19000, 19012))
        or TRAIN_PER_SIZE != 2048 or TRAIN_RESET_BASE != 20000
        or TRAIN_PER_SEED != 4096
        or EPISODE_CAP != 40 or MAX_DIAGNOSTIC != 11520
        or MAX_NATIVE != 19712 or EXPECTED_UPDATES != 8066
        or OBS_SHAPE != (5, 5) or ACTIONS != 4
    ):
        raise ValueError("G2-E6-P0B preregistration changed")
    possible_train_seeds = [
        set(range(TRAIN_RESET_BASE + 1000 * index,
                  TRAIN_RESET_BASE + 1000 * (index + 1)))
        for index in range(len(SEEDS))
    ]
    if (any(group.intersection(RESET_SEEDS) for group in possible_train_seeds)
            or possible_train_seeds[0].intersection(possible_train_seeds[1])):
        raise ValueError("TRAIN/diagnostic/cross-seed reset collision")
    p2.verify_source()
    path = Path(inspect.getfile(DQNAgent))
    raw = path.read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()
    if blob != DQN_SOURCE_GIT_BLOB_SHA1:
        raise ValueError("DQN source changed; refuse native execution")


def _public(view: np.ndarray) -> np.ndarray:
    value = np.asarray(view)
    if value.shape != OBS_SHAPE or value.dtype != np.int8:
        raise ValueError("Only native 5x5 int8 public observation is legal")
    if np.any(value < 0) or np.any(value > 8):
        raise ValueError("Illegal public tile value")
    return value.copy()


def _greedy(network, public: np.ndarray) -> int:
    with torch.no_grad():
        value = network(torch.as_tensor(_public(public)).unsqueeze(0))
    if value.shape != (1, ACTIONS) or not torch.isfinite(value).all():
        raise ValueError("Invalid frozen DQN output")
    return int(value.argmax(dim=1).item())


def _weights(network, seed: int, stage: str) -> dict[str, np.ndarray]:
    state = network.state_dict()
    if set(state) != set(DQN_WEIGHT_SHAPES):
        raise ValueError("Unexpected DQN weight names")
    result = {}
    for name, expected_shape in DQN_WEIGHT_SHAPES.items():
        value = state[name].detach().cpu().numpy().astype(np.float32, copy=True)
        if value.shape != expected_shape or not np.isfinite(value).all():
            raise ValueError("Invalid DQN weight shape or value")
        result[f"seed{seed}_{stage}_{name.replace('.', '_')}"] = value
    return result


def _weight_digest(weights: dict[str, np.ndarray], seed: int, stage: str) -> str:
    prefix = f"seed{seed}_{stage}_"
    digest = hashlib.sha256()
    names = sorted(name for name in weights if name.startswith(prefix))
    expected = {
        f"{prefix}{name.replace('.', '_')}": shape
        for name, shape in DQN_WEIGHT_SHAPES.items()
    }
    if set(names) != set(expected) or any(weights[name].shape != expected[name]
                                             for name in names):
        raise ValueError("Unexpected DQN weight names or shapes")
    for name in names:
        value = weights[name]
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Invalid DQN weight dtype or value")
        digest.update(name.encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _empty_numeric() -> dict[str, list]:
    return {key: [] for key in FIELDS}


def _arrays(numeric: dict[str, list], phase: int) -> dict[str, np.ndarray]:
    """Split TRAIN and diagnostic BEFORE writing, preventing E2-style mixed reads."""
    if set(numeric) != set(FIELDS) or phase not in (0, 1):
        raise ValueError("Unexpected numeric field or phase")
    n = len(numeric["executed_action"])
    if n > MAX_NATIVE or any(len(values) != n for values in numeric.values()):
        raise ValueError("Numeric row count/budget mismatch")
    if any(value not in (0, 1) for value in numeric["phase_index"]):
        raise ValueError("Unregistered trace phase")
    indices = [i for i, value in enumerate(numeric["phase_index"]) if value == phase]

    def selected(name):
        return [numeric[name][i] for i in indices]

    result = {
        "public_pre_action": np.asarray(selected("public_pre_action"), dtype=np.int8)
        .reshape(-1, *OBS_SHAPE),
        "public_post_action": np.asarray(selected("public_post_action"), dtype=np.int8)
        .reshape(-1, *OBS_SHAPE),
        "executed_action": np.asarray(selected("executed_action"), dtype=np.int8),
        "native_reward": np.asarray(selected("native_reward"), dtype=np.float32),
        "terminated": np.asarray(selected("terminated"), dtype=np.int8),
        "truncated": np.asarray(selected("truncated"), dtype=np.int8),
        "episode_id": np.asarray(selected("episode_id"), dtype=np.int32),
        "step_in_episode": np.asarray(selected("step_in_episode"), dtype=np.int8),
        "reset_seed": np.asarray(selected("reset_seed"), dtype=np.int32),
        "policy_seed": np.asarray(selected("policy_seed"), dtype=np.int8),
        "size": np.asarray(selected("size"), dtype=np.int8),
        "layout_index": np.asarray(selected("layout_index"), dtype=np.int8),
        "actor_index": np.asarray(selected("actor_index"), dtype=np.int8),
        "phase_index": np.asarray(selected("phase_index"), dtype=np.int8),
    }
    if any(value.dtype.kind == "O" or not np.isfinite(value).all()
           for value in result.values()):
        raise ValueError("Object or nonfinite numeric trace forbidden")
    return result


def _episode(env, initial, choose, cap: int, numeric: dict[str, list],
             meta: tuple[int, int, int, int, int, int, int], after_step=None) -> dict:
    """Meta: episode ID, reset seed, policy seed, size, layout, actor, phase.

    choose() and after_step() receive ONLY public views/action/reward/done.
    Evaluator-only info is summarized locally and never reaches the learner.
    """
    if not 1 <= cap <= EPISODE_CAP or len(numeric["executed_action"]) + cap > MAX_NATIVE:
        raise ValueError("Native action budget exceeded")
    episode_id, reset_seed, policy_seed, size, layout, actor, phase = meta
    if (policy_seed not in SEEDS or size not in SIZES or layout not in (0, 1)
            or actor not in range(4) or phase not in (0, 1)):
        raise ValueError("Invalid episode metadata")
    observation = _public(initial)
    trace = hashlib.sha256()
    event_counts: Counter[str] = Counter()
    total_reward = 0.0
    terminated = truncated = False
    for step in range(cap):
        action = choose(observation.copy())
        if type(action) is not int or not 0 <= action < ACTIONS:
            raise ValueError("Invalid actor action")
        next_view, reward, terminated, truncated, info = env.step(action)
        after = _public(next_view)
        reward = float(reward)
        if reward not in (0.0, 1.0):
            raise ValueError("Unexpected native sparse reward")
        done = bool(terminated or truncated)
        trace.update(observation.tobytes())
        trace.update(bytes((action,)))
        trace.update(np.asarray([reward], dtype="<f8").tobytes())
        trace.update(after.tobytes())
        for event in info.get("events", ()):
            if event in EVENTS:
                event_counts[event] += 1
        values = (observation, after, action, reward, int(terminated), int(truncated),
                  episode_id, step, reset_seed, policy_seed, size, layout, actor, phase)
        for key, value in zip(FIELDS, values, strict=True):
            numeric[key].append(value)
        # Log the actual native action BEFORE a possible optimizer exception.
        if after_step is not None:
            after_step(observation.copy(), action, reward, after.copy(), done)
        total_reward += reward
        observation = after
        if done:
            break
    if terminated and total_reward != 1.0:
        raise ValueError("Goal termination without native reward")
    return {
        "episode_id": episode_id, "reset_seed": reset_seed, "policy_seed": policy_seed,
        "size": size, "layout_index": layout, "actor_index": actor, "phase_index": phase,
        "steps": step + 1, "reward": total_reward,
        "goal_success": bool(terminated and total_reward == 1.0),
        "terminated": bool(terminated), "truncated": bool(truncated),
        "partial_final": not bool(terminated or truncated),
        "events": {name: event_counts[name] for name in EVENTS},
        "public_trajectory_sha256": trace.hexdigest(),
    }


def _train_seed(seed_index: int, numeric: dict[str, list],
                episode_id: int) -> tuple[dict, object, object, dict, int]:
    """Exactly 4,096 true actions; each seed's replay and RNG are isolated."""
    seed = SEEDS[seed_index]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = DQNAgent(observation_shape=OBS_SHAPE, actions=ACTIONS,
                     learning_rate=0.0003, gamma=0.99, device="cpu")
    untrained = copy.deepcopy(agent.online).eval()
    counts = {size: 0 for size in SIZES}
    training = []
    trained = updates = 0
    while trained < TRAIN_PER_SEED:
        size = min(SIZES, key=lambda value: (counts[value], value))
        cap = min(EPISODE_CAP, TRAIN_PER_SIZE - counts[size])
        if cap < 1:
            raise ValueError("Training size allocation exceeded")
        reset_seed = TRAIN_RESET_BASE + 1000 * seed_index + len(training)
        if len(training) >= 1000 or reset_seed in RESET_SEEDS:
            raise ValueError("TRAIN/diagnostic or cross-seed reset collision")
        world = WorkshopEnv(size=size, view_size=5, max_steps=EPISODE_CAP,
                            start_mode="seeded_start_chamber", layout_mode="default")
        try:
            initial, _ = world.reset(seed=reset_seed)

            def choose(public):
                epsilon = 1.0 - 0.9 * trained / TRAIN_PER_SEED
                return agent.select_action(public, epsilon)

            def update(before, action, reward, after, done):
                nonlocal trained, updates
                agent.buffer.add(Transition(before, action, reward, after, done))
                if agent.learn(batch_size=64) is not None:
                    updates += 1
                trained += 1
                counts[size] += 1
                if trained % 128 == 0:
                    agent.sync_target()

            row = _episode(world, initial, choose, cap, numeric,
                           (episode_id, reset_seed, seed, size, 0, 0, 0), update)
        finally:
            world.close()
        training.append(row)
        episode_id += 1
    if counts != {size: TRAIN_PER_SIZE for size in SIZES} or updates != TRAIN_PER_SEED - 63:
        raise ValueError("Frozen TRAIN actions/optimizer count mismatch")
    return ({"training": training, "train_actions_by_size": counts,
             "optimizer_updates": updates}, untrained,
            copy.deepcopy(agent.online).eval(),
            {**_weights(untrained, seed, "untrained"),
             **_weights(agent.online, seed, "trained")}, episode_id)


def _diagnose(seed_index: int, untrained, trained, numeric: dict[str, list],
              episode_id: int) -> tuple[dict, int]:
    seed = SEEDS[seed_index]
    evaluations = {}
    for size_index, size in enumerate(SIZES):
        evaluations[str(size)] = {}
        for layout_index, layout in enumerate(LAYOUTS):
            evaluations[str(size)][layout] = {}
            for actor_index, arm in enumerate(ARMS, 1):
                rows = []
                for index, reset_seed in enumerate(RESET_SEEDS):
                    world = WorkshopEnv(size=size, view_size=5, max_steps=EPISODE_CAP,
                                        start_mode="seeded_start_chamber", layout_mode=layout)
                    rng = np.random.default_rng(29000 + 10000 * seed_index
                                                + 1000 * size_index
                                                + 100 * layout_index + index)
                    try:
                        initial, _ = world.reset(seed=reset_seed)

                        def choose(public):
                            if arm == "uniform_random":
                                return int(rng.integers(ACTIONS))
                            return _greedy(trained if arm == "trained_greedy" else untrained,
                                           public)

                        row = _episode(world, initial, choose, EPISODE_CAP, numeric,
                                       (episode_id, reset_seed, seed, size, layout_index,
                                        actor_index, 1))
                    finally:
                        world.close()
                    rows.append(row)
                    episode_id += 1
                evaluations[str(size)][layout][arm] = rows
    return evaluations, episode_id


def run(output_dir: Path) -> dict:
    """One-shot native acquisition; NEVER called by tests or on import."""
    verify_registration()
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite existing acquisition")
    output_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    numeric = _empty_numeric()
    weights: dict[str, np.ndarray] = {}
    summary = {
        "study": "G2-E6-P0B", "status": "running",
        "source_sha": os.getenv("GITHUB_SHA", "unavailable"),
        "workshop_git_blob_sha1": p2.SOURCE_GIT_BLOB_SHA1,
        "dqn_git_blob_sha1": DQN_SOURCE_GIT_BLOB_SHA1,
        "versions": {"numpy": np.__version__, "torch": torch.__version__},
        "max_native_actions": MAX_NATIVE, "native_actions": 0,
        "native_train_actions": 0, "native_diagnostic_actions": 0,
        "optimizer_updates": 0, "teacher_queries": 0, "seed_results": {},
        "promotion": False, "size13_native_calls": 0,
        "native_resets": 0, "numeric_weight_sha256": {},
    }
    episode_id = 0
    try:
        for seed_index, seed in enumerate(SEEDS):
            training, untrained, trained, snapshot, episode_id = _train_seed(
                seed_index, numeric, episode_id
            )
            weights.update(snapshot)
            summary["numeric_weight_sha256"][str(seed)] = {
                stage: _weight_digest(snapshot, seed, stage)
                for stage in ("untrained", "trained")
            }
            evaluation, episode_id = _diagnose(
                seed_index, untrained, trained, numeric, episode_id
            )
            summary["seed_results"][str(seed)] = {**training, "diagnostic": evaluation}
            summary["optimizer_updates"] += training["optimizer_updates"]
        if summary["optimizer_updates"] != EXPECTED_UPDATES:
            raise ValueError("Optimizer update count mismatch")
        summary["native_train_actions"] = sum(
            value == 0 for value in numeric["phase_index"]
        )
        summary["native_diagnostic_actions"] = sum(
            value == 1 for value in numeric["phase_index"]
        )
        if (summary["native_train_actions"] != 8192
                or summary["native_diagnostic_actions"] > MAX_DIAGNOSTIC
                or len(numeric["executed_action"]) > MAX_NATIVE):
            raise ValueError("Native action ceiling or TRAIN allocation mismatch")
        summary["status"] = "complete"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary["native_actions"] = len(numeric["executed_action"])
        summary["native_resets"] = len(set(numeric["episode_id"]))
        summary["native_train_actions"] = sum(v == 0 for v in numeric["phase_index"])
        summary["native_diagnostic_actions"] = sum(v == 1 for v in numeric["phase_index"])
        files = {
            "train_public_numeric.npz": _arrays(numeric, 0),
            "diagnostic_public_numeric.npz": _arrays(numeric, 1),
            "frozen_dqn_weights.npz": weights,
        }
        summary["numeric_file_sha256"] = {}
        for name, arrays in files.items():
            np.savez_compressed(output_dir / name, **arrays)
            summary["numeric_file_sha256"][name] = hashlib.sha256(
                (output_dir / name).read_bytes()
            ).hexdigest()
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps({"status": result["status"], "native_actions": result["native_actions"],
                      "optimizer_updates": result["optimizer_updates"],
                      "promotion": result["promotion"]}))


if __name__ == "__main__":
    main()
