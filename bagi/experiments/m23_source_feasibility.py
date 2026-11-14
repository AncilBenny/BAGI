"""M23 source TRAIN-only physical-mechanism feasibility; zero target actions/fits.

This measures genuine executed source actions and evaluator-only effects. It does
NOT train or score an M23 world model, open UnlockPickup FINAL, or claim transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from bagi.experiments.m23_study_contract import (
    MONITOR_LAYOUTS,
    TRAIN_LAYOUTS,
    validate_split_contract,
)

SOURCE_ENV_IDS = {
    "doorkey": "MiniGrid-DoorKey-6x6-v0",
    "fetch": "MiniGrid-Fetch-5x5-N2-v0",
    "putnear": "MiniGrid-PutNear-6x6-N2-v0",
}
HORIZON = 256
OUT = "runs/m23_compositional_transfer"
MIN_EFFECTIVE_EVENTS = 8  # TRAIN-only scientific feasibility, not a training quota


class PublicObjectExplorer:
    """Generic public source explorer with memory of OWN observable effects."""

    def __init__(self) -> None:
        self.public_carrying_evidence = False

    @staticmethod
    def _front_obj(observation: dict) -> int:
        from minigrid.core.constants import OBJECT_TO_IDX

        if set(observation) != {"image", "direction", "mission"}:
            raise ValueError("Explorer saw nonpublic observation fields")
        image = np.asarray(observation["image"])
        if image.shape != (7, 7, 3):
            raise ValueError("Unexpected public MiniGrid image shape")
        front = int(image[3, 5, 0])
        if not 0 <= front < len(OBJECT_TO_IDX):
            raise ValueError("Unexpected native object index")
        return front

    def choose(self, observation: dict, rng: np.random.Generator) -> int:
        from minigrid.core.constants import OBJECT_TO_IDX

        front = self._front_obj(observation)
        pickable = {OBJECT_TO_IDX["key"], OBJECT_TO_IDX["ball"], OBJECT_TO_IDX["box"]}
        if not self.public_carrying_evidence and front in pickable:
            return 3
        if (self.public_carrying_evidence
                and front in {OBJECT_TO_IDX["empty"], OBJECT_TO_IDX["floor"]}
                and rng.random() < 0.35):
            return 4
        if front == OBJECT_TO_IDX["wall"]:
            return int(rng.integers(2))
        return 2 if rng.random() < 0.72 else int(rng.integers(2))

    def update(self, before: dict, action: int, after: dict) -> None:
        from minigrid.core.constants import OBJECT_TO_IDX

        previous, following = self._front_obj(before), self._front_obj(after)
        pickable = {OBJECT_TO_IDX["key"], OBJECT_TO_IDX["ball"], OBJECT_TO_IDX["box"]}
        if action == 3 and previous in pickable and following != previous:
            self.public_carrying_evidence = True
        if action == 4 and following in pickable and previous not in pickable:
            self.public_carrying_evidence = False


def run_source_episode(task: str, seed: int, *, split: str = "train") -> tuple[dict, dict]:
    if split not in ("train", "monitor") or task not in SOURCE_ENV_IDS:
        raise ValueError("M23 source acquisition requires TRAIN/MONITOR source task")
    layouts = TRAIN_LAYOUTS[task] if split == "train" else MONITOR_LAYOUTS[task]
    if seed not in layouts:
        raise ValueError(f"M23 source acquisition requires registered {split.upper()} layout")

    import gymnasium as gym
    import minigrid  # noqa: F401

    from bagi.experiments.m14_public_open_door import TraversableOpenDoorController
    from bagi.experiments.m20_fetch_physics import PublicFetchTeacher

    rng = np.random.default_rng(8_800_000 + seed)
    controller = (
        TraversableOpenDoorController() if task == "doorkey"
        else PublicFetchTeacher() if task == "fetch" else PublicObjectExplorer()
    )
    env = gym.wrappers.TimeLimit(gym.make(SOURCE_ENV_IDS[task]), max_episode_steps=HORIZON)
    rows = {
        "observation_image": [], "observation_direction": [], "mission": [],
        "action": [], "next_image": [], "next_direction": [],
        "reward": [], "terminated": [], "truncated": [],
    }
    effects = Counter()
    actions = []
    total_reward = 0.0
    try:
        observation, _ = env.reset(seed=seed)
        for _ in range(HORIZON):
            if set(observation) != {"image", "direction", "mission"}:
                raise ValueError("Source observations must be public MiniGrid only")
            action = int(controller.choose(observation, rng))
            if action not in range(7):
                raise ValueError("Invalid native executed action")
            world = env.unwrapped  # evaluator ONLY, never passed to controller
            before_carrying = world.carrying
            front = world.grid.get(*world.front_pos)
            was_locked = bool(getattr(front, "is_locked", False))
            following, reward, terminated, truncated, _ = env.step(action)
            actions.append(action)
            total_reward += float(reward)
            if (action == 3 and before_carrying is None and front is not None
                    and world.carrying is front):
                effects[f"{front.type}_pickups"] += 1
            if (action == 4 and before_carrying is not None
                    and world.carrying is None
                    and world.grid.get(*world.front_pos) is before_carrying):
                effects[f"{before_carrying.type}_drops"] += 1
            if (action == 5 and front is not None and was_locked
                    and not front.is_locked and front.is_open):
                effects["effective_door_unlocks"] += 1
            rows["observation_image"].append(np.array(observation["image"], copy=True))
            rows["observation_direction"].append(int(observation["direction"]))
            rows["mission"].append(str(observation["mission"]))
            rows["action"].append(action)
            rows["next_image"].append(np.array(following["image"], copy=True))
            rows["next_direction"].append(int(following["direction"]))
            rows["reward"].append(float(reward))
            rows["terminated"].append(bool(terminated))
            rows["truncated"].append(bool(truncated))
            controller.update(observation, action, following)
            observation = following
            if terminated or truncated:
                break
    finally:
        env.close()
    if not actions or len(rows["action"]) != len(actions):
        raise ValueError("Missing source executed-action evidence")
    return {
        "seed": seed, "native_actions": len(actions),
        "action_histogram": [actions.count(i) for i in range(7)],
        "executed_action_sha256": hashlib.sha256(bytes(actions)).hexdigest(),
        "native_reward_sum": total_reward, "events": dict(effects),
    }, rows


def execute_source_feasibility(output: str) -> dict:
    from bagi.experiments.m1_goal_learning import admitted_source_commit

    validate_split_contract()
    source_sha = admitted_source_commit()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "summary.json").exists():
        raise FileExistsError("Refusing to overwrite M23 source evidence")
    started = time.perf_counter()
    report = {
        "study": "m23_source_feasibility_only", "source_commit": source_sha,
        "source_tasks": {}, "target_environment_actions": 0,
        "target_final_episodes": 0, "fresh_model_fits": 0,
        "optimizer_step_calls": 0,
    }
    for task in SOURCE_ENV_IDS:
        summaries = []
        dataset = {}
        for seed in TRAIN_LAYOUTS[task]:
            summary, observations = run_source_episode(task, seed)
            summaries.append(summary)
            for name, values in observations.items():
                dataset.setdefault(name, []).extend(values)
        total = sum(r["native_actions"] for r in summaries)
        histogram = [sum(r["action_histogram"][i] for r in summaries) for i in range(7)]
        if sum(histogram) != total:
            raise ValueError("Source native action histogram mismatch")
        effects = Counter()
        for row in summaries:
            effects.update(row["events"])
        public_path = root / f"{task}-public-train-only.npz"
        if public_path.exists():
            raise FileExistsError("Refusing to overwrite public TRAIN data")
        np.savez_compressed(public_path, **{
            key: np.asarray(values, dtype="U128" if key == "mission" else None)
            for key, values in dataset.items()
        })
        report["source_tasks"][task] = {
            "episodes": len(summaries),
            "native_actions": total,
            "action_histogram": histogram,
            "native_effects": dict(effects),
            "executed_action_hashes": [r["executed_action_sha256"] for r in summaries],
            "public_data_sha256": hashlib.sha256(public_path.read_bytes()).hexdigest(),
            "native_reward_sum": sum(r["native_reward_sum"] for r in summaries),
        }
    source = report["source_tasks"]
    report["actual_native_actions"] = sum(r["native_actions"] for r in source.values())
    report["source_mechanism_coverage_pass"] = bool(
        source["doorkey"]["native_effects"].get("effective_door_unlocks", 0)
        >= MIN_EFFECTIVE_EVENTS
        and (
            source["fetch"]["native_effects"].get("key_pickups", 0)
            + source["fetch"]["native_effects"].get("ball_pickups", 0)
        ) >= MIN_EFFECTIVE_EVENTS
        and source["putnear"]["native_effects"].get("box_pickups", 0)
        >= MIN_EFFECTIVE_EVENTS
    )
    report["elapsed_seconds"] = time.perf_counter() - started
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_sha": source_sha, "actual_native_actions": report["actual_native_actions"],
        "native_source_effects": {k: v["native_effects"] for k, v in source.items()},
        "source_mechanism_coverage_pass": report["source_mechanism_coverage_pass"],
        "fresh_model_fits": 0, "optimizer_step_calls": 0,
    }, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=f"{OUT}/summary.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("M23 source acquisition requires explicit --execute")
    output = Path(args.output)
    if output.name != "summary.json" or str(output.parent) != OUT:
        raise SystemExit("M23 output must be the registered isolated location")
    execute_source_feasibility(str(output.parent))


if __name__ == "__main__":
    main()
