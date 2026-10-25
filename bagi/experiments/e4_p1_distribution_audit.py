"""Read-only diagnosis of saved G2-E4-P1 PPO action probabilities.

No MiniGrid import, simulator actions, new training, pickle or stochastic
rollout. Statistics refer only to the public observations visited by each
*already completed deterministic* evaluation trajectory. They do not predict
what would happen along genuinely stochastic trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np

ARCHIVE_SHA256 = "b3cecce2f9cb15092490f893ba8b0df16a9424a6d1b010747b07094cdccfa712"
TASKS = ("MiniGrid-Empty-Random-6x6-v0", "MiniGrid-DoorKey-5x5-v0")
SEEDS = (17, 29)
ARMS = ("untrained", "trained")
ACTION_NAMES = ("left", "right", "forward", "pickup", "drop", "toggle", "done")


def policy_probabilities(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Pure NumPy inference for SB3 default two-layer Tanh actor MLP."""
    obs = np.asarray(arrays["public_pre_action"], dtype=np.float64)
    if obs.ndim != 2 or not np.isfinite(obs).all():
        raise ValueError("Expected finite flat PUBLIC observations")

    def weights(name: str) -> np.ndarray:
        tensor = np.asarray(arrays["policy__" + name.replace(".", "__")])
        if tensor.dtype.kind not in "biuf" or not np.isfinite(tensor).all():
            raise ValueError("Unsafe or nonfinite policy weights")
        return tensor.astype(np.float64, copy=False)

    hidden1 = np.tanh(
        obs @ weights("mlp_extractor.policy_net.0.weight").T
        + weights("mlp_extractor.policy_net.0.bias")
    )
    hidden2 = np.tanh(
        hidden1 @ weights("mlp_extractor.policy_net.2.weight").T
        + weights("mlp_extractor.policy_net.2.bias")
    )
    logits = hidden2 @ weights("action_net.weight").T + weights("action_net.bias")
    if logits.shape != (len(obs), len(ACTION_NAMES)) or not np.isfinite(logits).all():
        raise ValueError("Unexpected actor output or nonfinite logits")
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def summarize_probabilities(arrays: dict[str, np.ndarray]) -> dict:
    probs = policy_probabilities(arrays)
    actual = np.asarray(arrays["actions"], dtype=np.int64)
    if len(actual) != len(probs) or len(actual) == 0:
        raise ValueError("Missing or mismatched action records")
    deterministic = np.argmax(probs, axis=1)
    agreement = int(np.count_nonzero(deterministic == actual))
    if agreement != len(actual):
        raise ValueError("Saved PPO action records did not match saved policy logits")
    top = np.max(probs, axis=1)
    second = np.partition(probs, -2, axis=1)[:, -2]
    entropy = -np.sum(probs * np.log(probs), axis=1)
    return {
        "recorded_deterministic_actions": len(actual),
        "policy_action_reproduction_matches": agreement,
        "action_histogram": np.bincount(actual, minlength=len(ACTION_NAMES)).tolist(),
        "mean_action_probability": [round(float(x), 6) for x in probs.mean(axis=0)],
        "mean_top_action_probability": round(float(top.mean()), 6),
        "median_top_action_probability": round(float(np.median(top)), 6),
        "median_top_two_probability_gap": round(float(np.median(top - second)), 6),
        "mean_normalized_entropy": round(float(entropy.mean() / math.log(7)), 6),
        "number_of_distinct_public_views_visited": len(
            {row.tobytes() for row in arrays["public_pre_action"]}
        ),
    }


def audit(archive: Path) -> dict:
    observed_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    if observed_sha != ARCHIVE_SHA256:
        raise ValueError("Frozen native ZIP SHA-256 is different from registered archive")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(TASKS) * len(SEEDS) * 3 + 1:
            raise ValueError("Unexpected number of files in frozen artifact")
        if sum(item.file_size for item in z.infolist()) > 30_000_000:
            raise ValueError("Oversized archive")
        if "summary.json" not in names:
            raise ValueError("Missing summary")
        summary = json.loads(z.read("summary.json"))
        if summary.get("status") != "complete" or summary.get("study") != "G2-E4-P1":
            raise ValueError("Wrong or incomplete research study")
        observed: list[dict] = []
        total = 0
        for row in summary["results"]:
            task, seed = row["task"], row["model_seed"]
            if task not in TASKS or seed not in SEEDS:
                raise ValueError("Unexpected task or model seed")
            for arm in ARMS:
                entry = next(
                    item for item in summary["trace_artifacts"]
                    if item["task"] == task and item["model_seed"] == seed and item["arm"] == arm
                )
                name = entry["file"]
                if name not in names:
                    raise ValueError("Missing archived policy evidence")
                data = z.read(name)
                if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                    raise ValueError("Archived numeric NPZ checksum changed")
                with np.load(io.BytesIO(data), allow_pickle=False) as stored:
                    arrays = {key: stored[key].copy() for key in stored.files}
                if any(value.dtype.kind not in "biuf" for value in arrays.values()):
                    raise ValueError("Refusing nonnumeric or pickle-like evidence")
                stats = summarize_probabilities(arrays)
                observed.append({"task": task, "model_seed": seed, "arm": arm, **stats})
                total += stats["recorded_deterministic_actions"]
        if len(observed) != 8 or total != 15_760:
            raise ValueError("Incomplete PPO deterministic evaluation records")
        return {
            "study": "G2-E4-P1 deterministic-policy probability postmortem",
            "source_archive_sha256": observed_sha,
            "source_native_run": 40212708157,
            "source_artifact_id": 12606845219,
            "total_saved_model_actions_exactly_reproduced": total,
            "action_names": list(ACTION_NAMES),
            "records": observed,
            "limitations": [
                "Offline diagnostic on TEST observations already visited by greedy "
                "policies; never used for training or hyperparameter selection.",
                "High probability entropy at these observations does NOT demonstrate "
                "stochastic policy success on different, unobserved trajectories.",
                "Raw TRAIN transitions were not preserved; training successes come "
                "from the independently audited JSON/source ledger.",
                "No simulator actions, sampling rollouts, new model training, or teacher "
                "calls occur in this analysis.",
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Reproduced", result["total_saved_model_actions_exactly_reproduced"],
          "frozen policy actions")


if __name__ == "__main__":
    main()
