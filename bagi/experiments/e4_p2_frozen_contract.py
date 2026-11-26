"""Read-only, fail-closed G2-E4-P2 numeric-weight provenance contract.

This module performs no environment resets, policy inference, training, or
simulator steps. P2 is a separately preregistered frozen-policy comparison,
not a new learning algorithm or an established skill.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from bagi.experiments import learned_competence_e4 as p1
from bagi.experiments.e4_p1_distribution_audit import ARCHIVE_SHA256

SOURCE_RUN_ID = 40212708157
SOURCE_ARTIFACT_ID = 12606845219
TASKS = ("MiniGrid-Empty-Random-6x6-v0", "MiniGrid-DoorKey-5x5-v0")
MODEL_SEEDS = (17, 29)
TEST_SEEDS = tuple(range(27000, 27010))
ARMS = ("greedy_trained", "stochastic_trained", "stochastic_untrained", "uniform_random")
CAPS = {TASKS[0]: 144, TASKS[1]: 250}
MAX_TEST_ACTIONS = 31520
MAX_TRAIN_ACTIONS = 0
RNG_SEED_BASE = 27_000_000
SOURCE_ARMS = ("random", "untrained", "trained")
ACTOR_KEYS = (
    "mlp_extractor.policy_net.0.weight",
    "mlp_extractor.policy_net.0.bias",
    "mlp_extractor.policy_net.2.weight",
    "mlp_extractor.policy_net.2.bias",
    "action_net.weight",
    "action_net.bias",
)


def verify_registration() -> None:
    """Reject changed P2 conditions and accidental reuse of old held-out starts."""
    if (
        TASKS != p1.TASKS
        or MODEL_SEEDS != (17, 29)
        or TEST_SEEDS != tuple(range(27000, 27010))
        or set(TEST_SEEDS) & set(p1.TEST_SEEDS)
        or ARMS != (
            "greedy_trained", "stochastic_trained",
            "stochastic_untrained", "uniform_random",
        )
        or CAPS != {TASKS[0]: 144, TASKS[1]: 250}
        or MAX_TEST_ACTIONS != 31520
        or MAX_TRAIN_ACTIONS != 0
        or RNG_SEED_BASE != 27_000_000
        or SOURCE_RUN_ID != 40212708157
        or SOURCE_ARTIFACT_ID != 12606845219
    ):
        raise ValueError("G2-E4-P2 frozen preregistration was modified")


def episode_rng_seed(
    task_index: int, model_index: int, arm_index: int, reset_seed: int,
) -> int:
    """Independent, prospectively fixed per-episode action-sampling stream."""
    verify_registration()
    if (
        task_index not in range(len(TASKS))
        or model_index not in range(len(MODEL_SEEDS))
        or arm_index not in range(len(ARMS))
        or reset_seed not in TEST_SEEDS
    ):
        raise ValueError("Unregistered G2-E4-P2 sampling condition")
    return (
        RNG_SEED_BASE + task_index * 1_000_000 + model_index * 100_000
        + arm_index * 10_000 + reset_seed - TEST_SEEDS[0]
    )


def validated_state_dict(
    archived: dict[str, np.ndarray], reference: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Match EVERY numeric NPZ tensor to an already constructed SB3 policy.

    The caller may pass the returned arrays to torch.from_numpy and
    load_state_dict(strict=True); this function never deserializes objects.
    """
    expected = {"policy__" + key.replace(".", "__"): key for key in reference}
    if set(archived) != set(expected):
        raise ValueError("Frozen policy tensor keys differ from SB3 state dict")
    output: dict[str, np.ndarray] = {}
    for encoded, name in expected.items():
        value = np.asarray(archived[encoded])
        ref = reference[name]
        target = np.asarray(ref.detach().cpu().numpy() if hasattr(ref, "detach") else ref)
        if (
            value.dtype.kind not in "biuf"
            or not np.isfinite(value).all()
            or value.shape != target.shape
            or value.dtype != target.dtype
        ):
            raise ValueError("Unsafe, nonfinite, or incompatible frozen policy tensor")
        output[name] = np.array(value, copy=True)
    return output


def archive_weights(path: Path) -> dict[tuple[str, int, str], dict[str, np.ndarray]]:
    """Verify the EXACT official P1 ZIP and all twelve inner NPZs before use.

    This is read-only; it never extracts files, executes checkpoint code,
    loads pickles, constructs a simulator, or touches P2 evaluation seeds.
    """
    verify_registration()
    raw = path.read_bytes()
    if len(raw) > 50_000_000 or hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Frozen P1 ZIP SHA256 mismatch or oversized archive")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        if (
            len(names) != 13 or len(set(names)) != 13
            or "summary.json" not in names
            or sum(item.file_size for item in z.infolist()) > 100_000_000
        ):
            raise ValueError("Unexpected or oversized frozen artifact members")
        summary = json.loads(z.read("summary.json"))
        if (
            summary.get("study") != "G2-E4-P1"
            or summary.get("status") != "complete"
            or summary.get("tasks") != list(TASKS)
            or summary.get("model_seeds") != list(MODEL_SEEDS)
            or summary.get("test_seeds") != list(p1.TEST_SEEDS)
            or summary.get("arms") != list(SOURCE_ARMS)
            or summary.get("native_action_ledger") != {
                "train_actions": 65536, "test_actions": 23110,
            }
        ):
            raise ValueError("Frozen source experiment metadata mismatch")
        rows = summary.get("results", [])
        expected_pairs = {(task, seed) for task in TASKS for seed in MODEL_SEEDS}
        if (
            len(rows) != 4
            or {(row["task"], row["model_seed"]) for row in rows} != expected_pairs
        ):
            raise ValueError("Missing or duplicate source trained models")
        entries = summary.get("trace_artifacts", [])
        expected = {
            (task, seed, arm) for task in TASKS
            for seed in MODEL_SEEDS for arm in SOURCE_ARMS
        }
        if (
            len(entries) != 12
            or {(e["task"], e["model_seed"], e["arm"]) for e in entries} != expected
            or set(names) != {"summary.json"} | {e["file"] for e in entries}
        ):
            raise ValueError("Missing or duplicated source numeric trace artifacts")
        policies: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
        for entry in entries:
            task, seed, arm = entry["task"], entry["model_seed"], entry["arm"]
            filename = f"{task.replace('-', '_')}_seed{seed}_{arm}.npz"
            if entry["file"] != filename:
                raise ValueError("Source NPZ name does not match registered condition")
            blob = z.read(filename)
            if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
                raise ValueError("Source NPZ SHA256 checksum mismatch")
            with np.load(io.BytesIO(blob), allow_pickle=False) as data:
                arrays = {name: data[name].copy() for name in data.files}
            if any(
                value.dtype.kind not in "biuf" or not np.isfinite(value).all()
                for value in arrays.values()
            ):
                raise ValueError("Unsafe or nonfinite source numeric artifact")
            weights = {key: value for key, value in arrays.items()
                       if key.startswith("policy__")}
            if arm == "random":
                if weights:
                    raise ValueError("Random control unexpectedly contains policy weights")
            else:
                required = {"policy__" + key.replace(".", "__") for key in ACTOR_KEYS}
                if not required <= set(weights):
                    raise ValueError("Frozen source actor weights are incomplete")
                policies[(task, seed, arm)] = weights
        if len(policies) != 8:
            raise ValueError("Frozen source trained/untrained weights incomplete")
        return policies
