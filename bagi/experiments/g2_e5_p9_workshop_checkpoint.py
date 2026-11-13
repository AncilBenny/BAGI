"""P9: extract only frozen TRAIN-derived E2 DQN tensors, never EVAL arrays.

The historical source ZIP contains EVAL arrays, but this converter only
materializes twelve named weight tensors from its lazy numeric NPZ. It never
calls a simulator, fits a model, or reads EVAL arrays as numpy values.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

SOURCE_SHA256 = "b000a02a2eacfb8aa07befa151e0bbfb2c5bf7fd430f8bd896bd11f90f57436a"
CHECKPOINT_SHA256 = "40c8d2d6cd8e591dd0fc9c23a7aeb354113b00565666d82ee88598efe64b56d6"
SOURCE_MEMBERS = {"summary.json", "public_policy_numeric.npz"}
SOURCE_FIELDS = {
    "observation", "next_observation", "action", "reward", "terminated",
    "truncated", "episode_index", "seed_index", "layout_index", "actor_index",
}
SHAPES = {
    "net_0_weight": (128, 25), "net_0_bias": (128,),
    "net_2_weight": (128, 128), "net_2_bias": (128,),
    "net_4_weight": (4, 128), "net_4_bias": (4,),
}
WEIGHTS = {f"seed{seed}_{name}": shape
           for seed in (17, 29) for name, shape in SHAPES.items()}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validated(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(tensors) != set(WEIGHTS):
        raise ValueError("Missing or unexpected E2 frozen DQN tensor")
    result = {}
    for name, shape in WEIGHTS.items():
        a = tensors[name]
        if (a.dtype != np.dtype("float32") or a.shape != shape
                or not np.isfinite(a).all()):
            raise ValueError(f"Invalid E2 frozen DQN tensor {name}")
        result[name] = np.ascontiguousarray(a)
    return result


def _manifest(source_sha256: str) -> dict:
    return {
        "schema": "BAGI_P9_E2_FROZEN_TRAIN_DQN_WEIGHTS_V1",
        "source_run_id": 40185492543,
        "source_artifact_id": 12596191672,
        "source_zip_sha256": source_sha256,
        "source_code_blob": "315c36dfc36c509522d1de5d20dead7757adbbb8",
        "source_contains_mixed_eval_arrays": True,
        "converter_materialized_eval_arrays": False,
        "training_native_steps_per_seed_from_source_contract": 2048,
        "training_optimizer_updates_per_seed_from_source_contract": 1985,
        "frozen_after_training_before_evaluation_in_source": True,
        "observation_shape": [5, 5],
        "actions": 4,
        "tensor_names": list(WEIGHTS),
    }


def _entry(name: str, data: bytes, archive: zipfile.ZipFile) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, data, compresslevel=9)


def _encode(tensors: dict[str, np.ndarray], source_sha256: str) -> bytes:
    tensors = _validated(tensors)
    body = io.BytesIO()
    with zipfile.ZipFile(body, "w") as archive:
        _entry("manifest.json", (json.dumps(_manifest(source_sha256), sort_keys=True)
                                 + "\n").encode("utf-8"), archive)
        for name in WEIGHTS:
            data = io.BytesIO()
            np.save(data, tensors[name], allow_pickle=False)
            _entry(f"{name}.npy", data.getvalue(), archive)
    return body.getvalue()


def extract_checkpoint(source: Path, destination: Path, *,
                       expected_source_sha256: str = SOURCE_SHA256,
                       expected_checkpoint_sha256: str | None = CHECKPOINT_SHA256) -> dict:
    """Read only twelve TRAIN-derived named tensors from pinned mixed source.

    No source summary, source EVAL array, or private evaluator value is read.
    Hash overrides are ONLY for synthetic tests; production defaults
    are exact frozen hashes. Refuse to overwrite existing checkpoints.
    """
    if destination.exists():
        raise FileExistsError("Refusing to overwrite a frozen checkpoint")
    raw = source.read_bytes()
    if _sha(raw) != expected_source_sha256:
        raise ValueError("Original mixed E2 source ZIP SHA256 mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as outer:
        if set(outer.namelist()) != SOURCE_MEMBERS or len(outer.namelist()) != 2:
            raise ValueError("Unexpected original E2 source ZIP members")
        if outer.getinfo("public_policy_numeric.npz").file_size > 500_000:
            raise ValueError("Original E2 source numeric member exceeds cap")
        with np.load(io.BytesIO(outer.read("public_policy_numeric.npz")),
                     allow_pickle=False) as numeric:
            if set(numeric.files) != SOURCE_FIELDS | set(WEIGHTS):
                raise ValueError("Original E2 source numeric member keys changed")
            # Access only weight keys; numpy's NpzFile is lazy. No EVAL array
            # key is ever indexed or materialized in this converter.
            tensors = {name: numeric[name].copy() for name in WEIGHTS}
    checkpoint = _encode(tensors, expected_source_sha256)
    digest = _sha(checkpoint)
    if expected_checkpoint_sha256 is not None and digest != expected_checkpoint_sha256:
        raise ValueError("Derived frozen checkpoint ZIP SHA256 mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(checkpoint)
    return {"status": "checkpoint_extracted", "checkpoint_sha256": digest,
            "source_sha256": expected_source_sha256,
            "frozen_tensors": len(WEIGHTS), "source_eval_arrays_materialized": 0,
            "new_native_actions": 0, "new_optimizer_updates": 0}


def load_checkpoint(path: Path, *, expected_sha256: str = CHECKPOINT_SHA256,
                    expected_source_sha256: str = SOURCE_SHA256) -> dict[str, np.ndarray]:
    """Strict safe-numeric reader; never uses torch.load or pickle."""
    raw = path.read_bytes()
    if _sha(raw) != expected_sha256:
        raise ValueError("Frozen E2 checkpoint ZIP SHA256 mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        required = {"manifest.json"} | {f"{name}.npy" for name in WEIGHTS}
        if set(archive.namelist()) != required or len(archive.namelist()) != len(required):
            raise ValueError("Unexpected checkpoint member or possible EVAL leakage")
        if json.loads(archive.read("manifest.json")) != _manifest(expected_source_sha256):
            raise ValueError("Frozen E2 checkpoint provenance mismatch")
        tensors = {}
        for name in WEIGHTS:
            member = archive.getinfo(f"{name}.npy")
            if member.file_size > 100_000:
                raise ValueError("Oversized frozen E2 tensor")
            tensors[name] = np.load(io.BytesIO(archive.read(member)), allow_pickle=False)
    return _validated(tensors)


def frozen_q_network(path: Path, seed: int):
    """Reconstruct existing DQN public actor from safe, strictly pinned arrays."""
    if seed not in (17, 29):
        raise ValueError("Unregistered frozen DQN policy seed")
    import torch

    from bagi.agents.dqn import QNetwork

    weights = load_checkpoint(path)
    model = QNetwork((5, 5), actions=4)
    state = {name.replace("_", "."): torch.from_numpy(weights[f"seed{seed}_{name}"].copy())
             for name in SHAPES}
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
