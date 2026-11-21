"""G2-E5: derive once and strictly load E2 actor0-only public TRAIN rows.

The ONE-TIME converter necessarily materializes mixed EVAL arrays while
extracting actor0. Consumers read only the newly isolated, pinned archive.
This is data-provenance instrumentation, not a new model or experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

SOURCE_ZIP_SHA256 = "b000a02a2eacfb8aa07befa151e0bbfb2c5bf7fd430f8bd896bd11f90f57436a"
ISOLATED_ZIP_SHA256 = "8d3f167ce977f62d3397ff37c553d8948b94bc03ac6d88df8a5a6e5fe86aac83"
SCHEMA = "BAGI_E2_ACTOR0_PUBLIC_TRAIN_V1"
FIELDS = (
    "observation", "next_observation", "action", "episode_index", "seed_index",
    "actor_index",
)
SOURCE_ROWS = 7296
TRAIN_ROWS = 4096
SOURCE_ACTOR_COUNTS = {"0": 4096, "1": 1600, "2": 1600}
TRAIN_SEED_COUNTS = {"0": 2048, "1": 2048}
EXPECTED_EPISODES = {0: tuple(range(52)), 1: tuple(range(92, 144))}
ZIP_MEMBERS = {"manifest.json", "train_numeric.npz"}


def _digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _validate_train(data: dict[str, np.ndarray]) -> None:
    if set(data) != set(FIELDS):
        raise ValueError("TRAIN archive has unexpected or missing fields")
    if any(v.dtype.kind not in "biuf" for v in data.values()):
        raise ValueError("Object or non-numeric arrays are forbidden")
    if any(v.shape[0] != TRAIN_ROWS for v in data.values()):
        raise ValueError("Unregistered TRAIN row count")
    for key in ("observation", "next_observation"):
        arr = data[key]
        if (arr.shape != (TRAIN_ROWS, 5, 5) or arr.dtype.kind not in "iu"
                or not np.isfinite(arr).all() or np.any(arr < 0) or np.any(arr > 8)):
            raise ValueError("Invalid native public observation")
    if (data["action"].shape != (TRAIN_ROWS,)
            or data["action"].dtype.kind not in "iu"
            or np.any(data["action"] < 0) or np.any(data["action"] > 3)):
        raise ValueError("Invalid executed TRAIN actions")
    for key in ("episode_index", "seed_index", "actor_index"):
        if data[key].shape != (TRAIN_ROWS,) or data[key].dtype.kind not in "iu":
            raise ValueError("Invalid native TRAIN metadata")
    if np.any(data["actor_index"] != 0):
        raise ValueError("EVAL rows are forbidden, including sentinel EVAL actors")
    for seed, expected in EXPECTED_EPISODES.items():
        mask = data["seed_index"] == seed
        if int(mask.sum()) != TRAIN_SEED_COUNTS[str(seed)]:
            raise ValueError("TRAIN seed count mismatch")
        episodes = data["episode_index"][mask]
        seen, counts = np.unique(episodes, return_counts=True)
        if not np.array_equal(seen, np.array(expected)):
            raise ValueError("Unregistered TRAIN episode identifiers")
        if not np.array_equal(counts, np.array([40] * 51 + [8])):
            raise ValueError("TRAIN episode lengths changed")
        # Episodes are contiguous and strictly ordered within each original policy seed.
        starts = np.flatnonzero(np.r_[True, episodes[1:] != episodes[:-1]])
        if not np.array_equal(episodes[starts], np.array(expected)):
            raise ValueError("TRAIN episode rows interleaved or out of order")
    if not np.isin(data["seed_index"], (0, 1)).all():
        raise ValueError("Unregistered policy seed index")
    # A next view must equal the next recorded current view within each episode.
    # This checks temporal integrity, not whether hidden physical state changed.
    within = data["episode_index"][1:] == data["episode_index"][:-1]
    if not np.array_equal(data["next_observation"][:-1][within],
                          data["observation"][1:][within]):
        raise ValueError("Temporal public transition chain mismatch")


def _manifest(npz: bytes) -> dict:
    return {
        "schema": SCHEMA,
        "source_run_id": 40185492543,
        "source_artifact_id": 12596191672,
        "source_zip_sha256": SOURCE_ZIP_SHA256,
        "derived_train_rows": TRAIN_ROWS,
        "source_total_rows": SOURCE_ROWS,
        "source_actor_counts": SOURCE_ACTOR_COUNTS,
        "train_seed_counts": TRAIN_SEED_COUNTS,
        "selection": "actor_index==0",
        "fields": list(FIELDS),
        "numeric_npz_sha256": _digest(npz),
        "derivation_accessed_mixed_eval_arrays": True,
        "consumer_archive_contains_eval_rows": False,
    }


def _encode_archive(data: dict[str, np.ndarray]) -> bytes:
    _validate_train(data)
    numeric = io.BytesIO()
    np.savez_compressed(numeric, **{key: data[key] for key in FIELDS})
    npz = numeric.getvalue()
    body = io.BytesIO()
    with zipfile.ZipFile(body, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, blob in (
            ("manifest.json", (json.dumps(_manifest(npz), sort_keys=True, indent=2)
                               + "\n").encode()),
            ("train_numeric.npz", npz),
        ):
            # Stable ZIP timestamps permit exact-byte reproducibility of the derived artifact.
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, blob)
    return body.getvalue()


def derive_actor0(source_zip: Path, output_zip: Path) -> dict:
    """Read historical MIXED archive once, disclose EVAL access, output TRAIN only.

    No native simulator is called. Never reads source summary or saved weights.
    The source ZIP digest must match the registered E2 run *before* any member read.
    """
    if output_zip.exists():
        raise FileExistsError("Refusing to overwrite an existing derived archive")
    source = source_zip.read_bytes()
    if _digest(source) != SOURCE_ZIP_SHA256:
        raise ValueError("Mixed original E2 source SHA256 mismatch")
    with zipfile.ZipFile(io.BytesIO(source)) as outer:
        if set(outer.namelist()) != {"summary.json", "public_policy_numeric.npz"}:
            raise ValueError("Mixed original E2 member schema changed")
        with np.load(io.BytesIO(outer.read("public_policy_numeric.npz")),
                     allow_pickle=False) as numeric:
            actor = numeric["actor_index"]
            if actor.shape != (SOURCE_ROWS,) or actor.dtype.kind not in "iu":
                raise ValueError("Invalid mixed E2 actor array")
            counts = {str(int(a)): int((actor == a).sum()) for a in np.unique(actor)}
            if counts != SOURCE_ACTOR_COUNTS:
                raise ValueError("Mixed E2 original TRAIN/EVAL partition changed")
            selected = actor == 0
            # Important: each of these six source arrays includes EVAL rows
            # when materialized. This is ONE-TIME extraction; never claim zero
            # source EVAL array reads for the converter itself.
            data = {}
            for key in FIELDS:
                original = numeric[key]
                if original.shape[0] != SOURCE_ROWS:
                    raise ValueError("Unexpected mixed E2 numeric rows")
                data[key] = original[selected].copy()
    blob = _encode_archive(data)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    output_zip.write_bytes(blob)
    return {
        "isolated_zip_sha256": _digest(blob),
        "source_zip_sha256": SOURCE_ZIP_SHA256,
        "source_eval_rows_materialized_during_derivation": SOURCE_ROWS - TRAIN_ROWS,
        "output_eval_rows": 0,
        "output_train_rows": TRAIN_ROWS,
        "new_native_actions": 0,
        "new_training_updates": 0,
    }


def load_isolated_train(path: Path, *, expected_sha256: str = ISOLATED_ZIP_SHA256
                        ) -> dict[str, np.ndarray]:
    """Fail-closed future consumer: verify exact artifact bytes BEFORE array read.

    The expected hash override exists for synthetic tests; production callers
    must use the fixed constant pinned after verifying real extraction.
    """
    raw = path.read_bytes()
    if _digest(raw) != expected_sha256:
        raise ValueError("Isolated TRAIN archive SHA256 mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if set(archive.namelist()) != ZIP_MEMBERS or len(archive.namelist()) != 2:
            raise ValueError("Not an isolated actor0-only archive")
        if archive.getinfo("train_numeric.npz").file_size > 5_000_000:
            raise ValueError("TRAIN archive is unexpectedly large")
        manifest = json.loads(archive.read("manifest.json"))
        npz = archive.read("train_numeric.npz")
    if manifest != _manifest(npz):
        raise ValueError("TRAIN manifest provenance or NPZ digest mismatch")
    with np.load(io.BytesIO(npz), allow_pickle=False) as numeric:
        if set(numeric.files) != set(FIELDS):
            raise ValueError("Isolated archive contains unexpected fields")
        data = {key: numeric[key].copy() for key in FIELDS}
    _validate_train(data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, help="Published, mixed E2 source ZIP")
    parser.add_argument("--derive", type=Path, help="Destination for TRAIN-only ZIP")
    parser.add_argument("--verify", type=Path, help="Verify an existing isolated ZIP")
    args = parser.parse_args()
    if args.source is not None and args.derive is not None and args.verify is None:
        result = derive_actor0(args.source, args.derive)
    elif args.verify is not None and args.source is None and args.derive is None:
        data = load_isolated_train(args.verify)
        result = {"status": "isolated_train_verified", "train_rows": len(data["action"]),
                  "eval_rows_read": 0, "native_actions": 0}
    else:
        parser.error("Provide --source and --derive together, OR --verify only")
        return
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
