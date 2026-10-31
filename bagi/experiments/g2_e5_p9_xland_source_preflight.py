"""P9 fail-closed audit of historical XLand BC *TRAIN source*, not model weights.

Read-only. No environment, policy import/fit, teacher query or TEST collection.
A successful audit does NOT authorize P9 native actions without frozen weights.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

OUTER_SHA = {
    "reuse19": "5992e5f3a14d8acf1d625b6df34f7ae1f8d18043d0ebfedf11353b71f0e6efd7",
    "reuse33": "adbbb1a1d67f07fcb4efeb94d1188e6b5c793720cd6cb5c3b3cd9f506a85d80a",
    "reuse36": "6ca2bf2d3102f3cd68c60394e8eb42af236cf47fcdd6f6e3abe5104f800d0c64",
}
INNER_SHA = {
    "reuse19": {
        "summary.json": "71c8c72e31574289447b399c21e475e523e5d4eccceea998cc7ddd9ef7105892",
        "action_provenance_seed17.npz": "25d24168163fbc189f1afbd8454e3948deae73d4924cfc83f49ba35ebb5b4882",
        "action_provenance_seed29.npz": "66d5a546875e8b373f5ec42926b46b683e0f29c885a24f207f66386f1dfa4607",
    },
    "reuse33": {
        "summary.json": "eaf2f99a2dae59f3e4fe78a1f296edf72afbe09bc107cbeb382fcb69f3eaf57d",
        "train_source_complete_demos.npz": "b405e1ba816f424179dbf65a93a31e5a5db5b471bd8079671ecfd99215688efd",
    },
    "reuse36": {
        "summary.json": "a178131f01724a9c14cd1ea16e45b8bd62eb17e447b6e751e2297264402277af",
        "diverse_train_physical.npz": "e6c605bc9a109e2e366eb50635361bae9cd4c93563c282d787640694280d75da",
    },
}
SOURCE19_FIELDS = {
    "obs": ((291, 54), "float32"),
    "expert_label": ((291,), "int64"),
    "executed_action": ((291,), "int64"),
    "prior_executed_actions": ((291, 4), "int64"),
    "round": ((291,), "int64"),
    "episode": ((291,), "int64"),
    "step_in_episode": ((291,), "int64"),
    "done": ((291,), "bool"),
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verified_members(path: Path, expected: str,
                     inner: dict[str, str]) -> dict[str, bytes]:
    """Hash entire archive *before* decoding any member or array."""
    raw = path.read_bytes()
    if len(raw) > 1_000_000 or _digest(raw) != expected:
        raise ValueError("Unregistered outer XLand TRAIN artifact SHA256 or size")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if (len(members) != len(inner)
                or {item.filename for item in members} != set(inner)
                or sum(x.file_size for x in members) > 2_000_000
                or any(x.file_size > 1_000_000 for x in members)):
            raise ValueError("Unexpected archive members/size/duplicate")
        decoded = {}
        for name, sha in inner.items():
            member = archive.read(name)
            if _digest(member) != sha:
                raise ValueError(f"Unreviewed historical TRAIN member {name}")
            decoded[name] = member
    return decoded


def checked_npz(payload: bytes,
                schema: dict[str, tuple[tuple[int, ...], str]]) -> dict[str, np.ndarray]:
    """Numeric-only, fixed exact names/shapes/dtypes; forbid pickle and eval fields."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as array:
        if set(array.files) != set(schema):
            raise ValueError("Unexpected XLand source numeric fields (possible EVAL leak)")
        output = {}
        for key, (shape, dtype) in schema.items():
            value = array[key]
            if (value.shape != shape or value.dtype != np.dtype(dtype)
                    or (value.dtype.kind == "f" and not np.isfinite(value).all())):
                raise ValueError(f"Invalid source numeric TRAIN field {key}")
            output[key] = np.ascontiguousarray(value)
    return output


def record_digest(rows: dict[str, np.ndarray]) -> str:
    """Exact historical REUSE-19 numeric record hashing protocol."""
    digest = hashlib.sha256()
    for name in sorted(rows):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(rows[name]).tobytes())
    return digest.hexdigest()


def audit(reuse19: Path, reuse33: Path, reuse36: Path) -> dict[str, object]:
    members = {
        name: verified_members(path, OUTER_SHA[name], INNER_SHA[name])
        for name, path in (("reuse19", reuse19), ("reuse33", reuse33),
                           ("reuse36", reuse36))
    }
    prior = json.loads(members["reuse19"]["summary.json"])
    records = prior.get("model_results", [])
    if (prior.get("heldout_teacher_labels") not in (False, 0)
            or prior.get("training_tasks_only") != ["hold", "near", "putdown"]
            or sorted(x.get("model_seed") for x in records) != [17, 29]):
        raise ValueError("REUSE-19 TRAIN metadata or model-seed provenance invalid")
    result = {}
    for seed in (17, 29):
        rows = checked_npz(members["reuse19"][f"action_provenance_seed{seed}.npz"],
                           SOURCE19_FIELDS)
        digest = record_digest(rows)
        meta = next(x for x in records if x["model_seed"] == seed)
        if (meta.get("numeric_record_sha256") != digest
                or meta.get("stored_teacher_labels_match_logged_visits") is not True
                or np.any(rows["expert_label"] >= 6)
                or np.any(rows["expert_label"] < 0)
                or np.any(rows["executed_action"] >= 6)
                or np.any(rows["executed_action"] < 0)
                or np.any(rows["prior_executed_actions"] > 6)
                or np.any(rows["prior_executed_actions"] < 0)):
            raise ValueError("REUSE-19 source record provenance/action mismatch")
        result[str(seed)] = {"rows": 291, "numeric_record_sha256": digest}
    teacher_schema = {"obs": ((248, 115), "float32"),
                      "expert_label": ((248,), "int64"),
                      "done": ((248,), "bool")}
    small = checked_npz(members["reuse33"]["train_source_complete_demos.npz"],
                        teacher_schema)
    diverse = checked_npz(members["reuse36"]["diverse_train_physical.npz"], {
        "obs": ((707, 115), "float32"),
        "expert_label": ((707,), "int64"),
        "done": ((707,), "bool"),
    })
    for source in (small, diverse):
        if np.any(source["expert_label"] < 0) or np.any(source["expert_label"] >= 6):
            raise ValueError("Unregistered physical TRAIN teacher label")
    if (int(np.count_nonzero(diverse["expert_label"] == 3)) != 66
            or int(np.count_nonzero(diverse["expert_label"] == 4)) != 66
            or int(diverse["done"].sum()) != 66):
        raise ValueError("REUSE-36 physical TRAIN label coverage changed")
    # No source summary with EVAL metadata is JSON-parsed for REUSE-33/36;
    # summary bytes are hash-verified. TRAIN teacher labels are never returned.
    return {
        "study": "G2-E5-P9", "preflight": "XLand historical TRAIN provenance",
        "status": "SOURCE_VERIFIED_WEIGHTS_UNAVAILABLE_STOP",
        "outer_zip_sha256": dict(OUTER_SHA),
        "inner_file_sha256": dict(INNER_SHA),
        "source_numeric_seed_records": result,
        "physical_teacher_train_rows": {"reuse33": 248, "reuse36": 707},
        "frozen_bc_weight_files_in_sources": 0,
        "bc_actor_reconstruction_fits_per_two_seeds_and_two_arms": 4,
        "total_nominal_source_exposures_if_retrained": 37776,
        "source_summaries_with_eval_metadata_parsed": 0,
        "training_source_summary_parsed": 1,
        "native_actions": 0, "native_resets": 0,
        "bc_fits_performed": 0, "optimizer_steps": 0,
        "teacher_queries": 0, "original_eval_arrays_accessed": 0,
        "authorized_to_evaluate_native_p9": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse19", type=Path, required=True)
    parser.add_argument("--reuse33", type=Path, required=True)
    parser.add_argument("--reuse36", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.reuse19, args.reuse33, args.reuse36)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "native_actions": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
