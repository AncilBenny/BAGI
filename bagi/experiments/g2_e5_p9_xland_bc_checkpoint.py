"""P9 XLand reconstructed BC actor snapshot: safe numeric-only checkpoint gate.

This reuses existing torch Module state_dict and NumPy's pickle-free .npy.
Nothing here trains BC, imports its optional runtime or calls a simulator.
An export is NOT a historical pretrained checkpoint, and NEVER licenses P9
native evaluation without separate true weight provenance and seed audit.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bagi.experiments.g2_e5_p9_xland_source_preflight import OUTER_SHA

SCHEMA = "BAGI_P9_XLAND_BC_RECONSTRUCTED_NUMERIC_V1"
MODEL_SEEDS = (17, 29)
ARM = "expanded_22_plus_72_train_layouts"
MAX_TENSORS_PER_ACTOR = 256
MAX_TOTAL_UNCOMPRESSED = 32_000_000
MAX_CHECKPOINT_BYTES = 32_000_000
MAX_MEMBER_BYTES = 8_000_000
_PROBE_ROWS = 128
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*\Z")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def public_probe_inputs() -> np.ndarray:
    """Deterministic 115-wide non-native public-format probes (not episode data)."""
    rng = np.random.default_rng(20261026)
    x = rng.integers(0, 2, size=(_PROBE_ROWS, 115), dtype=np.int8).astype(np.float32)
    # Original upstream public XLand input ends with 4 executed command slots,
    # each one-hot of 7 choices (6 native actions + PAD 6).
    x[:, -28:] = np.eye(7, dtype=np.float32)[
        rng.integers(0, 7, size=(_PROBE_ROWS, 4))
    ].reshape(_PROBE_ROWS, 28)
    return x


def _policy_logits(model: torch.nn.Module) -> np.ndarray:
    """Use documented SB3 get_distribution, not private actor/evaluator data."""
    if not callable(getattr(model, "get_distribution", None)):
        raise TypeError("BC actor missing upstream policy distribution API")
    model.eval()
    with torch.no_grad():
        distribution = model.get_distribution(torch.from_numpy(public_probe_inputs()))
        categorical = getattr(distribution, "distribution", None)
        logits = getattr(categorical, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("BC actor did not return categorical logits")
    result = logits.detach().to("cpu").numpy()
    if result.shape != (_PROBE_ROWS, 6) or not np.isfinite(result).all():
        raise ValueError("BC actor distribution shape or logits invalid")
    return result.astype(np.float32, copy=True)


def _validated_state(model: torch.nn.Module) -> dict[str, np.ndarray]:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Only a real torch Module state_dict may be exported")
    state = model.state_dict()
    if not 1 <= len(state) <= MAX_TENSORS_PER_ACTOR:
        raise ValueError("BC state_dict tensor count invalid")
    out: dict[str, np.ndarray] = {}
    total = 0
    for key in sorted(state):
        value = state[key]
        if not _NAME.fullmatch(key) or not isinstance(value, torch.Tensor):
            raise ValueError("Unsafe BC state_dict tensor name or object")
        tensor = value.detach().to("cpu")
        if tensor.dtype != torch.float32 or tensor.ndim == 0:
            raise ValueError("BC checkpoint requires ordinary float32 tensors")
        array = np.ascontiguousarray(tensor.numpy())
        total += array.nbytes
        if array.nbytes > MAX_MEMBER_BYTES or not np.isfinite(array).all():
            raise ValueError("Invalid, nonfinite or oversized BC tensor")
        out[key] = array
    if total > MAX_TOTAL_UNCOMPRESSED:
        raise ValueError("BC checkpoint tensor budget exceeded")
    return out


def _npy(a: np.ndarray) -> bytes:
    data = io.BytesIO()
    np.save(data, a, allow_pickle=False)
    return data.getvalue()


def _member(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, payload, compresslevel=9)


def encode_reconstructed(policies: dict[int, torch.nn.Module]) -> bytes:
    """Snapshot *already reconstructed* BC policies; no fit/optimizer calls.

    The manifest explicitly denies equivalence with historically unarchived
    weights. Only an independently verified historical checkpoint could lift it.
    """
    if set(policies) != set(MODEL_SEEDS):
        raise ValueError("Requires both registered learned-policy seeds")
    tensors: dict[str, bytes] = {}
    models: dict[str, dict[str, Any]] = {}
    for seed in MODEL_SEEDS:
        state = _validated_state(policies[seed])
        # Probe a detached actor: export must not switch live TRAIN actors to eval.
        probes = _policy_logits(copy.deepcopy(policies[seed]))
        record: dict[str, Any] = {"tensors": []}
        for i, (key, arr) in enumerate(state.items()):
            name = f"seed{seed}/{i:03d}.npy"
            payload = _npy(arr)
            tensors[name] = payload
            record["tensors"].append({
                "name": key, "member": name, "shape": list(arr.shape),
                "sha256": _sha(payload),
            })
        member = f"seed{seed}/probe_logits.npy"
        raw = _npy(probes)
        tensors[member] = raw
        record["probe_logits_sha256"] = _sha(raw)
        models[str(seed)] = record
    manifest = {
        "schema": SCHEMA, "arm": ARM, "seed_order": list(MODEL_SEEDS),
        "registered_source_zip_sha256": dict(OUTER_SHA),
        "bc_train_rows_per_seed_reference": 787, "bc_epochs_per_seed_reference": 12,
        "nominal_train_exposures_per_seed_reference": 9444,
        "actual_training_input_equivalence_verified": False,
        "reference_recipe_not_execution_proof": True,
        "status": "RECONSTRUCTED_WEIGHTS_NOT_HISTORICALLY_VERIFIED",
        "historical_weight_identity_verified": False,
        "authorized_to_run_native_p9": False,
        "actor_observation_width": 115, "actor_n_actions": 6,
        "probe_is_native_observation": False,
        "models": models,
    }
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        _member(archive, "manifest.json", (json.dumps(manifest, sort_keys=True)
                                           + "\n").encode("utf-8"))
        for name, body in sorted(tensors.items()):
            _member(archive, name, body)
    result = out.getvalue()
    if len(result) > MAX_CHECKPOINT_BYTES:
        raise ValueError("Compressed BC snapshot exceeds byte budget")
    return result


def save_reconstructed(destination: Path,
                       policies: dict[int, torch.nn.Module]) -> dict[str, Any]:
    """Create immutable-by-hash local output without overwriting prior weights."""
    if destination.exists():
        raise FileExistsError("Refusing to overwrite frozen BC checkpoint")
    payload = encode_reconstructed(policies)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents races at the file-system boundary.
    with destination.open("xb") as f:
        f.write(payload)
    return {"status": "RECONSTRUCTED_WEIGHTS_NOT_HISTORICALLY_VERIFIED",
            "sha256": _sha(payload), "bytes": len(payload),
            "authorized_to_run_native_p9": False, "bc_fits_during_export": 0}


def _read_checkpoint(raw: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    if len(raw) > MAX_CHECKPOINT_BYTES:
        raise ValueError("Oversized BC checkpoint")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise ValueError("Malformed/duplicate BC members")
        if sum(x.file_size for x in archive.infolist()) > MAX_TOTAL_UNCOMPRESSED:
            raise ValueError("Uncompressed BC checkpoint exceeds budget")
        if any(x.file_size > MAX_MEMBER_BYTES or x.is_dir() for x in archive.infolist()):
            raise ValueError("Unsafe BC archive member")
        manifest = json.loads(archive.read("manifest.json"))
        if (manifest.get("schema") != SCHEMA or manifest.get("arm") != ARM
                or manifest.get("seed_order") != list(MODEL_SEEDS)
                or manifest.get("registered_source_zip_sha256") != OUTER_SHA
                or manifest.get("bc_train_rows_per_seed_reference") != 787
                or manifest.get("bc_epochs_per_seed_reference") != 12
                or manifest.get("nominal_train_exposures_per_seed_reference") != 9444
                or manifest.get("actual_training_input_equivalence_verified") is not False
                or manifest.get("reference_recipe_not_execution_proof") is not True
                or manifest.get("actor_observation_width") != 115
                or manifest.get("actor_n_actions") != 6
                or manifest.get("historical_weight_identity_verified") is not False
                or manifest.get("authorized_to_run_native_p9") is not False
                or manifest.get("probe_is_native_observation") is not False
                or manifest.get("status") != "RECONSTRUCTED_WEIGHTS_NOT_HISTORICALLY_VERIFIED"
                or set(manifest.get("models", {})) != {"17", "29"}):
            raise ValueError("Nonconforming P9 BC actor provenance manifest")
        expected = {"manifest.json"}
        for seed in MODEL_SEEDS:
            item = manifest["models"][str(seed)]
            tensors = item.get("tensors")
            if not isinstance(tensors, list) or not 1 <= len(tensors) <= MAX_TENSORS_PER_ACTOR:
                raise ValueError("Invalid BC model tensor count")
            for i, tensor in enumerate(tensors):
                name = f"seed{seed}/{i:03d}.npy"
                if tensor.get("member") != name or not _NAME.fullmatch(tensor.get("name", "")):
                    raise ValueError("BC model tensor member/name mismatch")
                expected.add(name)
            expected.add(f"seed{seed}/probe_logits.npy")
        if set(names) != expected:
            raise ValueError("Unexpected/missing BC member")
        bodies = {name: archive.read(name) for name in expected - {"manifest.json"}}
    return manifest, bodies


def load_reconstructed(path: Path, *, expected_sha256: str,
                       policies: dict[int, torch.nn.Module]) -> dict[str, Any]:
    """Strict numeric-only read into already-instantiated upstream BC modules.

    Refuses any unpinned checkpoint. Never invokes torch.load, pickle, BC.fit,
    optimizer, a native world, or a teacher. Restored models remain STOP-only.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Explicit expected full SHA256 required")
    if set(policies) != set(MODEL_SEEDS):
        raise ValueError("Both actor seeds required")
    data = path.read_bytes()
    if _sha(data) != expected_sha256:
        raise ValueError("BC checkpoint exact SHA256 mismatch")
    manifest, bodies = _read_checkpoint(data)
    new_states = {}
    for seed in MODEL_SEEDS:
        original = _validated_state(policies[seed])
        item = manifest["models"][str(seed)]
        if [x["name"] for x in item["tensors"]] != list(original):
            raise ValueError("Fresh BC architecture/state keys do not match")
        state = {}
        for meta in item["tensors"]:
            raw = bodies[meta["member"]]
            if _sha(raw) != meta["sha256"]:
                raise ValueError("BC state member hash mismatch")
            arr = np.load(io.BytesIO(raw), allow_pickle=False)
            if (arr.dtype != np.dtype("float32") or list(arr.shape) != meta["shape"]
                    or arr.shape != original[meta["name"]].shape
                    or not np.isfinite(arr).all()):
                raise ValueError("Restored BC tensor invalid")
            state[meta["name"]] = torch.from_numpy(arr.copy())
        log_raw = bodies[f"seed{seed}/probe_logits.npy"]
        if _sha(log_raw) != item["probe_logits_sha256"]:
            raise ValueError("BC probe logits hash mismatch")
        expected_logits = np.load(io.BytesIO(log_raw), allow_pickle=False)
        if (expected_logits.shape != (_PROBE_ROWS, 6)
                or expected_logits.dtype != np.dtype("float32")
                or not np.isfinite(expected_logits).all()):
            raise ValueError("Invalid independent BC logits manifest")
        new_states[seed] = (state, expected_logits)
    # Probe ALL candidates on detached modules before changing either live actor.
    # A well-formed archive can contain false logits even with valid inner/outer
    # hashes; parity failure must not silently replace a previously live policy.
    for seed in MODEL_SEEDS:
        states, expected_logits = new_states[seed]
        candidate = copy.deepcopy(policies[seed])
        candidate.load_state_dict(states, strict=True)
        fresh_logits = _policy_logits(candidate)
        if (not np.allclose(fresh_logits, expected_logits, rtol=0, atol=1e-6)
                or not np.array_equal(fresh_logits.argmax(axis=1),
                                      expected_logits.argmax(axis=1))):
            raise ValueError("BC exported/restored probe distribution/action mismatch")
    # Commit the validated states only after both seeds passed the probe gate.
    for seed in MODEL_SEEDS:
        policies[seed].load_state_dict(new_states[seed][0], strict=True)
        policies[seed].eval()
        for param in policies[seed].parameters():
            param.requires_grad_(False)
    return {"status": "RECONSTRUCTED_WEIGHTS_NOT_HISTORICALLY_VERIFIED",
            "sha256": expected_sha256,
            "probe_rows_per_seed": _PROBE_ROWS,
            "restored_policy_seeds": list(MODEL_SEEDS),
            "historical_weight_identity_verified": False,
            "authorized_to_run_native_p9": False,
            "bc_fits_during_load": 0, "native_actions": 0}
