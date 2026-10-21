"""Read-only audit of a pinned third-party pretrained SB3 checkpoint.

No unpickling, torch.load, SB3.load or third-party serialized code execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import yaml

MAX_DOWNLOAD_BYTES = 8_000_000
MAX_METADATA_BYTES = 2_000_000
MAX_MEMBER_BYTES = 20_000_000
REQUIRED_MEMBERS = frozenset(("data", "policy.pth", "_stable_baselines3_version"))


def verify_and_inspect_archive(data: bytes, expected_hash: str) -> dict:
    """Read only plain JSON and text of a SHA256-verified SB3 ZIP."""
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_hash:
        raise ValueError("Pinned archive SHA256 mismatch; no archive contents inspected")
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ValueError("Archive is too large")
    with zipfile.ZipFile(BytesIO(data), "r") as archive:
        members = archive.infolist()
        names = [entry.filename for entry in members]
        if not REQUIRED_MEMBERS.issubset(names):
            raise ValueError("Missing standard SB3 checkpoint metadata or model entries")
        if len(names) != len(set(names)) or len(names) > 30:
            raise ValueError("Unexpected duplicate or numerous ZIP entries")
        for entry in members:
            name = entry.filename
            if (name.startswith("/") or "\\" in name or
                    ".." in Path(name).parts or entry.file_size > MAX_MEMBER_BYTES):
                raise ValueError("Unsafe or oversized ZIP member")
        if archive.getinfo("data").file_size > MAX_METADATA_BYTES:
            raise ValueError("Checkpoint JSON metadata exceeds safe size")
        # Serialized Python objects remain inert JSON strings.
        with archive.open("data") as handle:
            info = json.loads(handle.read(MAX_METADATA_BYTES + 1).decode("utf-8"))
        if not isinstance(info, dict):
            raise TypeError("Expected a JSON metadata object")
        with archive.open("_stable_baselines3_version") as handle:
            sb3_version = handle.read(256).decode("utf-8").strip()
    return {
        "archive_sha256_verified": digest,
        "archive_size_bytes": len(data),
        "archive_entries": names,
        "original_sb3_version": sb3_version,
        "metadata_field_names": sorted(info),
        "serialized_python_fields_not_loaded": sorted(
            key for key, value in info.items()
            if isinstance(value, dict) and ":serialized:" in value
        ),
        "execution_or_deserialization_performed": False,
    }


def download_pinned(repo: str, revision: str, filename: str) -> bytes:
    if repo != "sb3/ppo-MiniGrid-DoorKey-5x5-v0":
        raise ValueError("Unexpected checkpoint repository")
    if revision != "90b0f9903be52f016a4f9f378d027f441350444f":
        raise ValueError("Unexpected, unreviewed checkpoint revision")
    if filename != "ppo-MiniGrid-DoorKey-5x5-v0.zip":
        raise ValueError("Unexpected checkpoint filename")
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "BAGI-readonly-audit/1.0"}
    )
    with urllib.request.urlopen(request, timeout=65) as response:
        content = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise ValueError("Remote artifact exceeded maximum allowed bytes")
    return content


def probe_wrappers(env_id: str) -> dict:
    """Use the existing upstream MiniGrid wrappers, not a custom adapter."""
    import gymnasium as gym
    import minigrid  # noqa: F401
    from gymnasium.wrappers import FlattenObservation
    from minigrid.wrappers import FlatObsWrapper, ImgObsWrapper

    shapes = {}
    for name, wrapper in (
        ("published_recipe_flat", lambda env: FlatObsWrapper(env)),
        ("bagi_current_image", lambda env: FlattenObservation(ImgObsWrapper(env))),
    ):
        env = wrapper(gym.make(env_id))
        try:
            observation, _ = env.reset(seed=123)
            shapes[name] = {
                "observation_shape": list(observation.shape),
                "observation_dtype": str(observation.dtype),
                "action_count": int(env.action_space.n),
            }
        finally:
            env.close()
    return shapes


def run(config_path: str) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    upstream = config["upstream"]
    archive = verify_and_inspect_archive(
        download_pinned(upstream["repository"], upstream["revision"], upstream["filename"]),
        upstream["sha256"],
    )
    shapes = probe_wrappers(config["environment"])
    same = (shapes["published_recipe_flat"]["observation_shape"] ==
            shapes["bagi_current_image"]["observation_shape"])
    report = {
        "research_type": "static compatibility audit of EXISTING pretrained model",
        "source_model": upstream,
        "static_archive": archive,
        "current_minigrid_wrapper_contracts": shapes,
        "input_shapes_match": same,
        "safe_to_directly_evaluate": False,
        "blocked_by": [
            ("Original gym_minigrid FlatObsWrapper input contract must be reconciled "
             "with current Farama MiniGrid; matching only shape is insufficient."),
            ("Published checkpoint requires VecNormalize stats. Its external "
             "vec_normalize.pkl is NOT downloaded or deserialized."),
            ("SB3 archive observation/action spaces contain serialized Python objects; "
             "this audit leaves them inert."),
            "Model weights were never loaded; no competence or reward replicated.",
        ],
        "next_gate": (
            "Reuse the upstream RL Zoo evaluation recipe and establish a safe, "
            "version-correct normalization and serialization path in an isolated "
            "nonprivileged runtime before attempting pretrained inference."
        ),
    }
    destination = Path(config["output_dir"]) / "summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Verified official checkpoint SHA256: {archive['archive_sha256_verified']}")
    print(f"Published-like wrapper: {shapes['published_recipe_flat']}")
    print(f"BAGI image-only wrapper: {shapes['bagi_current_image']}")
    print(f"Pretrained inference blocked pending review; report: {destination}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrained_checkpoint_audit.yaml")
    run(parser.parse_args().config)
