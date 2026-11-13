"""Strict P0B numeric artifact packager; NEVER loads numerical NPZ arrays.

An independent later audit must inspect arrays with np.load(allow_pickle=False).
Here we only verify source files, registered summary contracts, hashes and ZIP
membership, without materializing mixed TRAIN/diagnostic observation arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

MEMBERS = (
    "summary.json",
    "train_public_numeric.npz",
    "diagnostic_public_numeric.npz",
    "frozen_dqn_weights.npz",
)
NUMERIC_MEMBERS = MEMBERS[1:]
ZIP_NAME = "bagi-g2-e6-p0b-public-only.zip"
MAX_MEMBER_BYTES = 64 * 1024 * 1024
WORKSHOP_SHA = "3a91b058d3461ce4ecb489e93939c88f624557f4"
DQN_SHA = "4c5fac098e53e9c81884d36e14b894679b40513d"


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def _registered_episode(
    row: object, *, seed: int, size: int, layout: int,
    actor: int, phase: int, reset: int,
) -> tuple[int, int] | None:
    """Validate summary-only public-run metadata; never deserialize NPZ."""
    if not isinstance(row, dict):
        return None
    expected = {
        "policy_seed": seed, "size": size, "layout_index": layout,
        "actor_index": actor, "phase_index": phase, "reset_seed": reset,
    }
    if any(type(row.get(name)) is not int or row[name] != value
           for name, value in expected.items()):
        return None
    steps, episode_id = row.get("steps"), row.get("episode_id")
    if (type(steps) is not int or not 1 <= steps <= 40
            or type(episode_id) is not int or episode_id < 0):
        return None
    return steps, episode_id


def _complete_registered_coverage(summary: dict) -> bool:
    """No full-run label without BOTH learned seeds and 288 diagnostic episodes."""
    records = summary.get("seed_results")
    if not isinstance(records, dict) or set(records) != {"17", "29"}:
        return False
    seen_ids: set[int] = set()
    total_train = total_diagnostic = total_updates = 0
    for seed_index, seed in enumerate((17, 29)):
        data = records[str(seed)]
        if not isinstance(data, dict):
            return False
        if data.get("train_actions_by_size") != {"9": 2048, "11": 2048}:
            return False
        if type(data.get("optimizer_updates")) is not int:
            return False
        if data["optimizer_updates"] != 4033:
            return False
        total_updates += data["optimizer_updates"]
        training = data.get("training")
        if not isinstance(training, list) or not training or len(training) > 1000:
            return False
        sizes = {9: 0, 11: 0}
        for index, row in enumerate(training):
            if not isinstance(row, dict) or row.get("size") not in sizes:
                return False
            size = row["size"]
            checked = _registered_episode(
                row, seed=seed, size=size, layout=0, actor=0, phase=0,
                reset=20000 + 1000 * seed_index + index,
            )
            if checked is None or checked[1] in seen_ids:
                return False
            steps, episode_id = checked
            seen_ids.add(episode_id)
            sizes[size] += steps
        if sizes != {9: 2048, 11: 2048}:
            return False
        total_train += sum(sizes.values())

        diagnostic = data.get("diagnostic")
        if not isinstance(diagnostic, dict) or set(diagnostic) != {"9", "11"}:
            return False
        for size in (9, 11):
            layouts = diagnostic[str(size)]
            if not isinstance(layouts, dict) or set(layouts) != {"default", "mirror_x"}:
                return False
            for layout_index, layout in enumerate(("default", "mirror_x")):
                arms = layouts[layout]
                expected_arms = ("trained_greedy", "untrained_greedy", "uniform_random")
                if not isinstance(arms, dict) or set(arms) != set(expected_arms):
                    return False
                for actor_index, actor in enumerate(expected_arms, 1):
                    rows = arms[actor]
                    if not isinstance(rows, list) or len(rows) != 12:
                        return False
                    for index, row in enumerate(rows):
                        checked = _registered_episode(
                            row, seed=seed, size=size, layout=layout_index,
                            actor=actor_index, phase=1, reset=19000 + index,
                        )
                        if checked is None or checked[1] in seen_ids:
                            return False
                        steps, episode_id = checked
                        seen_ids.add(episode_id)
                        total_diagnostic += steps
    return (
        total_train == 8192
        and total_updates == 8066
        and total_diagnostic == summary.get("native_diagnostic_actions")
        and total_train == summary.get("native_train_actions")
        and total_train + total_diagnostic == summary.get("native_actions")
        and len(seen_ids) == summary.get("native_resets")
    )



def package(run_dir: Path, bundle_dir: Path) -> dict:
    """Package one completed/failed acquisition, fail closed on any mismatch."""
    if bundle_dir.exists():
        raise FileExistsError("Refusing to overwrite an existing P0B bundle")
    bundle_dir.mkdir(parents=True)
    report = {"study": "G2-E6-P0B", "status": "invalid", "zip_sha256": None}
    try:
        if not run_dir.is_dir() or run_dir.is_symlink():
            raise ValueError("Missing or unsafe numeric run directory")
        files = {p.name: p for p in run_dir.iterdir()}
        if set(files) != set(MEMBERS):
            raise ValueError("Missing, extra, or unknown P0B run file")
        for path in files.values():
            if path.is_symlink() or not path.is_file():
                raise ValueError("P0B artifact must contain only regular files")
            if not 0 < path.stat().st_size <= MAX_MEMBER_BYTES:
                raise ValueError("P0B numeric evidence exceeds file-size ceiling")
        summary = json.loads(files["summary.json"].read_text(encoding="utf-8"))
        if summary.get("study") != "G2-E6-P0B":
            raise ValueError("Unexpected study identifier")
        if summary.get("status") not in ("complete", "failed"):
            raise ValueError("Invalid or still-running acquisition")
        if (summary.get("workshop_git_blob_sha1") != WORKSHOP_SHA
                or summary.get("dqn_git_blob_sha1") != DQN_SHA):
            raise ValueError("Native source does not match registered SHA-1")
        if summary.get("promotion") is not False or summary.get("size13_native_calls") != 0:
            raise ValueError("Unregistered holdout access or promotion")
        for key, limit in (("native_actions", 19712),
                           ("native_train_actions", 8192),
                           ("native_diagnostic_actions", 11520),
                           ("optimizer_updates", 8066)):
            value = summary.get(key)
            if type(value) is not int or not 0 <= value <= limit:
                raise ValueError(f"Invalid numerical budget field: {key}")
        if summary["status"] == "complete" and (
            summary["native_train_actions"] != 8192
            or summary["optimizer_updates"] != 8066
            or summary["native_action_count_exact"] is not True
            or summary["native_step_exceptions"] != 0
            or summary["recorded_actions"] != summary["native_actions"]
            or summary["native_train_actions"] + summary["native_diagnostic_actions"]
            != summary["native_actions"]
        ):
            raise ValueError("Complete run lacks exact native accounting")
        if summary["status"] == "complete" and not _complete_registered_coverage(summary):
            raise ValueError("Complete run lacks registered per-seed diagnostic coverage")
        expected = summary.get("numeric_file_sha256")
        if not isinstance(expected, dict) or set(expected) != set(NUMERIC_MEMBERS):
            raise ValueError("Missing numeric member SHA-256 manifest")
        for name in NUMERIC_MEMBERS:
            if not isinstance(expected[name], str) or _digest(files[name]) != expected[name]:
                raise ValueError(f"Corrupted numeric member: {name}")

        zip_path = bundle_dir / ZIP_NAME
        with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in MEMBERS:
                archive.write(files[name], arcname=name)
        with zipfile.ZipFile(zip_path, "r") as archive:
            if archive.namelist() != list(MEMBERS) or archive.testzip() is not None:
                raise ValueError("ZIP member list or compressed data is corrupted")
        digest = _digest(zip_path)
        (bundle_dir / (ZIP_NAME + ".sha256")).write_text(
            f"{digest}  {ZIP_NAME}\n", encoding="ascii"
        )
        report.update({
            "status": "packaged",
            "run_status": summary["status"],
            "zip_sha256": digest,
            "members": list(MEMBERS),
            "original_test_eval_arrays_materialized": False,
        })
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (bundle_dir / "packaging_status.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    args = parser.parse_args()
    result = package(args.run_dir, args.bundle_dir)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
