"""Regression tests for reproducible guided/random data collection."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from bagi.experiments.collect_guided import run as collect_guided


def _random_rows(path: Path, guided_episodes: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["episode"]) >= guided_episodes:
                normalized = dict(row)
                normalized["episode"] = int(row["episode"]) - guided_episodes
                rows.append(normalized)
    return rows


def _collect(tmp_path: Path, guided_episodes: int) -> Path:
    output_dir = tmp_path / f"guided_{guided_episodes}"
    config = {
        "seed": 42,
        "environment": {"size": 9, "max_steps": 20, "view_size": 5},
        "collection": {
            "guided_episodes": guided_episodes,
            "random_episodes": 3,
            "guided_seed_base": 42,
            "random_seed_base": 100_042,
            "output_dir": str(output_dir),
        },
    }
    config_path = tmp_path / f"collect_{guided_episodes}.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    collect_guided(str(config_path))
    return output_dir / "transitions.jsonl"


def test_random_background_is_identical_across_guided_budgets(tmp_path: Path) -> None:
    no_guidance = _collect(tmp_path, guided_episodes=0)
    with_guidance = _collect(tmp_path, guided_episodes=5)

    assert _random_rows(no_guidance, 0) == _random_rows(with_guidance, 5)
