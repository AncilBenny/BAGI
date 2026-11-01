"""Reuse-only XLand task split checks; optional package is isolated from core CI."""
import pytest

pytest.importorskip("jax")
pytest.importorskip("xminigrid")

from bagi.experiments.xland_composition_smoke import disjoint_tasks, fingerprint, ruleset_for


def specification(name, source, dest):
    return {
        "name": name,
        "source_tile": "SQUARE",
        "source_color": source,
        "result_tile": "BALL",
        "result_color": dest,
    }


def test_ruleset_ids_are_distinct_and_reproducible():
    config = {"tasks": {
        "train": [specification("t1", "BLUE", "RED")],
        "held_out": [specification("t2", "BLUE", "GREEN")],
    }}
    train, held = disjoint_tasks(config)
    assert train[0]["fingerprint"] != held[0]["fingerprint"]
    assert train[0]["fingerprint"] == fingerprint(ruleset_for(config["tasks"]["train"][0]))
    assert int(train[0]["ruleset"].goal[0]) == 1  # AgentHoldGoal, not immediate proximity
    assert int(train[0]["ruleset"].rules[0, 0]) == 2  # AgentNearRule


def test_duplicate_task_leakage_is_rejected():
    one = specification("same", "BLUE", "RED")
    with pytest.raises(ValueError, match="overlap"):
        disjoint_tasks({"tasks": {"train": [one], "held_out": [one]}})


def test_no_op_ruleset_is_rejected():
    value = {
        "name": "no_op", "source_tile": "BALL", "source_color": "RED",
        "result_tile": "BALL", "result_color": "RED",
    }
    with pytest.raises(ValueError, match="transform"):
        ruleset_for(value)
