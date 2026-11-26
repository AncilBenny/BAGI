"""Test XLand's existing composition primitives, not a custom rule engine."""
import pytest

pytest.importorskip("jax")
pytest.importorskip("xminigrid")

from bagi.experiments.xland_structural_composition import (
    KINDS,
    ORACLE_SUFFIXES,
    construct_ruleset,
)


def test_structural_train_test_rules_do_not_overlap():
    tasks = {kind: construct_ruleset(kind) for kind in KINDS}
    assert [int(v) for v in tasks["hold"].rules[:, 0]] == [1]
    assert [int(v) for v in tasks["near"].rules[:, 0]] == [2]
    assert [int(v) for v in tasks["hold_then_near"].rules[:, 0]] == [1, 2]
    assert [int(tasks[name].goal[0]) for name in KINDS] == [1, 1, 1]


def test_control_suffixes_are_shorter_and_action_types_differ():
    assert ORACLE_SUFFIXES["hold"] == (3,)
    assert ORACLE_SUFFIXES["near"] == (0, 3)
    assert ORACLE_SUFFIXES["hold_then_near"] == (3, 4, 0, 3)
    assert 4 not in ORACLE_SUFFIXES["hold"]
    assert 4 not in ORACLE_SUFFIXES["near"]


def test_rejects_an_unreviewed_task_type():
    with pytest.raises(ValueError, match="Unapproved"):
        construct_ruleset("make_up_a_rule")
