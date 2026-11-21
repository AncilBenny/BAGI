"""Prospective replication contracts; synthetic tests are not native results."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import yaml

from bagi.experiments import general_brain_stagnation_confirmation as confirm
from bagi.experiments import general_brain_stagnation_g2 as original


def test_untouched_confirmation_layouts_and_unchanged_budgets():
    confirm.validate_registration()
    original.validate_registration()
    assert confirm.FRESH_LAYOUTS == tuple(range(11800, 11820))
    assert not set(confirm.FRESH_LAYOUTS) & set(original.FRESH_LAYOUTS)
    assert confirm.MODEL_SEEDS == original.MODEL_SEEDS == (17, 29)
    assert confirm.TASKS == original.TASKS == ("hold", "near", "putdown")
    assert confirm.CONDITIONS == original.CONDITIONS == ("direct", "stagnation_escape")
    assert confirm.CAP == original.CAP == 40
    assert confirm.PATIENCE == original.PATIENCE == 3
    assert confirm.INPUT_WIDTH == original.INPUT_WIDTH == 115
    assert confirm.MAX_NATIVE_LEARNER_ACTIONS == 2 * 2 * 3 * 20 * 40 == 9600
    assert original.MAX_NATIVE_LEARNER_ACTIONS == 4800


@pytest.mark.parametrize("function", [
    "execute_episode", "ranked_frozen_policy", "public_signature",
    "validated_public_features", "physical_transition_changed", "summarize_cases",
])
def test_algorithm_functions_identical_to_first_trial(function):
    # Semantic AST identity: no retuning, changed action logic, hidden-state
    # inputs or learner architecture between D3 and independently frozen D4.
    left = ast.dump(ast.parse(inspect.getsource(getattr(original, function))),
                    include_attributes=False)
    right = ast.dump(ast.parse(inspect.getsource(getattr(confirm, function))),
                     include_attributes=False)
    assert left == right


def test_confirmation_run_training_and_evaluator_same_code():
    # The run retains identical native policy/action loops; only selected
    # prospective layouts and explicit budget/report labels may differ.
    o = inspect.getsource(original.run)
    n = inspect.getsource(confirm.run)
    replacements = {
        '"G2-D3 matched public-stagnation escape vs same frozen policy"':
            '"G2-D4 prospective independent replication of identical G2-D3 frozen-policy intervention"',
        'print(f"G2-D3 model_seed=': 'print(f"G2-D4 model_seed=',
        'up to 4800 real native learner steps for 120 fixed episodes':
            'up to 9600 real native learner steps for 240 fixed episodes',
        'Two model seeds and ten source-disjoint layouts':
            'Two model seeds and twenty source-disjoint layouts',
    }
    for before, after in replacements.items():
        assert o.count(before) == 1
        o = o.replace(before, after)
    assert o == n


def test_native_workflow_source_check_and_isolated_cpu():
    root = Path(__file__).parents[1]
    wf = root / ".github/workflows/bagi-general-stagnation-confirmation.yml"
    data = yaml.load(wf.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert data["on"]["push"]["branches"] == ["setup/initial-platform"]
    job = data["jobs"]["native-stagnation-confirmation"]
    assert job["runs-on"] == "ubuntu-latest"
    assert int(job["timeout-minutes"]) <= 25
    steps = '\n'.join(str(x.get("run", "")) for x in job["steps"])
    assert 'requirements/xland-reuse31.txt' in steps
    assert 'python -m bagi.experiments.general_brain_stagnation_confirmation' in steps
    assert 'tests/test_general_brain_stagnation_confirmation.py' in steps
    assert 'gh run download 40110945017' in steps
    assert '--source-dir runs/d3_source' in steps
    assert '--old-teacher-dir runs/d3_old_teacher' in steps
    assert '--diverse-dir runs/d3_diverse_teacher' in steps


def test_evaluator_cases_frozen_layouts_no_replacements():
    cases = [{"layout": i, "task": "putdown", "eligible": False,
              "native_actions": 0, "native_reward": False,
              "physical_placement": False, "target_publicly_seen": False,
              "effective_blue_pickups": 0, "effective_blue_drops": 0,
              "native_drop_action_attempts": 0,
              "suspected_stagnation_steps": 0, "escaped_actions": 0,
              "false_public_stagnation_alerts": 0,
              "trailing_drop_action_run": 0} for i in confirm.FRESH_LAYOUTS]
    assert confirm.summarize_cases(cases, "putdown")["eligible"] == 0
    with pytest.raises(ValueError, match="Missing/replaced"):
        confirm.summarize_cases(cases[1:], "putdown")
