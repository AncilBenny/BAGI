"""M1 is an opt-in mode of the EXISTING gated research runner, not a new CI job."""

from pathlib import Path

import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github" / "workflows" / "bagi-research-loop.yml"
)


def test_m1_and_legacy_reuse_exact_daily_quota_and_a_single_research_job():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert workflow["concurrency"]["group"] == "bagi-research-loop"
    # PyYAML may parse the YAML 1.1 key "on" as the boolean True.
    triggers = workflow.get("on", workflow.get(True))
    dispatch = triggers["workflow_dispatch"]["inputs"]["study"]
    assert dispatch["options"] == ["legacy", "m1"]
    assert dispatch["default"] == "legacy"
    assert dispatch["required"] is True
    assert triggers["schedule"] == [{"cron": "17 * * * *"}]
    jobs = workflow["jobs"]
    assert list(jobs) == ["audit", "budget", "research"]
    assert jobs["research"]["needs"] == ["audit", "budget"]
    assert jobs["research"]["timeout-minutes"] == 45
    assert "needs.budget.outputs.eligible == 'true'" in jobs["research"]["if"]
    assert jobs["research"]["env"]["BAGI_RESEARCH_STUDY"].endswith(
        "inputs.study || 'legacy' }}"
    )
    assert "bagi_research_budget.py research" in str(jobs["budget"]["steps"])
    assert "ci_snapshot(api, sha)" in str(jobs["audit"]["steps"])
    assert "ci_snapshot(api, sha)" in str(jobs["research"]["steps"])


def test_m1_installs_pinned_upstream_rl_only_when_selected_and_no_legacy_fit():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = {row["name"]: row for row in workflow["jobs"]["research"]["steps"]
             if "name" in row}
    m1 = steps["Run admitted M1 public-observation goal-learning trial"]
    assert m1["timeout-minutes"] <= 32
    assert m1["env"]["BAGI_M1_ADMITTED"] == "1"
    assert "env.BAGI_RESEARCH_STUDY == 'm1'" in m1["if"]
    assert "steps.evaluated.outputs.cache-hit != 'true'" in m1["if"]
    assert "--execute-budgeted-research" in m1["run"]
    install = steps["Install BAGI with pinned existing RL implementations (M1)"]
    assert "env.BAGI_RESEARCH_STUDY == 'm1'" in install["if"]
    assert "pip install -e '.[dev,rl]'" in install["run"]
    for name in (
        "Restore causal-v2 checkpoint cache",
        "Build causal-v2 checkpoint on cache miss",
        "Run oracle search ceiling",
        "Run learned planner candidate sweep",
        "Run grounded imagination-gap diagnostic",
        "Install BAGI (legacy study)",
    ):
        assert "env.BAGI_RESEARCH_STUDY == 'legacy'" in steps[name]["if"]
    assert "make check" in steps["Verify research platform"]["run"]


def test_outputs_preserve_source_provenance_partial_evidence_and_study_cache():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = {row["name"]: row for row in workflow["jobs"]["research"]["steps"]
             if "name" in row}
    assert "m1_goal_learning/summary.json" in steps["Capture research metadata"]["run"]
    assert "study_mode.txt" in steps["Capture research metadata"]["run"]
    artifact = steps["Upload research results"]["with"]
    assert "runs/m1_goal_learning/*.npz" in artifact["path"]
    assert "runs/m1_goal_learning/progress.json" in artifact["path"]
    restore = next(row for row in workflow["jobs"]["research"]["steps"]
                   if row.get("id") == "evaluated")
    assert "env.BAGI_RESEARCH_STUDY" in restore["with"]["key"]
    assert "env.BAGI_RESEARCH_STUDY" in steps.get(
        "Reuse any successful evaluation of this exact revision", restore
    )["with"]["key"]
