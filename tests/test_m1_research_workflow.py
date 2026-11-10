"""M1 is an opt-in mode of the EXISTING gated research runner, not a new CI job."""

from pathlib import Path

import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github" / "workflows" / "bagi-research-loop.yml"
)


def test_m1_and_legacy_share_ci_and_one_serialized_runner_without_daily_quota():
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
    assert list(jobs) == ["audit", "research"]
    assert jobs["research"]["needs"] == "audit"
    assert "timeout-minutes" not in jobs["research"]
    assert jobs["research"]["if"] == "needs.audit.result == 'success'"
    assert triggers["workflow_dispatch"]["inputs"]["force_retrain"]["type"] == "boolean"
    assert jobs["research"]["env"]["BAGI_RESEARCH_STUDY"].endswith(
        "inputs.study || 'legacy' }}"
    )
    assert "budget" not in jobs
    assert jobs["research"]["env"]["BAGI_RESEARCH_SOURCE_SHA"] == "${{ needs.audit.outputs.research_sha }}"
    assert "ci_snapshot(api, sha)" in str(jobs["audit"]["steps"])
    assert "ci_snapshot(api, sha)" in str(jobs["research"]["steps"])


def test_m1_installs_pinned_upstream_rl_only_when_selected_and_no_legacy_fit():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = {row["name"]: row for row in workflow["jobs"]["research"]["steps"]
             if "name" in row}
    m1 = steps["Run admitted M1 public-observation goal-learning trial"]
    assert "timeout-minutes" not in m1
    assert "env.BAGI_RESEARCH_STUDY == 'm1'" in m1["if"]
    assert "steps.evaluated.outputs.cache-hit != 'true'" in m1["if"]
    assert "--execute" in m1["run"]
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


def test_dispatch_guard_selects_study_before_training():
    import os
    import subprocess
    import sys

    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["audit"]["steps"]
    guard = next(row for row in steps
                 if row.get("name") == "Fail closed on missing or unknown manual study")
    assert guard["env"]["BAGI_DISPATCH_STUDY"] == "${{ inputs.study }}"
    assert guard["run"] == "python3 scripts/bagi_research_dispatch.py"
    assert steps.index(guard) < next(i for i, row in enumerate(steps)
                                     if row.get("id") == "verify")
    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"
    cases = (
        ("schedule", "", "refs/heads/main", True),
        ("workflow_dispatch", "legacy", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m1", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "", "refs/heads/setup/initial-platform", False),
        ("workflow_dispatch", "unregistered", "refs/heads/setup/initial-platform", False),
        ("workflow_dispatch", "m1", "refs/heads/main", False),
        ("push", "m1", "refs/heads/setup/initial-platform", False),
    )
    for event, study, ref, admitted in cases:
        env = {**os.environ, "GITHUB_EVENT_NAME": event,
               "BAGI_DISPATCH_STUDY": study, "GITHUB_REF": ref}
        run = subprocess.run([sys.executable, str(script)], env=env,
                             capture_output=True, text=True, timeout=5, check=False)
        assert (run.returncode == 0) is admitted
        assert ("BAGI research dispatch HOLD" in run.stderr) is not admitted
