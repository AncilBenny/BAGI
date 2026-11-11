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
    assert dispatch["options"] == ["legacy", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]
    assert dispatch["default"] == "legacy"
    assert dispatch["required"] is True
    assert triggers["schedule"] == [{"cron": "17 * * * *"}]
    assert triggers["push"] == {
        "branches": ["setup/initial-platform"],
        "paths": [
            ".github/bagi-m1-once.json",
            ".github/bagi-m2-once.json",
            ".github/bagi-m3-once.json",
            ".github/bagi-m4-once.json",
            ".github/bagi-m5-once.json",
            ".github/bagi-m6-once.json",
            ".github/bagi-m6-repair-once.json",
            ".github/bagi-m7-once.json",
            ".github/bagi-m8-once.json",
        ],
    }
    jobs = workflow["jobs"]
    assert list(jobs) == ["audit", "research"]
    assert jobs["research"]["needs"] == "audit"
    assert "timeout-minutes" not in jobs["research"]
    assert jobs["research"]["if"] == "needs.audit.result == 'success'"
    assert triggers["workflow_dispatch"]["inputs"]["force_retrain"]["type"] == "boolean"
    assert jobs["research"]["env"]["BAGI_RESEARCH_STUDY"] == (
        "${{ needs.audit.outputs.study }}"
    )
    assert jobs["audit"]["outputs"]["study"] == "${{ steps.mode.outputs.study }}"
    assert next(step for step in jobs["audit"]["steps"]
                if step.get("name") == "Fail closed on missing or unknown manual study"
                )["id"] == "mode"
    assert "budget" not in jobs
    assert jobs["research"]["env"]["BAGI_RESEARCH_SOURCE_SHA"] == "${{ needs.audit.outputs.research_sha }}"
    assert "ci_snapshot(api, sha)" in str(jobs["audit"]["steps"])
    assert "bagi_wait_for_research_ci.py" in str(jobs["audit"]["steps"])
    assert workflow["jobs"]["audit"]["steps"][0]["with"]["fetch-depth"] == 2
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
    assert "env.BAGI_RESEARCH_STUDY == 'm2'" in install["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm3'" in install["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm4'" in install["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm5'" in install["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm6'" in install["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm7'" in install["if"]
    assert "env.BAGI_RESEARCH_STUDY == 'm8'" in install["if"]
    m2 = steps["Run admitted M2 extended goal-learning and sampling diagnostic"]
    assert "env.BAGI_RESEARCH_STUDY == 'm2'" in m2["if"]
    assert "configs/m2_goal_learning.yaml" in m2["run"]
    assert "--execute" in m2["run"]
    m3 = steps["Run admitted M3 independent goal-memory replication"]
    assert "env.BAGI_RESEARCH_STUDY == 'm3'" in m3["if"]
    assert "configs/m3_goal_learning.yaml" in m3["run"]
    m4 = steps["Run admitted M4 evaluation-only frozen policy diagnostic"]
    assert "env.BAGI_RESEARCH_STUDY == 'm4'" in m4["if"]
    assert "m4_frozen_policy_eval" in m4["run"]
    m5 = steps["Run admitted M5 continual task-retention PPO study"]
    assert "env.BAGI_RESEARCH_STUDY == 'm5'" in m5["if"]
    assert "m5_continual_retention" in m5["run"]
    m6 = steps["Run admitted M6 DoorKey episode-continuity ablation"]
    assert "env.BAGI_RESEARCH_STUDY == 'm6'" in m6["if"]
    assert "m6_doorkey_continuity" in m6["run"]
    assert "--execute" in m6["run"]
    m7 = steps["Run admitted M7 frozen DoorKey native-action sampling diagnostic"]
    assert "m7_frozen_doorkey_sampling" in m7["run"]
    assert "--execute" in m7["run"]
    m8 = steps["Run admitted M8 fresh DoorKey PPO entropy ablation"]
    assert "m8_doorkey_entropy" in m8["run"]
    assert "--execute" in m8["run"]
    assert "sha256sum -c -" in steps[
        "Retrieve verified M6 numeric checkpoint artifact for M7"
    ]["run"]
    assert "--execute" in m5["run"]
    assert "--execute" in m4["run"]
    assert "sha256sum -c -" in steps[
        "Retrieve verified M3 numeric checkpoint artifact for M4"
    ]["run"]
    assert "--execute" in m3["run"]
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
    assert "runs/m2_goal_learning/progress.json" in artifact["path"]
    assert "runs/m2_goal_learning/*.npz" in artifact["path"]
    assert "runs/m3_goal_learning/progress.json" in artifact["path"]
    assert "runs/m3_goal_learning/*.npz" in artifact["path"]
    assert "runs/m3_goal_learning/summary.json" in steps["Capture research metadata"]["run"]
    assert "runs/m4_frozen_policy/summary.json" in steps["Capture research metadata"]["run"]
    assert "runs/m4_frozen_policy/progress.json" in artifact["path"]
    assert "runs/m5_continual_retention/progress.json" in artifact["path"]
    assert "runs/m5_continual_retention/*.npz" in artifact["path"]
    assert "runs/m5_continual_retention/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
    assert "runs/m6_doorkey_continuity/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
    assert "runs/m6_doorkey_continuity/progress.json" in artifact["path"]
    assert "runs/m6_doorkey_continuity/*.npz" in artifact["path"]
    assert "runs/m7_frozen_doorkey_sampling/progress.json" in artifact["path"]
    assert "runs/m8_doorkey_entropy/progress.json" in artifact["path"]
    assert "runs/m8_doorkey_entropy/*.npz" in artifact["path"]
    assert "runs/m7_frozen_doorkey_sampling/summary.json" in steps[
        "Capture research metadata"
    ]["run"]
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
        ("workflow_dispatch", "m2", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m3", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m4", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m5", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m6", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m7", "refs/heads/setup/initial-platform", True),
        ("workflow_dispatch", "m8", "refs/heads/setup/initial-platform", True),
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

def test_one_shot_push_request_rejects_modified_or_misbound_files(tmp_path):
    import json
    import os
    import subprocess
    import sys

    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"
    ref = "refs/heads/setup/initial-platform"

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, timeout=5
        ).strip()

    git("init", "-q")
    git("config", "user.name", "BAGI source regression")
    git("config", "user.email", "noreply@example.invalid")
    (tmp_path / "README.md").write_text("test\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "test parent")
    parent = git("rev-parse", "HEAD")
    request_file = tmp_path / ".github" / "bagi-m1-once.json"
    request_file.parent.mkdir()
    request = {
        "study": "m1", "force_retrain": True,
        "expected_parent_sha": parent,
        "request_id": "BAGI-M1-2026-11-11",
    }
    request_file.write_text(json.dumps(request) + "\n", encoding="utf-8")
    git("add", ".github/bagi-m1-once.json")
    git("commit", "-qm", "research(m1): request one fresh fit")
    sha = git("rev-parse", "HEAD")
    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps({
        "ref": ref, "before": parent, "after": sha,
    }), encoding="utf-8")
    env = {
        **os.environ, "GITHUB_EVENT_NAME": "push", "GITHUB_REF": ref,
        "GITHUB_SHA": sha, "GITHUB_EVENT_PATH": str(event_file),
        "BAGI_DISPATCH_STUDY": "",
    }
    def invoke():
        return subprocess.run(
            [sys.executable, str(script)], env=env, cwd=tmp_path,
            capture_output=True, text=True, check=False, timeout=6
        )

    assert invoke().returncode == 0
    request["force_retrain"] = False
    request_file.write_text(json.dumps(request), encoding="utf-8")
    assert invoke().returncode == 2
    request["force_retrain"] = True
    request_file.write_text(json.dumps(request), encoding="utf-8")
    env["GITHUB_SHA"] = "a" * 40
    assert invoke().returncode == 2
    env["GITHUB_SHA"] = sha
    event_file.write_text(json.dumps({
        "ref": ref, "before": "f" * 40, "after": sha,
    }), encoding="utf-8")
    assert invoke().returncode == 2


def test_one_shot_m2_request_requires_exact_source_and_expected_payload(tmp_path):
    import json
    import os
    import subprocess
    import sys

    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, timeout=5
        ).strip()

    git("init", "-q")
    git("config", "user.name", "BAGI source regression")
    git("config", "user.email", "noreply@example.invalid")
    (tmp_path / "README.md").write_text("test\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "test parent")
    parent = git("rev-parse", "HEAD")
    one_shot = tmp_path / ".github" / "bagi-m2-once.json"
    one_shot.parent.mkdir()
    payload = {
        "study": "m2",
        "force_retrain": True,
        "expected_parent_sha": parent,
        "request_id": "BAGI-M2-2026-11-11",
    }
    one_shot.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    git("add", ".github/bagi-m2-once.json")
    git("commit", "-qm", "research(m2): request fresh extended fit")
    sha = git("rev-parse", "HEAD")
    event = tmp_path / "event.json"
    event.write_text(json.dumps({
        "ref": "refs/heads/setup/initial-platform",
        "before": parent, "after": sha,
    }), encoding="utf-8")
    env = {
        **os.environ,
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/setup/initial-platform",
        "GITHUB_SHA": sha, "GITHUB_EVENT_PATH": str(event),
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
    }

    def invoke():
        return subprocess.run(
            [sys.executable, str(script)], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=6, check=False
        )

    assert invoke().returncode == 0
    assert (tmp_path / "output.txt").read_text() == "study=m2\n"
    payload["request_id"] = "unauthorized-repeat"
    one_shot.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert invoke().returncode == 2
    payload["request_id"] = "BAGI-M2-2026-11-11"
    one_shot.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    env["GITHUB_REF"] = "refs/heads/main"
    assert invoke().returncode == 2


def test_one_shot_m3_push_is_bound_to_request_only_new_commit(tmp_path):
    import json
    import os
    import subprocess
    import sys

    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, timeout=5
        ).strip()

    git("init", "-q")
    git("config", "user.name", "BAGI test")
    git("config", "user.email", "noreply@example.invalid")
    (tmp_path / "README.md").write_text("replication\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "initial")
    parent = git("rev-parse", "HEAD")
    request = tmp_path / ".github" / "bagi-m3-once.json"
    request.parent.mkdir()
    payload = {
        "study": "m3", "force_retrain": True,
        "expected_parent_sha": parent, "request_id": "BAGI-M3-2026-11-11",
    }
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    git("add", ".github/bagi-m3-once.json")
    git("commit", "-qm", "one-shot independent replication")
    sha = git("rev-parse", "HEAD")
    event = tmp_path / "event.json"
    event.write_text(json.dumps({
        "ref": "refs/heads/setup/initial-platform",
        "before": parent, "after": sha,
    }), encoding="utf-8")
    env = {
        **os.environ, "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/setup/initial-platform",
        "GITHUB_SHA": sha, "GITHUB_EVENT_PATH": str(event),
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
    }

    def invoke():
        return subprocess.run(
            [sys.executable, str(script)], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=6, check=False,
        )

    assert invoke().returncode == 0
    assert (tmp_path / "output.txt").read_text() == "study=m3\n"
    payload["force_retrain"] = False
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert invoke().returncode == 2
    payload["force_retrain"] = True
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    env["GITHUB_REF"] = "refs/heads/main"
    assert invoke().returncode == 2


def test_one_shot_m4_push_is_add_only_and_exact_commit_bound(tmp_path):
    import json
    import os
    import subprocess
    import sys

    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, timeout=5
        ).strip()

    git("init", "-q")
    git("config", "user.name", "BAGI M4 test")
    git("config", "user.email", "noreply@example.invalid")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "base")
    parent = git("rev-parse", "HEAD")
    request_path = tmp_path / ".github" / "bagi-m4-once.json"
    request_path.parent.mkdir()
    payload = {
        "study": "m4", "force_retrain": True,
        "expected_parent_sha": parent, "request_id": "BAGI-M4-2026-11-11",
    }
    request_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    git("add", ".github/bagi-m4-once.json")
    git("commit", "-qm", "M4 frozen evaluation request")
    sha = git("rev-parse", "HEAD")
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({
        "ref": "refs/heads/setup/initial-platform", "before": parent, "after": sha,
    }), encoding="utf-8")
    env = {
        **os.environ, "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/setup/initial-platform",
        "GITHUB_SHA": sha, "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
    }

    def invoke():
        return subprocess.run(
            [sys.executable, str(script)], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=6, check=False,
        )

    assert invoke().returncode == 0
    assert (tmp_path / "output.txt").read_text() == "study=m4\n"
    payload["force_retrain"] = False
    request_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert invoke().returncode == 2
    payload["force_retrain"] = True
    request_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    env["GITHUB_REF"] = "refs/heads/main"
    assert invoke().returncode == 2


def test_one_shot_m5_push_is_bound_to_single_added_file_and_exact_parent(tmp_path):
    import json
    import os
    import subprocess
    import sys

    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, timeout=5
        ).strip()

    git("init", "-q")
    git("config", "user.name", "M5 source guard")
    git("config", "user.email", "noreply@example.invalid")
    (tmp_path / "README.md").write_text("parent\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "init")
    parent = git("rev-parse", "HEAD")
    path = tmp_path / ".github" / "bagi-m5-once.json"
    path.parent.mkdir()
    payload = {
        "study": "m5", "force_retrain": True,
        "expected_parent_sha": parent, "request_id": "BAGI-M5-2026-11-11",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    git("add", ".github/bagi-m5-once.json")
    git("commit", "-qm", "request M5 training")
    head = git("rev-parse", "HEAD")
    event = tmp_path / "event.json"
    event.write_text(json.dumps({
        "ref": "refs/heads/setup/initial-platform",
        "before": parent, "after": head,
    }), encoding="utf-8")
    environ = {
        **os.environ, "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/setup/initial-platform",
        "GITHUB_SHA": head, "GITHUB_EVENT_PATH": str(event),
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
    }

    def invoke():
        return subprocess.run(
            [sys.executable, str(script)], cwd=tmp_path, env=environ,
            capture_output=True, text=True, timeout=6, check=False,
        )

    assert invoke().returncode == 0
    assert (tmp_path / "output.txt").read_text() == "study=m5\n"
    payload["force_retrain"] = False
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert invoke().returncode == 2
    payload["force_retrain"] = True
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    environ["GITHUB_REF"] = "refs/heads/main"
    assert invoke().returncode == 2


def test_one_shot_m6_push_is_add_only_and_exact_source_bound(tmp_path):
    import json
    import os
    import subprocess
    import sys

    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, timeout=5
        ).strip()

    git("init", "-q")
    git("config", "user.name", "M6 test")
    git("config", "user.email", "noreply@example.invalid")
    (tmp_path / "README.md").write_text("source\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "init")
    parent = git("rev-parse", "HEAD")
    request = tmp_path / ".github" / "bagi-m6-once.json"
    request.parent.mkdir()
    payload = {
        "study": "m6", "force_retrain": True,
        "expected_parent_sha": parent, "request_id": "BAGI-M6-2026-11-11",
    }
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    git("add", ".github/bagi-m6-once.json")
    git("commit", "-qm", "request M6 native study")
    head = git("rev-parse", "HEAD")
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({
        "ref": "refs/heads/setup/initial-platform",
        "before": parent, "after": head,
    }), encoding="utf-8")
    environ = {
        **os.environ, "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/setup/initial-platform",
        "GITHUB_SHA": head, "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
    }

    def invoke():
        return subprocess.run(
            [sys.executable, str(script)], cwd=tmp_path, env=environ,
            capture_output=True, text=True, timeout=6, check=False,
        )

    assert invoke().returncode == 0
    assert (tmp_path / "output.txt").read_text() == "study=m6\n"
    payload["force_retrain"] = False
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert invoke().returncode == 2
    payload["force_retrain"] = True
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    environ["GITHUB_REF"] = "refs/heads/main"
    assert invoke().returncode == 2


def test_one_shot_m6_repair_is_new_add_only_request_and_source_bound(tmp_path):
    import json
    import os
    import subprocess
    import sys

    script = WORKFLOW.parents[2] / "scripts" / "bagi_research_dispatch.py"

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, timeout=5
        ).strip()

    git("init", "-q")
    git("config", "user.name", "M6 repair test")
    git("config", "user.email", "noreply@example.invalid")
    (tmp_path / "README.md").write_text("source\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "initial")
    parent = git("rev-parse", "HEAD")
    request = tmp_path / ".github" / "bagi-m6-repair-once.json"
    request.parent.mkdir()
    payload = {
        "study": "m6", "force_retrain": True,
        "expected_parent_sha": parent,
        "request_id": "BAGI-M6-REPAIR-2026-11-11",
    }
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    git("add", ".github/bagi-m6-repair-once.json")
    git("commit", "-qm", "request M6 corrected runner")
    head = git("rev-parse", "HEAD")
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({
        "ref": "refs/heads/setup/initial-platform",
        "before": parent, "after": head,
    }), encoding="utf-8")
    env = {
        **os.environ, "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/setup/initial-platform",
        "GITHUB_SHA": head, "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
    }

    def invoke():
        return subprocess.run(
            [sys.executable, str(script)], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=6, check=False,
        )

    assert invoke().returncode == 0
    assert (tmp_path / "output.txt").read_text() == "study=m6\n"
    payload["request_id"] = "BAGI-M6-2026-11-11"
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert invoke().returncode == 2
    payload["request_id"] = "BAGI-M6-REPAIR-2026-11-11"
    payload["force_retrain"] = False
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert invoke().returncode == 2
