"""Fail-closed GitHub Actions preflight for BAGI, read-only GitHub REST.

Use with GITHUB_TOKEN/GITHUB_REPOSITORY on the scheduled workflow and
GITHUB_OUTPUT to pass verified run/artifact identifiers to subsequent jobs.
Requires no third-party Python packages, executes no research, and never
executes files extracted from evidence artifacts.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

BRANCH = "setup/initial-platform"
SHA = re.compile(r"^[0-9a-f]{40}$")
ALLOWED_WORKFLOWS = frozenset({"BAGI Research Loop", "BAGI E6 Imagination Gap"})
CI_PATH = ".github/workflows/ci.yml"
MAX_ZIP_BYTES = 12 * 1024 * 1024


class GateError(RuntimeError):
    pass


def latest_ci_success(runs: list[dict], sha: str, *, event: str,
                      merge_sha: str | None = None) -> bool:
    """Require the latest run of one event; never let events mask each other."""
    if event not in {"push", "pull_request"} or not SHA.fullmatch(sha):
        return False
    if event == "pull_request" and (not merge_sha or not SHA.fullmatch(merge_sha)):
        return False
    relevant = [r for r in runs if r.get("name") == "CI"
                and r.get("path") == CI_PATH and r.get("event") == event
                and r.get("head_branch") == BRANCH]
    if not relevant or any(r.get("status") != "completed" for r in relevant):
        return False
    latest = max(relevant, key=lambda r: (r.get("created_at", ""), r.get("id", 0)))
    allowed = {sha} if event == "push" else {sha, merge_sha}
    return (latest.get("head_sha") in allowed
            and latest.get("status") == "completed" and latest.get("conclusion") == "success")


def discover_milestone_pr(api: Github, sha: str) -> int:
    """Require exactly one open PR from this repository's protected branch."""
    owner = api.repo.split("/", 1)[0]
    query = urllib.parse.urlencode({"state": "open", "head": f"{owner}:{BRANCH}",
                                    "base": "main", "per_page": 100, "page": 1})
    records = api.request("pulls?" + query)
    # Multiple candidates (including a full page) are ambiguous. Never pick the
    # newest PR or silently fall back to a historical PR when discovery fails.
    if not isinstance(records, list) or len(records) != 1:
        raise GateError("expected exactly one open milestone PR")
    pr = records[0]
    try:
        number = pr["number"]
        valid = (isinstance(number, int) and not isinstance(number, bool) and number > 0
                 and pr["state"] == "open" and pr["head"]["ref"] == BRANCH
                 and pr["head"]["sha"] == sha
                 and pr["head"]["repo"]["full_name"] == api.repo
                 and pr["base"]["ref"] == "main"
                 and pr["base"]["repo"]["full_name"] == api.repo)
    except (KeyError, TypeError) as exc:
        raise GateError("malformed milestone PR discovery response") from exc
    if not valid:
        raise GateError("ambiguous or stale milestone PR discovery")
    return number


def ci_snapshot(api: Github, sha: str) -> dict:
    """Require one current same-repository PR and its exact synthetic merge."""
    pr_number = discover_milestone_pr(api, sha)
    pr = api.request(f"pulls/{pr_number}")
    base = api.request("branches/main")["commit"]["sha"]
    if (pr.get("state") != "open" or pr.get("merged") is not False
            or pr.get("number") != pr_number or pr.get("mergeable") is not True
            or pr["head"]["ref"] != BRANCH or pr["head"]["sha"] != sha
            or pr["head"]["repo"]["full_name"] != api.repo
            or pr["base"]["ref"] != "main" or pr["base"]["sha"] != base
            or pr["base"]["repo"]["full_name"] != api.repo):
        raise GateError("ambiguous or stale PR source/base provenance")
    merge = api.request(f"git/ref/pull/{pr_number}/merge")["object"]["sha"]
    if not SHA.fullmatch(merge) or pr.get("merge_commit_sha") != merge:
        raise GateError("stale PR merge ref")
    commit = api.request(f"git/commits/{merge}")
    if commit.get("sha") != merge or [p["sha"] for p in commit["parents"]] != [base, sha]:
        raise GateError("PR merge parents do not match current base and source")
    selected = {}
    for event in ("push", "pull_request"):
        query = urllib.parse.urlencode({"event": event, "branch": BRANCH, "per_page": 100})
        runs = api.request("actions/workflows/ci.yml/runs?" + query)["workflow_runs"]
        if not latest_ci_success(runs, sha, event=event, merge_sha=merge):
            raise GateError(f"latest {event} CI is not successful for current provenance")
        selected[event] = max(
            (r for r in runs if r.get("name") == "CI" and r.get("path") == CI_PATH
             and r.get("event") == event and r.get("head_branch") == BRANCH),
            key=lambda r: (r.get("created_at", ""), r.get("id", 0)))
    run = selected["pull_request"]
    associations = run.get("pull_requests", [])
    if len(associations) != 1 or associations[0].get("number") != pr_number:
        raise GateError("ambiguous PR run association")
    manifest = api.ci_provenance(run)
    expected = {"repository": api.repo, "event": "pull_request", "pr_number": pr_number,
                "source_sha": sha, "base_sha": base, "merge_sha": merge,
                "run_id": run["id"], "run_attempt": run["run_attempt"]}
    if manifest != expected:
        raise GateError("CI artifact does not prove current PR-merge provenance")
    return {"pr_number": pr_number, "source_sha": sha, "base_sha": base, "merge_sha": merge,
            "push_run_id": selected["push"]["id"], "pr_run_id": run["id"],
            "push_attempt": selected["push"]["run_attempt"], "pr_attempt": run["run_attempt"]}


def first_valid_artifact(artifacts: list[dict], sha: str, zip_reader) -> dict | None:
    """Require the artifact's *internal* commit marker to match current HEAD."""
    for artifact in artifacts:
        if artifact.get("expired") or not isinstance(artifact.get("id"), int):
            continue
        if not artifact.get("name", "").startswith(("bagi-research-", "bagi-")):
            continue
        try:
            marker = zip_reader(artifact["id"])
        except (GateError, OSError, ValueError, zipfile.BadZipFile):
            continue
        if marker == sha:
            return artifact
    return None


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward the GitHub bearer token to an artifact storage host."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and (
            urllib.parse.urlparse(newurl).netloc.lower()
            != urllib.parse.urlparse(req.full_url).netloc.lower()
        ):
            redirected.remove_header("Authorization")
        return redirected


class Github:
    def __init__(self, repo: str, token: str):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise GateError("invalid GITHUB_REPOSITORY")
        if not token:
            raise GateError("GITHUB_TOKEN missing")
        self.repo = repo
        self.token = token
        self.opener = urllib.request.build_opener(SafeRedirectHandler())

    def request(self, endpoint: str, *, binary: bool = False) -> bytes | dict | list:
        url = "https://api.github.com/repos/" + self.repo + "/" + endpoint
        request = urllib.request.Request(url, headers={
            "Authorization": "Bearer " + self.token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "bagi-evidence-gate/1.0",
        })
        for attempt in range(3):
            try:
                with self.opener.open(request, timeout=15) as response:
                    if binary:
                        raw = response.read(MAX_ZIP_BYTES + 1)
                        if len(raw) > MAX_ZIP_BYTES:
                            raise GateError("artifact zip exceeds safe size")
                        return raw
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                # Auth, billing, permission and missing-object problems are never retried.
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise GateError(f"GitHub API HTTP {exc.code} for {endpoint}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == 2:
                    raise GateError(f"GitHub API unavailable at {endpoint}") from exc
            time.sleep(2**attempt)
        raise GateError("unreachable retry state")

    def ci_provenance(self, run: dict) -> dict:
        name = f"bagi-ci-provenance-{run['id']}-{run['run_attempt']}"
        items = self.request(f"actions/runs/{run['id']}/artifacts?per_page=100")
        if items.get("total_count", 0) > 100:
            raise GateError("ambiguous CI artifact pagination")
        artifacts = [a for a in items["artifacts"] if a.get("name") == name]
        if len(artifacts) != 1 or artifacts[0].get("expired") is not False:
            raise GateError("missing unique current CI provenance artifact")
        raw = self.request(f"actions/artifacts/{artifacts[0]['id']}/zip", binary=True)
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if archive.namelist() != ["ci_provenance.json"]:
                    raise GateError("ambiguous CI provenance archive")
                if archive.getinfo("ci_provenance.json").file_size > 4096:
                    raise GateError("CI provenance manifest too large")
                return json.loads(archive.read("ci_provenance.json"))
        except (zipfile.BadZipFile, UnicodeError) as exc:
            raise GateError("invalid CI provenance archive") from exc

    def marker(self, artifact_id: int) -> str:
        data = self.request(f"actions/artifacts/{artifact_id}/zip", binary=True)
        assert isinstance(data, bytes)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            candidates = [n for n in archive.namelist() if n in (
                "research_branch_commit.txt", "research-loop/research_branch_commit.txt"
            )]
            if len(candidates) != 1:
                raise GateError("artifact lacks a unique source-commit marker")
            if archive.getinfo(candidates[0]).file_size > 128:
                raise GateError("commit marker too large")
            marker = archive.read(candidates[0]).decode("ascii").strip()
            if not SHA.fullmatch(marker):
                raise GateError("malformed artifact source commit")
            return marker


def inspect(api: Github) -> dict:
    branch = api.request("branches/" + urllib.parse.quote(BRANCH, safe=""))
    sha = branch["commit"]["sha"]
    if not SHA.fullmatch(sha):
        raise GateError("invalid protected branch HEAD")
    snapshot = ci_snapshot(api, sha)
    # Scheduled research workflows have main as their workflow head; their
    # internal research_branch_commit.txt establishes the executed source.
    records = api.request("actions/runs?per_page=100")["workflow_runs"]
    if any(r.get("name") in ALLOWED_WORKFLOWS and r.get("status") != "completed"
           for r in records):
        raise GateError("active research workflow")
    candidates = [r for r in records if r.get("name") in ALLOWED_WORKFLOWS
                  and r.get("status") == "completed" and r.get("conclusion") == "success"]
    candidates.sort(key=lambda r: (r.get("created_at", ""), r.get("id", 0)), reverse=True)
    for run in candidates[:30]:
        items = api.request(f"actions/runs/{run['id']}/artifacts")["artifacts"]
        artifact = first_valid_artifact(items, sha, api.marker)
        if artifact is not None:
            # Recheck source HEAD in case another writer moved the branch.
            again = api.request("branches/" + urllib.parse.quote(BRANCH, safe=""))["commit"]["sha"]
            if again != sha or ci_snapshot(api, sha) != snapshot:
                raise GateError("research branch changed during preflight")
            return {"research_sha": sha, "evidence_run_id": run["id"],
                    "evidence_artifact_id": artifact["id"],
                    "evidence_artifact_name": artifact["name"],
                    "cache_key": f"bagi-agent-reviewed-{sha}-{artifact['id']}"}
    raise GateError("no successful same-commit research artifact with verified marker")


def main() -> int:
    try:
        data = inspect(Github(os.environ.get("GITHUB_REPOSITORY", ""),
                              os.environ.get("GH_TOKEN", os.environ.get("GITHUB_TOKEN", ""))))
    except (GateError, KeyError, TypeError, ValueError) as exc:
        print(f"BAGI HOLD: {exc}", file=sys.stderr)
        return 2
    data["has_evidence"] = "true"
    for k, v in data.items():
        print(f"{k}={v}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            for k, v in data.items():
                print(f"{k}={v}", file=output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
