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
MAX_ZIP_BYTES = 12 * 1024 * 1024


class GateError(RuntimeError):
    pass


def latest_ci_success(runs: list[dict], sha: str) -> bool:
    """Latest CI run for this exact commit must be completed and successful."""
    ci = [r for r in runs if r.get("name") == "CI" and r.get("head_sha") == sha]
    if not ci:
        return False
    latest = max(ci, key=lambda r: (r.get("created_at", ""), r.get("id", 0)))
    return latest.get("status") == "completed" and latest.get("conclusion") == "success"


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

    def request(self, endpoint: str, *, binary: bool = False) -> bytes | dict:
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
    query = urllib.parse.urlencode({"head_sha": sha, "per_page": 100})
    runs = api.request("actions/runs?" + query)["workflow_runs"]
    if not latest_ci_success(runs, sha):
        raise GateError("latest exact-commit CI is not successful")
    if any(r.get("status") != "completed" and r.get("head_sha") == sha
           and r.get("name") in {"CI", "BAGI E6 Imagination Gap"} for r in runs):
        raise GateError("active CI or E6 run for current commit")
    # Scheduled research workflows have main as their workflow head; their
    # internal research_branch_commit.txt establishes the executed source.
    records = api.request("actions/runs?per_page=100")["workflow_runs"]
    candidates = [r for r in records if r.get("name") in ALLOWED_WORKFLOWS
                  and r.get("status") == "completed" and r.get("conclusion") == "success"]
    candidates.sort(key=lambda r: (r.get("created_at", ""), r.get("id", 0)), reverse=True)
    for run in candidates[:30]:
        items = api.request(f"actions/runs/{run['id']}/artifacts")["artifacts"]
        artifact = first_valid_artifact(items, sha, api.marker)
        if artifact is not None:
            # Recheck source HEAD in case another writer moved the branch.
            again = api.request("branches/" + urllib.parse.quote(BRANCH, safe=""))["commit"]["sha"]
            if again != sha:
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
