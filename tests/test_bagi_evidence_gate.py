"""Tests for the fail-closed, public-source BAGI evidence gate."""
import io
import sys
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from bagi_evidence_gate import GateError, Github, first_valid_artifact, inspect, latest_ci_success

SHA = 'a' * 40
OLD = 'b' * 40


class FakeAPI:
    def __init__(self, *, ci='success', artifact_sha=SHA, branch_sha=SHA):
        self.ci, self.artifact_sha, self.branch_sha = ci, artifact_sha, branch_sha

    def request(self, endpoint, *, binary=False):
        if endpoint.startswith('branches/'):
            return {'commit': {'sha': self.branch_sha}}
        if endpoint.startswith('actions/runs?head_sha='):
            return {'workflow_runs': [{'name': 'CI', 'head_sha': SHA,
                       'created_at': '2026-10-28', 'status': 'completed',
                       'conclusion': self.ci, 'id': 10}]}
        if endpoint == 'actions/runs?per_page=100':
            return {'workflow_runs': [{'name': 'BAGI Research Loop', 'id': 123,
                    'status': 'completed', 'conclusion': 'success', 'created_at': '2026-10-27'}]}
        if endpoint == 'actions/runs/123/artifacts':
            return {'artifacts': [{'id': 45, 'name': 'bagi-research-1', 'expired': False}]}
        raise AssertionError(endpoint)

    def marker(self, artifact_id):
        return self.artifact_sha


class GateTests(unittest.TestCase):
    def test_accepts_only_current_successful_ci(self):
        items = [{'name': 'CI', 'head_sha': SHA, 'status': 'completed',
                  'conclusion': 'success', 'created_at': '2026-10-27', 'id': 1},
                 {'name': 'CI', 'head_sha': SHA, 'status': 'completed',
                  'conclusion': 'failure', 'created_at': '2026-10-28', 'id': 2}]
        self.assertFalse(latest_ci_success(items, SHA))
        self.assertFalse(latest_ci_success(items, OLD))

    def test_refuses_stale_artifacts(self):
        items = [{'name': 'bagi-research-1', 'id': 1, 'expired': False}]
        self.assertIsNone(first_valid_artifact(items, SHA, lambda _: OLD))
        self.assertEqual(first_valid_artifact(items, SHA, lambda _: SHA)['id'], 1)

    def test_holds_when_ci_failed(self):
        with self.assertRaisesRegex(GateError, 'CI is not successful'):
            inspect(FakeAPI(ci='failure'))

    def test_holds_when_artifact_stale(self):
        with self.assertRaisesRegex(GateError, 'no successful same-commit'):
            inspect(FakeAPI(artifact_sha=OLD))

    def test_verified_current_artifact(self):
        result = inspect(FakeAPI())
        self.assertEqual(result['research_sha'], SHA)
        self.assertEqual(result['evidence_run_id'], 123)
        self.assertEqual(result['cache_key'], f'bagi-agent-reviewed-{SHA}-45')

    def test_marker_validation_rejects_missing_or_oversized_marker(self):
        def make_zip(name, body):
            fp = io.BytesIO()
            with zipfile.ZipFile(fp, 'w') as z:
                z.writestr(name, body)
            return fp.getvalue()
        api = object.__new__(Github)
        api.request = lambda endpoint, binary=False: make_zip('research_branch_commit.txt', SHA+'\n')
        self.assertEqual(api.marker(22), SHA)
        api.request = lambda endpoint, binary=False: make_zip('wrong.txt', SHA)
        with self.assertRaises(GateError):
            api.marker(22)
        api.request = lambda endpoint, binary=False: make_zip('research_branch_commit.txt', 'x' * 129)
        with self.assertRaises(GateError):
            api.marker(22)
