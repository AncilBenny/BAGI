"""Tests for the fail-closed, public-source BAGI evidence gate."""
import copy
import io
import json
import sys
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from bagi_evidence_gate import (
    GateError,
    Github,
    ci_snapshot,
    discover_milestone_pr,
    first_valid_artifact,
    inspect,
    latest_ci_success,
)

SHA = 'a' * 40
OLD = 'b' * 40


BASE = 'c' * 40
MERGE = 'd' * 40


class FakeAPI:
    repo = 'owner/BAGI'

    def __init__(self, *, ci='success', artifact_sha=SHA, branch_sha=SHA, pr_number=1):
        self.artifact_sha, self.branch_sha = artifact_sha, branch_sha
        self.pr = {'state': 'open', 'merged': False, 'number': pr_number, 'mergeable': True,
                   'head': {'ref': 'setup/initial-platform', 'sha': SHA,
                            'repo': {'full_name': self.repo}},
                   'base': {'ref': 'main', 'sha': BASE, 'repo': {'full_name': self.repo}},
                   'merge_commit_sha': MERGE}
        self.merge = MERGE
        self.parents = [BASE, SHA]
        self.runs = {event: [{'name': 'CI', 'path': '.github/workflows/ci.yml',
                             'head_branch': 'setup/initial-platform', 'head_sha': SHA,
                             'event': event, 'status': 'completed', 'conclusion': ci,
                             'created_at': '2026-10-28', 'id': i, 'run_attempt': 1,
                             'pull_requests': [{'number': pr_number}]}]
                     for i, event in enumerate(('push', 'pull_request'), 10)}
        self.manifest = {'repository': self.repo, 'event': 'pull_request', 'pr_number': pr_number,
                         'source_sha': SHA, 'base_sha': BASE, 'merge_sha': MERGE,
                         'run_id': 11, 'run_attempt': 1}
        self.open_prs = [self.pr]
        self.requests = []
        self.ci_reads = 0

    def request(self, endpoint, *, binary=False):
        self.requests.append(endpoint)
        if endpoint.startswith('pulls?'):
            return self.open_prs
        if endpoint == 'branches/main':
            return {'commit': {'sha': BASE}}
        if endpoint.startswith('branches/'):
            return {'commit': {'sha': self.branch_sha}}
        if endpoint == f"pulls/{self.pr['number']}":
            return self.pr
        if endpoint == f"git/ref/pull/{self.pr['number']}/merge":
            return {'object': {'sha': self.merge}}
        if endpoint.startswith('git/commits/'):
            return {'sha': self.merge, 'parents': [{'sha': p} for p in self.parents]}
        if endpoint.startswith('actions/workflows/ci.yml/runs?'):
            self.ci_reads += 1
            return {'workflow_runs': self.runs['pull_request' if 'event=pull_request&' in endpoint
                                               else 'push']}
        if endpoint == 'actions/runs?per_page=100':
            return {'workflow_runs': [{'name': 'BAGI Research Loop', 'id': 123,
                    'status': 'completed', 'conclusion': 'success', 'created_at': '2026-10-27'}]}
        if endpoint == 'actions/runs/123/artifacts':
            return {'artifacts': [{'id': 45, 'name': 'bagi-research-1', 'expired': False}]}
        raise AssertionError(endpoint)

    def ci_provenance(self, run):
        return self.manifest

    def marker(self, artifact_id):
        return self.artifact_sha


class GateTests(unittest.TestCase):
    def test_accepts_only_current_successful_ci(self):
        items = FakeAPI().runs['push']
        self.assertTrue(latest_ci_success(items, SHA, event="push"))
        failed = copy.deepcopy(items[0])
        failed.update(conclusion='failure', created_at='2026-10-29', id=99)
        items.append(failed)
        self.assertFalse(latest_ci_success(items, SHA, event="push"))
        self.assertFalse(latest_ci_success(items, OLD, event="push"))

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


class ProvenanceTests(unittest.TestCase):
    def test_both_event_modes_and_merge_checkout(self):
        for tested_sha in (SHA, MERGE):
            api = FakeAPI()
            api.runs['pull_request'][0]['head_sha'] = tested_sha
            self.assertEqual(ci_snapshot(api, SHA)['pr_run_id'], 11)

    def test_events_cannot_mask_each_other(self):
        for failed in ('push', 'pull_request'):
            api = FakeAPI()
            api.runs[failed][0]['conclusion'] = 'failure'
            with self.assertRaisesRegex(GateError, failed):
                ci_snapshot(api, SHA)

    def test_missing_stale_wrong_or_incomplete_event_holds(self):
        for event in ('push', 'pull_request'):
            for field, value in (('head_sha', OLD), ('event', 'workflow_dispatch'),
                                 ('path', 'evil.yml'), ('head_branch', 'other'),
                                 ('status', 'queued'), ('status', 'in_progress'),
                                 ('status', 'waiting'), ('conclusion', None)):
                with self.subTest(event=event, field=field, value=value):
                    api = FakeAPI()
                    api.runs[event][0][field] = value
                    with self.assertRaises(GateError):
                        ci_snapshot(api, SHA)
            api = FakeAPI()
            api.runs[event] = []
            with self.assertRaises(GateError):
                ci_snapshot(api, SHA)

    def test_latest_failure_and_active_old_run_block_success(self):
        for event in ('push', 'pull_request'):
            for status, conclusion in (('completed', 'failure'), ('queued', None)):
                api = FakeAPI()
                run = copy.deepcopy(api.runs[event][0])
                run.update(id=99, created_at='2026-10-29', status=status, conclusion=conclusion)
                api.runs[event].append(run)
                with self.assertRaises(GateError):
                    ci_snapshot(api, SHA)

    def test_ambiguous_or_stale_pr_snapshot_holds(self):
        for mutation in (
                lambda a: a.pr.update(mergeable=None),
                lambda a: a.pr.update(mergeable=False),
                lambda a: a.pr['head'].update(sha=OLD),
                lambda a: a.pr['base'].update(sha=OLD),
                lambda a: a.pr['head']['repo'].update(full_name='fork/BAGI'),
                lambda a: a.pr.update(merge_commit_sha=OLD),
                lambda a: setattr(a, 'merge', OLD),
                lambda a: setattr(a, 'parents', [BASE, OLD]),
                lambda a: a.runs['pull_request'][0].update(pull_requests=[]),
                lambda a: a.runs['pull_request'][0].update(pull_requests=[{'number': 2}]),
                lambda a: a.runs['pull_request'][0].update(
                    pull_requests=[{'number': 1}, {'number': 2}])):
            api = FakeAPI()
            mutation(api)
            with self.assertRaises(GateError):
                ci_snapshot(api, SHA)

    def test_manifest_must_match_every_provenance_field(self):
        for key in FakeAPI().manifest:
            api = FakeAPI()
            api.manifest[key] = 'wrong'
            with self.assertRaisesRegex(GateError, 'artifact'):
                ci_snapshot(api, SHA)

    def test_rechecks_ci_after_research_artifact_download(self):
        api = FakeAPI()
        def changed_marker(_):
            api.runs['push'][0]['status'] = 'queued'
            return SHA
        api.marker = changed_marker
        with self.assertRaises(GateError):
            inspect(api)

    def test_missing_ci_manifest_holds(self):
        api = FakeAPI()
        def absent(_):
            raise GateError('missing provenance')
        api.ci_provenance = absent
        with self.assertRaises(GateError):
            inspect(api)

    def test_ci_archive_requires_unique_unexpired_bounded_manifest(self):
        api = object.__new__(Github)
        run = {'id': 11, 'run_attempt': 1}
        name = 'bagi-ci-provenance-11-1'
        for entries in ([], [{'id': 1, 'name': name, 'expired': True}],
                        [{'id': i, 'name': name, 'expired': False} for i in (1, 2)]):
            api.request = lambda endpoint, entries=entries: {'artifacts': entries}
            with self.assertRaises(GateError):
                api.ci_provenance(run)
        for filename, body, valid in (('ci_provenance.json', '{}', True),
                                      ('wrong.json', '{}', False),
                                      ('ci_provenance.json', 'x' * 4097, False)):
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, 'w') as archive:
                archive.writestr(filename, body)
            def request(endpoint, binary=False, stream=stream):
                if binary:
                    return stream.getvalue()
                return {'artifacts': [{'id': 1, 'name': name, 'expired': False}]}
            api.request = request
            if valid:
                self.assertEqual(api.ci_provenance(run), json.loads(body))
            else:
                with self.assertRaises(GateError):
                    api.ci_provenance(run)


class MilestoneDiscoveryTests(unittest.TestCase):
    def test_future_prs_use_their_own_merge_ref_run_association_and_manifest(self):
        for number in (7, 42, 1234):
            with self.subTest(pr_number=number):
                api = FakeAPI(pr_number=number)
                snapshot = ci_snapshot(api, SHA)
                self.assertEqual(snapshot['pr_number'], number)
                self.assertIn(f'pulls/{number}', api.requests)
                self.assertIn(f'git/ref/pull/{number}/merge', api.requests)
                self.assertNotIn('pulls/1', api.requests)
                self.assertNotIn('git/ref/pull/1/merge', api.requests)
                self.assertEqual(api.requests[0], 'pulls?state=open&head=owner%3Asetup%2F'
                                 'initial-platform&base=main&per_page=100&page=1')

    def test_missing_multiple_or_malformed_discovery_holds_without_fallback(self):
        for records in ([], [{}, {}], [{}] * 100, {}, [None], [{}]):
            with self.subTest(records=records):
                api = FakeAPI(pr_number=42)
                api.open_prs = records
                with self.assertRaises(GateError):
                    ci_snapshot(api, SHA)
                self.assertFalse(any(p.startswith('pulls/') for p in api.requests))

    def test_discovery_rejects_wrong_source_target_fork_or_closed_pr(self):
        mutations = (
            lambda p: p.update(state='closed'),
            lambda p: p.update(number=0),
            lambda p: p.update(number=True),
            lambda p: p.update(number='42'),
            lambda p: p['head'].update(sha=OLD),
            lambda p: p['head'].update(ref='other'),
            lambda p: p['head'].update(repo=None),
            lambda p: p['head']['repo'].update(full_name='fork/BAGI'),
            lambda p: p['base'].update(ref='other'),
            lambda p: p['base']['repo'].update(full_name='other/BAGI'),
        )
        for mutation in mutations:
            api = FakeAPI(pr_number=42)
            mutation(api.pr)
            with self.assertRaises(GateError):
                discover_milestone_pr(api, SHA)

    def test_discovery_and_detail_must_agree(self):
        api = FakeAPI(pr_number=42)
        api.open_prs = [copy.deepcopy(api.pr)]
        api.pr['number'] = 43
        request = api.request
        def changed_detail(endpoint, **kwargs):
            if endpoint == 'pulls/42':
                return api.pr
            return request(endpoint, **kwargs)
        api.request = changed_detail
        with self.assertRaisesRegex(GateError, 'PR source/base'):
            ci_snapshot(api, SHA)

    def test_old_pr_association_or_manifest_cannot_authorize_new_milestone(self):
        for old_evidence in ('association', 'manifest'):
            api = FakeAPI(pr_number=42)
            if old_evidence == 'association':
                api.runs['pull_request'][0]['pull_requests'] = [{'number': 1}]
            else:
                api.manifest['pr_number'] = 1
            with self.assertRaises(GateError):
                ci_snapshot(api, SHA)

    def test_future_milestone_keeps_all_ci_failure_modes_closed(self):
        for event in ('push', 'pull_request'):
            for field, value in (('conclusion', 'failure'), ('head_sha', OLD),
                                 ('status', 'queued'), ('status', 'in_progress'),
                                 ('status', 'waiting'), ('conclusion', None)):
                with self.subTest(event=event, field=field, value=value):
                    api = FakeAPI(pr_number=42)
                    api.runs[event][0][field] = value
                    with self.assertRaises(GateError):
                        ci_snapshot(api, SHA)

    def test_pr_handoff_during_evidence_download_holds(self):
        api = FakeAPI(pr_number=42)
        def switched_pr(_):
            api.pr['number'] = 43
            api.runs['pull_request'][0]['pull_requests'] = [{'number': 43}]
            api.manifest['pr_number'] = 43
            return SHA
        api.marker = switched_pr
        with self.assertRaisesRegex(GateError, 'changed during preflight'):
            inspect(api)


    def test_future_milestone_missing_runs_or_ambiguous_associations_hold(self):
        for event in ('push', 'pull_request'):
            api = FakeAPI(pr_number=42)
            api.runs[event] = []
            with self.assertRaises(GateError):
                ci_snapshot(api, SHA)
        for associations in ([], [{'number': 42}, {'number': 43}]):
            api = FakeAPI(pr_number=42)
            api.runs['pull_request'][0]['pull_requests'] = associations
            with self.assertRaises(GateError):
                ci_snapshot(api, SHA)

    def test_future_milestone_stale_base_merge_or_manifest_holds(self):
        for mutation in (
                lambda a: a.pr['base'].update(sha=OLD),
                lambda a: a.pr.update(merge_commit_sha=OLD),
                lambda a: setattr(a, 'parents', [BASE, OLD]),
                lambda a: a.manifest.update(source_sha=OLD),
                lambda a: a.manifest.update(merge_sha=OLD),
                lambda a: a.pr.update(mergeable=None)):
            api = FakeAPI(pr_number=42)
            mutation(api)
            with self.assertRaises(GateError):
                ci_snapshot(api, SHA)
