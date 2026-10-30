"""Bound compute even across failures, source changes and workflow reruns."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from bagi_evidence_gate import GateError
from bagi_research_budget import WORK, budget_available

DAY = '2026-10-31'


class BudgetAPI:
    def __init__(self, kind='research', status='completed', conclusion='success'):
        self.records = {'total_count': 2, 'workflow_runs': [
            {'id': 10, 'created_at': DAY + 'T01:00:00Z', 'run_attempt': 1},
            {'id': 9, 'created_at': DAY + 'T00:00:00Z', 'run_attempt': 1}]}
        self.jobs = {'total_count': 1, 'jobs': [
            {'name': WORK[kind][1], 'status': status, 'conclusion': conclusion}]}
        self.requests = []

    def request(self, endpoint):
        self.requests.append(endpoint)
        if endpoint.startswith('actions/workflows/'):
            return self.records
        if endpoint == 'actions/runs/9/jobs?per_page=100':
            return self.jobs
        raise AssertionError(endpoint)


class BudgetTests(unittest.TestCase):
    def test_completed_failed_cancelled_and_pending_work_reserve_budget(self):
        for kind in WORK:
            for status, conclusion in (('completed', 'success'), ('completed', 'failure'),
                                       ('completed', 'cancelled'), ('in_progress', None),
                                       ('queued', None)):
                with self.subTest(kind=kind, status=status, conclusion=conclusion):
                    api = BudgetAPI(kind, status, conclusion)
                    self.assertFalse(budget_available(api, kind, 10, 1, DAY))
                    self.assertTrue(budget_available(api, kind, 10, 2, DAY))

    def test_skipped_jobs_and_first_attempt_do_not_consume_budget(self):
        for kind in WORK:
            api = BudgetAPI(kind, conclusion='skipped')
            self.assertTrue(budget_available(api, kind, 10, 1, DAY))
            self.assertIn('created=%3E%3D' + DAY, api.requests[0])
            self.assertFalse(any('/10/jobs' in p for p in api.requests))
            api.records['workflow_runs'] = api.records['workflow_runs'][:1]
            api.records['total_count'] = 1
            self.assertTrue(budget_available(api, kind, 10, 1, DAY))

    def test_rerunning_current_workflow_cannot_bypass_daily_cap(self):
        api = BudgetAPI(conclusion='skipped')
        api.records['workflow_runs'][0]['run_attempt'] = 2
        self.assertFalse(budget_available(api, 'research', 10, 1, DAY))

    def test_skipped_latest_attempt_cannot_erase_prior_compute(self):
        api = BudgetAPI(conclusion='skipped')
        api.records['workflow_runs'][1]['run_attempt'] = 2
        self.assertFalse(budget_available(api, 'research', 10, 1, DAY))

    def test_missing_ambiguous_truncated_or_wrong_date_history_fails_closed(self):
        baseline = BudgetAPI()
        for mutate in (
            lambda a: a.records.update(total_count=101),
            lambda a: a.records['workflow_runs'][0].update(run_attempt=None),
            lambda a: a.records['workflow_runs'][1].update(run_attempt=0),
            lambda a: a.records.update(total_count=3),
            lambda a: a.records['workflow_runs'][0].update(id=8),
            lambda a: a.records['workflow_runs'][1].update(id=10),
            lambda a: a.records['workflow_runs'][1].update(created_at='2026-10-30T00:00:00Z'),
            lambda a: a.jobs.update(total_count=101),
            lambda a: a.jobs.update(total_count=0, jobs=[]),
            lambda a: a.jobs.update(total_count=2),
            lambda a: a.jobs['jobs'][0].update(name='other'),
            lambda a: a.jobs['jobs'][0].update(status='unknown'),
        ):
            api = copy.deepcopy(baseline)
            mutate(api)
            with self.assertRaises(GateError):
                budget_available(api, 'research', 10, 1, DAY)

    def test_invalid_limits_do_not_enable_unbounded_compute(self):
        for limit in (0, -1, True, '1', None):
            with self.assertRaises(GateError):
                budget_available(BudgetAPI(), 'research', 10, limit, DAY)
        with self.assertRaises(GateError):
            budget_available(BudgetAPI(), 'unknown', 10, 1, DAY)
