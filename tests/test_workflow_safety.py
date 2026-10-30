"""Research is admitted through CI and budgets instead of blanket bans."""
import json
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class ScheduledSafetyTests(unittest.TestCase):
    def test_scheduled_research_requires_ci_and_compute_admission(self):
        for filename, cron, blocked in (
            ('bagi-research-loop.yml', '17 * * * *', 'research'),
            ('bagi-autonomous-agent.yml', '7 */2 * * *', 'gate'),
        ):
            with self.subTest(workflow=filename):
                data = yaml.safe_load((ROOT / '.github/workflows' / filename).read_text())
                triggers = data.get('on', data.get(True))
                self.assertEqual(set(triggers), {'schedule', 'workflow_dispatch'})
                self.assertEqual(triggers['schedule'], [{'cron': cron}])
                self.assertEqual(data['jobs'][blocked]['needs'], ['audit', 'budget'])
                self.assertEqual(data['jobs'][blocked]['if'],
                                 "needs.budget.outputs.eligible == 'true'")
                budget = data['jobs']['budget']
                self.assertEqual(budget['needs'], 'audit')
                self.assertEqual(budget['permissions'],
                                 {'contents': 'read', 'actions': 'read'})
                self.assertIn('bagi_research_budget.py', budget['steps'][1]['run'])
                audit = data['jobs']['audit']
                self.assertEqual(audit['permissions'], {'contents': 'read', 'actions': 'read'})
                self.assertEqual(audit['timeout-minutes'], 5)
                self.assertEqual(len(audit['steps']), 2)
                self.assertEqual(audit['steps'][0]['uses'], 'actions/checkout@v4')
                self.assertFalse(audit['steps'][0]['with']['persist-credentials'])
                code = audit['steps'][1]['run']
                self.assertIn('ci_snapshot(api, sha)', code)
                self.assertIn('this does not establish scientific progress', code)
                for forbidden in ('subprocess', 'pip install', 'make ', 'copilot ', 'git push'):
                    self.assertNotIn(forbidden, code)
                compile(code.split("<<'PYTHON'\n", 1)[1].rsplit('PYTHON', 1)[0], filename, 'exec')

    def test_autonomous_writes_still_require_evidence_and_validation(self):
        data = yaml.safe_load(
            (ROOT / '.github/workflows/bagi-autonomous-agent.yml').read_text())
        self.assertEqual(data['jobs']['propose']['needs'], 'gate')
        self.assertEqual(data['jobs']['validate']['needs'], 'propose')
        self.assertEqual(data['jobs']['commit']['needs'], ['propose', 'validate'])
        for job in ('propose', 'validate', 'commit'):
            self.assertNotIn('always()', data['jobs'][job]['if'])


    def test_policy_and_workflow_timeouts_agree(self):
        policy = json.loads((ROOT / '.github/research-policy.json').read_text())
        self.assertEqual(policy['schema_version'], 1)
        for key in ('research_runs_per_utc_day', 'ai_proposals_per_utc_day'):
            self.assertIs(type(policy[key]), int)
            self.assertGreater(policy[key], 0)
        research = yaml.safe_load(
            (ROOT / '.github/workflows/bagi-research-loop.yml').read_text())['jobs']['research']
        agent = yaml.safe_load(
            (ROOT / '.github/workflows/bagi-autonomous-agent.yml').read_text())['jobs']
        self.assertEqual(research['timeout-minutes'], policy['research_timeout_minutes'])
        self.assertEqual(agent['propose']['timeout-minutes'], policy['ai_timeout_minutes'])
        checkout = research['steps'][0]
        self.assertEqual(checkout['with']['ref'], '${{ needs.audit.outputs.research_sha }}')
        self.assertFalse(checkout['with']['persist-credentials'])
        self.assertIn('Recheck exact source and CI', research['steps'][1]['name'])
        self.assertIn('ci_snapshot(api, sha)', research['steps'][1]['run'])
        self.assertEqual(research['permissions'], {'contents': 'read', 'actions': 'read'})
        self.assertIn('bagi_evidence_gate.py', agent['gate']['steps'][1]['run'])
