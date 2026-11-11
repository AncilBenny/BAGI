"""No custom quotas; publication and branch-integrity checks remain intact."""
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class ScheduledSafetyTests(unittest.TestCase):
    def test_scheduled_research_requires_ci_without_daily_quotas(self):
        for filename, cron, blocked in (
            ('bagi-research-loop.yml', '17 * * * *', 'research'),
            ('bagi-autonomous-agent.yml', '7 */2 * * *', 'gate'),
        ):
            with self.subTest(workflow=filename):
                data = yaml.safe_load((ROOT / '.github/workflows' / filename).read_text())
                triggers = data.get('on', data.get(True))
                expected_triggers = {'schedule', 'workflow_dispatch'}
                if filename == 'bagi-research-loop.yml':
                    expected_triggers.add('push')
                self.assertEqual(set(triggers), expected_triggers)
                self.assertEqual(triggers['schedule'], [{'cron': cron}])
                self.assertEqual(data['jobs'][blocked]['needs'], 'audit')
                self.assertEqual(data['jobs'][blocked]['if'],
                                 "needs.audit.result == 'success'")
                self.assertNotIn('budget', data['jobs'])
                for job in data['jobs'].values():
                    self.assertNotIn('timeout-minutes', job)
                audit = data['jobs']['audit']
                self.assertEqual(audit['permissions'], {'contents': 'read', 'actions': 'read'})
                expected_steps = 4 if filename == 'bagi-research-loop.yml' else 2
                self.assertEqual(len(audit['steps']), expected_steps)
                self.assertEqual(audit['steps'][0]['uses'], 'actions/checkout@v4')
                self.assertFalse(audit['steps'][0]['with']['persist-credentials'])
                if filename == 'bagi-research-loop.yml':
                    guard = audit['steps'][1]
                    self.assertEqual(guard['name'],
                                     'Fail closed on missing or unknown manual study')
                    self.assertEqual(guard['run'], 'python3 scripts/bagi_research_dispatch.py')
                    self.assertEqual(guard['env']['BAGI_DISPATCH_STUDY'],
                                     '${{ inputs.study }}')
                code = audit['steps'][-1]['run']
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


    def test_research_remains_pinned_and_rechecked_without_quota_files(self):
        research = yaml.safe_load(
            (ROOT / '.github/workflows/bagi-research-loop.yml').read_text())['jobs']['research']
        agent = yaml.safe_load(
            (ROOT / '.github/workflows/bagi-autonomous-agent.yml').read_text())['jobs']
        checkout = research['steps'][0]
        self.assertEqual(checkout['with']['ref'], '${{ needs.audit.outputs.research_sha }}')
        self.assertFalse(checkout['with']['persist-credentials'])
        self.assertIn('Recheck exact source and CI', research['steps'][1]['name'])
        self.assertIn('ci_snapshot(api, sha)', research['steps'][1]['run'])
        self.assertEqual(research['permissions'], {'contents': 'read', 'actions': 'read'})
        self.assertIn('bagi_evidence_gate.py', agent['gate']['steps'][1]['run'])
        self.assertFalse((ROOT / '.github/research-policy.json').exists())
        self.assertFalse((ROOT / 'scripts/bagi_research_budget.py').exists())
