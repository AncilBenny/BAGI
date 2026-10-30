"""Scheduled maintenance must never restart stopped scientific work."""
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class ScheduledSafetyTests(unittest.TestCase):
    def test_schedules_only_reach_read_only_audit(self):
        for filename, cron, blocked in (
            ('bagi-research-loop.yml', '17 * * * *', 'research'),
            ('bagi-autonomous-agent.yml', '7 */2 * * *', 'gate'),
        ):
            with self.subTest(workflow=filename):
                data = yaml.safe_load((ROOT / '.github/workflows' / filename).read_text())
                triggers = data.get('on', data.get(True))
                self.assertEqual(set(triggers), {'schedule', 'workflow_dispatch'})
                self.assertEqual(triggers['schedule'], [{'cron': cron}])
                self.assertEqual(data['jobs'][blocked]['if'], '${{ false }}')
                audit = data['jobs']['audit']
                self.assertEqual(audit['permissions'], {'contents': 'read', 'actions': 'read'})
                self.assertEqual(audit['timeout-minutes'], 5)
                self.assertEqual(len(audit['steps']), 2)
                self.assertEqual(audit['steps'][0]['uses'], 'actions/checkout@v4')
                self.assertFalse(audit['steps'][0]['with']['persist-credentials'])
                code = audit['steps'][1]['run']
                self.assertIn('ci_snapshot(api, sha)', code)
                self.assertIn('Scientific progression remains HOLD', code)
                for forbidden in ('subprocess', 'pip install', 'make ', 'copilot ', 'git push'):
                    self.assertNotIn(forbidden, code)
                compile(code.split("<<'PYTHON'\n", 1)[1].rsplit('PYTHON', 1)[0], filename, 'exec')

    def test_autonomous_write_jobs_depend_on_blocked_gate(self):
        data = yaml.safe_load(
            (ROOT / '.github/workflows/bagi-autonomous-agent.yml').read_text())
        self.assertEqual(data['jobs']['propose']['needs'], 'gate')
        self.assertEqual(data['jobs']['validate']['needs'], 'propose')
        self.assertEqual(data['jobs']['commit']['needs'], ['propose', 'validate'])
        for job in ('propose', 'validate', 'commit'):
            self.assertNotIn('always()', data['jobs'][job]['if'])
