"""No custom quotas; publication and branch-integrity checks remain intact."""
import os
import subprocess
import sys
import tempfile
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
                condition = "needs.audit.result == 'success'"
                if filename == 'bagi-research-loop.yml':
                    condition += " && github.event_name != 'schedule'"
                self.assertEqual(data['jobs'][blocked]['if'], condition)
                self.assertEqual(data['concurrency']['group'], 'bagi-coordinated-development')
                self.assertFalse(data['concurrency']['cancel-in-progress'])
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
                self.assertIn('file=sys.stderr', code)
                self.assertIn('this does not establish scientific progress', code)
                for forbidden in ('subprocess', 'pip install', 'make ', 'copilot ', 'git push'):
                    self.assertNotIn(forbidden, code)
                compile(code.split("<<'PYTHON'\n", 1)[1].rsplit('PYTHON', 1)[0], filename, 'exec')

    def test_convergence_runtime_evidence_and_historical_native_workflow(self):
        runner = yaml.safe_load(
            (ROOT / '.github/workflows/bagi-research-loop.yml').read_text())
        contract = runner['jobs']['runtime_contract']
        self.assertEqual(contract['needs'], 'audit')
        self.assertEqual(contract['if'],
                         "needs.audit.result == 'success' && github.event_name == 'schedule'")
        self.assertEqual(contract['permissions'], {'contents': 'read', 'actions': 'read'})
        self.assertEqual(contract['steps'][0]['with']['ref'],
                         '${{ needs.audit.outputs.research_sha }}')
        self.assertIn('ci_snapshot(api, sha)', contract['steps'][1]['run'])
        self.assertIn('scientific_result', contract['steps'][3]['run'])
        self.assertIn('bagi-runtime-contract-', contract['steps'][4]['with']['name'])
        legacy = yaml.safe_load(
            (ROOT / '.github/workflows/bagi-general-brain-native-g1.yml').read_text())
        self.assertEqual(set(legacy.get('on', legacy.get(True))), {'workflow_dispatch'})
        self.assertIn('Historical G1 input unavailable', legacy['jobs']['two-real-worlds']
                      ['steps'][2]['run'])

    def test_autonomous_writes_still_require_evidence_and_validation(self):
        data = yaml.safe_load(
            (ROOT / '.github/workflows/bagi-autonomous-agent.yml').read_text())
        self.assertEqual(data['jobs']['propose']['needs'], 'gate')
        self.assertEqual(data['jobs']['validate']['needs'], 'propose')
        self.assertEqual(data['jobs']['commit']['needs'], ['propose', 'validate'])
        for job in ('propose', 'validate', 'commit'):
            self.assertNotIn('always()', data['jobs'][job]['if'])


    def test_agent_rechecks_real_multiroot_artifact_marker_and_rejects_ambiguity(self):
        data = yaml.safe_load(
            (ROOT / '.github/workflows/bagi-autonomous-agent.yml').read_text())
        steps = data['jobs']['propose']['steps']
        script = next(step['run'] for step in steps
                      if step.get('name') == 'Download only the previously verified same-commit artifact')
        self.assertIn('gh run download "$RUN_ID"', script)
        self.assertIn('test "$(git rev-parse HEAD)" = "$RESEARCH_SHA"', script)
        snippet = script.split("python3 - <<'PYTHON'\n", 1)[1].split('\nPYTHON', 1)[0]
        compile(snippet, 'agent-extracted-marker', 'exec')
        sha = 'a' * 40
        layouts = (
            'research_branch_commit.txt',
            'research-loop/research_branch_commit.txt',
            'artifacts/research-loop/research_branch_commit.txt',
        )
        def check(paths):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / 'evidence' / 'verified'
                root.mkdir(parents=True)
                for rel, content in paths:
                    path = root / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding='utf-8')
                return subprocess.run(
                    [sys.executable, '-c', snippet], cwd=temporary,
                    env={**os.environ, 'RESEARCH_SHA': sha},
                    capture_output=True, check=False,
                ).returncode
        for layout in layouts:
            with self.subTest(layout=layout):
                self.assertEqual(check([(layout, sha + '\n')]), 0)
        self.assertNotEqual(check([]), 0)
        self.assertNotEqual(check([(layouts[2], 'b' * 40)]), 0)
        self.assertNotEqual(check([(layouts[2], sha), (layouts[0], sha)]), 0)
        self.assertNotEqual(check([(layouts[2], sha + 'X' * 150)]), 0)

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
