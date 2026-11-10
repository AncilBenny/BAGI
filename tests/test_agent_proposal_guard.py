"""Repository proposals can grow without including secrets or losing history."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class ProposalGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.git('init', '-q')
        self.git('config', 'user.name', 'Guard Test')
        self.git('config', 'user.email', 'guard@example.invalid')
        log = self.root / 'docs/autonomous-research-log.md'
        log.parent.mkdir()
        log.write_text('original history\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'baseline')
        log.write_text('original history\nnew result\n')
        data = yaml.safe_load(
            (ROOT / '.github/workflows/bagi-autonomous-agent.yml').read_text())
        run = next(step['run'] for step in data['jobs']['validate']['steps']
                   if step.get('name') == 'Validate repository patch and branch snapshot')
        self.code = run.split("python -I - <<'PY'\n", 1)[1].split('\nPY\n', 1)[0]

    def git(self, *args):
        return subprocess.run(['git', *args], cwd=self.root, check=True,
                              capture_output=True, text=True)

    def guard(self):
        return subprocess.run([os.sys.executable, '-I', '-c', self.code], cwd=self.root, check=False,
                              capture_output=True, text=True)

    def test_accepts_larger_multifile_project_changes(self):
        for i in range(8):
            (self.root / f'module_{i}.py').write_text('value = 1\n')
        (self.root / 'docs/study.md').write_text('x' * 260_000)
        result = self.guard()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runner_outputs_are_not_packaged_as_project_changes(self):
        for name in ('proposal', 'evidence', 'runs', 'results', 'checkpoints', 'artifacts'):
            p = self.root / name
            p.mkdir()
            (p / 'runtime.txt').write_text('runtime output')
        self.assertEqual(self.guard().returncode, 0)
        names = self.git('diff', 'HEAD', '--name-only').stdout.splitlines()
        self.assertEqual(names, ['docs/autonomous-research-log.md'])

    def test_rejects_rewritten_history(self):
        (self.root / 'docs/autonomous-research-log.md').write_text('replacement\n')
        self.assertNotEqual(self.guard().returncode, 0)

    def test_rejects_credentials_and_symlinks(self):
        secret = self.root / '.env.secret'
        secret.write_text('not a real credential')
        self.assertNotEqual(self.guard().returncode, 0)
        self.git('reset', '-q')
        secret.unlink()
        (self.root / 'external.py').symlink_to('/etc/passwd')
        self.assertNotEqual(self.guard().returncode, 0)
