"""Read-only daily compute admission; skipped audits do not consume the budget."""
from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from bagi_evidence_gate import GateError, Github

WORK = {
    'research': ('bagi-research-loop.yml', 'research', 'research_runs_per_utc_day'),
    'agent': ('bagi-autonomous-agent.yml', 'propose', 'ai_proposals_per_utc_day'),
}


def budget_available(api, kind, current_run_id, limit, day):
    if kind not in WORK or type(limit) is not int or limit < 1:
        raise GateError('invalid research budget policy')
    workflow, job_name, _ = WORK[kind]
    records = api.request(
        f'actions/workflows/{workflow}/runs?created=%3E%3D{day}&per_page=100')
    if records['total_count'] > 100 or records['total_count'] != len(records['workflow_runs']):
        raise GateError('cannot prove daily budget across truncated run history')
    current = [r for r in records['workflow_runs'] if r['id'] == current_run_id]
    if len(current) != 1:
        raise GateError('cannot identify current workflow attempt')
    if any(type(r.get('run_attempt')) is not int or r['run_attempt'] < 1
           for r in records['workflow_runs']):
        raise GateError('invalid workflow attempt history')
    consumed = current[0]['run_attempt'] - 1
    for run in records['workflow_runs']:
        if run['id'] == current_run_id:
            continue
        if run['created_at'][:10] != day:
            raise GateError('unexpected run date in daily budget history')
        jobs = api.request(f"actions/runs/{run['id']}/jobs?per_page=100")
        if (jobs['total_count'] > 100 or not jobs['jobs']
                or jobs['total_count'] != len(jobs['jobs'])):
            raise GateError('missing or truncated job history for daily budget')
        matches = [j for j in jobs['jobs'] if j.get('name') == job_name]
        if len(matches) != 1:
            raise GateError('ambiguous research job in daily budget history')
        job = matches[0]
        if job.get('conclusion') == 'skipped' and job.get('status') == 'completed':
            # A skipped latest attempt does not erase earlier attempts' spend.
            consumed += run['run_attempt'] - 1
            continue
        if job.get('status') not in {'queued', 'in_progress', 'completed', 'waiting', 'pending'}:
            raise GateError('unknown research job state')
        # Failed, cancelled, pending, and rerun attempts all reserve/consume work.
        consumed += run['run_attempt']
    return consumed < limit


def main():
    try:
        kind = sys.argv[1]
        policy = json.loads((Path(__file__).resolve().parents[1]
                             / '.github/research-policy.json').read_text())
        if policy['schema_version'] != 1:
            raise GateError('unsupported research policy')
        api = Github(os.environ['GITHUB_REPOSITORY'], os.environ['GH_TOKEN'])
        available = budget_available(
            api, kind, int(os.environ['GITHUB_RUN_ID']), policy[WORK[kind][2]],
            datetime.now(UTC).date().isoformat())
    except (GateError, KeyError, TypeError, ValueError, IndexError, OSError) as exc:
        print(f'BAGI budget HOLD: {exc}', file=sys.stderr)
        return 2
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as output:
        print(f"eligible={'true' if available else 'false'}", file=output)
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as summary:
        print('Daily compute budget available.' if available else
              'Daily compute budget reached; this run only audits CI.', file=summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())
