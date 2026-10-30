# Research autonomy and damage prevention

This policy supersedes blanket operational bans in earlier engineering guidance.
Historical research logs and measurements remain unchanged. A negative result is
information for the next hypothesis, not a permanent prohibition on research.

## Proceed without repeated permission

Routine edits, dependency-compatible fixes, lint/tests, documentation, literature
review, diagnostics, and small reproducible experiments are allowed within the
current environment, budget and repository scope. Change one hypothesis at a time;
record the comparison, seeds, action budget, metrics and interpretation. P0B, P1,
P6, P9 and P10a follow-ups are allowed when they address a stated changed condition
or new hypothesis. Repeating a settled configuration without a scientific reason
is not useful progress; document that conclusion instead of burning compute.

## Preserve safeguards against concrete damage

- Never delete/rewrite append-only logs or overwrite historical result artifacts.
  Keep new outputs separate and make rollback possible.
- Never expose credentials, change access/billing/deployments, incur new external
  service charges, force-push, delete shared branches, or increase compute caps
  without explicit authorization for that action.
- Keep failed, stale, incomplete, missing and ambiguous CI evidence fail-closed
  for publication and scheduled execution. A CI failure does not block local work
  to investigate and fix that failure.
- Keep held-out data separate from training, tuning and model selection. Evaluation
  may use sealed labels through a fixed evaluator. Record any contamination and
  replace compromised evaluation splits before making generalization claims.
- Historical teacher counts, real-action budgets and failed gates remain facts.
  New measurements never retroactively change the interpretation of old results.

## Bounded scheduled work

The hourly research and two-hourly agent schedules remain configured. Work starts
only after current push and PR CI/provenance pass. Each workflow has serialized
concurrency; the research job is pinned to the audited source and rechecks CI.

Initial caps in `.github/research-policy.json` are one research attempt (45 minutes)
and one AI proposal (20 minutes) per UTC day. AI patch validation retains its
separate 25-minute timeout and commit publication its 5-minute timeout, in addition
to bounded read-only admission jobs. Successful evaluations are reused by exact
source SHA across days instead of automatically repeating settled runs. Failed/cancelled work counts; skipped
research jobs do not. Missing/ambiguous run history blocks compute. The AI still
needs a successful same-source research artifact, validates an allowlisted patch,
checks append-only history, tests changes and compares expected HEAD before push.
The policy and workflow timeout values must agree; budget changes are reviewed.

Only the bounded commands already present in the research loop are automatically
scheduled. Other follow-up experiments may be developed and run interactively
within an explicit small resource envelope; adding them to unattended automation
requires reviewing their data access, actions, runtime and cost first. A proposed
new acquisition with no defined resource envelope is not automatically admitted.

## Milestones

Follow AGENTS.md: test locally, verify push and PR merge provenance, integrate
validated milestones into main through the authorized PR, and record exact SHAs
and checks. Technical blockers should identify the repair needed, not freeze all
research. Green engineering checks are never proof of scientific advancement.
