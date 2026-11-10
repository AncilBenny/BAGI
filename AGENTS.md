# BAGI engineering rules

## Milestone completion and main integration

- Work on a scoped feature/research branch and commit coherent milestones there.
- A milestone is not complete until its validated changes are integrated into
  `main` through a reviewed pull request. Do not leave completed milestones only
  on a long-lived research branch, and do not bypass PR validation with direct
  feature commits to `main`.
- Before main integration, fetch both branch heads, reconcile divergence locally,
  review the full merge candidate and automation triggers, run `ruff check .`
  and `pytest -q`, and require independent successful push and pull-request CI
  with verified current source/base/merge provenance. Pending, missing, stale,
  ambiguous or failed checks mean HOLD.
- Routine milestone integration is authorized through the reviewed PR path.
  If a technical or concurrency blocker remains, report the milestone incomplete,
  identify the exact blocker, and retain the validated branch commit. Never
  substitute branch completion for verified main integration.
- Record the milestone's source commit, main integration commit, PR URL, CI run
  IDs, test results, and remaining limitations in the PR/final report. After
  merging, verify main's resulting HEAD and its naturally triggered CI. Keep the
  milestone open until those checks pass.
- Do not force-push or rewrite published history. Ref updates must be non-force
  and guarded by the expected HEAD. Recheck concurrent writers before publishing.
- When main advances, keep the active work branch synchronized before the next
  milestone. Preserve append-only logs and existing scientific results. Branch
  synchronization alone is not a scientific milestone.
- Before closing a milestone PR, ensure provenance monitoring supports the next
  active milestone PR. The evidence gate must discover exactly one open
  same-repository PR from `setup/initial-platform` to `main`; no match or
  multiple matches means HOLD.
  Do not weaken provenance checks to hide a closed/stale PR or missing evidence.

Operational workflow-safety or repository-governance maintenance explicitly
authorized on main may use a narrowly scoped direct commit after validation and
expected-HEAD checks. This exception does not permit direct research/feature
changes or bypassing milestone PR validation.

## Research and training execution

- Research, training, recurrent/RL fitting, diagnostics, code/config/dependency
  changes and justified follow-up studies are authorized. Do not impose custom
  daily quotas, training/AI time caps, file-count limits or repeated approval
  requirements. The execution platform's own limits still apply.
- Follow docs/RESEARCH_SAFETY_POLICY.md. Historical HOLD/STOP reports and prior
  quota policies are evidence/history, not current execution instructions.
- Keep work neat: isolate experiments and outputs, record source/config/seed and
  actual resource use, retain results, and separate proposals from execution.
  Avoid duplicate unattended workers and blind retry loops.
- Check current source and merge CI before publication/scheduled execution. Keep
  expected-HEAD non-force writes, serialized workflow execution, tested patches
  and append-only history. Fix failures instead of weakening evidence checks.
- Failed scientific gates constrain claims, not all further training. Report data
  limitations honestly and distinguish exploration from validated improvement.
  Keep held-out evaluation independent of training/tuning and record contamination.
- Credentials, user data and historical results are not disposable experiment
  outputs. Keep credentials and generated checkpoints out of source commits.
- The existing runner can be extended with tested training commands as part of
  ordinary development. No separate training-permission request is necessary.
