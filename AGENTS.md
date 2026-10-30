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
- Merge only within the user's authorization. If main integration is not
  authorized or is blocked, report the milestone as incomplete/HOLD, identify
  the exact blocker, and retain the validated branch commit. Never silently
  substitute branch completion for main completion.
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
  active milestone PR; the current evidence gate explicitly targets PR #1.
  Do not weaken provenance checks to hide a closed/stale PR or missing evidence.

Operational workflow-safety or repository-governance maintenance explicitly
authorized on main may use a narrowly scoped direct commit after validation and
expected-HEAD checks. This exception does not permit direct research/feature
changes or bypassing milestone PR validation.

## Scheduled automation and scientific evidence

- Scheduled jobs may audit CI and provenance while scientific HOLD/STOP is active.
  Green CI does not authorize an experiment, acquisition, training, AI session,
  scientific claim, or autonomous research commit.
- Keep P0B HOLD, P6/P10a STOP, P9 native HARD STOP, the failed frozen
  history-identifiability gate, and the size-13 heldout-label restriction binding.
- Do not re-enable blocked scientific jobs merely to make a scheduled run green.
  Changing a scientific stop requires a separately reviewed, explicitly authorized
  research decision with supporting evidence.
- Never modify/delete append-only research history. Do not rerun workflows or
  launch scientific compute without task-specific authorization.
