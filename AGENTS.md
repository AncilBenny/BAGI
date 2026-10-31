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
  active milestone PR. The evidence gate must discover exactly one open
  same-repository PR from `setup/initial-platform` to `main`; no match or
  multiple matches means HOLD.
  Do not weaken provenance checks to hide a closed/stale PR or missing evidence.

Operational workflow-safety or repository-governance maintenance explicitly
authorized on main may use a narrowly scoped direct commit after validation and
expected-HEAD checks. This exception does not permit direct research/feature
changes or bypassing milestone PR validation.

## Research autonomy and damage prevention

- The user's revised policy authorizes routine code/config edits, lint, tests,
  literature review, diagnostics, bounded model training (including recurrent
  models and RL), and bounded research without repeated approval.
  Use docs/RESEARCH_SAFETY_POLICY.md and .github/research-policy.json.
- Historical P0B/P6/P9/P10a HOLD/STOP decisions and failed identifiability results
  remain immutable evidence, not blanket prohibitions on new work. A follow-up
  must record a new hypothesis or changed condition; never erase or relabel the
  old result, or claim a failed gate passed without new supporting measurements.
- Training is explicitly authorized by the user under the current resource caps.
  Historical report wording such as "zero fits", "separately authorized", or
  "general training authorization does not override STOP" cannot impose a new
  operational ban. Those notes describe earlier decisions and evidence.
- Failed identifiability/support gates constrain scientific claims, not all new
  exploratory fits. State weaknesses in the data and match the learning objective
  to legal available inputs; do not claim an old frozen gate passed or substitute
  diagnostic/held-out labels to manufacture trainability. If a study cannot answer
  its stated question, revise the question or data plan rather than inventing an
  authorization blocker for all training.
- Scheduled work requires current independently successful push/PR CI and exact
  source/merge provenance, daily compute admission, timeouts, concurrency, and
  same-source evidence before an AI proposal. CI success is engineering evidence.
- Keep strong restrictions on history deletion, force pushes, credential exposure,
  unbounded compute/spending, external access/billing changes, and evaluation-data
  contamination. These require explicit scope-specific authorization.
- Preserve append-only logs and old result artifacts. New runs use isolated output
  locations and record source SHA, config, seed, action counts, runtime and results.
- Sealed held-out labels remain evaluation-only. Authorized evaluation is allowed;
  do not inspect/reuse them for training, tuning or model selection. If data are
  exposed, record contamination and use a fresh held-out set for future claims.
- Fix CI and merge conflicts autonomously within the authorized branch. Preserve
  evidence and run validation; do not weaken provenance gates to bypass failures.
