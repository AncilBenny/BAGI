# Research execution and orderly integration

The user authorizes research, model training (including supervised,
self-supervised, world-model, recurrent and RL methods), diagnostics, acquisitions
and code/config/dependency changes. No repository-imposed daily quota, custom job
runtime cap, file-count quota or repeated training-approval requirement remains.
Platform-enforced runtime, account and service limits still apply; this policy
cannot override them. Existing schedule cadences are unchanged.

This policy and AGENTS.md supersede operational bans in historical reports and
earlier quota policies. Earlier failed gates and P0B/P1/P6/P9/P10a findings remain
facts. They must not be reinterpreted as successful results or permanent bans on
new exploratory work. Choose useful objectives and report limitations candidly.

## Keep the work neat

1. Use a clear hypothesis and control, a reproducible configuration and seed, and
   isolated output locations. Record actual actions, optimizer updates and runtime.
2. Reuse compatible checkpoints when useful; an intentional rerun or fresh training
   is allowed using the research workflow's `force_retrain` dispatch input. Do not claim reused results are new measurements. Keep generated
   checkpoints, caches, downloaded evidence and credentials out of source commits.
3. Work on isolated branches/workspaces. Check current HEAD and concurrent writers
   before publishing; never overwrite another writer or force-push shared history.
4. Review the complete patch, run Ruff and pytest, and verify independent current
   push/PR CI with correct source/base/merge provenance. Fix conflicts and CI
   failures; do not bypass the checks. Local work to fix a failure can continue.
5. Commit coherent changes with clear messages, integrate completed milestones to
   main through the reviewed PR path, and verify the resulting main revision.
6. Preserve append-only logs and prior result artifacts. Record new results as new
   entries. Keep held-out evaluation independent from training/tuning so claims
   remain interpretable. Record contamination rather than hiding it.

## Execution arrangement

Scheduled research still checks CI and pins the evaluated source. Serialized
workflow groups prevent overlapping copies of the same workflow. Daily budget
jobs and custom timeout settings have been removed. The hourly research loop and
existing agent/task schedules may continue without waiting for a daily reset.

Proposal jobs prepare code; separate jobs validate and publish their patches.
There is no six-file or 250 KB proposal restriction. The patch guard keeps runner
outputs, private paths and rewritten research history out of commits. Extending
an execution command or training study is ordinary tested development, not a new
training-authorization request. Configure the runner to execute the intended study;
a proposal or cached checkpoint alone is not a completed training run.
