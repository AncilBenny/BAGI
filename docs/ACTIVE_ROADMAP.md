# BAGI active roadmap — convergence

This is the active development plan, not a claim of scientific progress.
See issue #4 and preserve all historical preregistrations, results and artifacts.

## Component map

| Existing component | Role now |
| --- | --- |
| `bagi/agents/general_brain.py` | Reusable goal/skill planning and verified effects |
| `bagi/runtime.py` | Common episode lifecycle and versioned outcome checkpoint |
| `bagi/models/world_model.py`, `bagi/planning/` | Candidates for runtime integration; validate interfaces before use |
| `bagi/agents/dqn.py` and maintained RL/BC implementations | Baselines and candidate learned skill providers |
| `bagi/experiments/`, Workshop, MiniGrid, XLand | Historical studies and diagnostics; retain reproducibility |
| `tests/`, CI | Runtime contracts and independent publication evidence |
| `docs/autonomous-research-log.md` | Append-only results; no retrospective edits |

## Milestones in order

1. **Convergence foundation.** Reconcile branch history, expose gate errors,
   serialize workers, establish one runtime entry point and current-provenance CI.
   Review and integrate via PR #1. Until reviewed integration, this remains open.
2. **Persistent learned state.** Integrate compatible trained policy and optimizer
   state, public observation/action histories and replay; test save/resume
   equivalence, explicit version compatibility and evaluation isolation. Outcome
   counts alone are not neural continual learning.
3. **Native sequential curriculum.** Reuse existing maintained implementations
   and adapters; choose two genuinely different task families. Compare carried
   state versus reset/fresh and frozen controls at matched real interactions and
   seeds. Register splits, physical goals and decision criteria before acquiring
   evaluation data; quantify transfer, forgetting and uncertainty.
4. **Generalize and plan.** Integrate a learned world model and reusable skills
   only where native evidence identifies a bottleneck; evaluate independent goals
   and environments. Advance capabilities based on evidence, not study numbering.

## Automation during convergence

Keep the two existing cron cadences and their provenance checks. Scheduled
research-loop ticks audit only instead of automatically fitting the legacy study;
explicit manual and source-bound requests remain available without arbitrary
training caps. The autonomous proposal worker should address this roadmap only
and continues to require verified evidence. Both use one concurrency group;
GitHub may replace a pending run, so do not rely on every tick for durable work.
Never weaken the evidence gate or create a duplicate worker. A future tested
runtime curriculum may replace the scheduled audit-only behavior.

## Known limitations

The current checkpoint schema persists skill reliability counts only, with an
explicit caller-supplied compatibility ID. Adapters must change that ID whenever
skill semantics, policy weights or observation contracts change. Frozen mode
freezes those counts, not arbitrary mutable policy internals. Use a separate
runtime with frozen policies and isolated held-out data for evaluation.
No new native learning or generalization is claimed by this consolidation.
