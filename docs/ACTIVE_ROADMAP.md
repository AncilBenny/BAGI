# BAGI active roadmap — convergence

This is the active development plan, not a claim of scientific progress.
See issue #4 and preserve all historical preregistrations, results and artifacts.

## Component map

| Existing component | Role now |
| --- | --- |
| `bagi/agents/general_brain.py` | Reusable goal/skill planning and verified effects |
| `bagi/runtime.py` | Common episode lifecycle and versioned outcome checkpoint |
| `bagi/models/world_model.py`, `bagi/planning/` | Candidates for runtime integration; validate interfaces before use |
| `bagi/agents/dqn.py`, `bagi/agents/persistent_dqn.py` | Existing DQN update algorithm exposed as a persistent runtime skill; other RL/BC adapters remain candidates |
| `bagi/experiments/`, Workshop, MiniGrid, XLand | Historical studies and diagnostics; retain reproducibility |
| `tests/`, CI | Runtime contracts and independent publication evidence |
| `docs/autonomous-research-log.md` | Append-only results; no retrospective edits |

## Milestones in order

1. **Convergence foundation.** Reconcile branch history, expose gate errors,
   serialize workers, establish one runtime entry point and current-provenance CI.
   Review and integrate via PR #1. Until reviewed integration, this remains open.
2. **Persistent learned state.** The existing DQN now has a runtime adapter with
   model/target/optimizer/replay/RNG and outcome-counter checkpointing. Verify
   resume, explicit adapter compatibility and evaluation isolation in CI. Integrate
   recurrent and world-model state only as needed by a native curriculum.
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
research-loop ticks audit CI and run engineering runtime contracts instead of automatically fitting the legacy study;
explicit manual and source-bound requests remain available without arbitrary
training caps. The autonomous proposal worker should address this roadmap only
and continues to require verified evidence. Both use one concurrency group;
GitHub may replace a pending run, so do not rely on every tick for durable work.
Never weaken the evidence gate or create a duplicate worker. A future tested native runtime curriculum may extend the scheduled contract path.
Engineering artifacts have their own evidence kind and never imply new science.
The historical G1 native workflow is manual-only: its three original artifact
runs returned zero artifacts on 2026-11-27. Keep its recorded results; do not
silently regenerate frozen inputs or reuse reserved evaluation layouts.

## Known limitations

The JSON outcome-only format is retained. The optional DQN bundle additionally
persists CPU neural and replay state; it is for trusted local run outputs and
requires a matching adapter compatibility ID. Use a new runtime constructed
with the restored learner, then restore its outcome snapshot. Frozen mode
skips learning hooks; callers must use the greedy policy and independent data
for evaluation. These integration tests do not establish native generalization.
