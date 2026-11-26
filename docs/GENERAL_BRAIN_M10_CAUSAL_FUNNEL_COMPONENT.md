# BAGI M10-D1 — evaluator-only causal opportunity counter (2026-11-11 UTC)

This source milestone adds a **pure counting component and unit tests**, not a new experiment or proof of learned competence. It implements part of the train-only diagnostic in `GENERAL_BRAIN_M10_PUBLIC_REPRESENTATION_PREREG.md`.

- Input: two independent auditor-only physical snapshots surrounding **one actually executed** native MiniGrid action, plus its native reward/termination flags.
- Counts pre-action key possession, key visibility, facing a locked door while carrying the key, valid toggle opportunities and attempts, effective pickup/opening events, native action frequencies, and first-event indices. Counts repeated effects as events, **not** unique solved episodes.
- Full physical success requires an observed pickup before an observed door opening, followed by a physically reached rewarded terminal goal. A reward alone is insufficient.
- This component is not connected to the PPO observation, policy, reward, rollout, or training loop. The environment-specific adapter and fresh train-only runs are still needed. The caller must verify that snapshots refer to the **same target door** and that key acquisition and goal flags reflect real simulator transitions. The component does not infer adjacency from public pixels and does not claim causal identification.
- Existing M1–M9 heldout streams remain excluded from tuning. M10 final layouts have not been opened.

**Validation:** dependency-free focused pytest unit tests exercised the equivalent source locally before publication (10 passed); full repository `ruff check .` and `pytest -q` could not be executed in the local environment because GitHub cloning and Ruff/dependency acquisition were unavailable. Independent push/PR CI must validate the committed source. No integration to main without review and postmerge CI.

**New work accounting:** 0 simulator actions, 0 optimizer updates, 0 model fits, 0 independent evaluation episodes, 0 teacher queries. No new workflow or schedule. Next step: connect auditor snapshots to fresh train-only native rollouts, verify action/effect identity and evaluate the frozen public-encoding comparison.
