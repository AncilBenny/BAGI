# G2-E6-P0 — Workshop size-diversity feasibility before learned-policy acquisition

**2026-10-27 UTC. Read-only evaluator-graph prerequisite; not a trained model, native rollout, learned goal completion or transfer result.** G2-E6-P0 Stage A remains **HOLD**. The source-shaped numerical table below was independently computed locally **without importing Gymnasium**. This commit adds a CI integration test using the actual source-pinned WorkshopEnv._observe and _can_enter; **native-source parity remains PENDING until that test passes**.

## Reuse-first question and provenance

Existing Workshop DQN TRAIN data cover one fixed size-9 geometry; the exact isolated actor0 TRAIN archive SHA256 8d3f167ce977f62d3397ff37c553d8948b94bc03ac6d88df8a5a6e5fe86aac83 is unavailable in this run. Do not reopen the original mixed TRAIN/EVAL archive or count its manifest as observations. Existing E1 Workshop data are random/scripted, not learned. Earlier P6 four-action-history prediction and P10a terminal-label prediction both STOP; do not repeat them. XLand E1 learned-BC public TRAIN has a partial source-eligibility pass, not a general-belief pass.

**Question:** Before inventing a Workshop layout generator or spending native training steps, does the existing WorkshopEnv(size=...) supply distinct, reachable, partially observed geometries at the unchanged 40-step cap? Null: reset seeds/mirror merely relabel the same geometry, or larger sizes make goals unreachable within 40. This is **evaluation/integration**, not a novel algorithm.

Reuse the CI-source-verified G2-E5-P2 evaluator graph and standard BFS, rather than implementing a new planner. Native Workshop source Git blob SHA1 **3a91b058d3461ce4ecb489e93939c88f624557f4** was unchanged at preflight HEAD 282d3c8bad00714d6b8fb76e04c6dab057bb2246. The new bounded audit covers **sizes 9, 11, 13**, layouts default/mirror_x, public 5x5 views, four actions and cap40. Mirrors are **symmetry controls, not independent geometries**. The graph calls neither native reset nor step. Evaluator latent states, goal coordinates and shortest paths are forbidden as policy/predictor inputs, training labels or teacher actions.

## Provisional source-shaped structural results

Each row applies identically to default and mirrored variants. The size-9 reference reproduces P2's published 13 starts, 79 reachable evaluator states, 57 public views, 2 ambiguous public-view/action keys and 15–19 shortest-path lengths. Size 11/13 numbers are **new source-shaped calculations**, not yet CI-confirmed native-source parity.

| Size | Legal starts | Distinct initial public views | Reachable evaluator states | Distinct public views | Ambiguous public-view/action keys | Privileged shortest actions | All starts ≤40? |
|---|---:|---:|---:|---:|---:|---:|---|
| 9×9 | 13 | 13 | 79 | 57 | 2 | 15–19 | Yes |
| 11×11 | 17 | **16** | 121 | 83 | **11** | 19–25 | Yes |
| 13×13 | 21 | **16** | 171 | 89 | **34** | 23–31 | Yes |

**Interpretation:** Changing size alters wall lengths, goal location, path distances and public aliasing while retaining the same switch→door→key→lock→goal causal rules and 4-action/5×5 interface. These are candidate **size-disjoint geometries**, not independent causal task families. More ambiguous public-view/action keys motivate an identifiability question but **do not demonstrate useful memory**. In particular, 17 or 21 nominal starts yield only **16 distinct initial public views** for sizes 11/13; reset seed numbers do not guarantee distinct observable initial states. Privileged paths within 40 are not learned navigation. Prior Workshop DQN/PPO pilots had no native rewarded TRAIN goals.

## Falsifiable native-source check and decision

New source: bagi/experiments/g2_e6_p0_workshop_size_native_parity.py, reusing G2-E5-P2's pinned native collision, observation and shortest-path helpers. A new integration test monkeypatches WorkshopEnv.reset and WorkshopEnv.step to raise, then demands exact six-case signatures, mirror parity, public-start counts and source blob identity. Changed source, changed metrics, skipped integration or failing CI **blocks promotion**. A local synthetic/source-shaped reproduction is not equivalent to this native-source integration.

**Decision now: source-shaped size feasibility promising; native-source parity PENDING; G2-E6-P0 Stage A HOLD.** No learned-policy data acquisition or neural fitting is launched. After source-parity CI passes, separately preregister a small public-only learned-policy Workshop data acquisition with exact matched native-step budgets, two model seeds, full chronological public before/after views and actually executed actions, independent sizes 9/11 for TRAIN and untouched size13 for later evaluation, positive/negative effect support and true native goal accounting. Specify frozen actor provenance and matched controls first; never use evaluator BFS as actor, teacher, oracle reward or heldout labels. Only after data eligibility may a matched observation-history versus reactive/shuffled/action-only/persistence predictor be considered. No Workshop size result establishes task-independent belief, BLUE placement, reusable skills or cross-domain parameter transfer.

**Resource ledger:** 0 native simulator steps/resets, model fits, optimizer/policy updates, teacher queries, original TEST/EVAL reads, evaluator fields in learner data, heldout expert labels, new workflows/automations, secrets, billing, deployments, merges or main writes. Only source-shaped CPU graph computation and source-parity instrumentation; existing CI triggered by the commit is not a scientific training run.

## Subsequent independent CI confirmation — 2026-10-27 UTC

Both HEAD CI runs for commit `a4069a5ae8195617038fcfb44ac3a5307d4f9402` **completed SUCCESS**: push `40505292506` job `122192935278`, and PR `40505301965` job `122192966139`. GitHub job logs independently show Ruff **All checks passed** and pytest **513 passed, 7 skipped** on both jobs, with Gymnasium installed; the native-method size-parity test was eligible to execute. Neither run published a research artifact. This upgrades the previously PENDING **source-pinned `_observe`/`_can_enter` parity prerequisite** to **PASS**. It does **not** add native reset/step evidence, goal competence, or learned cross-size behavior. G2-E6-P0 Stage A remains HOLD pending a real learned-policy public-transition acquisition. The separately frozen `GENERAL_BRAIN_G2_E6_P0B_WORKSHOP_DUAL_SIZE_LEARNED_ACQUISITION_PREREG.md` specifies the next one-shot design; no run is launched by this documentation.
