# M23 stage 8 — real native source physical task action selection: preregistered gate FAIL

**Executed:** 2026-11-15 ~05:21–05:24 UTC (10:51–10:54 IST).
**Source revision:** `2a8ab341b773d8e1bb30f816ad9685aa6e28c9e9`.
This experiment moves from local event prediction into real **physical
task-level decisions** on source DoorKey and Fetch; it does NOT
establish held-out UnlockPickup composition or task-independent
continual learning.

## Primary executed evidence, verified output and source provenance

- Existing serialized [BAGI Research Loop #43680572037](https://github.com/AncilBenny/BAGI/actions/runs/43680572037):
  source audit + real research job both **SUCCESS**.
  Independent exact-source push CI **#43680572114** and
  PR CI **#43680576080** passed. Research job passed Ruff and
  **797 pytest passed, 10 skipped, 44 subtests passed**.
- New immutable output artifact `bagi-research-678`,
  GitHub ID **14148172758**, SHA-256
  `84d4c335cdf7badf69d2f3e73951d808179b190550bd80e74db3c56260d7d4fa`,
  103,196 bytes, expires **2026-12-15 05:23:56 UTC**.
  **Independently downloaded and inspected:** the full ZIP SHA-256
  was recomputed and matched GitHub's digest exactly;
  `runs/m23_effect_action_selection/summary.json` and source commit
  marker were extracted and directly checked.
- Newly **executed** actual native actions:
  **48,655 source TRAIN-pilot + 24,397 independent source MONITOR
  = 73,052**. Across 576 true native episodes (DoorKey+Fetch,
  96 layouts/task, 3 control modes), all actions were executed
  through Gymnasium/MiniGrid, not predicted, teacher-labeled
  or repeated from cached artifacts.
  **0 new model fits, 0 optimizer updates, 0 imagined planner steps,
  0 UnlockPickup target-native actions, 0 target FINAL episodes.**
- Reused native source-only controllers:
  BAGI `TraversableOpenDoorController` and `PublicFetchTeacher`,
  plus maintained MiniGrid; comparison modes are unfiltered random
  action overrides, front-cell gating and publicly inferred
  executed-action-history gating. Fixed 0.65 public-only targeted
  attempted 3/4/5 override policy and existing 256-step native
  wrapper; no native private state was passed to controllers.
- Independent *evaluator-only* physics:
  `NativeDoorKeyAudit.full_physical_goal` verifies key pickup,
  effective door opening and actual goal arrival;
  `NativeFetchAudit.full_physical_goal` verifies physically
  carrying the mission-matched target after movement and
  non-adjacent target start. Native positive reward alone
  is **not** counted as a primary physical goal.

## Held-out source MONITOR outcomes (32 native layouts per task and mode)

| Task | Unfiltered physical goals | Front-only gate | History-memory gate |
| --- | ---: | ---: | ---: |
| DoorKey-6x6 | 4/32 | 4/32 | **9/32** |
| Fetch-5x5-N2 | 9/32 | **10/32** | 7/32 |
| **Combined** | **13/64** | **14/64** | **16/64** |

- History minus front physically completed goals:
  **DoorKey +5**, **Fetch −3**, pooled **+2/64**.
  Preregistered threshold ≥+8 combined and positive gains on
  **both** source tasks: **FAIL**. History was noninferior
  to unfiltered in total goals (16 vs 13), and the three-arm
  evaluation contained **3,047 real MONITOR proposals where public
  front rule and public-history rule disagreed** (registered
  minimum 16 satisfied). Nonvacuity alone is not success.
- Task-stratified paired-bootstrap 95% interval for total
  history-minus-front goal gain **[−5, +9]**. Interval includes zero,
  so even the pooled +2 should **not** be reported as a reliable
  improvement in physical task competence.
- MONITOR native action totals (all arms combined):
  **24,397** actual actions. By task/mode:
  DoorKey: unfiltered 7,699, front 7,843, history 6,959;
  Fetch: unfiltered 966, front 520, history 410.
  MONITOR accepted/rejected action override totals, per-run
  action SHA256, native physical effect events and all individual
  episode outcomes reside in the verified output artifact.
- Separate reward versus physical goal outcomes:
  DoorKey rewarded terminations = **4, 4, 9** for the three
  modes, coinciding with strict physical goals in that task.
  Fetch rewarded terminations = **16, 16, 17**, but strict
  physical goals = **9, 10, 7**. The 7, 6, and 10
  reward-only/nonqualifying episodes per mode demonstrate why
  reward alone would overstate this study's physical performance.
- The separate TRAIN-pilot physical goals (64 layouts/task/mode)
  were recorded in the artifact, but did **not** select the policy
  or change any frozen MONITOR rule. All three policies ran the
  same paired layouts, though their executed trajectories and
  action totals necessarily diverged after different decisions.

## Scientific diagnosis and decision

The frozen public-effect model, and simpler public-history rule,
correctly predicted *local* native physical effects on the prior
source-policy shift study (both F1=1.0). When used to select actual
actions, the history rule improved one source goal type but
**harmed a different mission**. A real physical interaction can be
possible yet irrelevant to the requested task: feasibility is not
goal relevance or planning. This stage decisively falsifies the
strong claim that accurate local event preconditions alone suffice
for cross-task physical competence.

**Next high-leverage study:** reuse maintained
mission-conditioned/hierarchical policy machinery (including BAGI's
source public Fetch mission parser and DoorKey controller, plus
maintained MiniGrid/BabyAI baselines), define a general public-input
goal-conditioned **skill-selection** comparison against both
source specialists and scratch baseline, and evaluate true
physical completions on independent source task layouts before any
zero-shot composed target final. Do **not** invent another
local effect predictor, lower the failed source action gate,
or treat hand-coded target sequencing as learned transfer.

Keep old M20–M23 results and sealed UnlockPickup
FINAL453001–453032 independent. Preserve existing hourly research
schedule; do not dispatch duplicates or merge PR #1 into main
without independent submitted review and exact-source CI provenance.
