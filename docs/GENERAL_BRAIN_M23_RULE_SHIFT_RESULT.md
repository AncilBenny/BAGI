# M23 stage 7 — frozen native-effect model policy-shift robustness PASS; public memory matches

**Executed:** 2026-11-15 05:03–05:05 UTC (10:33–10:35 IST).
**Result:** Registered source-only policy-shift robustness gate **PASS**,
and separately registered **prefer the simpler public-memory rule** gate
**PASS**. These do NOT demonstrate held-out UnlockPickup task completion,
physical continual transfer or general intelligence.

## Executed evidence and complete provenance

- Exact evaluated source commit:
  `78f2968fc28a36aea8a4ed86c55f20a346df67da`.
- Existing serialized [BAGI Research Loop #43679259230](https://github.com/AncilBenny/BAGI/actions/runs/43679259230)
  completed source audit and research jobs SUCCESS, with independently
  successful exact-source push CI **#43679259216** and
  PR CI **#43679262277**. Runner Ruff passed;
  **792 pytest passed, 10 skipped, 44 subtests passed**.
- **Frozen** Stage-6 source artifact `bagi-research-674`,
  ID **14134639518**, trained source commit
  `472ed53f37af64ae6c7e2ed9c12f74412f7ef34f`.
  Exact source ZIP SHA-256
  `de2270383c7de3d52dea54f4be208e0383b83d52e33001a27ba85cbff512e82e`
  rechecked by research job (`source.zip: OK`).
  Stage-6 internal registered PASS/source marker and all sixteen
  stage-6 checkpoint SHA-256 values were checked before strict numeric
  state loading with `numpy.load(allow_pickle=False)`.
  This stage performs **zero new model fits, zero optimizer.step calls,
  and zero new sample-training presentations**. Historical Stage-6
  16 fresh fits and 98,560 optimizer calls are NOT recounted.
- **9,591 new actual native MiniGrid source MONITOR actions**, executed
  once on 96 independent layouts: DoorKey602001–602032,
  Fetch612001–612032, PutNear622001–622032.
  The public-only controller used **0.65** probability of a
  front-cell-targeted 3/4/5 action override, versus Stage 6's
  0.35 uniformly random probe override. No private carrying,
  object IDs, global grid or native physics were used to select
  interventions. Independent native evaluator monitored actual
  pickup/drop/unlock and failures.
- Among **public-rule-eligible interaction attempts**, **811 effective
  real native physical events** and **4,152 ineffective native actions**
  (4,963 eligible attempts total). This is a nonvacuous hard-negative
  distribution shift; native effect and no-effect events are not
  merely inferred from reward or the predictor.
- Frozen probe-trained predictor macro-F1: **1.000000** on this subset
  for **all 8 model seeds**. Frozen success-only control macro-F1,
  respectively: **0.256001, 0.249423, 0.254346, 0.250893,
  0.326395, 0.251901, 0.254182, 0.300567**. The fixed
  source learned model met the prospective threshold against both
  the original public precondition rule and success-only control;
  runner explicitly reported `registered_shift_robustness_gate_pass=true`.
- A **simple public observation/action-history rule**, which tracks
  previously *visible* actual pickup/drop consequences and inferred
  carried object type, achieved **1.000000 macro-F1** on the SAME
  native hard-negative subset; runner reported
  `registered_public_history_reuse_preferred=true`.
  The history rule is not using native carrying or evaluator truth.
  It completely matched the frozen neural model on these source
  task families. This is the key reuse-first finding.
- No UnlockPickup target actions, no target FINAL episodes, no
  model-based planner rollouts, no physically executed composition
  successes and no independent demonstrations of task-independent
  continual learning were part of Stage 7.

## Artifact and limitations

- New output `bagi-research-677`, GitHub artifact ID **14148150366**,
  GitHub-reported ZIP SHA-256
  `45459e73692842da3f27ac347e1748efe1fffe8540675cb6385b329c41ed6284`,
  90,188 bytes, expires **2026-12-15 05:05:32 UTC**.
  Input ZIP integrity verified in research run; new output ZIP interior
  and independent per-trajectory hashes not separately downloaded
  here. Full per-task action histograms/native effects, model results,
  source SHA and versions are preserved in artifact.
- Perfect prediction on three source environments does **not** prove
  the same preconditions or memory suffice in unfamiliar layouts,
  objects or delayed hidden-state tasks beyond the tested domain.
  The neural model, heuristic and public-memory rule were evaluated
  on one shared 96-layout set; action decisions were made by the
  public source actor with pre-specified intervention policy, not by
  a model-based planner. Do not treat 4,963 temporally correlated
  interactions as IID trials or rewards as physical task success.

## Scientific decision — graduate from effect prediction to task-level action selection

The original stage-2/3/4 predictive failures and stage-5 vacuous
evaluation remain in append-only history. Stage 6 proved that
**actual failed-action data** enabled effect prediction; Stage 7
demonstrated shifted-policy robustness but also showed
**a maintained, transparent public-memory baseline is sufficient
for these local source effects**. Stop duplicating local effect
predictor fits here.

Next, reuse the simple observation-history rule to gate or plan
**actual executed actions**, compare with exposure-matched source
controllers and front-cell-only controls on fresh source task layouts,
and independently witness physical task completion, not just
local macro-F1. Only then construct a distinct preregistered
zero-shot compositional-policy evaluation; UnlockPickup
FINAL453001–453032 remain sealed. Maintain the existing research
schedule, independent result artifacts and PR review/CI requirements;
do not merge into main without submitted review.
