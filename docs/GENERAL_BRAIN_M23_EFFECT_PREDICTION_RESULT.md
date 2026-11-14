# M23 stage 5 actual result — apparent perfect macro-F1 is VACUOUS for effect/no-effect distinction

**Executed 2026-11-14 22:03–22:06 UTC (2026-11-15 IST).**
Do not interpret this source-only result as causal understanding, physical
compositional transfer, or UnlockPickup goal success.

## Full measured provenance

- Evaluated working source SHA `02876644ccb89597972307e1e5e70c6ed01b9919`.
  [Existing serialized research run #43645959845](https://github.com/AncilBenny/BAGI/actions/runs/43645959845)
  completed audit and research jobs SUCCESS after independent exact-source
  push CI **#43645959840** and PR CI **#43645963945** succeeded.
  Ruff and pytest in research job: **782 passed, 10 skipped, 44 subtests**.
- Newly created immutable output artifact `bagi-research-673`
  GitHub ID **14134336937**, GitHub-reported ZIP SHA-256
  `422058dc75e5059c5cd8d5091e5f26e33876cfe88acea50a1483b3acdf6d08d0`,
  5,163,787 bytes, expires 2026-12-14 22:06:38 UTC.
  GitHub metadata and job stdout verified. ZIP interior and all numeric
  checkpoint hashes were **not independently decompressed** for this report.
- Source TRAIN input: historical original M23 public transition artifact
  ID **14127428720** executed source
  `594c670a0d501fbd3cafe78ef47d59bef4733c0a`,
  **16,832 cached previously executed native source actions**.
  Input ZIP checksum `dd21e09790f91baffd07122712acdcf365aeb05534fc89d21a48001f605adadb`
  was independently rechecked (`source.zip: OK`) and internal markers and
  public-data hashes were verified by `load_source_artifact` in research.
  Cached actions are NOT new Stage 5 interactions.

## Fresh training and native source evaluation

- **16 fresh PyTorch fits:** 8 model seeds
  941/947/953/967/971/977/983/991 × action-aware versus current-action-masked
  same-size MLP, trained on public before/after front-object event labels
  for actual source interaction actions 3/4/5. 40 epochs, batch 128,
  Adam lr 0.001; **200 actual optimizer step calls per fit = 3,200 total**,
  with exact cached source interaction exposure counts in the artifact.
  `evaluator_native_effect` was **never** fitted as a label or learner input.
- **1,975 newly executed native actions** on 48 independent source MONITOR
  episodes DoorKey532001–532016, Fetch542001–542016, PutNear552001–552016.
  Only **69 actually executed interaction actions 3/4/5** were present.
  All **69/69** were witnessed as **effective native physical events**,
  so the monitor contained **zero failed/ineffective interaction actions**.
  The null/failed-effect support clause required at least 8 ineffective
  interactions: **FAIL / vacuous discrimination**.
- Native-effect four-class macro-F1 across supported classes:
  action-aware learner **1.0000** for **all 8 seeds**; action-masked learner
  **1.0000** for all 8; the no-training public object+action eligibility
  rule also **1.0000**. Accordingly aware minus rule **0.0000** and
  aware minus masked-action **0.0000**; neither >+0.05 in even one seed
  (registered ≥6/8). The correct independent registered result is **FAIL**.
  The score cannot tell whether the learned policy understands
  **why an action would fail**; all observed interactions worked.
  The action-masked model's perfect score is additional evidence that
  the class label can be inferred from the front-object precondition
  distribution without using the actual action.
- **Zero UnlockPickup target actions, zero target FINAL episodes, zero
  planner rollouts, zero physical composition claims.** No reward-only
  positive was used as a native effect witness.

## Decision and next nonvacuous experiment

This identifies a **data-collection selection bias**: the public source
teachers preferentially issue effective interaction actions. Stage 5
cannot establish a causal event predictor on no-effect counterexamples,
regardless of its perfect macro-F1. Do NOT revise Stage 5's threshold
or use the already-opened 532/542/552xxx source monitors for tuning.

Next experiment should use source-only, real executed **intervention
actions**, with an action policy whose decisions depend only on public
observations and its own executed history. Mix normal native source
teachers and randomized 3/4/5 probes from reachable TRAIN states;
measure both successful and ineffective events by an independent
native evaluator and select no model using closed FINAL data.
Use previously unseen source MONITOR seeds for fresh evaluation.
Preserve M22 separately, historical artifacts/logs, the current
hourly schedule and the reviewed PR-to-main restriction.
