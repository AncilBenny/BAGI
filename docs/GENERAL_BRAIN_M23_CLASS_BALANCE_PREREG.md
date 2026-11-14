# M23 stage 3 preregistration — isolate rare-object class imbalance

**Status:** prospective 2026-11-15; no stage-3 training or final evaluations.
Previous source-only world-model stage 2 failed at execution source
`1b0c72eedc17aa3e37b92dad482c8b5988dc044b` (run 43633738861).
The new question is whether ordinary categorical cross-entropy underweights
rare object transitions. This study is an **ablation/evaluation of an established
loss-weighting method**, not a new model or claimed breakthrough.

## Immutable inputs, baseline, hypothesis

Reacquire original TRAIN-only public transition ZIP from M23 source run 43631440997,
artifact ID **14127428720**, exact ZIP SHA256
`dd21e09790f91baffd07122712acdcf365aeb05534fc89d21a48001f605adadb`,
original source commit `594c670a0d501fbd3cafe78ef47d59bef4733c0a`.
Verify ZIP integrity, internal commit/mode markers, source event-coverage gate,
each TRAIN public dataset digest, and total **16,832** actually executed
TRAIN transitions before fitting. These are **cached** training transitions;
do not count their repeated presentations as new native actions.

Two parallel arms across the **same eight independent seeds**
941/947/953/967/971/977/983/991:

1. **ordinary**: literal stage-2 `PublicNextFramePredictor` and `fit_one`,
   full 7×7×3 public image, public direction/mission, 4 actually executed
   preceding native actions, current action, ordinary categorical losses.
2. **balanced_object**: identical architecture, initialization, minibatch order,
   objective heads, trajectory pool and output metrics; change only the
   next-object categorical cross-entropy to TRAIN-frequency inverse-square-root
   class weights, normalized to count-weighted mean 1 and clipped to [0.25, 8].
   Unobserved object classes have weight zero. No weight is learned from
   MONITOR/FINAL or from a private simulator physical field.

Both arms: 192-width two-ReLU-layer public predictor, Adam lr 0.001,
6 epochs, batch size 256, 396 optimizer-step calls and 100,992 cached
sample presentations per model. These are matched experimental settings,
not administrative caps. Report all actual updates and source SHA.

## NEW source-only monitor split and frozen gate

Never re-use stage-2 MONITOR for model selection or claim it is fresh.
Source-specific **new monitor** seeds: DoorKey 472001–472016,
Fetch 482001–482016, PutNear 492001–492016.
No seed overlaps earlier M15–M21 sealed FINAL ranges or M23 source TRAIN,
stage-2 MONITOR, M23 source FINAL, or target zero-shot FINAL.
Collect native public observations and **own executed** action histories
from 48 registered source episodes only. The same monitor transitions
must score both arms and all model seeds. External auditor may record
native pickup/drop/unlock effects separately but never feed them into
the learner. No UnlockPickup (target) actions, training, monitoring
or final layouts are allowed.

**Registered primary test:** on executed native actions 3/4/5,
balanced_object minus ordinary next-object *macro-F1*
(door/key/box/ball/wall) must exceed **+0.05 averaged over eight seeds**,
and exceed **+0.05 in ≥6/8 seed-matched contrasts**.
Secondary safety test: balanced all-cell next-object accuracy must not be
more than **0.05** below ordinary on average. Both conditions necessary.
Also report persistence macro-F1 on the same monitor split and changed-public
interaction subset, per-class precision/recall and actual event support.
When a class is absent, report support and avoid claiming its generalization.
Seed-blocked intervals with shared-layout caveat are required for a robust
improvement claim even if numerical gate passes.

**Interpretation:** passing the balanced-vs-ordinary gate would isolate a
specific source-prediction deficiency; it would **not** establish a
correct causal world model, MPC transfer, physical goal success or AGI.
Failing keeps M23 target FINAL 453001–453032 closed and suggests a more
structural partially observed/egocentric mechanism is needed. Do not
change earlier registered M23 stage-2 thresholds after seeing its failure.
No protected final task information may influence the model.

## Execution plan and accounting

Use existing serial BAGI Research Loop; admit `m23balance` only with a new
exact parent-SHA, one-file source-bound push request (force_retrain=true),
after independent source push/PR merge CI. Do not launch competing workflows
or change the hourly schedule. Write `runs/m23_class_balance/` outside
tracked history with numeric-only .npz checkpoints, checksums, full report,
action histograms and physical-source event counts. Record training and
monitor actual native actions separately, optimizer calls, version strings,
wall time, source/artefact SHA and uncertainty. Append results to
`docs/autonomous-research-log.md`; never edit historical entries.
Main integration remains held pending PR review and gate checks.
