# M23 stage 4 preregistration — does PUBLIC observation history improve source physical-effect prediction?

**Prospective, 2026-11-15. No stage-4 fits, optimizer calls, or newly
executed native actions at registration.** This is a distinct, source-only
diagnostic toward M23's compositional world-model objective, not an M22
encoder study and not UnlockPickup target evaluation. Do not retroactively
reopen failed stage-2 or stage-3 gates. Existing schedules unchanged.

## Research gap and reuse decision

M23 stage 2: 8 fresh PyTorch models/3,168 Adam calls and 2,331 new source
MONITOR actions. Public interaction next-object macro-F1 was **0.443**
versus persistence **0.604**; registered gate failed.
M23 stage 3: 16 new fits/6,336 Adam calls and 2,498 source MONITOR actions.
TRAIN-frequency balanced loss improved mean macro-F1 0.44984 → 0.51787,
but beat its matched MLP by >0.05 on only 4/8 seeds and all models remained
below persistence 0.63737. Earlier source TRAIN data had real effect
coverage: 16,832 native actions, 128 DoorKey unlocks, 127 Fetch pickups
and 34 PutNear box pickups. None demonstrates reliable learned physical
mechanism prediction or compositional transfer.

**Verified primary foundations and maintained code:**
- Hausknecht & Stone (2015), *Deep Recurrent Q-Learning for Partially
  Observable MDPs*, https://arxiv.org/abs/1507.06527; recurrency
  improves use of past evidence under partial observability but does not
  guarantee transfer.
- Kapturowski et al., ICLR 2019, *Recurrent Experience Replay in Distributed
  Reinforcement Learning*, https://openreview.net/forum?id=r1lyTjAqYX ;
  sequence boundaries, replay and recurrent-state staleness matter.
- Hafner et al., Nature 2025, *Mastering diverse control tasks through world
  models*, https://doi.org/10.1038/s41586-025-08744-2 ; action-conditioned
  recurrent world state motivates this test; avoid a GPU-heavy Dreamer port.
- BAGI existing PyTorch GRU, public mission vector, M23 verified public
  7×7×3 full-frame transition encoder/evaluator, and
  `bagi/experiments/m23_class_balance.py` balanced-CE MLP control.
  Maintained Farama MiniGrid / Gymnasium supply native action steps and
  native effect auditing. **Contribution type: integration and evaluation.**

Open question: Given the **same action-conditioned public input at the
current step**, can memory of earlier **public observations and own executed
actions** predict source changes more reliably than an architecture-identical
one-step recurrent predictor? If it cannot, larger history is not a
sufficient bottleneck fix.

## Data and model comparison frozen before execution

Reacquire only original M23 source TRAIN run
https://github.com/AncilBenny/BAGI/actions/runs/43631440997,
artifact ID 14127428720, original executed commit
`594c670a0d501fbd3cafe78ef47d59bef4733c0a`,
whole-ZIP SHA256
`dd21e09790f91baffd07122712acdcf365aeb05534fc89d21a48001f605adadb`.
Verify SHA256, internal source-commit and study-mode markers, data digests,
source native-effect coverage and 16,832 TRAIN actual executed actions
before any fit. **Those actions are cached**, not new stage-4 interactions.
Each sequence consists of 8 (or 1) consecutive public observations with
actual executed action history and current action. Left-pad only at
episode start, reset fully on true termination/truncation, and never
cross source task or episode boundaries. Hidden carrying fields,
simulator full grids/coordinates, teacher unexecuted actions and target
composition layouts are prohibited learner inputs.

Match 8 model seeds **941,947,953,967,971,977,983,991**, 6 epochs,
256 current-transition examples/minibatch, Adam lr=0.001, and the **same
TRAIN-only class weights** (inverse sqrt frequency clipped 0.25..8)
with identical object, color, state, direction and reward heads.
The registered arms on identical source transitions:
1. **balanced MLP**: exact existing `fit_weighted` from stage 3,
   fresh fit; public current frame and 4 preceding executed actions.
2. **GRU-1**: maintained `torch.nn.GRU` with one current public feature,
   96-wide learned input projection and 96-wide hidden state, zero reset
   each example. A recurrent-architecture/capacity control.
3. **GRU-8**: **same** GRU architecture, initialization seed, optimizer
   settings and labels, but 8 consecutive public input steps, including
   earlier images and actual actions. No replayed hidden state crosses
   episode boundaries. No fixed externally imposed compute quota.

All models use exactly the same current action supervision and the same
cached source transitions, **396 actual optimizer updates per fit** when
the 16,832-record source is validated; report true calls, not planned
counts. Report training presentations, checkpoint hashes and runtime.
Differences between GRU-1 and GRU-8 isolate temporal observation history;
MLP has different capacity and is a secondary comparator, not a
causal architecture-controlled contrast.

## Fresh independent source MONITOR and registered gates

Use exactly sixteen **previously unused** native source layouts per task:
DoorKey **502001–502016**, Fetch **512001–512016**, PutNear
**522001–522016**. These are not the prior 432/442/462xxx
or 472/482/492xxx monitor layouts, 431/441/461xxx TRAIN,
433/443/463xxx source FINAL, 451/452/453xxx UnlockPickup,
or earlier sealed M15–M21 FINAL. Run all source MONITOR episodes using
the existing public-only source actors; count and hash **actually executed**
actions and audit true pickup/drop/door unlock outside the learner.
Execute each MONITOR trajectory only once and evaluate all three arms and
all seeds on the same frozen trajectories. Minimum **16** actual
3/4/5 interaction steps for interpretable gate; record per-object support.

Primary source-only gate must satisfy **both**:
- GRU-8 − GRU-1 > **+0.05** mean action3/4/5 next-object
  macro-F1 over eight paired seeds, and >+0.05 in **≥6/8** seeds.
- GRU-8 − balanced MLP > **+0.05** mean on same metric, and
  >+0.05 in **≥6/8** seeds.

Additional necessary check: GRU-8 − observation-persistence
macro-F1 > **0.0** in ≥6/8 seeds. Report ordinary all-cell
accuracy, no-object-changes vs public changes on interaction steps,
per-class recall/precision, physical-effect action support, and a
paired seed-block bootstrap interval. Failure of any condition = FAIL
without threshold revision or posthoc seed filtering. A numerical
pass would show source next-observation prediction, NOT a true
causal-state solution, physical task completion, general intelligence,
or transfer to unseen composed UnlockPickup.

If prediction stays below persistence or lacks physical interaction
support, STOP before applying this model to the target FINAL layouts.
A subsequent separate preregistration can test counterfactual
action-effect or egocentric-motion disentanglement on *new source*
monitors; do not reuse these results as unseen evaluations.

## Execution path and provenance

Adapt `bagi/experiments/m23_public_world_model.py` and existing
`m23_class_balance` without creating a parallel algorithm package.
Implement isolated `bagi/experiments/m23_belief_memory.py` and tests.
Extend the **existing** serialized workflow/dispatcher for `m23memory`,
preserving hourly schedule, source verification, and independent push +
PR merge CI. Only after Ruff/pytest and exact-source checks pass,
create a one-file source-parent-bound push request with
`force_retrain=true`; never race an active worker.
Store numeric checkpoints/summary under ignored
`runs/m23_belief_memory/`; upload model hashes, actual optimizer
steps, library versions, native effects, action histograms, input
artifact SHA, exact source commit, runtime and independent outcome.
Append results to `docs/autonomous-research-log.md`, preserve old
evidence and secrets. PR #1 requires independent review before
merging into main. No target-final or Workshop/XLand evidence is used.
