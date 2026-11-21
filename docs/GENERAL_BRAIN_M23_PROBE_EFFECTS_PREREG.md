# M23 stage 6 preregistration — public-source intervention data for nonvacuous causal-effect prediction

**Prospective 2026-11-15. Stage 5 is archived separately as a FAILED,
vacuous evaluation; stage 6 has not executed at registration.** Existing
M22 and all old M23 study data/thresholds remain unchanged. Native
UnlockPickup target FINAL453001–453032 remains sealed.

## Primary evidence and established reuse

Stage 5 executed source commit `02876644ccb89597972307e1e5e70c6ed01b9919`,
research run 43645959845, 16 fresh PyTorch fits / 3,200 Adam calls,
1,975 new source MONITOR native actions. Native-effect macro-F1 was
1.0 for learned action-aware, masked-action and simple precondition
rule because **all 69 recorded 3/4/5 interactions were physically
effective**, with **zero ineffective** interaction trials. The
registered gate required at least eight ineffective trials and failed;
the perfect score was **not** causal knowledge.

Maintained native execution and public data: Farama MiniGrid
https://github.com/Farama-Foundation/Minigrid/blob/main/minigrid/minigrid_env.py
and action semantics
https://github.com/Farama-Foundation/Minigrid/blob/main/minigrid/core/actions.py;
BAGI original public source dataset, maintained PyTorch and M23
public-effect classifier and independent native outcome witness.
The experiment is an **integration/active-data evaluation**, not
original RL or a MuZero reimplementation. The open BAGI-specific
question is whether actually executed failed precondition tests,
alongside successful source interactions, allow a public-input
effect predictor to estimate physical effects beyond a trivial
front-object precondition rule.

## Data collection — actual executed interventional source actions

Source environments only: DoorKey-6x6, Fetch-5x5-N2 and PutNear-6x6-N2.
Freeze new **TRAIN** source layouts:
DoorKey **571001–571128**, Fetch **581001–581128**,
PutNear **591001–591128**. New untouched independent **MONITOR**
source layouts DoorKey **572001–572032**, Fetch **582001–582032**,
PutNear **592001–592032**. Seeds not present in earlier M23,
M15–M21 FINAL or target UnlockPickup. The TRAIN and MONITOR source
tasks are each sampled with the **same fixed public actor** plus
**independent 0.35 probability at each step of replacing its proposed
action with a uniformly drawn native interaction 3/4/5**. This is
a physically executed randomized action intervention on a reachable
source state, not teacher labels or imagined samples. A no-effect
attempt is intentionally valuable evidence, even if it delays the
mission. Controllers see only public observations and their own
actually executed actions. Each source episode follows maintained
native terminal/truncation and existing 256-step wrapper.

The experiment MUST record native **attempts** and actual key/ball/box
pickups, drops, and locked-door opens separately with evaluator-only
state. Keep full public transition corpora with mission, direction,
image and real executed-action history. Native inventory/carrying,
full global grid and simulator object identity remain **evaluation-only**;
they do not enter model features or supervised labels.
TRAIN supervision is the visible **before/after public front-cell**
4-class effect proxy from stage 5. Do not fit against native
effect-oracle labels. Report public/native agreement on the
independently acquired MONITOR: if <0.8, FAIL coverage validity.

## Comparators and matched learning exposure

Historical verified source TRAIN artifact ID **14127428720**,
source SHA `594c670a0d501fbd3cafe78ef47d59bef4733c0a`,
whole ZIP SHA256
`dd21e09790f91baffd07122712acdcf365aeb05534fc89d21a48001f605adadb`
supplies the **original 16,832 previously executed (cached) native
source actions**. Validate ZIP and internal source marker, study
mode, data hashes and original source coverage before use.

Use all original positive-biased source interaction TRAIN examples
plus newly executed randomized intervention TRAIN interactions.
The only scientific difference between the matched learned arms:
1. **probe-trained**: original source interaction examples + new
   executed intervention TRAIN examples with PUBLIC before/after
   supervised effect proxy.
2. **success-only matched**: same original source pool **repeated
   to exactly the same number of source sample presentations**
   using a frozen per-seed permutation/resampling; no new
   intervention examples. This is an exposure-matched data-coverage
   control, not the claim of new physical actions.
3. Nonlearning public action/front-object eligibility heuristic
   and always-no-effect baseline remain independent references.

Same PyTorch stage-5 action-aware model architecture, public feature
contract, TRAIN-only effect-class weights derived from **pooled
probe TRAIN** distribution and applied unchanged to both arms,
8 model seeds 941,947,953,967,971,977,983,991, Adam .001,
40 full training epochs, minibatch128 and identical number of
optimizer-step calls and source presentations per fit.
A policy's previous executed-action features never include
unexecuted teacher labels. Do not tune weights or action-intervention
rates against any MONITOR/FINAL episodes.
Report actual native TRAIN actions separately from reused 16,832,
actual new MONITOR actions, actual optimizer updates and label
presentations; these settings are scientific comparisons, not a
custom cap on authorized research.

## Independent physical outcome and preregistered decision

Run the MONITOR source episodes **once** and evaluate all
sixteen learned models on identical public inputs. Native-state
full physical pickup/drop/unlock witnesses are private to the
evaluator and never used to choose learner action labels or score
predicted image changes. Score native 4-class macro-F1 on executed
action3/4/5, report classwise precision/recall/support, zero-effect
specificity, true-event recall, native effect counts, intervention
counts, an action-histogram and action-sequence hashes. At least
**8 witnessed effective** and **8 witnessed ineffective**
interaction steps needed for nonvacuous evaluation; otherwise HOLD.

Registered strict primary gate: probe-trained minus success-only
native effect macro-F1 > **+0.05 mean** and >+0.05 in **≥6/8**
paired seeds, AND probe-trained minus precondition rule
> **+0.05 mean** and >+0.05 in **≥6/8** seeds. Mean correct
positive-event recall >=**0.6** and native/public-proxy
agreement >=**0.8**, with seed-block 95% bootstrap uncertainty
reported; never claim a robust gain if its interval includes zero.
No favorable seed selection, monitor retuning or changed threshold.

This is a **source-effect** test only. A pass would support real
precondition estimation, not native UnlockPickup composition success.
Physical target success and independent compositional transfer must
still be tested through a *separate preregistered planner*,
source and held-out FINAL protections, with actual executed actions
rather than reward-only results. A failure remains evidence and
does not prohibit subsequent methods on NEW studies.

## Execution and governance

Extend existing serialized BAGI Research Loop and research
dispatcher with an exact-parent single-file `m23probe` request,
`force_retrain=true`. Run Ruff/pytest and independent
current-source push/PR CI BEFORE publishing the request. No
competing workflow or schedule change. Store new native source
TRAIN and MONITOR data, native-effect witnesses, model numeric
checkpoints, hashes and result under ignored
`runs/m23_probe_effects/`. Record true actions, epochs,
Adam updates, package versions, source and ZIP SHA, current
run IDs, per-seed outcomes, failure limitations. Append actual
measurements to the append-only research log and preserve
historical artifacts. Main is not integrated without PR #1
review and exact CI/merge provenance.
