# M23 preregistration — compositional causal transfer via learned world models

**Status: prospective and not yet executed (2026-11-15).** Prepared relative to research
HEAD 4f6a80c1221da652cd540136ff7845700cdbf915. The actual training-source SHA,
configuration, environment versions, seed ranges, executed actions and checkpoint
provenance must be recorded again at execution. This is a study separate from M22
(public spatial input ablation); M22 and the existing research schedule are unchanged.

## Objective, established work and reuse-first decision

M20/M21 yielded replay retention of 250/256 versus 216/256 physical DoorKey
successes under matched exposure; M21 failed to replicate Fetch forward transfer
(naive sequential 36/128 versus scratch 38/128). Those are archived measurements,
not M23 results. The new open question is whether an agent can learn reusable
**action-conditioned environmental effects** from source tasks and plan the
physical completion of a composition that was never demonstrated end-to-end.

Existing work checked:
- Kessler et al., *The Effectiveness of World Models for Continual Reinforcement
  Learning*, CoLLAs 2023: https://proceedings.mlr.press/v232/kessler23a.html
  (Continual-Dreamer on MiniGrid/MiniHack; source code compatibility requires audit).
- Liu et al., *Continual Reinforcement Learning by Planning with Online World
  Models*, ICML 2025: https://proceedings.mlr.press/v267/liu25p.html
  (online dynamics + planning; theoretical assumptions are not established for BAGI).
- Zhou et al., *Benchmarking World Models for Continual Learning on Compositional
  Tasks*, September 2026: https://arxiv.org/abs/2609.22055
  (motivates independent compositional transfer; do not transfer its all-task
  pretraining information into a claimed zero-shot experiment).
- Farama MiniGrid: https://minigrid.farama.org/main/environments/minigrid/UnlockPickupEnv/
  and https://minigrid.farama.org/environments/minigrid/PutNearEnv/
  (maintained native tasks and seven-action interface).

**Reuse:** maintained MiniGrid, Gymnasium, PyTorch; BAGI M14 public DoorKey
controller, M20 public Fetch actor and native witness, M15 behavior cloning,
M20/M21 matching/replay infrastructure, existing world model and MPC planner.
Contribution type is integration, adaptation and independent evaluation, **not
a new algorithm**. The existing Workshop model expects 2D categorical cells,
not native MiniGrid's partially observed 7x7x3 images, mission text, and own
action history. It needs a verified adapter and independent held-out event
prediction before its weights/outputs can be treated as a MiniGrid world model.
Do not blindly port a GPU-sized Dreamer implementation to a small CPU benchmark.

## Native tasks and the object-coverage confound

Target: **MiniGrid-UnlockPickup-v0** (unlock a locked door, then pick up the
specified colored box in the other room). Primary source:
**MiniGrid-DoorKey-6x6-v0** and **MiniGrid-Fetch-5x5-N2-v0**.
Fetch exposes key/ball targets, but **not boxes**. A direct transfer failure
could therefore be unseen-object mechanics, not missing compositional ability.

Register two source-experience regimes *before* target evaluation:
- **A, diagnostic:** DoorKey + Fetch only. It tests both novel object and
  composition. It cannot independently support a pure-composition claim.
- **B, primary:** DoorKey + Fetch + **MiniGrid-PutNear-6x6-N2-v0**. This adds
  non-target box pickup/drop mechanism experience without UnlockPickup task
  demonstrations, target reward, or target layouts. The agent still must
  compose these effects behind a locked door.

First verify source B actually produces **effective native box pickups** in
TRAIN data. The presence of a box in an observation is not proof the policy
has learned the pickup effect. The source-only comparisons within A and B
must see identical task-specific real transition pools. Do not select the
regime or model after seeing any target FINAL outcome.

## Learner and comparison rules

Gather actual native executed source actions, not labels the teacher might
have suggested. Every source record contains only public image, direction,
mission, own executed-action history, action, next public observation,
reward and termination flags. Privileged info is for evaluator-only physical
witnesses and posthoc debugging. Do not input full native grids, actual
agent coordinates, private carrying fields, hidden object IDs or an oracle
goal-stage plan into a learner or world model.

Use the **same full public observation, mission tokenizer, action-history
contract and logged transition pool** for all source-trained comparisons,
so differences cannot be attributed to the M22 front-cell restriction:
1. Source dynamics model + short-horizon receding-horizon model-predictive
   planner, with separately counted imagined actions and actual native actions.
2. Source-only public BC, matched on source transitions, labels, optimizer
   exposures and visible information.
3. Source-only BC with rehearsal, exposure-matched (record source/target
   presentations separately, never fabricate additional native interactions).
4. Native random reference, as an honest low floor.
5. Target-only scratch after **target TRAIN adaptation**; never use scratch
   target results to select the source-only zero-shot controller.

A maintained RecurrentPPO baseline can supplement these controls, but its
extra online samples must be recorded and cannot be called exposure-matched.
A scripted target skill, mission-specific hand-coded sequence, pretrained
target mission vocabulary inferred from target trials, or hidden-state planner
does **not** count as learning to compose. All controllers may read legitimate
public instructions under one fixed tokenizer. Report failures on unseen
"box" tokens and syntax rather than silently fixing them after final testing.

**TRAIN-only feasibility:** validate task IDs/action meanings; show native
positive examples of real key acquisition, locked door opening, effective
correct-object pickup and effective box pickup, and held-out source event
prediction. If source mechanisms are missing, record the true source actions
and coverage limitation, expand source-only acquisition as a separately
specified step, and do not open FINAL target layouts or use a privileged target
oracle. Prior failed scientific gates are not bans on new work.

## Frozen seeds, independent eval and physical outcome

Model seeds: **941, 947, 953, 967, 971, 977, 983, 991**. Source and target
layout sets are defined in bagi/experiments/m23_study_contract.py:

| Task | TRAIN | MONITOR | independently frozen FINAL |
| --- | --- | --- | --- |
| DoorKey | 431001–431128 | 432001–432016 | 433001–433032 |
| Fetch | 441001–441128 | 442001–442016 | 443001–443032 |
| PutNear | 461001–461128 | 462001–462016 | 463001–463032 |
| UnlockPickup | 451001–451128 (adapt only) | 452001–452016 (adapt only) | 453001–453032 (zero-shot only) |

These ranges must not overlap M20/M21 or any earlier sealed FINAL layouts.
The primary ZERO-SHOT evaluation is **8 independent model seeds x 32 FINAL
target layouts = 256 paired native episodes per arm per source regime**.
Freeze source model weights, planner hyperparameters and selection before any
target FINAL episode. Report per-seed and per-layout success paired by episode
and seed-blocked uncertainty; avoid treating shared layouts as 256 IID samples.

**Independent evaluator-only full physical chain:** confirm that the start
has a locked door and the target box behind it, no already-complete target,
at least one effective forward step, actual native key acquisition, then a
real native door-unlocking/opening effect, then correct requested box physically
carried after an executed pickup action, in strict time order, and positive
terminal reward. Record key/door/box native-effect event traces, actual action
counts, imagined planning actions, reward and physical success **separately**.
Planned actions, labels, mere reward or initially adjacent objects cannot
substitute for witnessed effective state changes. This gate is implemented
as a pure contract in m23_study_contract.py; the actual MiniGrid native
witness and independent source-only feasibility tests are not implemented yet.

**Preregistered necessary primary numerical gate, source regime B:**
zero-shot world-model+planner must exceed **both** exposure-matched source BC
and source BC+rehearsal by at least **16/256 nonvacuous full physical goals**,
with positive gain in **at least six of eight model seeds** for each
comparison. Report source-task retention and a seed-blocked 95% uncertainty
interval; do not claim reliable improvement if this interval includes zero.
Positive reward without physical chain is not a pass. No posthoc gate
change or final-data tuning is allowed. Success on one MiniGrid composition
does not demonstrate task-independent learning or real-world competence.

After zero-shot evaluation, target TRAIN adaptation may be examined using
**newly preregistered, separate FINAL layouts**; do not use the locked zero-shot
453xxx layouts again as an adaptation selector. A further composition such
as BlockedUnlockPickup is independent future replication, not a hidden
M23 pass condition.

## Provenance, execution and integration

Next steps are: (1) implement independent native UnlockPickup audit,
source-only collector and event coverage; (2) adapt/reuse a maintained
public-input world model and planner, with source-only event prediction smoke
tests; (3) run Ruff, pytest and real native smoke episodes in an isolated
checkout; (4) extend the existing **serialized** BAGI Research Loop
dispatcher/workflow with explicit source-bound M23 force_retrain and verified
CI/PR provenance; and (5) execute and archive true source fits and actions,
then only when preregistered preconditions pass open untouched FINAL layouts.

Do not create a duplicate workflow or alter existing schedule. Keep
checkpoints, credentials, trajectories and generated outputs in ignored
runs/m23_compositional_transfer paths. Record source SHA, exact config,
package versions, action traces/hashes, actual interaction and optimizer
updates per arm, wall time, model hashes, independent outcomes, failure
analysis and artifact IDs. Append actual results to
docs/autonomous-research-log.md; never overwrite M20/M21 final evidence.
Use setup/initial-platform through lease-guarded, non-force publication
after reviewing any concurrent workers. Main integration requires a
submitted PR review, Ruff/pytest, independent current push and PR
merge-provenance CI and postmerge verification.

**At preregistration:** new M23 native actions 0; fits 0; optimizer updates
0; target FINAL episodes 0. The locally validated pure contract test
does **not** mean MiniGrid mechanics were executed or scientific results exist.
