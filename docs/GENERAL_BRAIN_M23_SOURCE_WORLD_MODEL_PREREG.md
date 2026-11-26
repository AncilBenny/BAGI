# M23 stage 2 preregistration — fit native public next-state model from verified source data

**Prospective, 2026-11-15; zero new model fits when locked.** This is one component of
the separately registered M23 causal-composition hypothesis, NOT the final
UnlockPickup experiment. Preserve prior M23 source event evidence and all M20/M21
FINAL layouts. Source availability gate passed in research run 43631440997.

## Reuse/provenance contract

Use the exact TRAIN-only public action/observation archive produced from source
commit 594c670a0d501fbd3cafe78ef47d59bef4733c0a, GitHub Actions
artifact 14127428720 (bagi-research-669), GitHub-reported SHA256
dd21e09790f91baffd07122712acdcf365aeb05534fc89d21a48001f605adadb.
The existing serialized research workflow must acquire the archive with
GitHub's read-only artifact API, verify whole-ZIP SHA256, source SHA marker,
study-mode marker, reported per-task public data digests and 16,832 true
TRAIN executed native actions before any optimizer call. Do not deserialize
pickles or uncontrolled model archives. The source data are cached historical
executed actions; they are **not new native interactions** during stage 2.
Do not re-acquire TRAIN or copy old results as new training.

Reuse maintained Farama MiniGrid 3.2.x, Gymnasium, PyTorch nn.Module/Adam,
and BAGI's existing hashed public-mission vector from M5. The predictor
uses all 7x7x3 **public** cells, direction, public mission token counts,
the preceding four **OWN executed** native actions, and the current
actually executed action. It learns next public object/color/state
categoricals, next direction and reward from true successor public observations.
No simulator full map, agent absolute position, hidden carrying field, evaluator
door/box effects or future target layout enters the learner.

## Training setup

One shared feed-forward prediction network across DoorKey6x6,
Fetch5x5-N2 and PutNear6x6-N2 source TRAIN transitions; two 192-wide
ReLU hidden layers, separate categorical next-cell/direction and reward
heads. Eight independent model initialization/shuffle seeds are exactly
941,947,953,967,971,977,983,991. Six epochs over the unchanged 16,832
cached source action transitions per seed; 256-example minibatches, Adam
learning rate 0.001. Training settings are **method-specific matched-comparison
choices, not administrative ceilings**. All minibatch presentations and
optimizer-step calls must be measured and reported, not inferred from
old runs. Save numeric-only PyTorch state arrays outside the repository with
SHA256; do not commit credentials or trained checkpoints.

## Independent public monitor and registered decision

Use fresh native episodes from source MONITOR layouts only (DoorKey
432001–432016, Fetch 442001–442016, PutNear 462001–462016), not closed
source FINAL or any UnlockPickup target layout. Controller may use only
public observations and own executed-action history. The external physical
event auditor records source pickup/unlock/drop effects but those are not
training inputs. Track all actual MONITOR native actions separately from
cached TRAIN actions. Use the same held-out transitions for each model seed;
do not count shared layouts as independent data.

Primary diagnostic: public next-object class macro-F1 (classes
door/key/box/ball/wall) conditional on native interaction action
3/4/5, compared against the **persistence baseline** which simply reuses
the current public image. Average macro-F1 advantage must exceed 0.05 and
at least 6/8 independent model seeds must each exceed 0.05 to pass the
source-prediction gate. Also report all-cell next-object accuracy,
event-action support, changed-public-interaction subsets, baseline metrics,
numeric checkpoint hashes and per-seed loss. Success of this gate is only
evidence of better public prediction; it is NOT evidence of actual native
causal-effect prediction or composed goal completion. If it fails, record
failure rather than tuning from target FINAL or asserting world-model transfer.

## Integration, outputs and scope

Register separate one-shot m23wm mode in the **existing** serialized
BAGI Research Loop with source-bound request and force_retrain.
No duplicate workflows, custom daily quotas or unreviewed main merge.
Write ignored outputs under runs/m23_world_model, including numeric checkpoints,
full report, dataset source and new monitor action accounting. Append actual
results (or failure) to docs/autonomous-research-log.md after execution,
preserving earlier entries; list exact fit counts, optimizer updates, source
commit, artifact digest, CI run IDs, and limitations. This test is stage 2:
it leaves the independently frozen UnlockPickup zero-shot FINAL
453001–453032 **unopened** and keeps M22 separate.
