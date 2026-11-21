# M23 stage 6 — real randomized source interventions: registered gate PASS (limited)

**Execution timestamp:** 2026-11-14 22:21–22:26 UTC / 2026-11-15 IST.
**Evidence:** actual native MiniGrid source actions and independent evaluator-only
physical-event witnesses, not target-task success or reward-only metrics.

## Provenance and exact resource accounting

- Evaluated source revision `472ed53f37af64ae6c7e2ed9c12f74412f7ef34f`.
- Existing serialized research workflow
  [#43647434248](https://github.com/AncilBenny/BAGI/actions/runs/43647434248)
  completed both audit and research successfully; exact-commit independent
  [push CI #43647434245](https://github.com/AncilBenny/BAGI/actions/runs/43647434245)
  and [PR CI #43647438891](https://github.com/AncilBenny/BAGI/actions/runs/43647438891)
  passed. Research runner Ruff passed, pytest **786 passed, 10 skipped,
  44 subtests passed**.
- Historical, **cached** TRAIN input: original source public-transition artifact
  #14127428720, executed original source
  `594c670a0d501fbd3cafe78ef47d59bef4733c0a`,
  containing **16,832 prior actually executed source native actions**.
  The new runner independently checked full input ZIP SHA-256
  `dd21e09790f91baffd07122712acdcf365aeb05534fc89d21a48001f605adadb`
  (`source.zip: OK`). Those actions are **not** stage-6 new interactions.
- **32,855 newly executed source TRAIN native actions** on DoorKey571001–571128,
  Fetch581001–581128, PutNear591001–591128, plus **8,197 newly executed
  source MONITOR native actions** on DoorKey572001–572032,
  Fetch582001–582032, PutNear592001–592032: **41,052 new real actions**.
  Source TRAIN and MONITOR used a registered probability 0.35 randomized
  native 3/4/5 action override on top of public source actors, with actual
  executed probes, not teacher labels or imagined transitions.
- **16 fresh PyTorch fits**: eight seeds
  941/947/953/967/971/977/983/991 x probe-trained vs historical
  success-only resampled, exposure-matched public-source control.
  **6,160 actual Adam optimizer-step calls per fit = 98,560 total**,
  Adam learning rate 0.001, 40 epochs, 128 batch. Source sample
  presentations and individual checkpoint hashes are stored in the
  immutable artifact; totals were not independently extracted here.
  Hidden native effects were NEVER fed as labels or features.
- Output artifact `bagi-research-674`, GitHub artifact ID **14134639518**,
  GitHub-reported ZIP digest
  `sha256:de2270383c7de3d52dea54f4be208e0383b83d52e33001a27ba85cbff512e82e`,
  expires 2026-12-14 22:26:03 UTC. The artifact ZIP and individual
  checkpoint digests were not independently extracted in this report.
- **No UnlockPickup target native actions, zero target FINAL evaluations,
  zero planner imagined actions or downstream physical composition successes.**

## Source MONITOR outcome: valid nonvacuous physical interaction coverage

- Across 96 independent MONITOR source episodes:
  **234 native witnessed effective interactions** and **4,269 ineffective
  attempted interactions** (4,503 total native actions 3/4/5).
  The old stage-5 failure, 69 effective and zero ineffective attempts,
  is no longer an obstacle to evaluating no-effect detection.
- Native four-class event macro-F1 (pickup, drop, unlock, no effect),
  against evaluator-only true physical changes, per seed:

| Seed | Probe-trained F1 | Success-only control F1 | Public front-cell/action rule F1 |
| --- | ---: | ---: | ---: |
| 941 | 0.999041 | 0.155875 | 0.540354 |
| 947 | 0.999041 | 0.149218 | 0.540354 |
| 953 | 0.999041 | 0.154539 | 0.540354 |
| 967 | 0.999041 | 0.142115 | 0.540354 |
| 971 | 1.000000 | 0.141582 | 0.540354 |
| 977 | 0.999041 | 0.135826 | 0.540354 |
| 983 | 1.000000 | 0.158505 | 0.540354 |
| 991 | 0.999041 | 0.141648 | 0.540354 |

- Runner reported **mean probe-minus-success-only +0.85186764**,
  **mean probe-minus-public-rule +0.45892659**.
  All eight probe models beat the two controls by >+0.05.
  Physical-positive and ineffective counts exceed the frozen
  ≥8/≥8 coverage requirement, and the runner returned
  `registered_probe_gate_pass: true`. Do not modify this preregistered
  gate or relabel historical failed studies.

## Interpretation and limitations

The experiment **supports** that executing and learning from
unsuccessful interaction attempts dramatically improves local source
native-effect prediction under the same randomized-probe data-generating
regime. This is a genuine new mechanism and meaningful BAGI-specific
measurement, not a general task-independent continual-learning
breakthrough. Native-event labels were reserved for evaluator scoring
while training targets came from the public pre/post front-cell proxy.

The near-perfect score could exploit easy negatives in source layouts,
unbalanced no-effect frequency, the fact that TRAIN and MONITOR share
source task and action-override policies, and image-visible local
preconditions. Performance under **changed intervention policies,
visually plausible but physically ineffective 'hard negatives', unobserved
carrying conditions, different objects/environments, or actual
model-based planning** has **not** been tested. 8 model seeds share one
monitor distribution, and nominal 4,503 action steps are temporally
correlated; do not treat them as IID confidence evidence or
task-transfer proof. An external replication with a different probe
policy and harder conditional negatives is now the next useful gate,
before original UnlockPickup FINAL453001–453032 is opened.

M22 remains independent; Workshop/XLand are diagnostics. The original
append-only results and source final layouts are preserved. No
generated checkpoints or credentials belong in commits. Maintain
existing workflow schedule; no competing research workflow or
unreviewed merge to main.
