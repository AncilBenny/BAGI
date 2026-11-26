# M23 stage 3 — fresh source-only object-class balance ablation result

**Outcome: registered primary source-prediction improvement gate FAIL.**
Executed on 2026-11-14 UTC (2026-11-15 IST). This is a genuine
source-world-model research measurement, NOT held-out UnlockPickup success,
not a general continual-learning result, and not a rerun of M20/M21.

## Exact provenance

- Executed `setup/initial-platform` commit:
  `1037d5816cf6b9265c89ed25e9abc6cfe6a645c8`.
- Existing serialized research run
  [#43640420904](https://github.com/AncilBenny/BAGI/actions/runs/43640420904)
  audited and executed SUCCESS. Exact-source push CI **#43640420868** and
  PR CI **#43640427656** both passed. Research job passed Ruff and
  pytest: **772 passed, 10 skipped, 44 subtests passed**.
- New immutable output artifact `bagi-research-671`, ID **14131992013**,
  GitHub-reported SHA-256
  `442b0059bb1e61068c9c5e809e6598b91d0da598ea35c572fcff5da59e4d6857`,
  16,402,260 bytes, expiry 2026-12-14 21:05:10 UTC.
  Output ZIP interior and per-checkpoint hashes were not independently
  opened in this report. Job stdout supports the numerical results.
- Reused original source TRAIN public dataset: 16,832 **cached native
  executed actions** from artifact #14127428720, original source
  `594c670a0d501fbd3cafe78ef47d59bef4733c0a`. Runner reverified
  original source.zip SHA-256:
  `dd21e09790f91baffd07122712acdcf365aeb05534fc89d21a48001f605adadb`
  (reported `sha256sum: OK`); input source marker and per-task digest are
  checked in code. Reused TRAIN actions were not counted as newly executed.

## New work and paired outcomes

- **16 fresh model fits**: eight seeds 941/947/953/967/971/977/983/991
  × ordinary cross-entropy and TRAIN-derived class-balanced object
  cross-entropy. Matched 6 epochs, batch 256, Adam lr .001, identical
  source transitions, model initialization and shuffle per paired seed.
  **396 actual optimizer steps per fit, 6,336 total**, with **100,992
  cached sample presentations per fit, 1,615,872 total**.
- **2,498 newly executed native SOURCE MONITOR actions** on fresh disjoint
  DoorKey472001–472016, Fetch482001–482016 and PutNear492001–492016
  layouts. There were **70 real executed action3/4/5 interaction steps**.
  Exactly the same monitor transitions evaluated all models.
  Zero UnlockPickup target actions, zero target FINAL episodes, zero MPC
  deployment/imagined planning actions.
- Macro-F1 evaluated for public next-object door/key/box/ball/wall on
  executed interaction steps, versus the identical observation-persistence
  reference `0.637373` on the new MONITOR split:

| Model seed | Ordinary next-object F1 | Balanced next-object F1 | Persistence F1 |
|---|---:|---:|---:|
| 941 | 0.435057 | 0.578778 | 0.637373 |
| 947 | 0.430649 | 0.484215 | 0.637373 |
| 953 | 0.399062 | 0.588610 | 0.637373 |
| 967 | 0.478283 | 0.482088 | 0.637373 |
| 971 | 0.417814 | 0.517365 | 0.637373 |
| 977 | 0.480209 | 0.494358 | 0.637373 |
| 983 | 0.481630 | 0.480481 | 0.637373 |
| 991 | 0.476042 | 0.517065 | 0.637373 |

- Mean ordinary F1 **0.44984329**, balanced **0.51786993**;
  paired improvement **+0.06802664**. Improvement >+0.05 on only **4/8**
  seeds (required ≥6/8), despite average gain exceeding +0.05.
  Thus the preregistered conjunctive gate **FAILED** without needing to
  infer its additional all-cell-accuracy condition. All eight balanced
  models still underperformed observation persistence on this metric.
- Model-seed and layout uncertainty are not IID: both arms/seed pairs
  evaluate on the same 48 environments; only 70 interaction steps
  enter the macro-F1 measurement. Output artifact contains full metrics,
  per-class supports, source physical event counts, seed bootstrap and
  numeric model/checkpoint hashes, but interior fields are **not
  independently verified by this report**.

## Interpretation

Class balancing seems to address part of the objective's minority-object
bias, but it does **not** solve held-out source action-effects prediction.
Do not repeat hyperparameter sweeps against this now-observed monitor or
revise its registered thresholds. A later study should separate egocentric
camera-motion prediction from action-induced object/door/carrying effects
using established action-conditioned structured predictors or belief-state
models, validated on *newly frozen* source data. Do not open or use
UnlockPickup FINAL453001–453032 until source causal predictions support
planning and all preregistered evaluation protections hold.

Historical M20/M21 source and FINAL results remain intact; M22 remains
a separate encoder experiment. Keep the existing hourly research and
autonomous schedules; no duplicate workflow, force push, credentials,
generated checkpoints in source, or unreviewed integration into main.
