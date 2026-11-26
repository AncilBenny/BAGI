# BAGI M16 — sequential public DoorKey BC transfer and rehearsal: independent native result

**2026-11-12 UTC. BOTH preregistered gates FAIL.** M16 completed a fresh, source-bound, independent physical evaluation. This is **within-family 6×6→8×8**, not cross-domain or task-independent continual learning. The learned replay arm nonetheless showed an exploratory +12/48 target-goal difference over naive fine-tuning, which is **not** the registered transfer comparison, and cannot be declared a successful M16 gate.

## Frozen design and provenance

- Pre-execution frozen registration: `docs/GENERAL_BRAIN_M16_SEQUENTIAL_BC_PREREG.md` and `configs/m16_sequential_public_bc.yaml`; run was admitted via add-only `.github/bagi-m16-once.json`. Registered M15 final layouts remain CLOSED.
- Source commit `7222dd125d37b35f01683cd683f4172799ec2d59`, exact source-bound push; research workflow [run #43282081200](https://github.com/AncilBenny/BAGI/actions/runs/43282081200) **SUCCESS**, with successful CI for this same source push [#43282080690](https://github.com/AncilBenny/BAGI/actions/runs/43282080690) and PR merge [#43282094961](https://github.com/AncilBenny/BAGI/actions/runs/43282094961). Earlier prereg source commit `cc9108f` failed Ruff import order; fixed in `514a436`, then both independent CI events passed. No failed CI is hidden or reclassified as scientific failure.
- [Preserved M16 run artifact](https://github.com/AncilBenny/BAGI/actions/runs/43282081200/artifacts/13972862655), artifact ID `13972862655`, digest `sha256:c9136fd4eb170675e6866b7981d3991c7b5bdee74478a1bc222c724f88a3e400`. Contains immutable numeric-only source/scratch/naive/replay checkpoint NPZ files (two model seeds), train_0/train_1 real public-action trace NPZ, run progress and independently audited summary JSON. Not committed into source. GitHub artifact retention until **2026-12-12 UTC**, subject to service policy. Summary config SHA256 `91a944aa3ba987cc9060360548db3b0a812b1b32ce667b173a8f0d3d03616243`.
- Fresh installed versions Gymnasium **1.4.0**, MiniGrid **3.2.0**, NumPy **2.4.6**, PyTorch **2.15.0**. Source MLP and public feature contract reused from M15, scripted teacher from M14, physical evaluator M10. No private grid/position/carrying entered policy; native auditor only reads private state after actual executed actions.
- Fresh actual **107,229 native actions**: 8,775 training-teacher actions (6×6 3,763, 8×8 5,012), 24,497 monitor and 73,957 independent-final actions. **3,000 actual optimizer-step calls**, **8 fresh fits**, **480 independent final episodes** across 10 actor conditions×2 sizes×24 layouts, plus 160 monitor episodes. Measured runner study elapsed **32.103 seconds** (excludes CI installation/queue). Two model seeds 547/563, source 32 fresh TRAIN layouts per size, independent FINAL seeds 233001–233024 and 243001–243024, with no tuning on FINAL.
- Source 6×6 training had 27/32 scripted-teacher physical goals; target 8×8 22/32. Executed action histograms source `[1231,1153,1315,32,0,32,0]` and target `[1369,1312,2273,32,0,26,0]`. Training trace SHA256: 6×6 `dc63b8b6c22a910d60c864703fc6145a33bf7b16635cf93b9c760b94eec3f861`, 8×8 `b6f712e0941c050e92a9712073e1ad2e550d68e08c4bc6ba3d71ee05d591ad22`. Each source fit had 300 optimizer calls and 37,630 cached target-label presentations; each target fit had 400 calls and 50,120 target presentations. Replay added 50,120 cached source presentations per seed without more target updates or native acquisition. Totals count source stage once per seed, not once per fine-tune arm.

## Independent final full physical goals, not merely reward

All numbers are actual **key pickup→effective door opening→rewarded goal** native audited episodes. Columns are two model seeds and pooled, denominator 24 per seed, 48 pooled.

| Learned arm | 6×6 seed547 | 6×6 seed563 | 6×6 pooled | 8×8 seed547 | 8×8 seed563 | 8×8 pooled |
|---|---:|---:|---:|---:|---:|---:|
| Frozen source-only | 22 | 23 | **45/48** | 13 | 11 | 24/48 |
| Scratch target-only | 18 | 19 | 37/48 | 12 | 15 | **27/48** |
| Naive source→target | 23 | 18 | **41/48** | 14 | 13 | **27/48** |
| Replay source→target | 22 | 21 | **43/48** | 18 | 21 | **39/48** |

Independent diagnostic scripted teacher: 23/24 goals on 6×6, 17/24 on 8×8; native uniform random: 0/24 each. All full physical goal counts match native rewarded terminations in this run; **reward agreement does not replace physical audit**. Every learned arm has nonzero native action execution and audited key/door events. Per-episode physical evidence, actual action counts and rewards are in the artifact summary.

## Frozen gate decisions

1. **Forward size-transfer gate FAIL:** naive 8×8 **27/48**, scratch 8×8 **27/48**. Difference **0**, required ≥**+4**. Source training did NOT improve target over scratch under this exact registered comparison.
2. **Rehearsal/retention gate FAIL:** replay source **43/48**, frozen source **45/48** (−2, satisfies no-more-than-4 loss), naive source **41/48** (replay +2, required ≥+4). Replay target **39/48** versus naive target **27/48** (+12, satisfies no-more-than-4 degradation). The required old-task replay-over-naive retention improvement is missing. Naive old-task forgetting was only −4/48 relative to frozen, leaving little room for a +4 improvement.

**Exploratory, not preregistered success:** replay target +12/48 over naive on the same 48 8×8 layout/model-seed cells. This comparison changes effective cached source exposure (50,120 extra source labels per model) and does not isolate an abstract continual-learning mechanism. No confidence interval or significance test was preregistered; model seed n=2 and stochastic rollouts are small and correlated. Treat as a hypothesis for a NEW study with independent seeds/layouts and matched label-exposure controls, not a retroactive M16 pass.

## Limits, integration and next useful test

- Both tasks are MiniGrid DoorKey and use engineered public front-cell/mission/direction features plus a hand-coded pickup-evidence bit; this does not establish general learning across tasks. Training labels come from a scripted controller; neither autonomous skill discovery nor cross-domain transfer was demonstrated.
- Native action and optimizer accounting are real. Replayed cached labels are explicitly additional training exposure, not additional environment actions. Teacher itself failed some physical goals on new layouts, so policy ceiling and demonstration quality remain confounds.
- The current reviewed PR path is [PR #1](https://github.com/AncilBenny/BAGI/pull/1). Research result documentation is a new source change and needs new push/PR CI; no merge to main until independently reviewed and exact provenance is green. Do not bypass review.
- Next falsifiable study: preregister replay vs equal-cached-label-exposure **target-only oversampling** and fresh third-party/changed-task generalization, or first isolate target-domain improvement with ≥3 new model seeds and fresh FINAL layouts. Keep all M16 FINAL layouts CLOSED to training, tuning, and future confirmatory evaluation.
