# M23 stage 4 result — source-only recurrent belief-history prediction FAILED registered gate

**Execution:** 2026-11-14 21:34–21:41 UTC (2026-11-15 IST). This is a
new source-only MiniGrid world-model prediction experiment, NOT physical
completion of held-out UnlockPickup or evidence of general intelligence.
Historical failed gates remain unchanged.

## Exact executable provenance and outputs

- Exact source revision:
  `716b79fc743489bc4268407560cb96e76e8d6803` on
  `setup/initial-platform`, admitted from one-file exact-parent request
  `.github/bagi-m23memory-once.json`, with `force_retrain=true`.
- [Existing serialized research run #43643513734](https://github.com/AncilBenny/BAGI/actions/runs/43643513734)
  audit + research jobs **SUCCESS**. Exact-commit independent push CI
  **#43643513676** and PR CI **#43643518563** passed. The research job
  ran `make check`: Ruff all checks passed; **777 pytest passed,
  10 skipped, 44 subtests passed**.
- Historical, reused source TRAIN artifact `bagi-research-669`,
  ID **14127428720**, source
  `594c670a0d501fbd3cafe78ef47d59bef4733c0a`.
  Training runner independently checked exact artifact ZIP checksum
  `dd21e09790f91baffd07122712acdcf365aeb05534fc89d21a48001f605adadb`
  (`source.zip: OK`) and source record member markers/digests
  through `load_source_artifact`. The 16,832 previously executed TRAIN
  actions were **reused/cached**, not newly executed in this study.
- New output artifact **`bagi-research-672`**, GitHub ID
  **14133287513**, GitHub-reported ZIP SHA-256
  `ceea3065a82f2b41ebdfb2f1abfa38e55983240f02ccfc8c202ea2d2838e30f9`,
  18,676,158 bytes, expiry 2026-12-14 21:41:04 UTC.
  Output ZIP interior and all 24 numeric checkpoint member hashes were
  **not independently downloaded/extracted** by this report author.
  Runner stdout confirms the outcomes below; detailed version strings,
  physical source event histograms and checkpoint digests are preserved
  within the artifact.

## Actual newly executed work

- One new source MONITOR collection on native DoorKey layouts
  **502001–502016**, Fetch **512001–512016**, PutNear **522001–522016**,
  all disjoint from previous monitored/frozen layouts. **2,250 actual
  newly executed native actions**, including **66 executed 3/4/5
  interaction actions** on 48 episodes. Actions and evaluator-only
  key/ball/box/door native effects are archived separately from reward;
  source effect histogram interior not independently verified here.
- **24 fresh PyTorch model fits:** eight model initialization/shuffle
  seeds 941,947,953,967,971,977,983,991, each trained for balanced
  MLP, one-public-frame GRU and eight-public-frame GRU.
  6 epochs, batch 256, Adam learning rate 0.001, same original source
  TRAIN trajectories and TRAIN-derived class weights.
  **396 actual optimizer-step calls and 100,992 repeated cached
  sample presentations per fit**: **9,504 optimizer updates** and
  **2,423,808 cached transition presentations** total.
  Both GRU arms used the same 96-hidden-unit recurrent architecture,
  with episode boundary resets. The MLP differs in capacity and
  serves as a secondary reference.
- **Zero target UnlockPickup environment actions, zero target FINAL
  episodes, zero MPC planning/imagination actions.**
  Workshop/XLand diagnostics were not used as transfer evidence.

## Frozen new MONITOR result (higher macro-F1 is better)

All values are next-public-object macro-F1 (door/key/box/ball/wall)
conditional on **actually executed** interaction actions 3/4/5.
The unchanged no-learning persistence baseline is 0.611225 across
all eight seeds.

| Model seed | Balanced MLP | GRU current frame (1) | GRU public-history (8) |
| --- | ---: | ---: | ---: |
| 941 | 0.529593 | 0.324488 | 0.321474 |
| 947 | 0.498744 | 0.321171 | 0.320622 |
| 953 | 0.519776 | 0.344888 | 0.341223 |
| 967 | 0.492444 | 0.323264 | 0.322541 |
| 971 | 0.516602 | 0.322559 | 0.336473 |
| 977 | 0.515799 | 0.322333 | 0.320022 |
| 983 | 0.480668 | 0.322324 | 0.332999 |
| 991 | 0.535019 | 0.323192 | 0.321676 |
| **Mean** | **0.511081** | **0.325527** | **0.327129** |

- **Architecture-matched GRU8 minus GRU1:** **+0.0016014** mean;
  required >+0.05 mean and >+0.05 in ≥6/8 seeds.
  Actual >+0.05 in **0/8**. FAILED.
- **GRU8 minus class-balanced MLP:** **−0.1839519** mean;
  required >+0.05 and ≥6/8 independent seeds.
  Actual >+0.05 in **0/8**. FAILED.
- **GRU8 versus no-learning persistence:** **−0.2840966** mean;
  needed positive advantage in at least 6/8 seeds; actual **0/8**.
- **Overall preregistered conjunctive source-memory gate: FAIL**,
  without changing thresholds or selecting favorable seeds.
  66 interaction steps, three source environments and eight shared
  initialization seeds are too little evidence for claims about
  broader recurrent world-model families.

## Scientific interpretation and next independent question

The tested fixed eight-frame GRU does **not** substantially improve
held-out source physical-interaction **public-image predictions**.
The almost equal GRU1/GRU8 scores suggest merely enlarging observed
history under this particular objective/architecture is insufficient.
Do NOT generalize this outcome to all belief-state models, recurrent
state-space architectures, full-sequence recurrent replay or MPC
competence; underfitting, egocentric camera-motion aliasing, sparse
interaction support, and imperfect model selection remain plausible.

Next mechanism-focused study should distinguish **camera-induced
visual changes** from **object/door/pocket physical effects** using
action-conditioned, evaluator-sealed event targets only for independent
scoring, with fresh source layouts and a suitable maintained
structured dynamics baseline. No retuning on the now-open 502/512/522xxx
monitors or any previously closed FINAL set. Before any target
UnlockPickup zero-shot composition, require independently verified
source causal-effect predictive utility and a separate preregistered
plan. Maintain existing schedule, preserve history/checkpoints,
avoid competing workflow, and hold `main` merge until PR #1 has a
submitted review and provenance-valid CI.
