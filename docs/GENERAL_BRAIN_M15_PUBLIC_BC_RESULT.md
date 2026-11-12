# BAGI M15 — learned public-action behavior cloning, independent physical DoorKey results

**2026-11-12 UTC. Registered M15 balanced-over-ordinary pilot: FAIL. Learned physically verified goals: YES.** Two task-family-specific learned policies achieved 67/96 (ordinary behavior cloning) and 66/96 (class-balanced behavior cloning) independently audited full key→door→goal episodes. The pooled balanced improvement was **−1 goal**, not the preregistered +4. This is **learned closed-loop physical competence within a shared DoorKey task family**, NOT a demonstration of task-independent continual learning, arbitrary transfer, autonomous skill discovery, or AGI.

## Protocol, primary literature, baseline, and execution provenance

- Frozen BEFORE training: `docs/GENERAL_BRAIN_M15_PUBLIC_BC_PREREG.md`, `configs/m15_public_bc.yaml`, using established supervised cross-entropy behavior cloning, PyTorch Adam and two-layer MLP, and Farama MiniGrid 3.2.0 DoorKey6×6/8×8 native action environments. Prior physically competent *hand-programmed* public controller: `docs/GENERAL_BRAIN_M14_PHYSICAL_GOAL_RESULT.md`; external evaluator-only physical audit: `NativeDoorKeyAudit`. The actor never receives private grid/carrying/door state, external outcome flags, post-hoc evaluator rewards, or the teacher's action at evaluation. Its one-bit pickup-observed memory feature IS hand specified and derived solely from its own executed action and public successor observation. The learned policy is not a generic policy architecture validated across unrelated tasks.
- Original source-bound request `f517acb555cf865f8863b7c5bfb31ee6450abc32`; [research run 43270819544](https://github.com/AncilBenny/BAGI/actions/runs/43270819544) interrupted during train-only monitor when BAGI rejected legitimate simultaneous native termination and time-limit truncation. Its artifact `bagi-research-653` (ID 13969450021, ZIP SHA256 `0f8adb9cf44490e95d968219b4ccd0d68d66b03edd136dcb110d4eb04350c2cb`) preserves 8,087 training actions, 4 fits, and 2,560 optimizer steps, but **no independent final episode/result**.
- Corrective commit `a1f9fedeb2c4926b7363a6dc4fba88eb4ff5026a`: native auditor accepts simultaneous `terminated=True`/`truncated=True`, preserving unmodified native flags and independent physical success evidence. Source push CI [43275493070](https://github.com/AncilBenny/BAGI/actions/runs/43275493070), PR CI [43275508763](https://github.com/AncilBenny/BAGI/actions/runs/43275508763) both passed Ruff/pytest, with PR merge-provenance success.
- **Completed independent M15 source:** `0e24d34cabe4e50791a67b220a8931a7aec8cb88` from separate ADD-only repair request `.github/bagi-m15-repair-once.json`, exact parent/source guard, unchanged single serialized workflow. Its push CI [43275968274](https://github.com/AncilBenny/BAGI/actions/runs/43275968274) and PR CI [43275981232](https://github.com/AncilBenny/BAGI/actions/runs/43275981232) passed Ruff, pytest, and merge-provenance where applicable. [Research run 43275968276](https://github.com/AncilBenny/BAGI/actions/runs/43275968276) audit and compute jobs completed SUCCESS, including research-platform `make check`.
- **Primary artifact:** `bagi-research-654`, ID **13969804611**, ZIP SHA256 **`238ddbdca5fb7d7ed568393f64b17e2b9a3df108f4116ac740fc42e6da19beff`**. Verified against GitHub's stored digest; ZIP CRC passed, source SHA and study `m15` agree, four numeric-only `.npz` model member manifests and per-model SHA256 verified with `allow_pickle=False`, numeric training data SHA256 verified. Versions recorded: Gymnasium1.4.0, Farama MiniGrid3.2.0, PyTorch2.15.0, NumPy2.4.6; frozen config SHA256 `d4dc9e219c4a12918074e1c0f38a8f2f06e27f6dc2145ddd99e1a5a3000a2705`.

## Actual fresh native actions and optimizer accounting

| Stage | Actual native actions | Episodes | Training optimizer steps |
|---|---:|---:|---:|
| Fresh public teacher TRAIN acquisition | **8,087** | 64 | 0 |
| TRAIN-only monitor (six conditions per task) | **15,069** | 96 | 0 |
| Independently held-out FINAL (six conditions per task) | **47,707** | 288 | 0 |
| Model fitting, four fresh matched fits | 0 | 0 | **2,560** |
| **Total** | **70,863** | **448** | **2,560** |

- Real TRAIN label histogram for native actions 0–6: **[2328, 2304, 3327, 64, 0, 64, 0]**; genuine public teacher episode goal count 56/64 is TRAIN-only, not independent. 8,087 public feature vectors of width 89 with actual executed action labels; no privileged teacher queries. All four PyTorch Adam fits ran 10 epochs, batch128, lr0.003, hidden64, 6,215 parameters each. Every fit executed **640 optimizer step calls and 80,870 cached label presentations**. Two initialization seeds 521/533, ordinary and sqrt-frequency-weighted class-balanced cross-entropy; balanced maximum class weight 8.0. There were zero new simulator interactions DURING optimizer training and zero cached evaluation substitutions.
- The completed fresh run reproduced the prior interrupted run's five `.npz` files **byte-for-byte** (the numeric teacher dataset and four checkpoints); this is deterministic fresh execution, not evidence of five different datasets or an independent training replication. Fresh **monitor and independent final** physical measurements only belong to successful run 43275968276.
- Registered TRAIN seeds 211001–211032/221001–221032; MONITOR 212001–212008/222001–222008; FINAL 213001–213024/223001–223024. Earlier M9–M14 independent final layouts and now M15 FINAL are **closed to all tuning**. The trained models were frozen before monitor and final. Episode outputs preserve actual native action histograms, seed/RNG mode and external physical-event evidence.

## Independent FINAL physical-event funnel (24 episodes per task/condition)

| Task | Actor | Effective key pickup | Effective door opening | **Fully physically verified goals** |
|---|---|---:|---:|---:|
| 6×6 | Public scripted teacher | 24 | 24 | **24** |
| 6×6 | Uniform native random | 22 | 4 | **0** |
| 6×6 | Ordinary BC, seed521 | 24 | 23 | **20** |
| 6×6 | Balanced BC, seed521 | 24 | 24 | **22** |
| 6×6 | Ordinary BC, seed533 | 24 | 23 | **21** |
| 6×6 | Balanced BC, seed533 | 24 | 24 | **22** |
| 8×8 | Public scripted teacher | 24 | 18 | **16** |
| 8×8 | Uniform native random | 12 | 0 | **0** |
| 8×8 | Ordinary BC, seed521 | 24 | 22 | **14** |
| 8×8 | Balanced BC, seed521 | 24 | 19 | **13** |
| 8×8 | Ordinary BC, seed533 | 24 | 21 | **12** |
| 8×8 | Balanced BC, seed533 | 24 | 20 | **9** |

- **Ordinary learned BC:** 6×6 pooled 41/48, 8×8 pooled 26/48, total **67/96 = 69.8%** independently verified physical full goals. **Balanced learned BC:** 6×6 pooled44/48, 8×8 pooled22/48, total **66/96 = 68.8%**. Uniform random 0/48; scripted teacher 40/48 on its separate random action streams. These policy streams use the same new layouts but different seeded stochastic action sampling; episode outcomes are not matched-action controlled comparisons. The per-episode external `full_physical_goal` equals the rewarded-terminal indicator in this artifact, but **the goal claim is grounded in actual pickup, door opening and native rewarded terminal separately**, not the reward alone.
- The preregistered primary gate required balanced ≥10/96, gain ≥4 against ordinary, and ≥1 physical goal for every balanced task-size/model-seed. Balanced achieved66 and every seed/task was positive, but the observed gain was **−1**; **the overall gate FAILS**. Class weighting was not beneficial overall on these held-out layouts. No post-hoc class-weight tuning or model selection is licensed on these final seeds.
- **Exploratory uncertainty, not a new preregistered gate:** comparing balanced versus ordinary on each of 48 layout clusters (24 per size, pooling two policy seeds), a deterministic stratified-by-size cluster bootstrap (20,000 draws, RNG seed85124) gives observed difference −1.04 percentage points and approximate 95% interval **[−11.46,+9.38] pp**. Same-layout outcomes are correlated; simple independent-binomial confidence intervals would understate shared-layout dependence. This interval does not demonstrate a gain or prove equivalence, and covers stochastic action differences as well as weighting effects.

## Interpretation and next decision

1. **Genuine acquired task-family policy:** a PyTorch model trained solely on real public teacher actions independently completes numerous *physical* key→door→goal episodes on new 6×6 and 8×8 layouts. This directly clears the earlier concern that BAGI had never demonstrated a learned policy with meaningful closed-loop native physical success. The scripted source teacher, 89 engineered features and single hand-specified history bit are still substantial assistance.
2. **Rare-action imbalance hypothesis not supported:** balanced and ordinary models perform similarly overall; balanced success in 8×8 deteriorates. More fitting/tuning on the now-closed M15 FINAL cannot honestly establish a better model.
3. **What is NOT demonstrated:** task-independent skill discovery, continual retention, causal transfer beyond shared DoorKey mechanics, robust 8×8 performance, or general intelligence. The next genuinely open question requires a preregistered *new* task/seed family and a strong simple public-history/behavior-cloning baseline measuring learning, retention and transfer, rather than continually editing DoorKey rules to increase goal counts. Workshop/XLand remain diagnostics, and independent physical action effects must continue to be audited.
4. **Governance:** this is a completed branch research result, NOT yet a completed integrated milestone. PR [#1](https://github.com/AncilBenny/BAGI/pull/1) still needs an approved independent review, current push and PR CI and postmerge main CI before merging. Preserve existing schedule; source/checkpoints/traces remain outside the repository.

