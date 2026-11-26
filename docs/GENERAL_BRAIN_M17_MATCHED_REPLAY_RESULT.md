# BAGI M17 — independent matched-exposure source replay confirmation

**2026-11-12 UTC / 2026-11-13 IST. Preregistered M17 primary gate: FAIL.** M17 executes fresh native actions and 20 new PyTorch fits. On independent DoorKey8×8 physical goals, replay of old 6×6 labels achieved **93/128** versus **90/128** for exactly exposure-matched extra target-only labels. The +3 goals (= **+2.34 percentage points**) fell below the registered +12-goal threshold, although 3/4 model seeds individually favored source replay. This experiment **does NOT establish a reliably distinct old-experience replay benefit** beyond the extra cached label exposure control, and does not prove cross-domain or task-independent continual learning.

## Source, maintained work, immutable evidence and validation

- Frozen preregistration: `docs/GENERAL_BRAIN_M17_MATCHED_REPLAY_PREREG.md` and `configs/m17_matched_replay.yaml` (config SHA256 **`77801790c475622035aaad4f6e1818b19ac15c622fed6c13374c4ed3b506ce5b`**). Established PyTorch CE+Adam and BAGI M15 policy/89 public features, M14 public-only scripted native demonstration and external M10 physical auditor; official Farama MiniGrid (source https://minigrid.farama.org/main/environments/minigrid/DoorKeyEnv/); Rolnick et al., NeurIPS2019 replay foundation (https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html). No new algorithm claimed.
- Exact working source and ADD-only execution request commit **`8ec0be8836273ef3e96d76f7222647adf155605e`** (parent prereg source `81a336b1d2a0a6132e24349a97e2c0e45b221a75`). Independent push CI **[43286577536](https://github.com/AncilBenny/BAGI/actions/runs/43286577536)** and PR CI **[43286590256](https://github.com/AncilBenny/BAGI/actions/runs/43286590256)** both succeeded with Ruff, pytest and tested PR-merge provenance. Existing serialized [research run 43286577800](https://github.com/AncilBenny/BAGI/actions/runs/43286577800) passed audit, its own source/CI recheck, platform checks and research execution. Existing hourly schedule unchanged.
- Preserved [artifact `bagi-research-656`](https://github.com/AncilBenny/BAGI/actions/runs/43286577800/artifacts/13976021013), ID **13976021013**, ZIP SHA256 **`2923ebc7b4489647e65a6f301632266178f9ff1d28d07fa8b1a2ecee57caa02d`**. Independently downloaded and verified GitHub digest, ZIP CRC, summary source SHA, both numeric-only training trace SHA256, all **20 numeric-only checkpoint SHA256**, each member's dtype/shape and numeric finite values using `allow_pickle=False`. No private evaluator fields, credentials or checkpoint bytes were committed.
- Actual versions recorded in run: Gymnasium **1.4.0**, Farama MiniGrid **3.2.0**, NumPy **2.4.6**, PyTorch **2.15.0**. Experiment work elapsed **99.226 seconds** (excluding GitHub setup and installation). New training and held-out evaluation are genuinely executed native simulator actions, **not** cached M16 trajectories or prior independent-layout results.

## Actual resource use and replay matching

| Execution stage | Fresh native actions | Native episodes | Optimizer-step calls |
|---|---:|---:|---:|
| SOURCE TRAIN, 6×6, 48 new public-teacher episodes | **4,594** | 48 | 0 |
| TARGET TRAIN, 8×8, 48 new public-teacher episodes | **7,000** | 48 | 0 |
| Train-only monitor (44 conditions) | **77,862** | 528 | 0 |
| Independent FINAL (44 conditions) | **203,258** | 1,408 | 0 |
| Four SOURCE and sixteen TARGET model fit phases | 0 | 0 | **10,240** |
| **Totals** | **292,714** | **2,032** | **10,240** |

- Source TRAIN seeds 251001–251048, target TRAIN 261001–261048. Source training action histogram native0–6: **[1379,1383,1737,48,0,47,0]**, 45/48 teacher TRAIN-only physical goals. Target histogram **[1885,1808,3216,48,0,43,0]**, 35/48 TRAIN-only physical goals. Native teacher actions are actually executed; privileged teacher queries **0**. SHA256 of source public-feature/actual-action trace `6d45e02cf3ce515a6b289f94f2df1b02ba825567bffc886fce3ef0b0c47a4989`; target `4922758a1c623c5dcc5c4463b32b0635b657f4ccbf1e61c6d4db1e229b7ca508`. Four initialization seeds: **601,607,613,619**.
- Four genuinely trained SOURCE 6×6 fits, each **360 optimizer calls and 45,940 cached source-label presentations**. Sixteen genuinely trained target fits, each **550 optimizer calls and 70,000 primary target-label presentations**. Every target-stage arm used the identical target dataset and model seed's target minibatch ordering. The two extra-example arms each received **70,000 additional cached label presentations per seed** via equal-weight target+extra-example CE within the same 550 optimizer steps; only extra domain differs (old 6×6 source versus 8×8 target). Total actual optimizer calls **4×360 + 16×550 = 10,240**. No extra real native steps were obtained during model optimization. Numeric-only checkpoints for all 20 saved model states.
- Independent FINAL layouts 6×6 **253001–253032**, 8×8 **263001–263032**, all now **CLOSED** to training, follow-up tuning, or repeated confirmatory final evaluation. Monitor layouts252001–252012/262001–262012 are not pooled into final evidence. All 44 conditions at each split have correctly reconciled per-episode native seven-action histograms, physical-effect evidence, and real actions.

## Independently audited physical full goals (four model seeds, 32 new layouts/size/seed)

| Learned model condition | 6×6 source-task goals /128 | 8×8 target-task goals /128 |
|---|---:|---:|
| Frozen source-only BC | **111** | **78** |
| Target-only scratch BC | **109** | **80** |
| Source→target naive BC | **121** | **83** |
| Source→target with extra cached TARGET labels | **121** | **90** |
| Source→target with extra cached old SOURCE labels | **124** | **93** |

All counts are **physical key pickup followed by effective door opening followed by native rewarded goal** as externally audited; they are NOT action-choice or game-reward-only statistics. Every returned full-physical-goal episode's event-order and native reward were separately reconciled; this particular benchmark's rewarded terminal matched physical full-goal counts on all episodes. The public scripted teacher achieved 31/32 on 6×6 and 22/32 on 8×8 on its separate action streams; native random achieved 1/32 and 0/32.

For 8×8, genuine effective pickups were source-only128, scratch128, naive128, extra-target127, old-source-replay128 of 128 learner episodes each. Effective door-opening episodes were source-only105, scratch100, naive111, extra-target118, old-source-replay110. Positive pickup and opening counts do not automatically establish goal success.

### Registered gate verdict — FAIL

- Primary pooled 8×8 **source replay minus exposure-matched target extra = 93−90 = +3/128**, versus preregistered requirement **≥+12/128**. Thus **FAIL** even though positive seed differences occurred in 3/4 seeds, meeting the second half of the conjunctive gate.
- Per model seed (each out of32 independent final 8×8 layouts): seed601 **25 vs23 (+2)**; seed607 **20 vs27 (−7)**; seed613 **25 vs21 (+4)**; seed619 **23 vs19 (+4)**. This seed variability is substantive. Do not select the three favorable seeds and discard the negative one.
- Secondary descriptive: old-source replay 8×8 **+10** over naive, extra-target **+7** over naive, both with more cached label presentations than naive; source-only8×8 **78/128**, scratch **80/128**. On the 6×6 source task, naive **121** and replay **124** both exceed frozen source-only **111**, so **catastrophic forgetting was not demonstrated** in this experiment. Higher source-task scores after sequential training are not, by themselves, independent evidence of reusable skills.
- Post-measurement uncertainty diagnostic, **exploratory not a changed preregistration**: resampling 32 new 8×8 layout clusters with all four per-layout model seeds retained, 30,000 bootstrap draws with fixed RNG seed172026, source-minus-target goal-rate difference **+2.34 percentage points**, approximate cluster-bootstrap 95% interval **[−9.38,+13.28] pp**. Shared layout and stochastic policy trajectories limit inference; model-seed sampling n=4 is too small to claim a robust benefit or equality.
- Common pseudorandom-number streams were used for each layout/model seed across learner arms to reduce sampling variance, but realized actions/trajectories can differ. Label domains have different distribution and predictive information even at equal counts; the study estimates a matched-condition behavioral contrast, not a proof of an isolated neural learning mechanism.

## Interpretation, limitations, and research decision

M17 removes a key M16 confound: extra-target examples nearly reproduce the old-source-replay target gains. The data **do not support a large, reliable effect of old-source identity** within this shared DoorKey family and fixed public handcrafted feature/memory contract. The small +3 point difference and bootstrap uncertainty cannot justify declaring task-independent continual learning. The strong learned physical success is real for DoorKey under the hand-specified public pickup-evidence feature; it does not establish autonomous skill discovery, cross-domain continual retention, causal understanding or open-world general intelligence.

**Next most useful objective:** stop repeatedly iterating only same-mechanics DoorKey replay variants. Before another expensive fit, preregister a **different native task mechanism with independently auditable non-vacuous physical effects**, plus the maintained action-conditioned public encoder and a simple skill/history baseline. Measure true effects, goal completion, retained source performance and adapted target learning on fresh layouts. Workshop and XLand remain diagnostics, not substitutes for physical competence. A future study must not reuse M15/M16/M17 final layouts for tuning.

**Governance:** the research source is verified on `setup/initial-platform`. This result document and append-only log require fresh CI after publication. PR [#1](https://github.com/AncilBenny/BAGI/pull/1) remains open without approved review, so integration to `main` remains HOLD; do not force-merge. Preserve existing hourly schedules, artifacts and credentials.
