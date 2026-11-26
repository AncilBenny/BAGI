# BAGI M19 — independently audited reverse-order rehearsal: pooled retention signal, registered gate FAIL

**2026-11-13 UTC/IST.** Frozen preregistration: `docs/GENERAL_BRAIN_M19_REVERSE_REHEARSAL_PREREG.md`, `configs/m19_reverse_rehearsal.yaml`. Actual fresh native source GoToDoor8×8 → native target DoorKey6×6 study **completed**, but its **preregistered conjunctive primary gate FAILED**. Old-task source replay achieved **26/128** nonvacuous physical GoToDoor goals versus exposure-matched extra-target **13/128** (+13), meeting the minimum +12; new DoorKey target replay **124/128** versus naive **126/128** (deficit 2), meeting the maximum deficit 8. However, positive source-retention contrast was recorded in only **2/4** model seeds, below the required 3/4. No revision of the registered outcome.

## Immutable execution, evidence and reuse

- Exact admitted research source: `ef74b1b364a6efb800060d20261a323cafc9ffd5` (one-shot ADD-only `.github/bagi-m19-once.json`, with parent `acaae3789b7553a971f2243ffcbc47b28b16e80c`). Parent source push [CI #43372873782](https://github.com/AncilBenny/BAGI/actions/runs/43372873782) and independent PR [CI #43372878018](https://github.com/AncilBenny/BAGI/actions/runs/43372878018) SUCCESS Ruff, pytest, and exact PR synthetic-merge provenance. Execution-request push [CI #43373190225](https://github.com/AncilBenny/BAGI/actions/runs/43373190225) and PR [CI #43373197558](https://github.com/AncilBenny/BAGI/actions/runs/43373197558) also SUCCESS. The existing serialized [research run #43373190265](https://github.com/AncilBenny/BAGI/actions/runs/43373190265) passed source and independent CI admission, exact-source recheck, `make check`, fresh actual study execution and artifact upload.
- Preserved immutable `bagi-research-660` artifact **ID `14012913889`**, [run artifact link](https://github.com/AncilBenny/BAGI/actions/runs/43373190265/artifacts/14012913889), 604,132 bytes; GitHub digest and independently downloaded ZIP SHA256 **`5785206a87362e9a6129fa77a4fd96886d81aa890afb63345642880771d18f61`** match. ZIP CRC pass, 27 members, embedded `research_branch_commit.txt` exactly the executing SHA, `study_mode.txt` = `m19`. Both ignored numeric training datasets and **all 20 PyTorch numeric-only checkpoints** were verified by SHA256, safe `numpy.load(allow_pickle=False)`, expected tensor keys/dtypes/shapes, and finite values; all **96 native teacher episode action hashes** recomputed and matched using actual uint8 native action bytes. All 44 monitor and 44 final condition totals reconcile with per-episode native seven-action counts, physical goals and native reward.
- Frozen config SHA256 `1b319ebcdf30d5a9f84891a136f2d0acb0b112f58fac4bee38248db93b4c14c8`. Versions recorded from actual execution: Gymnasium **1.4.0**, official Farama MiniGrid **3.2.0**, NumPy **2.4.6**, PyTorch **2.15.0**. Experiment elapsed **117.001 s** excluding CI/setup/install. Public teacher privileged queries **0**. All 20 numeric learned models use existing tested PyTorch M15/17/18 `BCPolicy` and existing public 89-feature/action-history contract. No checkpoints or credentials were committed.
- Reused primary work: Rolnick et al., [*Experience Replay for Continual Learning*](https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html), official maintained Farama [GoToDoor](https://minigrid.farama.org/main/environments/minigrid/GoToDoorEnv/) and [DoorKey](https://minigrid.farama.org/main/environments/minigrid/DoorKeyEnv/), PyTorch CE/Adam, BAGI native M10/M18 independent physical audit, M14/M18 public teachers and M17/M18 matched cached example controls. This is an *adaptation and controlled evaluation*, **not** a newly invented continual-learning algorithm, full CLEAR replication, or cross-world transfer.

## Actually executed resources (all NEW M19 actions and fits)

| Stage | Actual native actions | Native episodes | Optimizer calls |
|---|---:|---:|---:|
| Source GoToDoor8×8 public-teacher TRAIN (48 new layouts) | **5,284** | 48 | 0 |
| Target DoorKey6×6 public-teacher TRAIN (48 new layouts) | **5,292** | 48 | 0 |
| TRAIN-only monitor (44 conditions) | **81,069** | 528 | 0 |
| Independently frozen FINAL (44 conditions) | **201,324** | 1,408 | 0 |
| Four SOURCE and sixteen TARGET PyTorch fit phases | 0 | 0 | **8,400** |
| **Total** | **292,969** | **2,032** | **8,400** |

- Source actual-action native histogram (0–6) `[1249,1271,2724,0,0,0,40]`; 38/48 source TRAIN layouts achieved nonvacuous public-teacher GoToDoor goal. Source numeric action/feature trace SHA256 `040a2081e23449cfd559696c1160ff583f3097d4731bb1243c9b86dda3de69ca`. Target native action histogram `[1644,1611,1941,48,0,48,0]`; 47/48 target TRAIN layouts achieved externally audited pickup→door→goal under reference. Target numeric trace SHA256 `ed5930df44d2dd667b9c5b2046619bf797939207b2f8e2ab1b70336adb006160`. TRAIN seeds: source291001–291048 and target301001–301048; monitored292001–292012/302001–302012; new independent final293001–293032/303001–303032. Four fresh model initialization seeds **701,709,719,727**.
- Each of four SOURCE fits made **420 actual optimizer calls** and **52,840 cached source example presentations**. Each of sixteen TARGET fits made **420 actual optimizer calls** and **52,920 primary target-label presentations**. For each model seed, both extra-target and source-replay arms got **52,920 additional cached label presentations** (equal counts, target minibatch sequence and weighted CE objective), not extra native simulator actions. Total optimizers `4×420+16×420=8,400`. Unchanged architecture hidden64, 7 native action logits and 89 public inputs; 10 matched training epochs per stage, batch128, Adam learning rate0.003 are preregistered *experimental matching parameters*, not a general training ceiling.

## Frozen independent physical outcome — no reward-as-skill substitution

| Arm | Old GoToDoor8×8 nonvacuous goals /128 | New DoorKey6×6 complete physical goals /128 |
|---|---:|---:|
| Frozen SOURCE-only | **25** | 0 |
| Target SCRATCH | 1 | **120** |
| SOURCE→TARGET naive | **12** | **126** |
| SOURCE→TARGET plus equal cached TARGET labels | **13** | **125** |
| SOURCE→TARGET plus old SOURCE replay | **26** | **124** |

- Native source GoToDoor success is scored only after **ACTUALLY executed `done=6`** at the **mission-color-matched door**, native termination, real earlier forward movement and non-target-adjacent initial spawn. Native reward independently counted: old-source replay **27/128 reward** versus **26/128 nonvacuous physical**: one reward-only/presatisfied case is correctly excluded. The other learned arm source successes equaled native rewarded counts (25,1,12,13 respectively). Scripted reference old-task FINAL achieved 26/32 nonvacuous physical vs 27/32 rewarded; uniform 0/32 physical vs1/32 rewarded. These numbers show why reward alone is insufficient.
- Native target DoorKey physical success requires externally audited true effective key pickup → door opening → rewarded physical goal. Target frozen-source 0/128, scratch120/128, naive126/128, extra-target125/128, source-replay124/128; physically verified goals equal terminal rewards *on this target*. Target actual pickup episode counts respectively **5,128,128,128,128**, effective door open counts **0,126,128,128,127**. Scripted teacher target 29/32 full physical goals and uniform 2/32.
- All model evaluation policies use the original hand-engineered **public** local front tile, heading, mission and own-executed-action pickup evidence; private position/target/pickup/door data are *evaluator-only*, never policy/teacher labels. Per-condition native action counts, histograms and per-episode native reward/physical outcome reside in the immutable workflow artifact. Same-layout model-seed RNG couples sampling for comparison but realized trajectories need not match.

### Conjunctive preregistered gate — **FAIL**

| Criterion | Measured | Required | Decision |
|---|---:|---:|---|
| Old GoToDoor replay minus equal-exposure extra-target physical goals | **+13/128** | ≥+12 | Meets |
| Independently initialized seeds with positive retention gain | **2 of 4** | ≥3 of 4 | **Fails** |
| New DoorKey naive minus replay physical goals | **2/128** | ≤8 | Meets |
| **Primary result** | | All three required | **FAIL** |

Per model seed old-task replay vs extra-target (out of32 new final layouts): **701:7 vs2 (+5); 709:8 vs0 (+8); 719:4 vs4 (0); 727:7 vs7 (0)**. The gain is concentrated in two seeds; do not select only favorable seeds. Target replay vs naive per seed: **701:31 vs31; 709:31 vs32; 719:30 vs32; 727:32 vs31**.

**Exploratory uncertainty, post-result not a revised gate:** resample 32 final layout clusters jointly retaining each layout's four model-seed cells, 20,000 bootstrap draws, RNG seed192026: old-task replay minus extra-target **+10.16 percentage points**, approximate 95% interval **[+0.78,+19.53] pp**; new-task replay minus naive **−1.56 pp**, approximate interval **[−5.47,+2.34] pp**. The layout-cluster interval does **not** account for uncertainty about the population of model initializations; only four seeds and concentrated effects prevent claims of robust general benefit.

## Interpretation, limitations and next evidence gate

M18 had severe DoorKey→GoToDoor naive forgetting, which rehearse-old-source recovered. In the reversed task order, naive GoToDoor retention **12/128** already exceeds equal-target **13/128** almost equally, old-source replay **26/128** is near frozen source **25/128** with target DoorKey still 124/128. This is useful but **seed-heterogeneous** within shared MiniGrid mechanics and the fixed engineered representation/public scripted demonstration contract. It does not establish open-world/task-independent continual learning, autonomous skill discovery, general causal understanding or AGI. Neither old nor new independent FINAL layouts should be reopened for tuning, and *all M15–M19 final streams are CLOSED*.

The next useful high-leverage study is a genuinely different *mechanical family* such as native Fetch/PutNear, requiring witnessed actual pickup/carry/drop or relative-object motion rather than only rewarded completion, or a matched recurrent-history baseline where public observation cannot reveal carried state. Reuse maintained Farama, PyTorch or SB3-Contrib and earlier real-action provenance/audits; preregister independent native physical outcome seeds, control labels/updates and evaluation before more fitting. Workshop and XLand remain diagnostics.

**Governance:** exact M19 execution source passed both independent CI events. This **new report and append-only log** are documentation updates requiring fresh exact reporting-HEAD push CI and PR CI verification after publication. PR [#1](https://github.com/AncilBenny/BAGI/pull/1) still has **no independent review**; main integration HOLD until reviewed PR, verified exact-source/base/merge provenance and postmerge main CI. Existing hourly research and autonomous schedules unchanged, no duplicate worker, no forced ref updates. The earlier autonomous gate's reporting-commit repair is on the research branch only, not yet deployed to trusted main.
