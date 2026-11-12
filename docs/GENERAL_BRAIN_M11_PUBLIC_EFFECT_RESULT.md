# BAGI M11 — executed public-effect continual-retention diagnostic

**2026-11-12 UTC.** Completed real native two-task acquisition + fresh supervised fitting, but the **preregistered pilot gate FAILS**. This is a task-independent *predictive component* on two MiniGrid task families, **not** a task-solving policy or demonstrated general intelligence.

## Execution and artifact provenance

- Source: source-registration commit `f0270da17a341c87c6643e5068ca39a45ef4f39c`; actual admitted ADD-only research-request commit **`be3038cecd84aa19075971f73795919594f11741`**. Existing serialized [BAGI Research Loop run 43206421804](https://github.com/AncilBenny/BAGI/actions/runs/43206421804) SUCCESS. Both exact-source admission and research jobs passed, including real M11 runner and artifact upload. Prior source push CI 43205967321, PR CI 43205974180; request push CI 43206421635 and PR CI 43206430429 all passed Ruff, pytest and relevant PR merge provenance.
- Artifact: **`bagi-research-648`**, GitHub ID **13938876793**; connector-downloaded ZIP SHA-256 **`2aeae76dc4f256cb1cdd938794d4360a26a9a09adba966f7f1e766ff0bf9795b`** matches GitHub artifact digest. ZIP CRC passed and summary's study/source markers equal `m11` / actual source. Configuration SHA-256 **`ef1101e95aa32f9a7971892d9b5619c8868fd94a5f1e262e69488c77898af4db`**, committed frozen file `configs/m11_public_effect_retention.yaml`.
- Pinned observed runtime: Gymnasium **1.4.0**, MiniGrid **3.2.0**, PyTorch **2.15.0**, NumPy **2.4.6**; `torch.nn.GRU`/Adam and BAGI `PublicGoalActions` reused, no custom optimizer or RL algorithm. Model seed initializations **317,331**. Each has one independently trained source GRU, copied source weights for naïve+replay continuation, and one same-seed target-only scratch control. Two source fits, four target continuations and two target-only fits, each **96 actual optimizer.step calls**: **768 optimizer updates** across all eight model states.
- All eight compressed **numeric-only NPZ checkpoint SHA-256, member arrays, dtype/shape manifests and finite-value checks PASSED** using `numpy.load(allow_pickle=False)`. The original numeric models remain in the Actions artifact; no pickle deserialization, generated checkpoints, private data, credentials or raw trajectories were committed.
- Measured in-run wall duration **25.134 s**, excluding separate Action runner/dependency-install overhead. **0 teacher queries**, **0 privileged physical effect labels**.

## Actual native acquisition, independent episode protocol

| Actual data | Episodes | Executed native actions | Positive PUBLIC image-change labels | Native rewarded terminations |
|---|---:|---:|---:|---:|
| MemoryS7 TRAIN, seeds131101–131124 | 24 | 3,947 | 1,442 | 6 |
| DoorKey8x8 TRAIN, seeds141101–141124 | 24 | 6,144 | 2,268 | 0 |
| MemoryS7 FINAL, seeds132101–132112 | 12 | 2,447 | 860 | 4 |
| DoorKey8x8 FINAL, seeds142101–142112 | 12 | 3,072 | 1,211 | 0 |
| **Total** | **72** | **15,610** | **5,781** | **10** |

Actual train acquisition: **10,091 native steps**, final held-out public-effect testing: **5,519 native steps**. Native action histogram totals match all steps; episode seeds and SHA-256 splits are retained in artifact. Final effect-evaluation seeds were first acquired only after all eight model fits and checkpoint exports; 12 distinct final episodes per task, **shared across model seeds** (not two distinct environment replications). The acting behavior policy used deterministic-seeded **uniform native random actions**, not the trained predictor or a skill controller. Native Memory rewarded terminations above are random-policy events, not BAGI's learned competence; DoorKey had none. They were **not externally physically audited** as key→door→goal successes.

The supervised target is only `next public 7x7x3 image changed` after each actually executed action. It cannot identify hidden inventory, actual effective door openings or task completion. The predictors saw the current public observation/action/history, not the next image or privileged simulator state at prediction time.

## Independent final Brier scores (lower is better)

All final metrics are transition-weighted across the **same** independent Memory and DoorKey episodes, separate from training collection.

| Initial model seed | Model checkpoint phase | MemoryS7 Brier | DoorKey8x8 Brier |
|---|---|---:|---:|
| 317 | Source frozen after Memory only | 0.21027 | 0.21945 |
| 317 | Naive fine-tuning after DoorKey | 0.15385 | 0.15471 |
| 317 | Cached Memory replay after DoorKey | **0.14494** | **0.15443** |
| 317 | DoorKey-only scratch | 0.21101 | 0.21409 |
| 331 | Source frozen after Memory only | 0.20665 | 0.22359 |
| 331 | Naive fine-tuning after DoorKey | 0.14680 | 0.15002 |
| 331 | Cached Memory replay after DoorKey | **0.13521** | **0.15076** |
| 331 | DoorKey-only scratch | 0.20541 | 0.20928 |
| — | **TRAIN-fitted action-conditioned empirical frequency baseline** | **0.04051** | **0.05767** |

Constant no-change baseline Brier equals independent positive-image-change prevalence: Memory **860/2447 = 0.35145**, DoorKey **1211/3072 = 0.39421**. Source-only trained GRUs have much worse calibration than the highly predictive, inexpensive action-conditioned empirical baseline. After adaptation, the GRUs improve substantially but **all variants still underperform action-only baseline on BOTH tasks**. This undermines any claim that the learned recurrent model already offers useful effect prediction over the trivial action mapping; further positive-event diagnosis and normalization should precede policy integration. Even action-only baseline prediction does NOT mean physical causal preconditions are learned.

## Hypothesis gate and limitations

The preregistered gate required, separately for both seeds, **replay old-task Brier improvement >=0.015** over naïve new-task fine-tuning and **target-task replay Brier cost <=0.020**, plus positive image-change events in both tasks. Observed Memory Brier replay improvements were **0.00891 and 0.01159**, respectively: both below 0.015. DoorKey Brier replay costs were **-0.00028 and +0.00073**, inside the maximum cost condition. Thus the **full gate FAILS**.

Additional interpretation: naive *did not exhibit catastrophic forgetting on this measurement*. Instead, adaptation to DoorKey improved Memory Brier over the source-frozen checkpoint in both seeds (source and target share observation coding and many action effects). Cached rehearsal improved Memory further, but this is a small supervised same-benchmark gain, not the restoration of a lost task skill. Replay consumed **14,830/14,915 additional cached source-presentation actions** (seeds317/331) for the same **96 target optimizer updates** per continuation; do not claim equal CPU/computation or new simulator interactions for that replay. Two model seeds and a single task order are insufficient for robust statistics.

**Next falsifiable step (NOT executed in this result):** train-only *action-conditional calibration/normalization* comparison and separately audit labels by action type and effective interaction precondition using fresh eligible data, while keeping all 24 M11 FINAL episodes untouched for tuning. Explicitly test whether a learned recurrent head beats the action-frequency control on **positive and negative public effects** before deploying it as an acting-policy belief or claiming task-independent continual skills. Additional unrelated task families and physical task-success verifiers remain required. Workshop and XLand stay diagnostics, not generality proof.

**Governance:** older M10-P1 and M9 negative results/independent evidence remain unchanged; this result is preserved only as source documentation plus append-only log, not new native actions from documentation. Existing workflow cadence unchanged; no new workflow or duplicate run. Main integration remains **HOLD** while PR #1 lacks an approved review and postmerge main CI.
