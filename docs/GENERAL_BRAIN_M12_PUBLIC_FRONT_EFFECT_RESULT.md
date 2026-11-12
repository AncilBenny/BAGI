# BAGI M12 — executed public forward-action effect prediction

**2026-11-12 UTC. Result: preregistered M12 pilot PASS for a narrowly defined PUBLIC observation-change prediction.** This is not proof of physical manipulation, learned control, cross-domain transfer, task-independent continual learning, or general intelligence.

## Source, prior research and artifact verification

- Study question was frozen **before fitting** in docs/GENERAL_BRAIN_M12_PUBLIC_FRONT_EFFECT_PREREG.md and configs/m12_public_front_effect.yaml. Original M12 implementation and registration commit 5e89daf81de4a241f6f938eb4a2be0dce1e2d35f; corrected/tested source-parent commit **f241534e444e93a0b790e39e61ee3881d9e3870b**. The actual ADD-only one-shot request and EXECUTED source commit is **b92c2aef8473a91760d3d500263410b4985f9483**.
- Existing serialized BAGI Research Loop [run 43212674445](https://github.com/AncilBenny/BAGI/actions/runs/43212674445) completed **successfully**, with separate exact-source and PR-merge-provenance audit and research jobs PASS. Source-parent push CI 43212278019 and PR CI 43212283492 successful; request push CI 43212674291 and PR CI 43212681098 successful (Ruff, pytest, current PR merge provenance). No duplicate workflow or schedule change.
- Original GitHub Actions artifact **bagi-research-649**, ID **13941424342**, download ZIP SHA256 **90304007dec262bb33825b9d74704de9106a7fa21d993e5f1f6bd50dcd1f31a5**, independently matching the GitHub digest. ZIP CRC passed; internal source and study marker match actual commit and M12. Config SHA256 **666a957ad69e1b2bec92c109b3f8b21d0c3a2189c19bf3f8e80eaba4424209fa**. Both numeric-only NPZ checkpoints verified with allow_pickle=False, exact SHA256, member names, shapes/dtypes and finite weights. No pickle model loading and no generated outputs in the repository.
- Reused, measured runtime versions: Gymnasium 1.4.0, MiniGrid 3.2.0, PyTorch 2.15.0 and NumPy 2.4.6; upstream public partial image and native actions, maintained nn.Linear and Adam. This is a documented *minimal integration/diagnostic*, not new algorithm research.

## Actual native actions, fits and split integrity

| Native source | Real episodes | Executed actions | Forward action2 examples | Forward public-change positives | Game-reward terminal episodes, unaudited |
|---|---:|---:|---:|---:|---:|
| MemoryS7 TRAIN, seeds151101–151132 | 32 | 5,930 | 896 | 412 | 5 |
| DoorKey8x8 TRAIN, seeds161101–161132 | 32 | 8,192 | 1,122 | 574 | 0 |
| MemoryS7 independent FINAL, seeds152101–152116 | 16 | 2,293 | 331 | 185 | 7 |
| DoorKey8x8 independent FINAL, seeds162101–162116 | 16 | 4,096 | 604 | 285 | 0 |
| **TOTAL** | **96** | **20,511** | **2,945** | **1,460** | **12** |

The final 32 layout episodes were acquired **only after both fresh model fits** and checkpoint exports. **14,122** new actions created the training dataset and **6,389** measured untouched independent effect evaluation. All per-action action/positive counts sum to each registered split and complete native-action total; all 96 seeds are unique and do not reuse M9–M11 final layouts. Data were collected by reproducible seeded **uniform random real native actions**. The fitted models did not act in the environment; these are not policy competence tests.

Two independent PyTorch logistic initialized model seeds 421/433, each with **141 parameters**: 140 action-interacted publicly visible front-cell object/color/state inputs + bias. Actual optimizer.step calls **672 each, 1,344 total**, and **169,464 cached training sample presentations per model**, 12 preregistered epochs, batch256, Adam lr0.01. Additional controls were TRAIN-fitted action-only Laplace-smoothed rates and TRAIN-fitted public front-object table shrunk to action-only rates (strength20), both with **0 optimizer steps**. **0 teacher queries**. Measured in-run wall time **4.691 seconds**, excluding setup/install overhead.
- Seed421 numerical checkpoint: **0b5a36a99e49f4b04792df01f504ff32ddc217b1a1a1914815bfdb2cb7917ece**.
- Seed433 numerical checkpoint: **ac956aad39219dd1bdc8453e1d20053a4d25749e97e87b211f1fc07eb4074201**.

## Independent native forward-action effect Brier scores (lower is better)

These are episode-disjoint PUBLIC image-change predictions for actual native forward action2, not physical goal success. Each model is evaluated on the **same** 16 held-out layouts/task.

| Predictor | MemoryS7 FINAL (185 positives, 146 negatives) | DoorKey8x8 FINAL (285 positives, 319 negatives) |
|---|---:|---:|
| TRAIN-fitted action-only frequency | 0.251471 | 0.249489 |
| TRAIN-fitted action×public-front-object table | 0.002470 | 0.000904 |
| Fresh logistic seed421 | **0.000221** | **0.000220** |
| Fresh logistic seed433 | **0.000221** | **0.000213** |

The preregistered M12 pilot required >=20 positive and >=20 negative independent forward action examples for **each** task, and >=0.010 forward Brier improvement over action-only for **both model seeds on both tasks**. Actual forward gains were **0.25125/0.25125** (Memory) and **0.24927/0.24928** (DoorKey), with coverage satisfied: **PASS**.

**Critically, the near-deterministic non-gradient front-object table almost matches the trained logistic model**, especially on DoorKey (0.000904 vs 0.00021). This tells us a simple *visible forward occupancy precondition* largely solves the registered prediction target within this simulator; it does NOT show a novel general-purpose causal learner or a reason to scale neural training on the same label. A positive pilot is not a robust measure of cross-domain transferable state learning.

## Action-specific negative/coverage findings

- Turns left/right (native0/1) changed the public partial-view image in **100%** of evaluated actions: Memory652/652 and DoorKey1169/1169. Thus they dominate the unexpectedly strong global action-only baseline seen in M11.
- All-actions independent Brier: action-only **0.036761 Memory / 0.059935 DoorKey**; learned seed421 **0.000657 / 0.010719**; front-object table **0.001138 / 0.013800**. These are PUBLIC view metrics only.
- Independent interaction actions pickup/drop/toggle3–5: Memory **0/984 positive changes**, DoorKey **101/1750**, comprising pickup52, drop49, toggle0; the DoorKey TRAIN split saw just **one** observable toggle change across 1198 native toggle attempts. Public image-change is an extremely sparse and incomplete proxy for effective physical toggling. No physical door-opening-success claim can be derived from these numbers.
- Random behavior achieved 5 Memory TRAIN and 7 Memory FINAL reward terminations, none in DoorKey; these reward events were **not** verified physical manipulation and **not** produced by a trained policy. Report separately from event-prediction accuracy.

## Decision and next open question

**Do not spend more training on this already-solved forward-image-change label.** The useful next unresolved task-independent effect question is whether a *public observation/action-history* predictor can distinguish genuinely valid manipulation preconditions and effective interactions rather than camera changes, with external private-state audit restricted to evaluation and no privileged training target. Need actual positive-event coverage under a public-only acquisition controller, an information-matched simple belief/history baseline, and independent entirely NEW evaluation layouts; otherwise the manipulation task is unidentifiable from scarce random successes. A future actionable study needs positive and negative **physical** outcomes measured separately from reward; another MiniGrid image-change metric does not count as general intelligence. Workshop/XLand remain diagnostic until independent cross-task policy success, retention and transfer exist.

Older M9/M10/M11 results, frozen gates and all independent layouts are preserved. M12 final seeds 152101–152116 and 162101–162116 now permanently closed to tuning, as are older heldouts. PR #1 remains open without reviews; integration to main **HOLD** pending reviewed PR and postmerge CI.
