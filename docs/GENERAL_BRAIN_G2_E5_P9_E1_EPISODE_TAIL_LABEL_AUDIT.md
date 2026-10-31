# G2-E5-P9 — E1 XLand TRAIN public-change label and episode-tail audit

**2026-10-26; retrospective source-only measurement, NOT an independent model fit or P9 native result. P9 remains STOP.** This audit reuses the already-published E1 TRAIN artifact and BAGI's unchanged `_shift_within_episodes` / public-feature contracts. It does not modify P8 scores, P9's frozen prospective protocol, actor weights, decision gates or action selection. The question is whether the P8/P9 public-visible-change target contains meaningful positive action-outcome coverage beyond trivial action/heading changes.

## Independently verified source and information boundary

- Native E1 run `40165691302`, artifact `12588790887` (`bagi-g2-e1-public-effect-native`), ZIP SHA-256 **`aaf8eb011763bf47e1d196771e257eb13c4eb35294bbeed2b71757ade525f073`**; downloaded and independently rehashed byte-for-byte.
- Outer ZIP members: `summary.json` and `public_effect_numeric.npz`. Opened only allowlisted `xland_train_*` numeric keys using NumPy `allow_pickle=False`; did **not** parse the summary or materialize any `*_test_*`, original E2 EVAL, private pocket/grid, heldout expert label, or native outcome arrays. The separately published TRAIN-only per-episode `hidden_without_public_change` diagnostic was aggregated **after** collection and never used as a predictor feature.
- E1 XLand TRAIN contains **180 episodes** across **15 layouts**, 90 frozen-BC and 90 random-behavior episodes, **3,244 actually executed actions** with **2,019 public-change labels**. The existing P8 within-episode shift selects **3,064** rows (3,244 minus one final action from each episode), including **1,897** positive public changes. Recomputed current/next public-feature inequality, excluding the four executed-action history slots, agrees with all 3,064 stored labels (**0 disagreements**).
- The 180 excluded **episode-tail** actions contain **122 positive public changes**. An episode tail is not necessarily a native terminal transition; it may be the 20-action cap. The excluded actions include **all four positive action-4 public changes** in the original E1 TRAIN labels. No claim is made that those four were effective BLUE drops or goal completions.

## Exact action support and label censoring

| Native action ID | All E1 TRAIN executed actions | All public changes | P8 shifted rows | P8 shifted public changes | Excluded tail actions / positive |
|---|---:|---:|---:|---:|---:|
| 0 | 937 | 766 | 903 | 737 | 34 / 29 |
| 1 | 640 | 640 | 603 | 603 | 37 / 37 |
| 2 | 566 | 566 | 540 | 540 | 26 / 26 |
| 3 | 358 | 43 | 317 | 17 | 41 / 26 |
| 4 | 444 | 4 | **416** | **0** | 28 / 4 |
| 5 | 299 | 0 | 285 | 0 | 14 / 0 |
| **Total** | **3,244** | **2,019** | **3,064** | **1,897** | **180 / 122** |

On P8's 3,064 within-episode rows, 1,870 show local-image change, 1,143 show heading change, and **zero** show changed public rule/goal instruction tokens. These components overlap: 1,116 rows changed both image and heading, 754 image only, 27 heading only; 1,167 unchanged. Actions 1 and 2 have a changed public heading on **every** included row (603/603 and 540/540). Actions 4 and 5 have **zero positive public-change labels** in the shifted training sample, although action 4 had four positive labels among excluded episode tails. Only action IDs 0 and 3 have nontrivial mixed labels in that shifted sample.

The E1 evaluator-only counter reports **117 native-state changes without a public-change label across 3,244 executed actions**, in **62/180 episodes** (84 such steps under frozen BC and 33 under random behavior). Its source definition compares grid, agent position/direction and pocket *after* a real step. These counts cover **all** executed actions, not only the 3,064 shifted rows, and are not linked here to particular action IDs, BLUE movement or rewarded success. They demonstrate an observational information limit; they do not establish whether any individual hidden change was inferable from prior public history.

## Interpretation, reuse decision and next falsifiable gate

**New bounded negative finding:** P8's favorable held-TRAIN Brier scores measure prediction of a target in which turns are always public changes, two action categories are always unchanged, and the four observed action-4 positive labels are all censored by episode-tail shifting. This does **not** invalidate the registered P8 numerical result, but it sharply limits interpretation as a learned general action-outcome predictor, and exposes a **prospective P9 TRAIN-label coverage risk** if the same shifted target is reused. No conclusion about true physical pickup/drop success or cross-domain transfer follows.

Reuse the existing E1 collector's directly recorded public-change labels, the strict TRAIN-only reader, established BCE/Laplace controls and source-group splits; do **not** invent a BLUE-specific predictor or silently revise the already-scored P8 experiment. Before any P9 native actions, separately preregister and test an **episode-tail-inclusive public-event training contract** against the unchanged shifted control with matched data/fit budgets and explicit positive-action support in both XLand and Workshop. Preserve all existing P9 actor-provenance, seed-history, calibration, public/evaluator isolation and 160-reset/4,480-action hard STOP gates. A positive event prediction would still not establish causal recovery or skill.

**Resource ledger:** one existing ~105 KB GitHub artifact downloaded/rehashed; 0 new simulator resets/actions, 0 model fits or optimizer steps, 0 teacher queries, 0 original E1 TEST/E2 EVAL array materializations, 0 unsafe pickle, 0 workflow/automation/dependency changes, 0 secrets/billing/deployments/merges/main writes. Existing push CI `40370144809` and PR CI `40370149012` both SUCCESS at preflight, no active branch Actions. No independently visible concurrent Copilot/ChatGPT branch writer; external sessions cannot be excluded.
