# G2-E6-P0B — TRAIN public-view-change is not verified action-effect learning (2026-10-29 IST)

**Decision:** read-only, retrospective TRAIN-only descriptive audit. **P0B frozen identifiability HOLD remains binding; no recurrent P1 model fit or native retry.** This is not preregistration, a prediction benchmark, evidence of a learned causal model, or a source to replace the failed frozen-history gate.

## Research preflight and prior evidence

Rechecked the protected \`setup/initial-platform\` HEAD \`72ab412432450ccf7dfb86badc1ed83f7e58d5e6\`, \`RESEARCH_PROTOCOL\`, \`RESEARCH_DECISION_GATES\`, \`REUSE_INVENTORY\`, \`GENERAL_AGENT_ARCHITECTURE\`, D3/D4, complete append-only research log and the P0B native-result, TRAIN-conflict and independent-episode/geometry-support reports. Post-public GitHub CI retry of run \`40602642611\` **attempt 2 succeeded** for this exact HEAD (539 passed, 7 skipped and Ruff passed); pull-request run \`40602648883\` remains marked failed from its earlier no-runner attempt, so do not silently claim all checks green. Other scheduled workflows have not been independently runtime-validated after visibility change. Last check found no active/queued/pending/waiting GitHub jobs and only the existing hourly ChatGPT research task enabled; recheck before any push.

Reused the immutable **one-shot** native artifact: research source commit \`1751989016ed14a82bfe110e27ec9a711c1be14f\`, run \`40578403185\`, artifact \`12768976012\`; SHA256 of the downloaded outer artifact ZIP \`182466a08a646b7f6d62a2b95a82f0a0c20e10d4313d71959e6ac840669cc7e3\`; inner four-member numeric ZIP \`fcdaaa9b3aa0f08d4c3ae350f291cefb1ae04921a299893b16d08be928a4f5dc\`. Read and verified **only** \`summary.json\` and \`train_public_numeric.npz\`; TRAIN member SHA256 \`3f31fb7a2ebaa6b617cfc0314af30f1d40a319747fd1f1e7326719df836a18d9\` matches the registered summary. Numeric loading used \`np.load(..., allow_pickle=False)\`. **Neither** the diagnostic NPZ nor frozen weight NPZ was opened; no evaluator-only hidden state, heldout label or size13 row was inspected. Original recorder source \`bagi/experiments/g2_e6_p0b_workshop_dual_size_acquisition.py\` at the registered commit (Git blob \`ec466dc62039dd8849382071e9a93b18dcb2cf5f\`) confirms that \`public_pre_action\`, \`executed_action\` and \`public_post_action\` are taken chronologically around an actual native \`env.step(action)\` and before/after the legal 5×5 public observation.

## Newly independently recomputed TRAIN-only outcome prevalence

For each row, define **visible-change** as \`np.any(public_pre_action != public_post_action)\` over the 25 legal int8 cells; this label deliberately does **not** mean a verified physical state transition or achievement. Stratify by Workshop size and executed action ID; do not pool sizes to qualify scientific gates. All 8,192 TRAIN actions use \`layout_index=0\` (default), two policy seeds 17/29 and 208 source episodes in aggregate.

| Size | Executed action ID | TRAIN rows | Visible change | Unchanged | Visible-change share |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 0 | 970 | 508 | 462 | 52.37% |
| 9 | 1 | 859 | 686 | 173 | 79.86% |
| 9 | 2 | 1,174 | 618 | 556 | 52.64% |
| 9 | 3 | 1,093 | 836 | 257 | 76.49% |
| 11 | 0 | 959 | 487 | 472 | 50.78% |
| 11 | 1 | 805 | 644 | 161 | 80.00% |
| 11 | 2 | 1,140 | 613 | 527 | 53.77% |
| 11 | 3 | 1,192 | 875 | 317 | 73.41% |
| **Total** | — | **8,192** | **5,267** | **2,925** | **64.29%** |

Independent seed×size tabulation: size9 seed17 **1,355/2,048** public changes, seed29 **1,293/2,048**; size11 seed17 **1,371/2,048**, seed29 **1,248/2,048**. Source-only cross-check of exact current public-view+executed-action key conflicts reproduced the previous audit: **2/155** at size9 and **7/173** at size11 have more than one next 5×5 public view. These are exploratory TRAIN counts and cannot replace the frozen trained-policy identifiability gate, which remains size9 **0/24** and size11 **1/28**, both required.

All **eight** size×action groups have a visible-change majority; therefore a deliberately trivial source-sample rule **always predict changed**, even separately by action, matches **2,648/4,096 (64.65%)** size9 and **2,619/4,096 (63.94%)** size11 TRAIN rows. This is a descriptive **in-sample prevalence**, **not** heldout accuracy, a fitted model, evidence of learnability or a strong baseline. The prevalence varies substantially by action; changes in the agent's *view* could arise from pose/motion rather than the intended external object-effect class. Also, **all 19,712 native TRAIN+diagnostic actions** in this one-shot study produced **zero goal rewards**, despite **8,066 optimizer updates** (original recorded execution; not new work). Thus success on a binary any-view-change target alone would be uninformative about useful, calibrated action-effect understanding or recovery.

## Scientific decision: reuse, falsifiable next test and STOP

**BAGI-specific gap:** still no independently repeated, geometry-diverse, *action-conditional specific outcome-class* support sufficient to validate a legal public belief/transition model. Build-versus-reuse: **reuse** the archived numerical public observations/executed actions, source recorder, existing Workshop DQN/PyTorch GRU, and maintained POMDP/MiniGrid/SB3-Contrib baselines. Do **not** add a BLUE-specific rule, propose a new network, or treat the shortcut binary visible-change prevalence as a measure of acquired effect knowledge.

Prospective *separate authorization only*: hypothesize that history-conditioned **specific next-view tile/event** prediction, not binary any-view-change, improves episode- and geometry-disjoint positive-class performance and calibrated next-view loss by at least 5% over the strongest same-data and same-compute reactive, persistence, action-only and shuffled-history controls **without increasing false-positive effects**. Keep independent seeds and geometry splits, public input only, actual executed actions, matched real-step budgets, and explicit negative controls. This is a **proposal**, not a preregistered study or achieved performance claim.

**Hard STOP before fitting or acquisition:** no model fit until each required size **and** source geometry has independently repeated positive/negative outcomes within defined specific public event classes in different episodes, disjoint validation episodes, valid exact-HEAD CI and independently checked provenance; otherwise report HOLD. P0B is already frozen and failed; do not retune it, rerun it or launder TRAIN outcomes into its diagnostic gate. P6/P10a STOP; P9 native HARD STOP; do not touch size13/heldout labels. XLand BLUE is strictly an external physical verification diagnostic, requiring effective pickup, later effective drop, initial nonadjacency, final adjacency and native reward.

**Established primary references, not BAGI results:** Ni, Eysenbach & Salakhutdinov (2022), recurrent RL for POMDPs: https://proceedings.mlr.press/v162/ni22a.html (author code https://github.com/twni2016/pomdp-baselines); Fujimoto, Meger & Precup (2019), offline extrapolation limits: https://proceedings.mlr.press/v97/fujimoto19a.html; maintained public-observation wrapper https://minigrid.farama.org/v2.4.0/api/wrapper/; maintained recurrent policy https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/recurrent/policies.py.

**New iteration budget:** 0 native actions/resets, 0 optimizer updates, 0 fits, 0 teacher queries, 0 size13 or heldout interactions, 0 workflow dispatches, 0 CI retries, 0 billing/secrets changes, 0 merges, 0 main writes. One append-only research-log entry and this evidence-only document are the only proposed tracked changes, subject to final expected-HEAD and concurrency checks. Original artifact sizes and its TRAIN/diagnostic budget are not newly incurred.
