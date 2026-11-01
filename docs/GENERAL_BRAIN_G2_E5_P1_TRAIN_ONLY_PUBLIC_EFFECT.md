# G2-E5-P1 — TRAIN-only public next-observation effect support (2026-10-26)

**Status: completed exploratory retrospective audit; NO prospective heldout test, native simulator call, policy optimization, physical action-effect validation, learned Workshop policy, or skill promotion.** This diagnostic uses *only* existing G2-E4-P4 TRAIN numeric arrays. The analysis and thresholds were **not preregistered before inspecting the TRAIN results**. It must not be treated as independent confirmation of a new learned general brain.

## Research question and reuse decision

BAGI aims to learn task-independent public observation/action effects, belief, recovery and transferable skills. G2-E1 XLand's public-visible-change predictor failed a TRAIN-only action-frequency baseline on disjoint TEST layouts, and G2-E5-P0 found only 193/3,244 exact (view,action) cross-TRAIN-layout matches; Workshop's learned DQN had no rewarded TRAIN goals. Before paying for another world-model/sequence architecture, ask the narrower **support** question: *do existing successfully rewarded navigation TRAIN traces contain repeated, predictable `(current public view, executed action) -> next public view` transitions, beyond action-only frequency and a shuffled-view control?*

**Reuse first:** the authoritative numeric-only G2-E4-P4 PPO TRAIN archive, NumPy, and the existing G2-E5-P0 group-disjoint conditional-frequency method. This is an established tabular empirical conditional-distribution estimator and retrospective diagnostic, **not** a novel algorithm, online controller, learned recurrent belief, causal model or skill. No new packages, pretrained weights, unsafe deserialization or source-specific BLUE logic. The frozen trained PPO is only the **behavioral source**; the predictor is fit offline to executed TRAIN transitions and never selects actions.

## Provenance and legal information boundary

- G2-E4-P4 preregistered source commit `793278a503f2b7b7ed59c46fcd37b91ff2001f7c`, successful native run `40230186023`, artifact `12611649923` (`bagi-g2-e4-p4-input-scaling`). The archived ZIP SHA256 **`15b8976859385c7f620364ca50c42a7d952ec85d26f9ab1faf537b19dbc28759`** and all four selected inner TRAIN NPZ checksums are independently verified **before** any arrays are read. Only `train_seed{17,29}_{raw,scaled}.npz` members and whitelisted `public_codes`, `actions`, `episode_index`, `terminated`, `truncated` arrays are loaded using `allow_pickle=False`. **No P4 TEST array, reward, label or model output is accessed.** Source archive contains TEST files, but this script never opens them.
- Each of four original P4 policies executed exactly **16,384 real native TRAIN actions** in upstream `MiniGrid-Empty-Random-6x6-v0` with public 147-value categorical view, actions 0..6, sparse native reward and no evaluator-private actor input. This audit does not repeat those actions. The predictor's **input** is only current public view plus the actually executed action; the **target** is the *next* public view, used only after the archived real transition. No future view is fed to an actor. Terminal/truncated boundaries are excluded, so no reset transition is mislabeled as a physical action outcome.
- The `raw` and `/10` policies have different action trajectories but use the same native `public_codes` representation for the predictor. Model seed17 and seed29 have **different TRAIN reset-seed streams**, beginning at **2,000,000** and **2,100,000** respectively. However, the 6x6 environment has a small finite set of repeated physical layouts and only **48 distinct TRAIN public views per condition**; distinct reset seeds **do not guarantee distinct layouts**. The archive does not provide per-episode physical layout hashes, so no layout-disjoint transfer claim is possible.

## Reproducible descriptive protocol

1. Within each seed/scale archive, take the current view and executed action from each row and its next view from the **next row only when the episode index is unchanged**. Exclude terminal or final partial rows (170, 188, 256 and 241 excluded respectively); retain **16,214 / 16,196 / 16,128 / 16,143** valid transitions.
2. Use five TRAIN-only episode-grouped folds (`episode_index % 5`). For each held-out TRAIN episode, fit the modal **exact next public view** from the other four TRAIN folds under (a) action-only, (b) exact current-public-view + action, and (c) same key after a deterministic global public-view shuffle (`rng seed 20261026`) that breaks observation/outcome correspondence. On unseen keys fall back to the action-only predictor. Report exact next-view accuracy, fraction of incorrectly predicted view cells and Laplace-smoothed Brier for whether the next view differs. The shuffle is a negative control, not a randomized causal intervention. This is **episode-disjoint TRAIN cross-fitting**, not heldout-TEST or layout-disjoint evaluation.
3. As an additional stronger TRAIN-only separation, fit on **all seed17 TRAIN transitions** and score **seed29 TRAIN transitions**, then reverse, separately for raw and scaled behavior. The TRAIN reset-seed ranges are disjoint, but exact public views and physical layouts can repeat. Report source exact-key count and target exact-key coverage; do not call this novel-state generalization or cross-domain transfer.
4. Interpret accuracy together with exact-key support. Predicting a common next view is not necessarily predicting a useful causal effect; cell mismatch can be tiny when views differ in only one tile. The predictor is never rolled out, used to recover a failed action, or evaluated on a new environment.

## Actual results (exploratory, existing TRAIN only)

### Episode-grouped TRAIN cross-fit

| Frozen PPO behavior | Usable TRAIN transitions | Distinct public views | Action-only next-view exact | View+action next-view exact | Shuffled-view+action next-view exact | View+action public-change Brier |
|---|---:|---:|---:|---:|---:|---:|
| Seed17 raw | 16,214 | 48 | 10.30% | **95.15%** | 8.77% | **0.000983** |
| Seed17 scaled | 16,196 | 48 | 12.63% | **95.64%** | 9.98% | **0.001192** |
| Seed29 raw | 16,128 | 48 | 15.27% | **95.61%** | 12.90% | **0.001435** |
| Seed29 scaled | 16,143 | 48 | 12.53% | **95.77%** | 10.92% | **0.001499** |

Action-only public-change Brier **0.049535 / 0.066869 / 0.089435 / 0.095581** in the same order. View+action exact-key coverage is **16,214 / 16,190 / 16,128 / 16,143** rows. Mean next-view cell mismatch with view+action is **0.000863 / 0.000854 / 0.000802 / 0.000758**. These tiny errors mainly reflect near-deterministic, repeatedly encountered public view/action pairs; they are not a measure of physical goal progress.

### Cross-model-seed TRAIN-to-TRAIN prediction

| Source TRAIN → target TRAIN | Behavior | Target rows / exact-key matches | Action-only exact | View+action exact |
|---|---|---:|---:|---:|
| 17 → 29 | Raw | 16,128 / 16,128 | 15.27% | **95.36%** |
| 29 → 17 | Raw | 16,214 / 16,214 | 10.55% | **95.02%** |
| 17 → 29 | Scaled | 16,143 / 16,143 | 11.65% | **95.89%** |
| 29 → 17 | Scaled | 16,196 / 16,196 | 10.19% | **95.89%** |

**Critical limit:** source TRAIN contains exactly **334 distinct (public view, action) keys**, and **100% of target TRAIN transitions reuse keys already present in source TRAIN**. Therefore these positive numbers demonstrate **repeated-input interpolation** under disjoint reset-seed ranges, not prediction on unseen observation-action combinations, disjoint physical layouts, hidden-state disambiguation, action recovery, learned Workshop competence or cross-domain transfer. The two model seeds and raw/scaled variants are not four independent benchmark domains.

## Decision and next gate

**Positive narrow descriptive result; NO GENERAL SKILL / MEMORY PROMOTION.** A legal current-public-view+executed-action lookup can predict *visible next observations* far better than an action-only prior on this heavily repeated easy-navigation TRAIN support. This is compatible with G2-E4-P3/P4's negative observation-conditioned **policy** result: learning a predictable effect table is not the same as using it to make goal-directed decisions. It also does not overturn G2-E1's negative source-disjoint XLand effect result.

**Next bounded prerequisite:** acquire or verify a genuinely rewarded independently learned Workshop policy on fresh starts **without privileged info or repeated first-visit-beta0.05 retuning**; separately require XLand physical effective pickup/drop/adjacency/native-reward audit. Only then preregister a *prospective* same-data, same-capacity, public-only reactive versus history/shuffled-history next-effect predictor with truly disjoint XLand layouts and Workshop conditions. The predictive gate must beat an action-only frequency baseline and a shuffled-input control **in both domains**, with independent native goals reported separately. If TRAIN coverage remains only repeated 334 easy-view keys or Workshop still lacks rewarded behavior, STOP before native cross-domain transfer claims.

**This iteration:** 0 new simulator actions, 0 new learner optimizer steps, 0 teacher queries, 0 new TEST probes, 0 new workflow or automation, no secrets/deployments and no changes to `main`. Reproduce using `python -m bagi.experiments.g2_e5_p1_train_effect_audit --source-zip ORIGINAL.zip --output results.json`; the frozen machine-readable result is in `docs/data/g2_e5_p1_train_effect.json`. Ten simulator-free synthetic contract tests passed locally. CI validates software, not science.
