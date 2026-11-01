# G2-E6-P0B — TRAIN-only independent-episode support and geometry limitation (2026-10-28 IST)

**Decision: P0B frozen history-identifiability HOLD; no P1 recurrent/history model fit.** This is a *posthoc descriptive* reanalysis of already acquired public TRAIN trajectories, not a new experiment, a new model, a change to the preregistered gate, or evidence of transferable belief learning. It extends the earlier `GENERAL_BRAIN_G2_E6_P0B_TRAIN_CONFLICT_SUPPORT_AUDIT.md` with a more stringent **other-episode** support test and an explicit geometry check.

## Provenance and exact public boundary

Reused source commit `1751989016ed14a82bfe110e27ec9a711c1be14f`, native run `40578403185`, artifact `12768976012`. Re-downloaded and checked outer ZIP SHA256 `182466a08a646b7f6d62a2b95a82f0a0c20e10d4313d71959e6ac840669cc7e3` and nested ZIP SHA256 `fcdaaa9b3aa0f08d4c3ae350f291cefb1ae04921a299893b16d08be928a4f5dc`. Read **only** `train_public_numeric.npz` with `allow_pickle=False`; no diagnostic NPZ, evaluator hidden state, size13, heldout label or original mixed E2 data entered this computation. No native simulator calls, policy fits, optimizer updates, teacher queries, workflows or new data collection.

Within each size independently, form `(exact current 5x5 public view, actually executed action)` keys and exact next-public-view outcomes. Use **only nonterminal steps 4..38**, so each row has four actual preceding `(public view, executed action)` pairs. Compare history lengths 0, 1, 2, 4 using exact equality; an observation has *independent-episode support* only if another **different episode ID** contains that same current key and preceding history. Count independently repeated outcomes by episode ID, not just repeated steps in one episode. No predictor or outcome label was fit. The same-source rows are exploratory, not a heldout accuracy estimate.

## Observed support collapse (TRAIN only)

| Exact-history length | Size 9: supported / all eligible | Size 11: supported / all eligible | Size 9: supported / conflicting-key rows | Size 11: supported / conflicting-key rows |
| --- | ---: | ---: | ---: | ---: |
| 0 | 3,550/3,578 | 3,539/3,578 | 122/122 | 337/337 |
| 1 | 3,462/3,578 | 3,429/3,578 | 121/122 | 328/337 |
| 2 | 3,024/3,578 | 2,873/3,578 | 116/122 | 278/337 |
| **4** | **946/3,578 (26.4%)** | **914/3,578 (25.5%)** | **31/122 (25.4%)** | **79/337 (23.4%)** |

The original current-view+action conflicting keys remain **2 in size9** and **7 in size11**, among 153 and 167 eligible keys at steps 4..38. Four-step signatures within those conflicting keys are singletons in **88/122** size9 and **254/337** size11 rows; stricter *other-episode* support is lower still because some signatures repeat only inside one episode. Among the 4-step supported conflicting-key rows, **31/31** size9 and **76/79** size11 have at least one matching *same-outcome* observation in another episode. This is a support statistic, **not** out-of-episode prediction accuracy.

At four preceding view/action pairs, size9 has **0** and size11 **2** exact-history groups with differing next views in different episodes. **Neither size has any four-history group in which every competing outcome is independently observed in at least two episodes.** At two preceding pairs the corresponding number of independently conflicting history groups is **3** size9 and **5** size11, with only **0** and **3** groups respectively having at least two independent episodes *per* outcome. Longer exact histories thus trade observed ambiguity for sparse support; they do not establish sufficient memory or learnable causal state.

## Geometry and generalization boundary

Every one of the **8,192 TRAIN rows** is `layout_index=0` (**default**) by the frozen P0B design: each model seed received 2,048 actions per size, all on that one layout family. Varying reset seeds and start positions yields 13/16 distinct initial views at sizes 9/11, **not independent layout geometries**. `mirror_x` exists only in the separately frozen diagnostic split and was not inspected for this analysis. Therefore no TRAIN conflicting key has cross-layout TRAIN support; no claim about prediction under a new geometry, an unrelated task family or XLand follows. This is a limitation of the registered acquisition, not a newly discovered source violation.

## Reuse, interpretation and next stop gate

Established partial-observation/belief methods (Kaelbling–Littman–Cassandra, 1998) and the existing PyTorch GRU / Workshop DQN and public numeric recorder remain preferred to a bespoke memory algorithm. This audit is **evaluation/diagnosis**, not invention. It cannot rule out representation-level generalization across *different* exact histories, but cannot validate it either. The preregistered **frozen** P0B identifiability gate still fails (size9 **0/24** vs size11 **1/28** qualifying frozen keys); TRAIN evidence may not replace it. The native Workshop outcome remains **zero goals across 19,712 original actions**.

A future *separately authorized* study must preregister independent episode **and geometry** splits, sufficient independently repeated positive/negative public effects per size, a no-fit-on-failed-support stop rule, and matched reactive/action-only/persistence/shuffled-history versus recurrent/finite-history predictors with equal TRAIN data, parameters/compute and real steps. It must measure event-positive errors and eventually learned-policy navigation/recovery on genuinely new tasks. **No such study or P1 fitting is authorized here**; P6/P10a STOP, P9 native HARD STOP, untouched size13 and XLand physical pickup→effective drop→adjacency+reward evaluator all remain binding. No BLUE-specific algorithmic logic is proposed.

Reproducibility: the independently computed numeric JSON report `BAGI_G2E6_P0B_TRAIN_geometry_history_support_readonly.json` is retained with the active research audit; the original GitHub Actions ZIP above is authoritative. All denominators, history definitions and hashes are stated here so another reader can independently recompute them from its TRAIN-only numeric member.
