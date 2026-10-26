# G2-E5-P8 — Retrospective fold robustness and calibration audit

**Status: completed, read-only reanalysis, 2026-10-26. NOT an independent replication, a prospective gate, a native intervention or learned skill.** The original registered P8 thresholds, data and model are unchanged.

## Provenance and audit

Research branch starting commit: c2e1f32223494310833194cb4cbdbd0df9ed959c. Push CI 40309194427 and PR CI 40309204366 both completed SUCCESS, including lint and pytest, no new scored CI artifacts. The immutable P8 TRAIN-only score CSV is [G2_E5_P8_TRAIN_ONLY.csv](research-results/G2_E5_P8_TRAIN_ONLY.csv), published Git blob c50ae226a7fff202b193b5c83f6cdd079fcf608d and full SHA256 3363c60cec8850f25b15c6570be4f538d5958b9cb0257cd4f78a0a5f021ef0b6. The 96-row numeric ledger and its 14-column schema were checked before analysis; the exact expected 4 datasets × 2 source-disjoint TRAIN folds × 2 model initializer seeds × 6 arms are present once each. Source row counts, group counts, public-change positives, score finiteness/ranges and untrained-baseline seed identity were verified. Full machine-readable 16-fold/seed checks and 4 dataset summaries are in [G2_E5_P8_POSTHOC_FOLD_ROBUSTNESS.json](research-results/G2_E5_P8_POSTHOC_FOLD_ROBUSTNESS.json). A local standard-library-only audit source and 7 simulator-free tests independently passed; these standalone files are available as conversation attachments, not claimed to be committed code. This is only reanalysis of already-published numerical metrics, **not** re-reading E1 TRAIN/TEST or the historical E2 source/EVAL mixed archive.

## Findings

For each dataset/fold/seed we recomputed percentage held TRAIN Brier reduction of reactive public-observation+executed-action model relative to each frozen action-only, shuffled TRAIN action, and Laplace-smoothed per-action baseline. This is **post-hoc group consistency** of the preregistered P8 gate, not a newly preregistered extra gate. Independent source units are 15 XLand layouts (8 versus 7 TRAIN groups), 10 random-only Workshop episodes (5 versus 5), and 52 DQN Workshop TRAIN episodes per seed (26 versus 26). The DQN episodes share one map geometry. Model seeds 17 and 29 score the **same** held-fold rows, not two independent native evaluation samples.

| TRAIN source | P8 gate across fold×initialization checks | Smallest Brier margin over Laplace | Reactive ECE10 worse than Laplace |
|---|---:|---:|---:|
| XLand, 15 source layouts | 4/4 pass | +5.76% | 4/4 |
| Workshop random only, 10 episodes (secondary) | **0/4 pass** | **−31.83%** | 2/4 |
| Workshop DQN seed17, 52 episodes same geometry | 4/4 pass | +22.81% | 4/4 |
| Workshop DQN seed29, 52 episodes same geometry | 4/4 pass | +17.97% | 4/4 |

**Positive but fragile inference:** all 12 *required* fold/initializer scores clear 5% Brier reduction over each registered control. The narrowest XLand advantage over simple per-action Laplace is **only 0.76 percentage points above the numerical 5% decision threshold**. On the independently gathered but small random-only Workshop source, the reactive predictor loses to the simple action-frequency control in every fold/seed. Critically, the reactive predictor's **10-bin ECE is worse than Laplace in ALL 12 required comparisons**, and original P8 source report also showed weaker log loss/calibration in several settings. Better Brier is NOT a license to use raw network probabilities as thresholds for recovery.

## What has not been established; next scientific gate

The **historical 144-supervised-step extra pre-CI XLand smoke fit** remains a P8 protocol deviation; the completed 48-fit result cannot be called perfectly clean. The original P0/P5/P6 E2 mixed-archive access correction and P7 isolated actor0-only derivation disclosure remain intact. The current audit did **not** inspect any original TEST/EVAL array, model checkpoint, teacher labels, heldout native effect state, native goal, or physical BLUE movement. This is not independent native replication and supplies no statistical confidence interval from source groups: the ledger stores per-group dispersion, not the individual paired group difference series required for a valid paired grouped bootstrap.

**Decision: do not promote the predictor or tune P8 after seeing its numbers.** The next falsifiable question is independent **passive native out-of-source public-change prediction with calibration** using fresh seeds and verified frozen behavior-policy actors. The strictly prospective [G2-E5-P9 native confirmation preregistration](GENERAL_BRAIN_G2_E5_P9_NATIVE_CONFIRMATION_PREREG.md) defines hard prerequisite STOPs before new native calls. Even success would precede, not substitute for, a separate learned recovery/native-goal intervention.

**This iteration resource account:** zero native simulator actions/resets, zero optimizer/policy updates or new model fits, zero E1 TEST arrays or original E2 mixed EVAL reads, zero expert calls or goal probes, zero new workflows/automations/dependencies/secrets/deployments/billing/main/merges. Only existing published CSV rows were parsed; one branch-only atomic documentation/results/log commit after an explicit HEAD recheck.
