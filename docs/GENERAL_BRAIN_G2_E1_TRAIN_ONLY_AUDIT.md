# G2-E1 TRAIN-only coverage and observation-aliasing audit (2026-10-25)

**Scope:** post-result forensic analysis of **TRAIN data only** from the already frozen, failed-promotion G2-E1 predictive gate. This is not a new fitted model, heldout evaluation, prospective G2-E2 result, controller intervention, or cross-domain transfer claim. Do not use the G2-E1 TEST data for any model selection.

## Provenance and reproducibility

- Source commit: 2c8b1140afee463b0dd0ca7d7792f525c26ec7a5; successful pinned CPU workflow: 40165691302; numeric artifact: 12588790887 (bagi-g2-e1-public-effect-native).
- Artifact ZIP SHA256 independently verified: aaf8eb011763bf47e1d196771e257eb13c4eb35294bbeed2b71757ade525f073.
- Analysis reads only the xland_train_* and workshop_train_* arrays from public_effect_numeric.npz using numpy.load(..., allow_pickle=False). No TEST labels, TEST predictions, or TEST feature arrays enter any computation in this report.
- For each domain, count TRAIN actions, public-change labels, and per-action and per-collection-behavior frequencies using recorded TRAIN group indices. A duplicate episode is identical concatenated bytes of its ordered public observations, executed actions, and labels (SHA256). For an observation/action key, compare exact float32 observation bytes plus int64 executed action. A **mixed-label key** has both public-change=0 and public-change=1 among its recorded occurrences. XLand is also checked after stripping the final 28 **executed-action-history** features, retaining the other 87 legal public features. This is a descriptive collision diagnostic, not proof of an underlying hidden-state cause.

## XLand TRAIN evidence

- **180 TRAIN episodes, 3,244 actual actions, 2,019 public-change positives (62.24%)**; 90 frozen-BC episodes / 1,502 actions / 1,243 positives (82.76%) and 90 uniform-random episodes / 1,742 actions / 776 positives (44.55%). All 180 TRAIN trajectories are byte-distinct under the stated definition.
- Exact TRAIN action -> positive/total: action0 **766/937 (81.75%)**; action1 **640/640 (100%)**; action2 **566/566 (100%)**; action3 **43/358 (12.01%)**; action4 **4/444 (0.90%)**; action5 **0/299 (0%)**. These are **public observation changes**, not physical action effectiveness. In particular, 0/299 for action5 does not establish hidden-state no-op.
- Important collection-policy heterogeneity: action3 was positive **37/63** in frozen BC versus **6/295** in uniform random; action0 **568/652** versus **198/285**; action4 **4/153** versus **0/291**. The frozen BC never issued action5 in TRAIN; uniform random issued it 299 times. The source behavior label is collection metadata, **not a legal extra public input** to the effect predictor.
- Full 115-feature observation + action: **2,847 unique keys**, 182 keys repeat (579 rows), **5 mixed-label keys spanning 11 rows**. After excluding only the final 28 executed-action-history features: **1,479 unique keys**, 649 repeat (2,414 rows), **38 mixed-label keys spanning 223 rows**. History therefore distinguishes many otherwise-colliding records, but neither this count nor label collisions establish causal hidden-state aliasing rather than unmodeled randomness/layout variation.
- Evaluator-only hidden-state change with unchanged public observation: **117 TRAIN actions**. These audit flags were not used in predictor fitting, action selection, or public-change labels.

## Workshop TRAIN evidence

- **20 TRAIN episodes, 360 actual actions, 281 positives (78.06%)**. The ten fixed G1 scripted episodes contain **160/160** public-change positives and collapse to **one** distinct trajectory; the ten random episodes contain **121/200** positives and are individually distinct. Thus only **11 distinct ordered trajectories** exist among 20 TRAIN episodes, with **9 duplicate episodes**. The script is **not a learned policy**.
- Exact TRAIN action -> positive/total: action0 **18/40**; action1 **133/133**; action2 **85/114**; action3 **45/73**. The script disproportionately contributes positives (action1 80/80, action2 60/60, action3 20/20). The random behavior has action1 53/53, action2 25/54, action3 25/53 and action0 18/40.
- Full 25-feature public observation + action: **67 unique keys**, 52 repeated keys (345 rows), **zero mixed-label keys**. This is only evidence about the small, highly repeated TRAIN support; it does not establish that Workshop dynamics are fully observable or transferable.
- Evaluator-only hidden change without public change: **zero reported TRAIN actions**.

## Interpretation, reuse decision and stop rule

This TRAIN-only audit explains why a **training-only action-frequency baseline is a serious required control**: several native action classes nearly determine the public-change label in the collected mixture. It does **not** isolate the mechanism of the frozen logistic model's heldout failure, establish that adding history is sufficient, or justify tuning on the already-inspected G2-E1 TEST set. Collection behavior changes action prevalence and, for some actions, conditional outcomes; Workshop's scripted repeats reduce independent coverage. Publicly unchanged observations can still conceal real physical changes.

**Existing work/reuse:** use the existing G2-E1 numeric archive, standard numpy grouped counts and exact-key diagnostics, established POMDP/belief-state principles already reviewed in docs/GENERAL_AGENT_ARCHITECTURE.md, and the registered simple action-frequency control. No new architecture, packages, teacher, learned policy or simulator sweep. Contribution type: **descriptive post-result TRAIN-only audit**; not algorithm invention.

**Open question:** can a legal public-history-conditioned model beat action-only frequency on **fresh, independently preregistered** layouts and a genuinely learned Workshop task policy at matched real-step budgets, while remaining calibrated under differing collection behaviors? Any new G2-E2 trial requires its own prospective frozen split, independent evaluation layouts, predeclared baselines and stop gate, and a reuse review before new native actions. No retuning or rerunning G2-E1 on its TEST layouts.

**This iteration:** 0 new native actions, 0 teacher queries, 0 model fits/updates, 0 TEST evaluations. The G2-E1 nonpromotion decision remains unchanged. Physical BLUE success was not tested in this audit; game reward alone is not physical success.
