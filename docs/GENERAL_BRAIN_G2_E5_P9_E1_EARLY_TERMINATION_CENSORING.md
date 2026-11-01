# G2-E5-P9 — E1 early-native-terminal public-label censoring (TRAIN-only)

**2026-10-26 retrospective result; P9 native evaluation remains HARD STOP.** This is a follow-up to the E1 episode-tail audit, not a new native run, trained predictor, causal action-effect, physical manipulation or transferable-skill claim.

## Source and contract

E1 run `40165691302`, artifact `12588790887`, downloaded again, ZIP SHA256 **`aaf8eb011763bf47e1d196771e257eb13c4eb35294bbeed2b71757ade525f073`** (independently verified, 105,254 bytes). Only allowlisted `xland_train_*` and `workshop_train_*` numeric fields were materialized with `allow_pickle=False`; no `*_test_*`, original E2 EVAL, privileged pocket/grid, expert labels or native evaluator summary were opened. On all 3,064 shifted XLand rows, recomputed next-row public-view inequality (excluding 28 executed-history bits) exactly agrees with direct stored public-change labels. The native `collect_public_episode` contract stops before cap only after `native_step` returns terminal; the registered XLand cap is 20.

## Falsifying finding

**30/180 XLand TRAIN episodes ended before 20 steps, hence received a native terminal signal. All 30 last actions have positive directly recorded public-change labels, but P8's next-row within-episode shift excludes all 30.** They are **26 action-3** last steps (18 hold, 8 near) and **4 action-4** last steps (all putdown). These four are *every positive action-4 label in the original 3,244-step E1 TRAIN archive*; P8 retained **0/416 action-4 positives**, while the original archive has **4/444**. Across all actions: original **3,244 / 2,019 positive**, shifted **3,064 / 1,897**, excluded tails **180 / 122**. Action 3: original **43/358**, shifted **17/317**, tails **26/41**. Early termination does not establish goal completion or genuine physical BLUE pickup/drop: native reward can be vacuous, and no physical success gate was used here. Cap-length 20 episodes cannot be classified as terminal versus time limit by this audit.

## Matched source-disjoint Laplace control (Brier, lower better)

Both per-action Laplace arms score the **same held-fold rows**; TRAIN rates use only opposite layout-parity fold rows. The shifted arm trains on within-episode rows, the tail-inclusive arm trains on all directly labeled executed TRAIN rows. No model fits, optimizer steps or new native actions.

| Held rows | n / positive | Shifted-trained | Tail-inclusive-trained |
|---|---:|---:|---:|
| XLand retained | 3,064 / 1,897 | **0.049624258** | 0.050084575 |
| XLand excluded tails | 180 / 122 | 0.174526031 | **0.158753287** |
| XLand all | 3,244 / 2,019 | 0.056554690 | **0.056114282** |
| XLand early-terminal tails | 30 / 30 | 0.903580880 | **0.802402462** |
| Workshop uniform-random retained | 190 / 114 | 0.187766267 | **0.187580055** |
| Workshop uniform-random tails | 10 / 7 | 0.158862711 | **0.151327070** |

P8's published shifted Laplace `0.04962426` is independently reproduced. Tail-inclusive training modestly improves excluded-tail scores but still predicts early-terminal steps very poorly; it slightly worsens XLand retained-row Brier. These retrospective, termination-selected comparisons are **not** evidence of better learned effects, causal success, or a prospective model gate. No P8 model weights, scores or preregistration were changed.

## Decision and resources

**P9 remains STOP** because its frozen source specifies 3,064 shifted rows; it cannot silently switch to 3,244. A separate prospective terminal-inclusive comparison must freeze matched policy/data/optimizer/native budgets, source splits, Laplace/action-only/shuffled controls, positive terminal-action support, calibration and genuine public/native physical-effect gates, across XLand and independently learned Workshop policies. XLand historical BC actor weight identity, pinned runtime parity, full candidate-seed clearance and evaluator isolation remain unresolved. No task-specific BLUE logic or new heuristic sweep is justified. P6 memory FAIL, P4 RND STOP and small D3/D4 physical results unchanged.

One existing E1 artifact re-downloaded/rehashed; **0 new simulator actions/resets, 0 fits/optimizer updates, 0 teacher queries, 0 TEST/EVAL array reads, 0 workflows/dependencies, no secrets/billing/deployments/merges/main writes**. Local simulator-free numerical audit used a source-reviewed E1 TRAIN allowlist equivalent (not the full byte-identical repo package), with **5/5 synthetic tests** and Python compilation passing; no repository-level test claim until CI. See adjacent numeric summary and conversation-local detailed ledger.
