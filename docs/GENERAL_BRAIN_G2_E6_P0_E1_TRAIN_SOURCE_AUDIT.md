# G2-E6-P0 — Independent E1 TRAIN source and public-history eligibility audit

**2026-10-27 UTC; read-only prerequisite result.** This follows the prospectively registered [G2-E6-P0 source-eligibility gate](GENERAL_BRAIN_G2_E6_P0_SOURCE_ELIGIBILITY_AND_BELIEF_PREREG.md). **Partial XLand source-eligibility evidence; overall P0 Stage A HOLD.** This is not a new model fit, new native trial, learned-belief success, goal-reaching success, or cross-domain transfer. It does not authorize P1 fitting or P9 native evaluation.

## Hash-pinned provenance and exact access boundary

- E1 real native run `40165691302`, artifact `12588790887` (`bagi-g2-e1-public-effect-native`), full downloaded ZIP **SHA256 `aaf8eb011763bf47e1d196771e257eb13c4eb35294bbeed2b71757ade525f073`**, rehashed locally and matching GitHub's published artifact digest. Producer: `bagi/experiments/public_effect_e1_native_runner.py` and `public_effect_xland_native.py`, source commit `2c8b1140afee463b0dd0ca7d7792f525c26ec7a5`.
- Exact original ZIP members `summary.json` and `public_effect_numeric.npz`. The audit checked the complete outer ZIP SHA before any member access, then opened the numeric NPZ using `numpy.load(..., allow_pickle=False)`. It **indexed and copied exactly 16 allowlisted `xland_train_*` and `workshop_train_*` arrays**: `observations`, `actions`, `labels`, `groups`, `episode_ids`, `layouts`, `behaviors`, `native_actions` for each domain. **Zero original `*_test_*` arrays and zero `*_hidden_without_public_change` evaluator arrays were indexed/materialized.** The compressed NPZ also contains TEST members; the full ZIP/NPZ bytes were necessarily downloaded/opened, so this is not a claim that the file contains no TEST data.
- Independently downloaded G2-D4 artifact `12576889147`, full ZIP **SHA256 `7fc26e3485a4f3c4cd6afdd92f81c8ab118ce3ae3a629e69fc823a470f0c5afb`**, contains **only `summary.json`** (no numeric public-transition arrays). Its summary holds aggregate/episode-level action and evaluator counts, not a verified per-step public before/after TRAIN transition archive. D4's real native evaluation remains valid on its original terms; **do not recycle D4 EVAL cases into P1 TRAIN**.
- No original mixed E2 Workshop archive was opened. The previously verified isolated E2 actor0 TRAIN-only ZIP SHA256 `8d3f167ce977f62d3397ff37c553d8948b94bc03ac6d88df8a5a6e5fe86aac83` is documented in `GENERAL_BRAIN_G2_E5_P7_E2_TRAIN_ISOLATION.md`, but its bytes are not stored in this repository and were **not available for a new independent E2 numeric audit**. The original E2 mixed source must not be loaded as a workaround: that historically materialized EVAL arrays.

## Exact E1 TRAIN public transitions, no simulator or fits

A source-episode boundary check verified each recorded `native_actions` count against its contiguous `groups` span. A next-view label was constructed **only** from the next actually recorded pre-action public observation **within the same episode**, excluding the last action of each episode. This gives **3,064 XLand** and **340 Workshop** next-view pairs; the existing public-change labels agree with the reconstructed non-history next-view changes on **3,064/3,064** and **340/340**, respectively. No unrecorded terminal after-view was fabricated.

| E1 TRAIN behavior | Episodes | Distinct source layouts | Actual recorded actions | Within-episode next-view pairs | Public-change positive labels (all actions) |
|---|---:|---:|---:|---:|---:|
| XLand **frozen learned BC** | 90 | 15 | **1,502** | **1,412** | **1,243** |
| XLand uniform random | 90 | 15 | 1,742 | 1,652 | 776 |
| Workshop uniform random | 10 | **1** | 200 | 190 | 121 |
| Workshop G1 public script (NOT learned) | 10 | **1** | 160 | 150 | 160 |

**Correction to the earlier provisional P0 uncertainty:** the E1 XLand TRAIN archive **does** contain 90 time-aligned episodes executed by the frozen learned `frozen_expanded_bc` actor over 15 source layouts, with legal public 115-feature observations (including the prior four executed actions), legal current actions and recorded episode boundaries. This satisfies the *XLand learned-policy source-availability* prerequisite for a future prediction-only audit. It does **not** establish independent learned-policy competence, causal physical effects, or a public-state belief representation.

## Frozen P0 repeated-key diagnostic: descriptive, not fitted prediction

For each behavior separately, use a **full 115-feature legal current XLand observation + actual current action** as the base key. This includes the public goal/context and already-embedded last four executed-action one-hots. Use the **first 87 non-history public features of the next recorded view** as the target. Repeated keys with differing next views indicate public aliasing in the archived TRAIN sample; not necessarily learnable state, and not necessarily physical movement. For a distinct test, condition on the preceding two **environmental public views only** (not evaluator state, not an additional prior-action oracle); pad missing episode history explicitly. These counts were independently cross-checked using tuple-based keys rather than only byte hashes.

| XLand TRAIN source | Base keys | Repeated keys (rows) | Conflicting next-view keys (rows) | Conflicting binary public-change keys (rows) | Cross-layout conflicting next-view keys |
|---|---:|---:|---:|---:|---:|
| Frozen learned BC | 1,091 | 143 (464) | **28 (81)** | **4 (9)** | **28** |
| Uniform random | 1,634 | 17 (35) | 2 (4) | 0 (0) | 2 |

Among the **28 learned-BC ambiguous next-view keys**, **12** have distinct preceding two-public-view histories; **10** are perfectly separated *in this same TRAIN sample* by those histories, while **18** remain ambiguous after conditioning. This is **only a candidate identifiability signal**: the number of ambiguous keys is small, the split is in-sample and may encode layout/history-specific coincidences, the binary visible-change conflicts are only four keys, and no model was fitted or tested on disjoint layouts. Do **not** report 10/28 as predictive accuracy or proof of belief learning. XLand learned-BC ambiguous actions were 7 keys on action0, 18 on action1 and 3 on action2; no manipulation-specific claim follows.

Workshop E1 random and scripted rows have **zero** conflicting next-view keys under the same base-key definition; they come from one fixed geometry and **neither is an independently learned Workshop behavior source**. They are not admissible cross-domain evidence for P1.

## Decision and next bounded falsification

**G2-E6-P0 Stage A: PARTIAL PASS for E1 learned-XLand public-source chronology; overall HOLD for two-domain learned-policy evidence.** The E1 XLand repeated-key conflicts motivate a strictly new, preregistered observation-history comparison but do not authorize fitting until the **isolated E2 actor0-only TRAIN bytes** are available for a read-only audit and the Workshop learned-policy source's geometry and outcome limitations are explicitly assessed. One fixed Workshop geometry cannot support geometry-generalization; any later acquisition of independent learned Workshop layouts must be separately preregistered with a small native budget, matched policy/step controls and no TEST leakage.

Preserve P6 and P10a **STOP**, P9 native **HARD STOP**, and the XLand physical BLUE evaluator-only pickup→later effective drop→final adjacent+native reward gate. No BLUE-specific controller logic or retuning. Resource ledger for this audit: **zero** new simulator actions/resets, teacher queries, model fits, optimizer updates, policy updates, original TEST/EVAL array reads, hidden-evaluator array reads, heldout expert-label reads, new workflows, secrets, billing, deployments, merges or main writes. Only two previously published ZIP artifacts were downloaded and independently SHA-verified; D4 summary was used for schema, not model fitting. A standalone read-only audit script and numeric JSON ledger were produced in the active research conversation; the source artifact is authoritative.
