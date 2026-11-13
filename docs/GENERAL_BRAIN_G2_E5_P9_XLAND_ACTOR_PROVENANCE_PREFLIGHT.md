# G2-E5-P9 — XLand frozen learned-actor provenance preflight

**2026-10-26 — READ-ONLY AUDIT, P9 STOP.** No P9 simulator reset, model fit, checkpoint conversion, expert label or evaluator query occurred. This does not amend the prospective P9 protocol.

## Scientific prerequisite and reuse decision

The task-independent P9 public-change predictor must be evaluated on actions taken by an **independently frozen, previously learned** XLand actor. Reuse upstream imitation.BC and BAGI's existing 115-feature public observation (local view + public rules/goal + four *actually executed* prior actions); do not invent a BLUE-specific policy or new learner. The unresolved issue is a verified *weight checkpoint*, not the existence of a TRAIN-only reconstruction recipe.

Inspected current research-branch source:

- bagi/experiments/general_brain_native_g1.py (Git blob d3095f621f5cfeb5a204a1eab440f26337fb5c06), source_policies: loads verified REUSE-19, REUSE-33 and REUSE-36 source rows, then calls diverse.fit_original_bc for **two seeds × two arms = four BC fits**.
- bagi/experiments/xland_diverse_physical_demos.py (blob e735674b178881839f715b9379b8908bf88ccaff), sample_fixed_extra / fit_original_bc: **291 original + 248 rehearsal + 248 physical TRAIN labels = 787 rows** per fit, 115 public features, six actions, upstream imitation.BC, 12 epochs, batch 32. Nominal sample exposures are **9,444 per fit / 37,776 for all four**. E1 uses the expanded arm.
- bagi/experiments/xland_diversity_confirmation.py (blob dcf77d9a37d5a095cbac0fc39eb66269f693c0ed), require_previous_training: verifies source digest and **training metadata**, including 787 rows and 9,444 exposures; **does not compare neural weight tensors to a pinned exported checkpoint**. A deterministic seed and equal metadata do not prove identical weights across library/runtime changes.
- bagi/experiments/xland_upstream_instruction_bc.py (blob 1af853a9ab5b3bb19e3f7aec69b17af03842b2f7): verifies the 291-row source action-provenance numeric archive and real executed history using allow_pickle=False.

**Finding:** these inspected sources establish a hash-guarded **TRAIN-data reconstruction recipe**, but not a separately pinned frozen BC **weight checkpoint**. A P9 scored runner must not silently call source_policies, perform four fresh BC fits and report zero actor training. This audit does not assert that no weight checkpoint exists elsewhere outside the inspected paths.

## Historical artifact provenance (API metadata only)

GitHub Actions API returned unexpired branch-specific artifacts. The following outer ZIP SHA256 values are **GitHub API-reported digests, not downloaded/recomputed hashes in this iteration**:

| Historical source | Run | Artifact | API-reported SHA256 |
|---|---:|---:|---|
| REUSE-19 executed-action TRAIN provenance | 39637067506 | 12362840074 | 5992e5f3a14d8acf1d625b6df34f7ae1f8d18043d0ebfedf11353b71f0e6efd7 |
| REUSE-33 physical TRAIN teacher | 40101923830 | 12563231248 | adbbb1a1d67f07fcb4efeb94d1188e6b5c793720cd6cb5c3b3cd9f506a85d80a |
| REUSE-36 expanded physical TRAIN teacher | 40110945017 | 12565959908 | 6ca2bf2d3102f3cd68c60394e8eb42af236cf47fcdd6f6e3abe5104f800d0c64 |
| E1 native effect source | 40165691302 | 12588790887 | aaf8eb011763bf47e1d196771e257eb13c4eb35294bbeed2b71757ade525f073 |

Existing **source-code** inner-file hash guards, NOT independently re-run here: REUSE-33 summary eaf2f99a2dae59f3e4fe78a1f296edf72afbe09bc107cbeb382fcb69f3eaf57d, numeric b405e1ba816f424179dbf65a93a31e5a5db5b471bd8079671ecfd99215688efd; REUSE-36 summary a178131f01724a9c14cd1ea16e45b8bd62eb17e447b6e751e2297264402277af, numeric e6c605bc9a109e2e366eb50635361bae9cd4c93563c282d787640694280d75da. Check actual bytes before any future reconstruction.

## Limited prospective-layout collision check

Arithmetic-only audit of ten **specifically inspected** historical XLand layout ranges (206 distinct IDs): REUSE-33 TRAIN 10400–10423 / EVAL 10600–10609; REUSE-36 TRAIN 10900–10971 / EVAL 11000–11019; REUSE-37 EVAL 11100–11119; G1 EVAL 11300–11304; D3 EVAL 11700–11709; D4 EVAL 11800–11819; E1 TRAIN 11900–11914 / TEST 12000–12009. P9 candidate 17100–17107 overlaps **none of these**. **This is not a complete all-repository/all-artifact seed-history audit**; do not certify the seeds untouched or substitute alternatives. Workshop candidates 28100–28115 were not exhaustively checked either.

## STOP and next smallest safe step

**P9 remains STOP before any native action.** A separate bounded offline preflight must download/re-hash REUSE-19/33/36 outer and inner artifacts, verify TRAIN-only public features and dependency compatibility, account for every BC reconstruction fit, export **numeric-only SHA-pinned frozen expanded-arm weights** without pickle/torch.load, and establish independent logit/action parity on deterministic public observations. Only a subsequent P9 scorer may load that verified frozen checkpoint without any BC refitting. Complete historical seed overlap, native dependencies, action/evaluator separation and per-episode hard budgets remain independent prerequisites.

P8's early smoke/provisional Brier and calibration limitations, P6 history gate FAIL, P4 RND STOP, and small D3/D4 physical gains are unchanged. Prediction of public observation change is not evidence of improved navigation, causal action selection, learned recovery or cross-domain transfer.

**Resource accounting:** 0 new native resets/actions, 0 BC/predictor fits/optimizer steps, 0 teacher queries, 0 source artifact downloads or original EVAL array reads, 0 workflows/automations, no unsafe pickle, billing, secrets, deployments, merges or main writes.
