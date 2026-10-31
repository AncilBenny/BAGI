# G2-E5-P9 — Independent XLand BC TRAIN sampling audit (2026-10-26)

**Read-only result:** The published expanded-arm TRAIN action histograms reproduce **exactly** for both policy seeds, but 787 with-replacement training draws contain only **401/414 distinct source-row indices**. This is not new policy training, historical actor-weight verification, native skill acquisition or cross-domain transfer. **P9 native EVAL remains STOP** under the unchanged [P9 prospective preregistration](GENERAL_BRAIN_G2_E5_P9_NATIVE_CONFIRMATION_PREREG.md).

## Reuse-first question and source verification

Before two separately accounted upstream `imitation.BC` reconstructions, quantify the effective data diversity of the **existing published** class-balanced source recipe. The original `xland_complete_demo_bc.sampled_indices` uses `torch.utils.data.WeightedRandomSampler` with replacement, inverse per-action label frequencies and exact seed offsets 0, +4100, +5300. The original `xland_diverse_physical_demos.sample_fixed_extra` selects 291 original executed-history TRAIN rows, 248 rehearsal rows from that same 291-row pool and 248 physical TRAIN rows from the concatenated 248+707=955-row teacher pool, for **787 draws per BC seed**, 12 epochs and 9,444 nominal sample exposures per fit. This audit reuses the exact sampling procedure, not a new learner or heuristic.

Three historical Actions ZIPs were downloaded and independently SHA256-verified before opening: REUSE-19 run `39637067506` / artifact `12362840074` = `5992e5f3a14d8acf1d625b6df34f7ae1f8d18043d0ebfedf11353b71f0e6efd7`; REUSE-33 run `40101923830` / artifact `12563231248` = `adbbb1a1d67f07fcb4efeb94d1188e6b5c793720cd6cb5c3b3cd9f506a85d80a`; REUSE-36 run `40110945017` / artifact `12565959908` = `6ca2bf2d3102f3cd68c60394e8eb42af236cf47fcdd6f6e3abe5104f800d0c64`. All four inner TRAIN NPZ hashes matched the earlier [independent byte verification](GENERAL_BRAIN_G2_E5_P9_XLAND_SOURCE_BYTE_VERIFICATION.md). Only allowlisted numeric fields were materialized with `np.load(allow_pickle=False)`. No original E2 mixed archive, E1 TEST numeric data, held-out composition expert labels, or original EVAL numeric arrays were accessed.

**Metadata-access disclosure:** REUSE-36 `summary.json` was parsed solely to check the original TRAIN histogram, source digest, rows and nominal exposures. That JSON also contains historical EVAL outcome metadata, which was consequently read but **not used** for sampling, training or scoring. REUSE-33 summary JSON was not parsed.

## Reproduced sampling evidence

| Source-index and action-label audit | BC seed 17 | BC seed 29 |
|---|---:|---:|
| Original 291-row source labels, actions 0–5 | 130,69,64,22,6,0 | 132,83,49,21,6,0 |
| **Final 787 sampled labels, actions 0–5** | **140,166,154,161,166,0** | **142,157,172,145,171,0** |
| Exact match to published REUSE-36 expanded-arm TRAIN histogram | **YES** | **YES** |
| Distinct original-source indices across 291+248 draws | 207/291 | 212/291 |
| Distinct physical-source indices across 248 draws | 194/955 | 202/955 |
| **Distinct source-row indices among 787 draws** | **401** | **414** |
| Repeated source-index draws | **386** | **373** |
| Physical draws from newer 707-row source | 201/248 | 188/248 |

Across both seeds, only **331 of the 955 physical TRAIN source indices** appear at least once; **65 indices** appear in both seeds' sampled physical subsets. These are **source-row index** counts, not independent environments, trajectories, semantic states, unique public observations or actually executed learner effects. The two label histograms match the historical published `results[].training.label_action_counts` exactly. This is strong evidence for reproduced *class-level sampling composition*, **not** proof of identical sampled-index sequences, full 115-feature BC training tensors, or neural weights. Action 5 has no teacher-label support in these particular sources; no conclusion about its necessity follows.

The numeric ledger, including each 291/248/248 block's action counts, sampled-index SHA256 digests and source SHA pins, is [G2_E5_P9_XLAND_BC_TRAIN_SAMPLING_AUDIT.json](research-results/G2_E5_P9_XLAND_BC_TRAIN_SAMPLING_AUDIT.json).

## Scientific interpretation and STOP

The published balancing procedure gives >140 nominal samples per action class 0–4 in both seed fits, while many draws repeat the same source indices. This does **not** establish that repetition caused earlier navigation/manipulation failures or that more labels, another architecture, a BLUE-specific controller or a heuristic sweep would help. The sampling audit cannot assess physical effects, state aliasing, goal-conditioned transfer or task-independent lifelong learning.

**Runtime limitation:** local sampling used Python **3.13.5**, PyTorch **2.10.0+cpu**, NumPy **2.3.5**, not historical pinned Python **3.11** / PyTorch **2.5.1+cpu** / `imitation==1.0.1`. Histogram parity was verified against published metadata, but **index-level parity under pinned runtime remains UNVERIFIED**. The base 291 public features were not reassembled with the task instruction/history encoder, and **no BC fit or optimizer step occurred**.

**Next smallest safe gate:** rerun index SHA parity under historical pinned CPU Torch, then only consider an explicitly preregistered, separately metered TRAIN-only two-actor BC reconstruction and real upstream categorical action/log-probability parity. Reconstructed actors would be **new** policies without independently preserved historical weight identity; the original P9 frozen-actor preregistration cannot silently substitute them. Complete XLand/Workshop candidate-seed overlap, native dependencies, public/evaluator isolation and native-step instrumentation remain independent STOPs. The P8 calibration caveat, P6 memory FAIL, P4 RND STOP and small D3/D4 physical results remain unchanged.

**Actual resources:** three small historical TRAIN ZIP downloads; one REUSE-36 historical summary read; numeric TRAIN-only sampling and hash checks; **0** new XLand/Workshop native actions/resets, **0** actor/predictor fits or optimizer steps, **0** teacher queries, **0** original E2 EVAL/E1 TEST numeric-array reads, **0** workflow/automation/dependency changes, no secrets, billing, deployments, merges or `main` edits. Starting HEAD `d74404319a84a3cfb0b5bd74c1eab9ea02b0464e`, push CI `40340950532` and PR CI `40340957281` both SUCCESS (lint/tests, no artifacts).
