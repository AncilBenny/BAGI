# G2-E5-P5 — Completed TRAIN-only public-transition identifiability result

**Result date:** 2026-10-26. **Status:** archived retrospective TRAIN-only diagnostic COMPLETED, source-hash verified and independently reproduced; **not** a native policy trial, model-fitting experiment, causal identification, held-out evaluation, proof of belief, or evidence of transfer. The [prospective preregistration](GENERAL_BRAIN_G2_E5_P5_PUBLIC_TRANSITION_PREREG.md) and frozen experiment code remain unchanged. This report logs results only after the successful preregistration and synthetic CI, never retroactively changes the gate.

## Immutable source and reproduction evidence

- Research branch `setup/initial-platform` starting commit `8a1df1f721a0fad8b42d5d8e01082dcd92b547be`; push Actions `40274495509`, job `121414239685`, and PR Actions `40274499755`, job `121414253431`, both completed **SUCCESS** including Ruff and pytest, no artifacts and no active runs at verification. Existing PR #1 untouched, no visible open research issues or competing writer at the preflight check.
- E1 source-only native archive from run `40165691302`, artifact `12588790887`, verified full ZIP SHA256 `aaf8eb011763bf47e1d196771e257eb13c4eb35294bbeed2b71757ade525f073`; E2 archived Workshop DQN TRAIN from run `40185492543`, artifact `12596191672`, verified ZIP SHA256 `b000a02a2eacfb8aa07befa151e0bbfb2c5bf7fd430f8bd896bd11f90f57436a`.
- Authoritative P5 code source Git blob `9aeaab498d4a0bfe34469e199beea95793ef548e`; tests source blob `748ad1f11b6819d245a62ddf95b3482648dea522`. Locally verified byte-for-byte against the branch's source blob hash and reran **five P5 synthetic tests: 5 passed** (Python 3.13.5, NumPy 2.3.5, pytest 9.0.2).
- Original P5 `run(e1,e2)` reproduced its prior recorded output **exactly**, using strict archive SHA gates, `np.load(..., allow_pickle=False)`, and TRAIN-only array whitelists. Output numeric JSON SHA256 `238b279669e7c2e2a31466298cc11ef53328f6973814e126c5fe750fed3dd0ec`. An independent, separately written TRAIN-only implementation also reproduced **all four dataset dictionaries exactly**; its JSON differs only in metadata (independent-run status/source SHA/limitation). Numeric counts are retained in [`docs/research-results/G2_E5_P5_TRAIN_ONLY.json`](research-results/G2_E5_P5_TRAIN_ONLY.json).

Reproduce read-only with the existing project module, the two exact published ZIPs and no upstream TEST imports:

```bash
python -m bagi.experiments.g2_e5_p5_public_transition_identifiability \
  --e1-zip /path/to/verified_e1.zip \
  --e2-zip /path/to/verified_e2.zip \
  --output /path/to/p5_train_only.json
```

## Frozen results: exact public-observation + actual executed action keys

`Ambig keys/rows` counts repeated keys for which more than one next **public** view occurred. `Repeated rows` counts observations with any same-key repetition; it is NOT a measured generalization score. Results preserve history widths 0 and 4, without changing source data.

| Archived TRAIN dataset | Rows / episodes | Unique keys (h0 → h4) | Repeated key rows (h0 → h4) | Ambiguous keys (h0 → h4) | Ambiguous rows (h0 → h4) | Cross-episode ambiguous rows (h0 → h4) |
|---|---:|---:|---:|---:|---:|---:|
| E1 XLand original BC + random | 3,064 / 180 | 2,708 → 2,708 | 525 → 525 | 29 → 29 | 87 → 87 | 87 → 87 |
| E1 Workshop random + duplicated fixed-script | 340 / 20 | 64 → 190 | 325 → 175 | 0 → 0 | 0 → 0 | 0 → 0 |
| E2 Workshop learned DQN TRAIN seed 17 | 2,048 / 52 | 118 → 1,572 | 2,023 → 737 | 1 → 1 | 27 → 2 | 27 → 2 |
| E2 Workshop learned DQN TRAIN seed 29 | 2,048 / 52 | 102 → 1,577 | 2,035 → 710 | 1 → 2 | 29 → 6 | 29 → 6 |

- XLand current public features **already contain the previous four actual executed-action encodings**; adding separately reconstructed history is redundant. P5 excludes exactly 28 history-feature bits from the **next-view target**, not from the legal acting/current-view input. Its 29 conflicting keys (87 rows) are cross-episode; this is **not** evidence of a transferable predictive latent model.
- E1 Workshop has 0 sampled contradictory next public views, but duplicated deterministic fixed-script episodes are not independent evidence and absent observed contradictions **do not establish observability**. The repetition count drops by 150 when adding history.
- For E2 Workshop DQN, 4-action history reduces ambiguous **rows** 27→2 and 29→6, but repeated-key coverage also declines 2,023→737 and 2,035→710; unique keys rise 118→1,572 and 102→1,577. Thus reduced ambiguous-row counts may reflect **key fragmentation**, not improved identifiability or predictive skill. Seed29 ambiguous **keys increase** 1→2. No supervised predictor was fitted or tested, so there is no history-vs-reactive Brier or next-state accuracy improvement to report.

## Scientific decision

**STOP / NO PROMOTION:** this dataset-level observation is necessary context for future reusable belief/effect modeling, but cannot establish learned memory, task-independent navigation, native Workshop goal completion, physical BLUE effects or generalization. Do not use these TRAIN descriptive statistics as a model-selection or TEST gate. Next prioritize a separately preregistered observation-only matched predictive study **with held-out source-disjoint real transitions**, strict repeated-key/effective-sample-size controls, reactive/history/shuffled-history/action-only baselines, plus independently trained Workshop policies and physical native XLand criteria; or obtain genuine learned Workshop rewards before attempting general cross-domain promotion. P4 RND remains STOP on incompatible/unproven upstream wheel and SB3 reward/update contract; no custom heuristic substitution.

**Actual additional resource count:** 0 new native TRAIN or EVAL actions/resets, 0 teacher queries, 0 trained model or policy optimizer updates, 0 TEST-array reads, 0 workflows/automations, no unsafe pickle, billing, secrets, deployment, `main` write or merges. Only two already-archived, SHA-pinned ZIPs read, five local synthetic tests and source-only numeric counting.
