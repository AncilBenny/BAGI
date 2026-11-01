# G2-E5-P9 — Independently verified XLand BC TRAIN-source bytes (2026-10-26)

**Decision: historical TRAIN source passed byte and schema verification; prospective P9 native evaluation remains HARD STOP.** This is a reproducibility/safety gate, **not** an XLand frozen-model checkpoint, BC policy restoration, new learning, evidence of native physical manipulation, or task-independent transfer. The original [P9 preregistration](GENERAL_BRAIN_G2_E5_P9_NATIVE_CONFIRMATION_PREREG.md) remains unchanged.

## Fresh evidence and exact provenance

Starting research branch `setup/initial-platform` HEAD `5428846c2290248bf5a4e0748ad90e2db570a00d`; its push CI `40323675869` and PR CI `40323682531` completed SUCCESS (including lint and tests), with zero active research Actions jobs at the successful preflight. Existing open PR #1 not touched. Mandatory research protocol, decision gates, reuse inventory, general architecture, D3/D4 physical native evidence, full append-only log, P8 results/P9 preregistration, and the previous [XLand actor-provenance STOP](GENERAL_BRAIN_G2_E5_P9_XLAND_ACTOR_PROVENANCE_PREFLIGHT.md) read before this verification.

For the FIRST time in the P9 chain, three exact historical GitHub Actions TRAIN archives were independently **downloaded, rehashed byte-for-byte, and verified** against the previously API-reported outer SHA256 and the old source-code inner-file SHA256 constants, without deserializing pickle or opening original E2 mixed-EVAL data:

| Training source | Original run / artifact | Outer ZIP SHA256 verified | Numeric TRAIN member SHA256 verified | Shape verified |
|---|---|---|---|---|
| REUSE-19, actual executed action history | `39637067506` / `12362840074` | `5992e5f3a14d8acf1d625b6df34f7ae1f8d18043d0ebfedf11353b71f0e6efd7` | seed17 `25d24168163fbc189f1afbd8454e3948deae73d4924cfc83f49ba35ebb5b4882`, seed29 `66d5a546875e8b373f5ec42926b46b683e0f29c885a24f207f66386f1dfa4607` | 291 rows×54 legal source floats per policy seed, 4 actual prior executed actions, integer labels/actions, boolean done |
| REUSE-33, physical source teacher | `40101923830` / `12563231248` | `adbbb1a1d67f07fcb4efeb94d1188e6b5c793720cd6cb5c3b3cd9f506a85d80a` | `b405e1ba816f424179dbf65a93a31e5a5db5b471bd8079671ecfd99215688efd` | 248 rows×115 source public floats, TRAIN expert actions, done |
| REUSE-36, diversified physical teacher | `40110945017` / `12565959908` | `6ca2bf2d3102f3cd68c60394e8eb42af236cf47fcdd6f6e3abe5104f800d0c64` | `e6c605bc9a109e2e366eb50635361bae9cd4c93563c282d787640694280d75da` | 707 rows×115, 66 action3, 66 action4, 66 episode completion markers |

Original summary JSON member hashes were also verified exactly: REUSE-19 `71c8c72e31574289447b399c21e475e523e5d4eccceea998cc7ddd9ef7105892`, REUSE-33 `eaf2f99a2dae59f3e4fe78a1f296edf72afbe09bc107cbeb382fcb69f3eaf57d`, REUSE-36 `a178131f01724a9c14cd1ea16e45b8bd62eb17e447b6e751e2297264402277af`. Only REUSE-19 TRAIN summary was JSON-parsed for its two seed-specific numeric-digest attestations; REUSE-33/36 summaries were hashed as bytes but **not JSON-parsed**, as they include historical EVAL outcome metadata. No numerical EVAL arrays or heldout composition teacher labels were accessed or fit.

The REUSE-19 original numeric record hashing protocol (sorted field name then contiguous array bytes) was independently recalculated. Both digests exactly equal their `model_results` metadata: seed17 `f893dc633a14bb9347c5e12a2ac5799f450b49a2daa700e4e67845f2176dd4ba`, seed29 `0735b160ef62a353b3fb52dd8df496630ffe5104ec10d24b1bfb2b22a9b83675`. This verifies the stored public TRAIN action-provenance rows, not BC parameters or an actor's efficacy.

## Machine-verifiable safety gate

New **standard-library + existing NumPy** reader `bagi/experiments/g2_e5_p9_xland_source_preflight.py` performs whole-ZIP SHA256 before opening, exact ZIP-member listing and size limits, per-member hash checks, numeric `np.load(allow_pickle=False)` field allowlists/shapes/dtypes and values, the two REUSE-19 content digests, and physical teacher label-completeness checks. It outputs only a source/provenance manifest; **the TEACHER labels are not written into the report**. Six simulator-free synthetic tamper tests passed locally (outer-hash mismatch, ZIP extra/duplicate member, valid-ZIP/altered-member mismatch, EVAL or object field, invalid arrays and record digest mutation). The actual three historical source archives all passed the same verifier; machine output is [G2_E5_P9_XLAND_TRAIN_PROVENANCE.json](research-results/G2_E5_P9_XLAND_TRAIN_PROVENANCE.json), SHA256 `e42ff1049382a72471a6a54610c0612cc2dd29917a9ca5e3acea17570c503a0f`. GitHub CI for this NEW module/tests is pending after its branch-only commit.

Reproduce from the exact downloaded source ZIPs without native environment, policy-model dependencies, or TEST arrays:

```sh
python -m bagi.experiments.g2_e5_p9_xland_source_preflight \
  --reuse19 /path/to/github-actions-artifact-12362840074.zip \
  --reuse33 /path/to/github-actions-artifact-12563231248.zip \
  --reuse36 /path/to/github-actions-artifact-12565959908.zip \
  --output /path/to/G2_E5_P9_XLAND_TRAIN_PROVENANCE.json
```

## Decisive limitation and next gate

There are **zero archived BC policy-weight files** among these verified source artifact members. They do NOT establish the originally learned frozen expanded-arm policy tensor state. Existing `source_policies` would retrain four upstream `imitation.BC` actors (two seeds×two control arms, 787 sampled rows×12 epochs = 37,776 nominal sample exposures). To reproduce only the expanded-arm P9 behavior actor would still require **two disclosed supervised BC reconstructions**, 18,888 nominal sample exposures, exact dependency versions, an immutable SAFE NUMERIC tensor export and independent original-policy logits/action parity. Never call reconstructed weights an untouched original frozen checkpoint without establishing identity. Neither actor training nor checkpoint export happened here. It is unsafe to run P9 native evaluations against a silently retrained actor, or to claim zero learning updates.

P9 also requires a **complete** seed-history overlap audit and a tested interaction-metered native public/evaluator boundary before any of its candidate XLand layouts `17100–17107` or Workshop reset seeds `28100–28115` are used. The earlier partial 206-layout check is explicitly NOT full clearance. Separate held-out learned-goal and genuinely physical BLUE placement criteria remain mandatory for any later intervention claims. P8 offline Brier numerically positive but calibration/protocol-limited; P6 history gate FAIL; P4 RND STOP.

**Actual additional resource accounting:** zero native XLand/Workshop resets or actions, zero BC/predictor fits and optimizer updates, zero teacher queries, zero model or evaluator goal scoring, zero original TEST/EVAL numeric-array reads, no pickle, no workflow/automation/dependency/secrets/deployment/billing/main/merge changes. THREE previously collected TRAIN ZIPs downloaded and byte-read; one REUSE-19 TRAIN source metadata JSON parsed, two old EVAL-bearing summary members hashed (not parsed). Only one bounded nonforce research-branch documentation/source/tests/log commit permitted.
