# G2-E5-P9 — Tracked-tree prospective seed collision audit (2026-10-26)

**Scope: static provenance preflight only; P9 native EVAL remains HARD STOP.**
This is an evaluation-integrity result, not new learned action effects, model improvement,
goal achievement, memory, cross-domain transfer, or authorization to run P9.

## Frozen source and reuse-first question

The prospective [P9 native protocol](GENERAL_BRAIN_G2_E5_P9_NATIVE_CONFIRMATION_PREREG.md)
reserves XLand layout IDs **17100–17107** and Workshop reset IDs **28100–28115**
but explicitly requires independent history clearance. The earlier
[XLand actor preflight](GENERAL_BRAIN_G2_E5_P9_XLAND_ACTOR_PROVENANCE_PREFLIGHT.md)
checked only ten historical XLand ranges (206 IDs), not the entire tracked research tree.
This bounded audit reused the existing GitHub tracked source and registration metadata;
no new simulator, model, seed substitution, teacher, or architecture was introduced.

The exact research-branch commit audited was
`70a8c5c51e5548bfe8e29def95468a950fafb492`,
tree `526955594f7cd4c86a608988b7a4740ef92da128`.
The GitHub recursive tree was complete (`truncated=false`), containing **376 blobs**.
Each of the 376 tracked blobs was read as text, including both CSV reports,
`.gitignore` and `Makefile`. Categories explicitly inspected:

| Category | Files |
|---|---:|
| `bagi/experiments/*.py` | 96 |
| `configs/*.yaml, *.yml, *.json` | 55 |
| `.github/workflows/*.yaml, *.yml` | 47 |
| `tests/*.py` | 89 |
| `docs/*.md`, including append-only log | 53 |
| Other tracked files (agents, worlds, results, root files, etc.) | 36 |
| **Total** | **376** |

The text audit searched **whole-number literals** in the candidate sets
(`17100..17107`, `28100..28115`) and checked simple Python
`range(integer, integer)` literals for intersection (exclusive upper bound).
The text-literal search found **eight lines** in total: four in existing P9
preregistration/provenance documents, three in the append-only research log,
and **one in a synthetic negative test**. No overlapping simple Python
`range(start, stop)` was found in the 372 extension-filtered text files
where that pattern was checked. No candidate literal was found in any of the
96 experiment scripts, 55 configurations or 47 workflows.

**Important false-positive classification:** `tests/test_e4_p3_live_vs_stale_observation.py`
line 68 supplies `(29, 28100)` **inside `pytest.raises(ValueError)`** to assert
that `paired_seed` rejects an unregistered seed. This is NOT a native reset.
The corresponding frozen E4-P3 runner explicitly registers
`TEST_SEEDS=range(28000, 28016)` for the **MiniGrid Empty-Random** task.
Its independently recorded successful native run `40224607234` used those
28000–28015 starts, not Workshop 28100. Thus treating the negative-test
literal as a previous Workshop 28100 rollout would be a false collision.
The E4-P3 numeric artifact was **not redownloaded or rehashed** here;
the earlier append-only log records artifact `12611060916` and outer SHA256
`0b8900a9bbd367ad0ec159080cd3b0e63c884ba03553c7875892bb014314f670`.

## Reproducibility and epistemic boundary

For an equivalent checked-out tree, enumerate `git ls-files`, read all
tracked text, and search whole-number tokens using
`(?<![0-9])(?:1710[0-7]|2810[0-9]|2811[0-5])(?![0-9])`.
Independently review every match in its execution context, rather than
treating test fixtures or documentation as real environment interactions.
Inspect simple `range(a,b)` literals for interval intersection, then
explicitly list dynamic expressions, imported seeds and archived runs as
**not certified**. This audit is not a sound whole-program static analysis.

**What passed:** no *literal tracked-source collision* was found in the
registered runnable experiments/configurations/workflows at the audited tree.
**What did NOT pass:** the protocol's requirement for a complete historical
artifact/branch/real-reset uniqueness audit. Archived GitHub Actions artifacts,
untracked/generated files, other branches, dynamically constructed seed
expressions, previous external jobs, and true native reset ledgers were not
exhaustively downloaded, rehashed or compared. Cross-domain equality of a
numeric seed also does not by itself imply equal physical starts.

**Scientific decision: P9 STOP, unchanged.** Do not run candidate layouts,
substitute seeds or promote passive public-change prediction. Additional
independent blockers remain: true historical XLand frozen BC weights or a
separately justified reconstruction (with all fits counted), real pinned
upstream actor logit/action parity, Workshop actor/native compatibility,
public-only predictor/evaluator separation and 160-reset/4,480-step metering.
P8 calibration/protocol caveats, P6 memory gate FAIL, P4 RND STOP and small
D3/D4 physical results remain unchanged.

## Resource and access ledger

Prior branch HEAD `70a8c5c51e5548bfe8e29def95468a950fafb492`;
both associated push CI `40350349655` and PR CI `40350356739`
completed SUCCESS, no published artifacts. Open PR #1 unchanged,
no visible reviews/comments/open issues or active research-branch Actions.
This iteration read tracked source/docs/metadata only: **0** new native
XLand/Workshop resets or actions, **0** actor/predictor fits or optimizer
steps, **0** teacher queries, **0** original EVAL/TEST numeric-array reads,
**0** artifact downloads, **0** workflow/automation/dependency changes,
no unsafe pickle, secrets, billing, deployments, merges or main writes.
External concurrent sessions cannot be ruled out by GitHub visibility alone.
