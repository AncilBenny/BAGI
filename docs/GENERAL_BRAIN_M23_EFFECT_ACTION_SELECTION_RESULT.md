# M23 stage 8 result — public-effect memory in actual native source-task action selection

**Executed 2026-11-15; preregistered source physical-goal gate FAIL.**
This report records historical real execution from GitHub Actions, not a new
training fit, an invented success, or a revision of the preregistered thresholds.
Read `docs/GENERAL_BRAIN_M23_EFFECT_ACTION_SELECTION_PREREG.md` for the
prospective design and `docs/GENERAL_BRAIN_M23_RULE_SHIFT_RESULT.md` for
the frozen stage-7 source-effect result.

## Exact provenance and checks

- Research source commit: `2a8ab341b773d8e1bb30f816ad9685aa6e28c9e9`
  on `setup/initial-platform`; parent stage-8 implementation source
  `e01e3dcb4e55e61c8f376736e01c58883cdb1560`.
- Serialized [BAGI research run 43680572037](https://github.com/AncilBenny/BAGI/actions/runs/43680572037):
  source admission, research and artifact upload completed successfully.
  Exact-source [push CI 43680572114](https://github.com/AncilBenny/BAGI/actions/runs/43680572114)
  and [PR CI 43680576080](https://github.com/AncilBenny/BAGI/actions/runs/43680576080)
  succeeded. Both CI logs executed `ruff check .` and `pytest -q`;
  the latter reported **797 passed, 10 skipped, 44 subtests passed**.
- Uploaded immutable artifact `bagi-research-678`, ID **14148172758**,
  ZIP SHA-256 **`84d4c335cdf7badf69d2f3e73951d808179b190550bd80e74db3c56260d7d4fa`**,
  GitHub-recorded expiry **2026-12-15 05:23:56 UTC**. ZIP checksum independently
  matched GitHub's digest; ZIP internal
  `runs/m23_effect_action_selection/summary.json` and
  `artifacts/research-loop/research_branch_commit.txt` checked.
  The internal source is the exact research commit above, and the study mode
  is `m23action`.
- MiniGrid native DoorKey-6x6 and Fetch-5x5-N2 task adapters, public
  `PublicEffectMemory`, existing public controllers and external
  `NativeDoorKeyAudit`/`NativeFetchAudit` were reused. Study execution used
  override probability **0.65**, with fixed train/monitor layout sets and
  policy/override RNG derivations recorded per episode in the artifact.

## True execution resource accounting

- New, actually executed source TRAIN feasibility actions: **48,655**
  across 64 layouts per task per action strategy (384 pilot episodes).
- New, actually executed independent source MONITOR actions: **24,397**
  across 32 layouts per task per strategy (192 monitor episodes).
- Total new native actions: **73,052**. All actions are issued MiniGrid
  actions; attempted overrides are not counted twice. Per-episode executed
  action hashes and seven-action histograms are retained in the artifact.
- **0 new model fits**, **0 Adam/other optimizer updates**, **0 imagined
  planner steps**, **0 target UnlockPickup actions** and **0 protected
  UnlockPickup FINAL episodes**. The inherited source controllers and
  public-memory rule were not trained during stage 8.
- The report's `elapsed_seconds` measures the experiment script
  (approximately **20.725 s**), not the full GitHub runner lifecycle.
  This is distinct from prior stage-6 (41,052 actions, 16 fits, 98,560
  updates) and stage-7 (9,591 actions, no fits) historical evidence.

## Frozen source MONITOR physical-goal outcomes

All rows below measure the **external auditor's full native physical goal**,
not only environment reward. Each task and mode uses 32 MONITOR layout seeds.

| Real action strategy | DoorKey physical goals | Fetch physical goals | Total / 64 |
| --- | ---: | ---: | ---: |
| Unfiltered 0.65 public overrides | 4 | 9 | 13 |
| Public front-rule gate | 4 | 10 | 14 |
| Public executed-history-memory gate | 9 | 7 | 16 |

The history-memory gate improved DoorKey versus front-rule by **+5/32**,
but worsened Fetch by **−3/32**, for **+2/64 total**. MONITOR had **3,047**
actual override decisions where the two public rules disagreed, above the
minimum support of 16. History-memory total 16/64 was noninferior to
unfiltered 13/64, but the preregistered requirements of **at least +8/64**
versus front rule **and positive gain on each task** were both unmet.
Task-stratified paired 20,000-replicate bootstrap 95% interval for the
total goal-count gain: **[−5, +9]**. The interval includes zero.

**Registered decision: FAIL**, as reported by
`registered_source_action_selection_gate_pass: false`. Do not reinterpret
a favorable aggregate difference, stage-7 prediction F1, or a successful
workflow as passed physical-control transfer.

### Reward and physical-outcome mismatch

On Fetch MONITOR, rewarded terminations versus independently verified
full physical goals were: unfiltered **16 vs 9** (7 reward-only),
front-rule **16 vs 10** (6 reward-only), and history-memory **17 vs 7**
(**10 reward-only**). Fetch's auditor additionally requires a mission-matched
physical pickup, non-initial adjacency and effective movement. On DoorKey,
rewarded/full-goal counts agreed for these MONITOR runs. Physical and reward
measurements must remain separate.

TRAIN feasibility episodes were descriptive and not used to adjust the
preregistered policy or MONITOR gate. Their full episode-level outcomes,
action counts, accepted/rejected overrides, and native effect witnesses
are preserved in the ZIP. Source MONITOR layouts are now exposed; never
retune on them or present reused evaluation as fresh.

## Interpretation and next separate study

Stage 7 established near-perfect prediction under a policy shift for
both frozen learned effects and a simpler public-history rule. Stage 8
tests a different question: whether that memory improves *executed*
source-task control. Under the registered 0.65 override policy the benefit
was task-dependent and statistically uncertain. No zero-shot task
composition, task-independent continual-learning improvement, general
navigation competence, or UnlockPickup physical-chain success was shown.

Preserve the FAIL and freeze this MONITOR. Research may continue under a
**separately preregistered** experiment: diagnose source navigation and
skill preconditions on *new TRAIN-only layouts*, then test an observation-
verified closed-loop option planner from the existing `GeneralBrain` core
against matched public-only nonplanning/assumed-effect controls on fresh,
disjoint source combinations. Use public observations and actual executed
action history for all acting policies, keep native goal/effect witnesses
outside decision inputs, and separate source outcomes from untouched target
FINAL453001–453032. Reuse maintained Farama MiniGrid and established
options/DAgger machinery rather than inventing an unrelated architecture.
Historical failed scientific gates do not prohibit a new prospectively
registered study.

No generated numeric traces, checkpoints or credentials are committed with
this report. Main integration remains **HOLD** until the single current
research-to-main PR obtains an independent submitted review and complete
current source/base/merge CI provenance; do not bypass that path.
