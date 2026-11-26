# BAGI-M4 — completed frozen-policy action-selection evaluation

**Outcome audited 2026-11-11 UTC; preregistered before any independent M4 episodes. This report is retrospective, not a revision to the preregistration.**

## Execution and artifact provenance

- Original registered experiment: docs/GENERAL_BRAIN_M4_FROZEN_POLICY_EVALUATION_PREREG.md, with source-only implementation and all-eight-checkpoint restoration preflight.
- Admitted execution source commit: 85fa5d04464019b14d100f7e60bdb6b8363cb478; configuration: configs/m4_frozen_policy.yaml, SHA256 b91096dd57c74ac0dae16f2ebf4afb45523de5043f75d460459883883dac4ad6.
- Existing serialized research workflow [run 42952169808](https://github.com/AncilBenny/BAGI/actions/runs/42952169808): SUCCESS. Exact-source push CI [42952169797](https://github.com/AncilBenny/BAGI/actions/runs/42952169797) and PR CI [42952175113](https://github.com/AncilBenny/BAGI/actions/runs/42952175113): SUCCESS with Ruff, pytest and PR merge-provenance check.
- Downloaded artifact bagi-research-635, ID **13825437331**, ZIP SHA256 **b53a1813f0799da3d5b2c530708f40c37dce71693278b3346715b0551a0c3a4f**, matching the GitHub artifact digest. Its research_branch_commit.txt agrees with the execution source and study_mode.txt is m4; record timestamp 2026-11-11T03:25:31Z.
- Upstream archived M3 checkpoint artifact ID 13823291153, source 99f7f194787c1f02014b692aed221957639b77e6, ZIP SHA256 4aa0656b6dd6d9aacd8a187246014100cb71368f194e5c21682182618d860739, original config SHA256 820aab08b951b1447e897225522b382555a12cbbaca0a021169d08d289070824. All eight frozen model policies were reconstructed from numeric-only NPZ, with exact member-name/shape/dtype/hash checking and no pickle. Their original deterministic M3 MemoryS7 results were **reproduced before any M4 independent test**: 48 separate verification episodes.
- M4 performed **0 new training simulator actions, 0 new model fits, 0 optimizer updates and 0 teacher queries**. Earlier M3 totals 524288 training actions / 16384 optimizer updates are **reused historical artifact metadata, NOT new M4 work**.
- Actually executed M4: **936 M3 reference-replay native actions, 48 reference resets; 21245 new independent native evaluation actions, 1152 new independent resets**, 14.154 seconds elapsed in evaluator. Each M4 independent episode has its own actual reset and executed actions, not imagined steps or cached success. Native goal success requires positive reward AND true termination, never reward alone.
- New independent MemoryS7 layouts: 98001..98012, and size13 zero-shot MemoryS13Random layouts: 99001..99012; both are disjoint from all M1/M2/M3 tested layouts. For EACH of eight frozen models and each task, 12 deterministic episodes plus 60 sampled episodes (5 repeated stochastic draws per layout). The 5 draws share the same layout and must NOT be represented as independent layouts.
- All 1152 episode success values, true terminal flags, task/layout seed lists, executed-action histogram sums, independent native step sum and reset count were independently recomputed from the downloaded JSON and reconciled exactly against the summary.

## Independently rewarded goal completions

The numerator/denominator below represent separate native rollouts, NOT independent model seeds or independent layouts for sampled repeats.

| PPO arm | M3 model seed | MemoryS7 deterministic | MemoryS7 sampled | Size13 deterministic | Size13 sampled |
|---|---:|---:|---:|---:|---:|
| reactive | 37 | 7/12 | 33/60 | 3/12 | 33/60 |
| reactive | 53 | 1/12 | 28/60 | 9/12 | 26/60 |
| action4 | 37 | 5/12 | 35/60 | 9/12 | 30/60 |
| action4 | 53 | 5/12 | 29/60 | 9/12 | 39/60 |
| frame4 | 37 | 7/12 | 25/60 | 0/12 | 17/60 |
| frame4 | 53 | 7/12 | 30/60 | 3/12 | 19/60 |
| recurrent | 37 | 7/12 | 25/60 | 3/12 | 29/60 |
| recurrent | 53 | 0/12 | 29/60 | 0/12 | 38/60 |

The preregistered primary same-task recurrent gate required sampled success rate minus deterministic success rate of **at least +0.25, separately for each recurrent model seed**. Observed recurrent seed37: 25/60 - 7/12 = **-0.1667**; seed53: 29/60 - 0/12 = **+0.4833**. **H-M4 FAILS** because both seeds did not exceed the threshold. Do not pool away the failed seed or change the threshold after observation.

As a secondary size13 diagnostic, recurrent seed37 sampled-minus-deterministic is +0.2333 (29/60 - 3/12); seed53 is +0.6333 (38/60 - 0/12). In contrast, action-history PPO seed37/53 size13 deterministic reached 9/12 and 9/12, versus sampled 30/60 and 39/60. This supports a useful transfer-control lead but not task-independent learning or recurrent memory advantage.

## Cluster-aware uncertainty and failure interpretation

Post-hoc **derived uncertainty calculation**, not new simulator data: for each recurrent model seed and task, form 12 layout-level contrasts, each defined as the average of its five sampled goal indicators minus its one deterministic goal indicator on the SAME layout. Run 30000 independent percentile-bootstrap draws of 12 whole layouts with replacement per model/task, NumPy default_rng(20261111 + model_seed), and take 2.5th and 97.5th percentiles. This respects within-layout dependence but cannot remove model-selection bias or fully quantify uncertainty from two model initializations.

| Recurrent model | MemoryS7 difference (95% layout-bootstrap interval) | Size13 difference (95% layout-bootstrap interval) |
|---|---:|---:|
| seed37 | -0.1667 (-0.3667, +0.0333) | +0.2333 (-0.0167, +0.4500) |
| seed53 | +0.4833 (+0.2667, +0.7000) | +0.6333 (+0.4000, +0.8333) |

These intervals are descriptive, conditional on each trained model and 12 layouts; they do not establish a general-policy mechanism. Significant differences in action selection may reflect determinism, action-distribution entropy, model initialization, and layout variation. Merely obtaining native navigation reward is not physically verified BLUE-square acquisition, tool-task completion, or persistent cross-domain capability.

M4 is an **evaluation-only study** and M3 remains the only source of these eight policy weights. The prior M2 transfer recurrence lead and M3 negative independent replication are preserved. No Workshop/XLand physical-task success is claimed.

## Evidence-driven next milestone

Do not keep rerunning stochastic action-selection experiments or claim the failed recurrent gate has passed. The task-independent BAGI gap is continual acquisition and retention of distinct, publicly specified goals under executed actions. Preregister a new controlled, multi-task continual-learning pilot using maintained upstream PPO and a second task family within MiniGrid before ever promoting cross-domain generality. See docs/GENERAL_BRAIN_M5_CONTINUAL_TASK_RETENTION_PREREG.md.

No prior research log entry or artifact was overwritten, no model/checkpoint/archive or credential was committed, and no schedule/workflow was added. This report itself requires current-HEAD Ruff, pytest and independent push/PR CI before the milestone is considered source-validated. Main integration still requires the reviewed PR path.
