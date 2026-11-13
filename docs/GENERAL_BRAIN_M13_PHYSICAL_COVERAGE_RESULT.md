# BAGI M13 — executed public-only physical interaction coverage

**2026-11-12 UTC. Result: preregistered M13 pilot PASS for genuine physical key/door interaction coverage. End-to-end physical-goal success: ZERO on independent final across all controllers.** This is an integration/coverage diagnostic with **0 learned models and 0 optimizer updates**, NOT evidence of task-independent continual learning, learned manipulation or general intelligence.

## Actual source and artifact provenance

- Registered source module/prereg: `bagi/experiments/m13_public_interaction_coverage.py`, `docs/GENERAL_BRAIN_M13_PUBLIC_INTERACTION_PREREG.md`, frozen `configs/m13_public_interaction_coverage.yaml`; implementation commit `eefef483409f364132ffcdb8ce40b31ae7d9fc6d`, registered existing-workflow commit `c2e284e7e5a7d2c3c65fdcd06d39ecd6ed40b503`.
- Exact execution from ADD-only one-shot request commit **`638bc6e0ef3d76a7ec6f37517d6823d9ae7bfb7b`**. Existing serialized [BAGI Research Loop run 43237067806](https://github.com/AncilBenny/BAGI/actions/runs/43237067806) completed SUCCESS. Prior tested source push CI **43236566970** and PR CI **43236579153** SUCCESS; exact request commit push CI **43237067694** and PR CI **43237080965** SUCCESS with Ruff, pytest and PR merge provenance.
- Actions artifact **`bagi-research-651` ID `13952747779`**. Independently downloaded ZIP SHA-256 **`904957127ad19e412f64a08a9d39d01f6fa6a6aa5b45d8b73847a16c78fb2131`**, identical to published artifact digest; ZIP CRC passed. Embedded source commit and study marker match `638bc6e0...` and `m13`. Config SHA-256 **`8d4923c0f33fc2d9f115553c970271187418bee3ecad03db9bfd24dfbc5e37d9`**.
- Executed versions: Gymnasium **1.4.0**, Farama MiniGrid **3.2.0**, NumPy **2.4.6**. Reused native DoorKey and independent BAGI `NativeDoorKeyAudit`. No model checkpoints, downloads of untrusted agents, teacher queries or private-state policy inputs.

## Real simulator actions and experiment splits

- **TRAIN-MONITOR:** 24 seed layouts per size (6x6 171001–171024; 8x8 181001–181024) × 3 fixed public controllers: **144 episodes and 36,805 executed native actions**. These layouts are diagnostic, never independent evidence.
- **INDEPENDENT FINAL:** 36 previously untouched layouts per size (6x6 172001–172036; 8x8 182001–182036) × 3 controllers: **216 episodes and 55,296 executed native actions**.
- **New executed M13 total: 360 episodes, 92,101 real native actions**. Run measured wall time **42.845 s** excluding CI/install. **0** optimizer-step calls, **0** fits, **0** teacher queries. Every episode's seven-action histogram sums to its native action count, all physical openings are preceded by an actual toggle action and all valid toggle attempts are subset of true valid-door opportunities. External private snapshots were used ONLY by the evaluator after executed actions and never passed into controller selection or public memory updates.
- Public controllers: `uniform` selects all 7 native commands; `reactive` reacts to visible key/door and explores without using key memory; `public_memory` keeps a one-bit **public** record of key disappearance following its **own** executed pickup. No learned weights, expert instruction, special private state, reward shaping or hidden coordinates. Controller random seeds were fixed in source; comparison is not an isolated causal memory intervention because trajectories diverge.

## Independent FINAL outcomes, 36 distinct layout episodes per size/mode

| DoorKey environment | Controller | Actual key pickup episodes | Physically opened door episodes | Valid door opportunity episodes | Physically complete key→door→rewarded goal episodes |
|---|---|---:|---:|---:|---:|
| 6x6 | uniform random | 31/36 | 5/36 | 8/36 | **0/36** |
| 6x6 | public reactive | 36/36 | 34/36 | 36/36 | **0/36** |
| 6x6 | public memory | 36/36 | **36/36** | 36/36 | **0/36** |
| 8x8 | uniform random | 28/36 | 4/36 | 9/36 | **0/36** |
| 8x8 | public reactive | 36/36 | 26/36 | 32/36 | **0/36** |
| 8x8 | public memory | 36/36 | **32/36** | 32/36 | **0/36** |

- Pooled independent effective-door openings: **public memory 68/72 episodes, reactive 60/72, uniform random 9/72**. Memory minus reactive +2/36 (6x6) and +6/36 (8x8), pooled +8. Memory pickups 36/36 each size. Thus the frozen primary **interaction-coverage gate PASSES** (>=4 pickups per size, nonnegative per-size door-opening difference, >=+2 pooled openings).
- Native toggle actions: 6x6 uniform1350 (5 valid), reactive57 (34 valid), memory36 (**36 valid**); 8x8 uniform1354 (4 valid), reactive33 (26 valid), memory32 (**32 valid**). Hence memory 68/68 toggles valid under frozen layouts, but the validity conditional is partly **designed** by its hand-coded public key/disappearance rule. It is not independent evidence that a model has learned the concept or calibrated a posterior.
- Full native key-pickup→door-open→goal successes **0/216 independent episodes** across all controllers; native rewarded goals **also 0/216**. A single prior **TRAIN-MONITOR** random episode (6x6 layout171015) had one physically verified full goal after197 real actions; this is NOT independent final success or learned behavior.
- External key-evidence audit: public-memory controller's evidence bit agreed with actual held key under these controls (0 false positive/negative action-steps) because it never selects drop. Uniform random controller had **6118 false-positive key evidence action-steps** in independent final (3574 6x6 +2544 8x8) after actions such as native drops; this is a known **non-updating evidence-memory bug** if one tries to reuse the simple bit under general actors.

## Important diagnosed controller limitation and next experiment

**Source-visible implementation limitation:** the M13 controller's `choose()` navigation fallback treats **all door-front objects** as blocked via `obj in (KEY, DOOR, WALL)`, without checking that an actually open door has public state0. A successful toggle opens the door physically, but the controller's navigation step cannot intentionally choose forward across an open door when the door is immediately in front. This is a credible post-opening control bottleneck consistent with the measured ZERO independent full-goal success; it is not formally proven to be the only cause, because stochastic navigation and layout geometry may matter too. Preserve this bug and zero-goal result as evidence: **do not retroactively alter M13 policy or refit on the 72 final layouts**.

**New falsifiable follow-up (not executed here):** preregister a distinct M14 physical task-goal experiment in which the same public controller may move forward when an actually **publicly observable opened door** is in front. Keep the original M13 reactive/public-memory controls and action budget, use wholly new TRAIN/FINAL layout seeds, external true pickup→door→goal audit, and compare unchanged controller versus this one policy-contract correction. Count actual physical goals, native actions and post-door progression, not reward alone. If no physically verified goal improvement occurs, stop assuming post-opening control was the primary limitation and investigate public navigation separately. This is still a hand-specified baseline and would not establish learned generality.

**Governance:** historical M9–M12 results, independent layouts and append-only log preserved. M13 independent FINAL is now CLOSED to subsequent training and tuning. No new model training is claimed, and no result file/model checkpoint was committed. Main integration HOLD pending independently validated current push/PR CI and reviewed PR #1.
