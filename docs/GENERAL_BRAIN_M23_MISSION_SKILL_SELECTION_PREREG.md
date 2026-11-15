# M23 Stage 9 preregistration — public mission-conditioned source skill selection

**Prospective, 2026-11-15. Not executed.** Registered after the independent M23 Stage 8 physical-goal gate **FAILED**. This is a new, source-only follow-up prompted by the observed DoorKey/Fetch asymmetry; Stage 8's opened 652/662xxx MONITOR data are never used to fit/select Stage 9 or to rescore Stage 8. The prior Stage 8 threshold, results and uncertainty remain unchanged. The original M23 `UnlockPickup` FINAL453001–453032 remains sealed.

## Hypothesis and scope

Correctly predicting whether a native interaction can affect the environment is insufficient to decide whether that interaction *advances a public mission*. Compare a frozen public-mission-relevance skill gate with the source-only action gates from Stage 8 on **new native source layouts**. This isolates goal relevance under randomized action interventions, not learnt zero-shot composition, not hierarchical reinforcement learning, and not an AGI result. The mission gate below is a transparent hand-authored reuse baseline, **not a trained policy**. It must never be described as evidence that an agent learned to select skills.

Reuse maintained MiniGrid, BAGI's public `TraversableOpenDoorController` and `PublicFetchTeacher`, `fetch_request` mission parser, Stage 7 `PublicEffectMemory` and Stage 8 public eligibility filter, and evaluator-only `NativeDoorKeyAudit`/`NativeFetchAudit`. All arms receive only standard public native `image`, `direction`, `mission` and their own executed-action/observation history, no grid, native carrying, target identity oracle, privileged coordinate, evaluator witness, reward, or future observation at action time. Controllers may use the public mission but are selected per *registered source task*; the new override gate itself must infer mission class from legitimate public text only.

## Frozen intervention and arms

Use `MiniGrid-DoorKey-6x6-v0` and `MiniGrid-Fetch-5x5-N2-v0` with existing 256-step wrapper. For each task/layout and independent arm, restart the same fixed seeded source actor and override RNG. At each actual step, with probability **0.65** draw the original Stage 8 front-targeted proposed native interaction 3/4/5; otherwise execute the actor action. A rejected proposal falls back to the actor's original choice. All modes update their own public-only memory after the *actually executed* action. Only native `env.step` calls count as physical interactions.

Compare four frozen modes:
1. `actor_only`: maintained mission-aware source actor, no overrides; non-learning high reference, not an oracle or a new learner.
2. `front_rule_gate`: accept override when the Stage 8 current-public-front-effect rule predicts nonzero.
3. `history_rule_gate`: accept override when the Stage 7 public executed-action-memory rule predicts nonzero.
4. `mission_skill_gate` (new): require the same public-memory nonzero predicted effect, **and** the following fixed public mission relevance:
   - For Fetch missions parsed by the maintained `fetch_request` grammar, accept native pickup action 3 only if publicly visible front object type **and color** match the requested object; reject all drop 4 and toggle 5 overrides (the verified Fetch goal is to carry the requested object).
   - For DoorKey missions in the maintained public `use the key to open the door ...` family, accept pickup 3 only when the visible front object is a key, accept toggle 5 only if the public effect memory predicts it effective, and reject drop 4 overrides. The existing source actor remains responsible for navigation; the gate does not know native state or prescribe a target end-to-end action sequence.
   - Reject unknown/ambiguous mission strings rather than inferring a hidden task ID. Never include UnlockPickup-specific rules, instructions or layouts.

Maintain source/environment RNG contract, actual action hashes, override acceptance/rejection, per-task mission-rule rejections, mode disagreement, 7-action histograms and native-effect counts. Distinguish environmental feasibility from mission relevance. Don't pretend actor-only and the three intervened arms have matched native actions after their trajectories diverge.

## Fresh layouts, evaluation and registered decision

Register only these untouched native source seed sets:
- DoorKey TRAIN-pilot **681001–681064**, MONITOR **682001–682064**;
- Fetch TRAIN-pilot **691001–691064**, MONITOR **692001–692064**.

Run four arms on all 64 TRAIN-pilot layouts per task for implementation/coverage feasibility; **no pilot-driven gate edits**. Then freeze the code and run the four arms on the 64 independent MONITOR layouts per task = **128 paired source task/layouts per mode** (512 MONITOR episodes in total). Do not re-open M23 older monitor/final layouts, M20/M21 finals or original target FINAL453xxx. Fixed proposal RNG `23_800_000+seed` and source actor RNG `8_800_000+seed`, as Stage 8. Any subsequent changed rule is a new prospective study and requires fresh MONITOR layouts.

**Primary endpoint:** strict independently audited full *physical* source task completion, not reward. DoorKey requires real key pickup, door opening and goal reaching; Fetch requires nonvacuous correct requested object pickup after movement with no initially adjacent target, as in maintained auditors. Record reward-positive but physically disqualified episodes separately.

**Necessary nonvacuous support:** at least **32 MONITOR override decisions** on which mission_skill_gate and history_rule_gate disagree in their public eligibility; otherwise mark INSUFFICIENT. **Numerical physical-goal gate:** relative to history_rule_gate, mission_skill_gate earns at least **12 additional full physical goals across 128 paired task/layouts**, strictly positive paired improvement in **each task**, and not fewer physical full goals than `actor_only` in total. The existing `front_rule_gate` is an additional specified reference, not a substitutable primary control. Report paired per-layout outcomes, stratified-by-task paired bootstrap 95% interval (20,000 resamples, seed 230917), and the number of task-discordant outcomes. A numerical pass with interval containing zero may be reported as passing the registered numerical gate but **not reliable improvement**. Report FAIL if sufficient disagreement but any numerical prerequisite is unmet, even if some rewarding episodes are observed. Do not retune using MONITOR.

The threshold is a new prospective criterion for a **different** mission-relevance intervention; it does not relax, retroactively change or replace Stage 8's failed +8/64 and per-task requirement. Physical-goal noninferiority to actor-only is deliberately conservative; a fail is useful evidence.

## Execution, provenance and limits

Implementation may add a Stage 9 experiment and tests, plus an explicit `m23skill` option to the **existing serialized** Research Loop dispatcher/workflow and one unique exact-parent one-shot request. Keep both existing cron schedules and concurrency groups byte-for-byte unchanged; never create a second research workflow. Require `ruff check .`, `pytest -q`, independent current push and PR CI with source/base/merge provenance before any research request. One-file request metadata must bind the tested exact parent. No duplicate or active workers; recheck HEAD and CI immediately before publication.

Save source/config, package versions, new source action and audit traces, per-episode JSON, actual action counts, elapsed wall time, artifact/checksum and any real optimizer/fitting counts to ignored `runs/m23_mission_skill_selection/`. Planned new fits/updates are **zero**; actual counts must be reported after execution. If no real run occurs, report **0 new native actions, 0 fits, 0 optimizer updates and no physical result**. Preserve prior checkpoints, reports and append-only scientific log. Main PR #1 remains review-HOLD until an independent submitted review and fully verified PR integration path; do not merge.

**Next decision:** if the mission-gated source policy remains worse than its actor-only control or cannot improve physical goals robustly, prioritize matched source-trained policy learning/skill selection using maintained BabyAI/Farama/RL reference training machinery as a separately preregistered milestone. A future original M23 source-only generic composition planner and sealed target zero-shot study require independent source mechanism/planning gates and cannot use target-specific hand-coded scripts.
