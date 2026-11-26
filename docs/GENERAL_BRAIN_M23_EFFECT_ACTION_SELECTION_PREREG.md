# M23 stage 8 preregistration — public-effect memory in actual source-task decisions

**Prospective 2026-11-15.** This is **not** the original M23 zero-shot
UnlockPickup composition FINAL evaluation. Existing M22, M20/M21 and all
M23 scientific results/gates remain independent and unchanged.

## Why this is the next useful experiment

Stage 6 learned native source local pickup/drop/unlock effect prediction
after **41,052 actual executed interventions** and 16 PyTorch fits
(98,560 optimizer calls); preregistered source effect gate PASS,
though no target-task success was measured.
Stage 7 evaluated those frozen numeric models with **9,591 new native
source actions** under changed public-action policy and recorded
**811 effective / 4,152 ineffective publicly eligible attempts**.
Both the frozen learned model and a **simpler public observation/action
history rule** achieved macro-F1 **1.0** on these physical effects;
registered model robustness and the rule-reuse preference both PASS.
Stage 7 performed zero new fits. **Do not keep refitting a local effect
predictor when a transparent public-memory rule already matches it.**

Maintained mechanisms verified: Farama MiniGrid DoorKey and Fetch native
environments and official action semantics; BAGI
`TraversableOpenDoorController`, `PublicFetchTeacher`,
`NativeDoorKeyAudit` and `NativeFetchAudit`, plus stage-7 public
memory and public rule-shift override. These are maintained/reused
components and auditable integration, not a new RL algorithm or
general planning invention. PutNear continues as source-mechanism
coverage; it is not a Stage 8 primary physical goal because the
existing repo has no independently verified full PutNear success
auditor. A goal audit would be required before counting PutNear task
success. Workshop/XLand remain diagnostics.

**True open question:** Does screening actual randomized interaction
actions with **past publicly observed pickup/drop effects**, rather
than just visible front-cell affordances, improve physical DoorKey
and Fetch mission completion in real MiniGrid episodes?

## Frozen candidate action selection and source splits

Use source tasks only: MiniGrid-DoorKey-6x6-v0 and
MiniGrid-Fetch-5x5-N2-v0. Original public-only source teacher is
the fixed controller for each task, with unchanged observation,
public mission, current/previous executed actions and RNG seed.
Reuse Stage 7's public-only 0.65 probability of overriding the source
controller's proposed action by a targeted native 3/4/5
interaction attempt selected from the visible front cell.
The controller never receives private grid/carrying/target state.

Compare exactly **three action strategies**:
1. **unfiltered:** execute every sampled override.
2. **front_rule_gate:** if an overridden action is predicted to have
   no effect by the frozen public current front-cell/action rule,
   reject the override and execute the original public source actor's
   proposed action; otherwise execute the override.
3. **history_rule_gate:** same override and fallback, but accept it
   only if the Stage-7 public observation/action history rule predicts
   an effect; update inferred carried item only after actual executed
   visible before/after effect. This is **not** native carrying state
   or evaluator truth.

All arms use the same source seed and independent action-policy/override
RNG seeds until trajectories diverge. The only difference between
front_rule_gate and history_rule_gate is the permitted *public
observation-history information*. Every arm receives a legitimate
native action on every time step. No imagined actions or action labels
count as native interactions. Report proposed overrides, accepted
and rejected overrides, rules-disagree counts, native action histograms,
executed-action sequence SHA256, native effects and physical task
completion separately from rewards.

REGISTERED source TRAIN feasibility layouts (no tuning):
DoorKey **651001–651064**, Fetch **661001–661064**.
REGISTERED independent source MONITOR layouts (no overlap or retuning):
DoorKey **652001–652032**, Fetch **662001–662032**.
No earlier M23 TRAIN/MONITOR/FINAL, M15–M21 FINAL or the sealed
UnlockPickup target FINAL453001–453032 are accessed.
Evaluate all 3 strategies on all 96 layouts per task, monitor paired
by identical native environment seed; TRAIN is descriptive pilot
only and must not be used to change the registered policy/gate.

## External physical outcome and preregistered gate

For DoorKey use the existing **NativeDoorKeyAudit.full_physical_goal**
which independently requires effective key pickup, door unlocking
and actual green-goal arrival, with correct initial state.
For Fetch use **NativeFetchAudit.full_physical_goal**, which verifies
physical pickup of mission-matched color/type after real movement,
not an initially adjacent target and not reward alone. Both auditors
may use `env.unwrapped` exclusively outside policy decisions.
Track total actual native action counts, stage-specific rewarded
episodes, positive/failed native interaction evidence and steps to
goal. Published reward-only success is not a substitute for
`full_physical_goal` in the primary outcome.

**Primary paired gate, MONITOR only:** history_rule_gate must achieve
at least **8 more full physical goals out of the 64 paired MONITOR
layout/task episodes** than front_rule_gate and at least **one extra
full physical goal in each of DoorKey and Fetch**, without reducing
total full-goal success versus unfiltered. Report per-layout paired
differences, two-task grouped uncertainty and 95% paired-bootstrap
interval; an interval including zero prevents a reliable transfer
claim even if the numerical gate passes.
**Necessary evaluability:** at least **16 actual override decisions
where front rule and history rule disagree**, counted on real
MONITOR public observations; otherwise report INSUFFICIENT and
do not infer value of memory. Also report negative/positive
effect events and reward vs physical-goal discrepancies.
These are decision/evidence thresholds, **not resource quotas**.

A pass would validate only that **public action-effect memory improves
source-task closed-loop decisions under interventions**; it cannot
establish zero-shot composition or continual-learning transfer.
If front_rule_gate and history_rule_gate tie, prefer the simpler
policy and stop claiming that additional memory is necessary in
these source tasks. If all policies fail source physical goals,
diagnose navigation/action-selection without opening target finals.

## Governance, output and next decision

Implement source-only `bagi/experiments/m23_effect_action_selection.py`
with existing native auditors, controlled public action override,
and tests for public/private separation and native task goals.
Reuse the **existing** serialized research workflow/dispatcher with
one registered `m23action` source-parent-bound request, no new
workflow, no schedule edits and no competing worker write.
Run Ruff and pytest, exact current push and PR CI before requesting
execution. Save numeric-only action/effect traces and JSON reports
under ignored `runs/m23_effect_action_selection/`, upload immutable
provenance, source SHA, package versions, true native action counts,
reward/physical outcomes and limitations. Zero model fits and zero
optimizer calls by design. Append executed results without editing
earlier evidence. Main integration remains held until independent
PR review and current CI/merge checks.

If the source physical decision gate proves useful, the next
distinct M23 step is a **preregistered task-independent composition
planner** reusing learned/observed action effects, followed by
TRAIN-only UnlockPickup feasibility and only later the original
sealed FINAL. A hand-coded target solution must never be labeled
learned zero-shot transfer.
