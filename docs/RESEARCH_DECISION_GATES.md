# BAGI evidence-driven research decision gates (after REUSE-24)

**Status:** 2026-10-21, XLand-MiniGrid-R1-9x9. This is a research
handover and preregistration roadmap, not evidence of transfer.
**Protected branch:** \`setup/initial-platform\` only; do not change \`main\`,
secrets, deployments, or workflows except bounded research tests. Recheck
head and concurrent experiments before pushing. Keep the EXISTING hourly
autonomous research automation; never add a duplicate.

## What is learned already?

| Study | Verified experimental finding | Decision consequence |
|---|---|---|
| REUSE-14, run 39603838885 | Privileged corrective expert recovered 24/24 scripted off-path TRAIN situations, but not learner-real wrong-held-object situations | The teacher is a conditional local labeling tool, not a general arbitrary-state solver |
| REUSE-15, run 39605165443 | Two rounds of existing upstream DAgger worked mechanically (291 actions/teacher labels per seed), but ordinary BC had zero independent source rewards | Reuse DAgger; do not assume aggregating examples alone makes closed-loop skill |
| REUSE-16/17, runs 39619756922 / 39621044776 | Class balancing rescued original-pool rare-action recall and generated some independently rewarded hold/near, with strong policy-seed variability | Label class imbalance mattered, but 100% original-label putdown recall was never sufficient |
| REUSE-18, run 39623603317 | Many putdown ACTION attempts without an effective pocket drop in previous tested layouts | Stop conflating action selection and successful actuator preconditions |
| REUSE-19, run 39637067506 | 83.1%/78.8% of teacher labels differ from ACTUALLY executed actions during DAgger beta0; four-action histories mostly unique | Only history of the agent's own executed commands can be a legal policy input |
| REUSE-20/21, runs 39638358164 / 39639959628 | XLand PUBLIC goal/rule wrapper missing from initial 54-input BC; task instructions improved some primitives but not robust putdown | All legitimate learners should receive a deliberate, consistent task-instruction observation contract |
| REUSE-22, run 39643131034 | Carried pocket can affect putdown despite identical standard image+heading+public task tokens; counterfactual empty-pocket states not claimed reachable | Instantaneous policy input omits a causal state; test valid history/stateful baselines |
| REUSE-23/26, runs 39644180240 / 39650414747 | First putdown benchmark reward was later proven VACUOUS on layout9604: objects started adjacent, 0 real pickups/drops/grid changes | RETRACT earlier learned-placement-skill claim; always screen preexisting adjacency and audit real manipulation |
| **REUSE-24, run 39646704229** | Ten untouched layouts per model/condition: 0/40 rewarded putdown, no correct BLUE square held on any putdown rollout; one genuine effective RED-ball pickup/drop under seed29 true history/layout9708 | Failure is upstream of rewarded placement: correct-object acquisition, recovery, stage navigation, and maybe memory |
| **REUSE-25, run 39649325255** | Real wrong-RED-ball learner state rejected by existing teacher's missing-grid-RED lookup; one legal native drop then 19 train-only correction actions recovered the putdown goal in 24 total steps | One REAL learner mistake is salvageable with existing BFS, but teacher coverage has a verified state gap; learned policy did not choose the rescue |

No heldout causal-chain completion is established. A positive supervised teacher-label
action recall is not proof of closed-loop causal understanding. Do not treat
learned hold/near rewards as compositional transfer.

## REUSE-24 measurement detail / reproducibility

- Frozen source \`REUSE-19\` numeric artifact: GitHub Actions \`39637067506\`,
  SHA256 verified per model seed; original policy artifact \`39644180240\`;
  current dedicated GitHub Actions \`39646704229\`, artifact
  \`bagi-frozen-skill-phase-replication\` ID \`12368116321\`.
- Two unchanged \`imitation.BC\` policy initialization seeds [17,29], each
  with 291 original train-only labels, 12 balanced-sampling epochs, 3492
  sample exposures; original 115 policy inputs (54 image+direction, 33
  PUBLIC task instructions, 28 true/sham history). No new training action
  or teacher labels. Full reference reward-success layout sets [9600..9604]
  reproduced exactly before evaluating new seeds [9700..9709].
- Fresh HOLD successes: sham seed17 1/10 (9705); real history seed17 1/10
  (9705); sham seed29 2/10 (9700,9706); real history seed29 0/10.
  Aggregate sham 3/20, real history 1/20.
- Fresh NEAR successes: sham seed17 1/10 (9705); real history seed17 2/10
  (9705,9706); sham/real seed29 both 0/10. Aggregate sham 1/20, real 2/20.
- Fresh PUTDOWN successes: 0/10 in each of the four combinations (**0/40**).
  There were 502 native pickup action attempts and 687 native putdown
  action attempts across the 40 putdown episodes, yet exactly ONE effective
  pickup and ONE effective drop (both seed29/real history/layout9708).
  That pickup held a RED ball, never the required BLUE square. No learner
  carried the required BLUE square in any putdown episode. The successful
  RED drop occurred 12 actions after pickup and never rewarded placement.
  Note: an initially near-adjacent square/ball pair is NOT necessarily
  a rewarded TileNearGoal terminal; stage adjacency must be interpreted
  alongside the upstream native reward, never substituted for it.
- Simulator interactions: 6437 actual reference+fresh evaluation steps,
  including previously tested references only to confirm exact model replay;
  do NOT count reference repetitions as new independent test layouts.
  Both dedicated experiment and code-commit CI checks passed.

## Critical benchmark-validity correction (REUSE-26)

**REUSE-26, run 39650414747** replayed the exact 13 upstream
native actions of the previous seed29/layout9604 "learned rewarded putdown".
The blue square began at [3,5] and the red ball at [2,5]: **already
adjacent** (Manhattan distance1). Throughout all 13 steps, XLand
showed **zero effective pickups, zero effective drops, zero grid modifications**
and the pocket stayed empty. Native action4 at the end nonetheless yielded
a real XLand reward **0.7075000405**. The current upstream TileNearGoal
predicate checks action4 and target-tile adjacency but does not itself
require a successful pickup or drop. Therefore the earlier REUSE-23
"first learner-controlled putdown" is a REAL GAME REWARD but
**NOT an acquired physical pickup-and-placement skill**.

This **explicitly supersedes** every earlier optimistic
"learned putdown skill" phrasing in research logs, inventory, and user
updates. It does not invalidate the observed hold/near skill rewards, the
REUSE-24 zero real putdown successes on fresh layouts, or the REUSE-25
manual-intervention full expert recovery.

**Mandatory for future physical-placement skill evaluation:**
(1) verify BLUE square and RED ball are *not* adjacent at reset;
(2) record that the learner **actually** picked up the BLUE square;
(3) record that it **actually** dropped that BLUE square;
(4) verify native goal reward on that episode.
Only (1)+(2)+(3)+(4) establishes a nonvacuous rewarded putdown
under our research definition. Preserve the native reward separately
as a benchmark score. This is a task-validity exclusion, not
posthoc cherry-picking: preregister and report excluded layouts before
future learner evaluations, and never silently redefine old denominators.


## What to do when learning reaches a bottleneck

**Do not respond to zero reward by indefinitely raising BC epochs, collecting
more teacher-reset demonstrations, designing a new neural network, or querying
the heldout composition oracle.** Identify and test the **first failed causal
milestone**, then choose the smallest existing-method intervention:

### Gate A — can the existing teacher correct learner-induced failures?

**Next task (REUSE-25 proposal):** under the untouched REUSE-24 frozen seed29
true-history model on putdown layout9708, replay the learner until the FIRST
native effective pickup (step4), record learner-visible observation, public
rule/goal, full state for evaluation ONLY and the actual held RED ball.
Query existing train-only \`correction_plan\` on this REAL learner-reachable
state. It currently has:

\`\`\`python
if not empty:
    raise ValueError("Unsupported object held in putdown task")
\`\`\`

The known case must be faithfully reproduced first. It is **not a proof**
the environment is unsolvable or that memory is ineffective: it is a
limitation of the existing teacher's corrective-domain coverage. Check
whether standard upstream XLand action4 can release the wrong held tile
into a valid free cell and whether reused BFS+existing teacher can finish
the same TRAIN putdown goal within remaining 40-step budget. Compare to an
expert-only reset solution on the same layout. If recovery impossible under
the remaining cap, record why; do not synthesize invalid corrective labels.
A successful recovery may justify a narrow teacher fallback and guarded tests
for wrong-held-object state, not a new planning architecture.

Resource gate: one predeclared learner policy+layout and existing native XLand
step calls; count real actions and teacher queries separately. Extend to at most
three **train-only** real learner-failure states if first succeeds. No
heldout-chain teacher labels or full-state learner features.

### Gate B — does targeted on-policy correction close the DATA gap?

Only after Gate A shows valid corrective labels for REAL learner states,
re-use pinned upstream \`imitation.SimpleDAggerTrainer\` and \`BC\`.
Preregister a small capped train-putdown corrective-state intervention,
matched against the same number of reset-state expert labels (do NOT
tune after seeing results). Keep public task instructions available in
BOTH policies, preserve train-only split and native 40-step cap, compare
actual effective BLUE-square pickups, effective square putdowns, rewarded
putdown completions and hold/near retention on untouched seeds. Use
identical teacher-query budget, model seeds, network and optimization
exposures, and explicitly report any unsupported expert queries instead
of silently filtering them out. A class-balanced label pool does NOT
guarantee task-phase or state coverage.

### Gate C — can existing *stateful* agents solve hidden inventory?

If teacher is valid but matched-feedforward BC still fails temporal
sequencing, test an already maintained recurrent baseline; **prefer
upstream XLand's published recurrent PPO training code** or SB3-Contrib
\`RecurrentPPO\` rather than implementing a new LSTM. Compare a matched
feedforward PPO control with identical legal PUBLIC image+heading+
goal/rules observation, task curriculum, native action space, source
evaluation seeds and explicitly capped real environment interactions.
Use correct episode-start resets for hidden states. Start with a
small environment-contract smoke test and escalate training only after
proving learning signal and tracking exact simulator steps. Avoid claiming
a zero-reward small PPO run refutes recurrent memory; sparse rewards and
insufficient training budget are alternative explanations.

Upstream references:
- https://github.com/dunnolab/xland-minigrid (official recurrent PPO baselines)
- https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html
- https://imitation.readthedocs.io/en/stable/algorithms/dagger.html

**Gate A limited feasibility result (REUSE-25 run 39649325255):** verified.
Original learner seed29/layout9708 picked wrong RED ball at step4 and
dropped it at step16 without reward. `correction_plan` failed because
it attempted to find RED on grid before checking that RED was held.
A separate privileged *diagnostic* native action4 released the RED ball;
the original unchanged BFS teacher then corrected it in 19 native actions
and rewarded the goal at step24 total. One rejected teacher query and
19 successful teacher calls; 97 simulator calls including controls.
This supports a **bounded teacher coverage repair** and controlled
on-policy correction study, not learned learner recovery or general teacher
validity. Check future wrong-object reachability separately.

### Gate D — only then test genuine composition

Do **not** promote heldout-chain success as a primary KPI until the same
learned agent independently passes all three training primitives, including
a reliable rewarded putdown, on held-out layouts under a preregistered
threshold (suggested feasibility gate: >=7/10 successes on EACH source
skill for EACH of two model seeds, repeated on disjoint confirmation
seeds). This numerical gate is a research choice, not a proven benchmark
standard. When passed, evaluate a behaviorally distinct, source-disjoint
compositional task, with no heldout expert trajectory/teacher actions in
training. Record ruleset hashes AND true behavior difference; prior
REUSE-5 found that different rule hashes can still share identical
solutions, which invalidates a purported transfer comparison.

## Operating discipline

- Failure results are **information** when they discriminate hypotheses:
  which object got picked up, whether it was held, whether putdown
  changed inventory, whether reward fired, whether the expert can recover.
  Keep one preregistered comparison at a time.
- Always cite exact source commit + successful GitHub run/artifact +
  actual real simulator steps; label oracle-only rewards separately from
  independent learned policy rewards.
- Abort concurrent branch write rather than force update; stay on
  \`setup/initial-platform\` only. No secrets/pickle/unreviewed taskset
  downloads, deployed services, or giant autonomous compute jobs.
