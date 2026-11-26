# G1: general outcome-checking brain across two native environments

**Preregistered before the first native Actions execution** (2026-10-24).

## Research question

Can the same domain-agnostic skill planner, public evidence sensor, and
cross-goal outcome-learning kernel operate safely across unrelated action
spaces and real environment dynamics without hidden simulator input? Does
replanning after a *publicly unverified* skill outcome improve task completion
at a bounded interaction cap, compared to assuming the action worked?

This is **interface and recovery validation**, not a claim of autonomous
skill discovery, a learned world model, general intelligence, or transfer of
trained policy weights between Workshop and XLand.

## Original shared algorithm

`bagi/agents/general_brain.py` supplies the same `Goal`, `Skill`, `GeneralBrain`
and bounded option planning in both environments. `PublicEvidenceSensor`
accepts **only** the public before/after observation, actually executed
action and public reward. Facts are never set by proposing an action. Facts
remain persistent when evidence is absent/offscreen but are retracted if
visibly contradicted. Public reward is named `native_reward_observed`, never
`physical_skill_succeeded`; externally inspected simulator state verifies
physical performance independently.

## Workshop (one narrow real-environment integration test)

- Native `WorkshopEnv` 9x9, 5x5 public view, cap40.
- **Only two unique deterministic geometries**, `default` and `mirror_x`,
  fixed start; do not label repeated seeds as layout generalization.
- Scripts are **handwritten domain adapters**, selected left/right from a
  publicly visible switch, **not learned policies**. The general core knows
  neither `SWITCH` nor keys, goals or tile numbers.
- Three conditions: (a) direct competent option script (16 actions),
  (b) explicitly unsafe *assume-effects* control (issue two known-blocked
  actions then execute the later sequence), (c) general verified/replanning
  brain (detect visible opened door, retry with alternative, follow remaining
  script). All capped at40 actual native actions.
- Main measure genuine native goal reward; external evaluator separately
  reports hidden door/key state **after** the run. The controller never gets
  `info`, hidden position, native flags or privileged state.
- This controlled failure fixture tests correct evidence and recovery wiring.
  It cannot prove learned navigation or autonomous generalization.

## XLand (archived published learned options, no fresh teacher)

- Frozen 115-wide public partial-grid+heading+goal/rule+four *actually
  executed* actions. No full grid or hidden pocket enters the learned policy.
- Reconstruct upstream `imitation.BC` baseline and expanded-demo policies on
  original model seeds17 and29 from SHA256-verified REUSE-19/33/36 TRAIN
  numeric artifacts; require exactly earlier published 787 sampled rows,
  12 epochs, 9444 nominal row exposures per BC arm, unchanged seed-specific
  action-label histograms. Zero new teacher queries or live training actions.
- Preregister **five NEW layouts 11300..11304** for each of three independent
  native skill tasks hold/near/putdown. Two seeds x two conditions x three
  tasks x five layouts = **60 episodes maximum**, cap40 each, at most 2400
  actual native XLand evaluator actions. No substituted seeds.
- (a) Direct expanded-demo BC for at most40 actions. (b) same expanded BC
  for at most20 actions, then original frozen BC for up to20 more if no
  *public reward evidence*. Same observation/input interface and total cap,
  with the same `GeneralBrain` retaining observed success/failure posterior
  across episodes (transient goal/belief reset per episode).
- Outcome metrics: native game reward on each task, physical putdown success
  only if **initially nonadjacent** BLUE/RED and actual BLUE pickup, later
  effective BLUE drop, final adjacency **and** native reward; separately
  pickup-episode count, drop count, action counts, option trajectories,
  posteriors, hold/near regression. Some native putdown rewards are vacuous;
  they **never** count as physical manipulation.
- The two XLand conditions differ in *policy schedule*; any difference is
  evidence about this particular controller integration, not isolated proof
  that its belief state or memory is optimal. Task order and tiny sample size
  limit causal inferences. There is no matching action-assumption condition
  for XLand because a failed native reward is already publicly measurable.

## Next gate

If the exact native experiment succeeds technically, audit each recorded
episode. Where the controller fails, measure missing observable success
signals and option recovery before proposing a learned state/effect encoder
(G2). Judge cross-domain architecture on lawful inputs and honest real step
counts; judge *learned* transfer only when there are actually learned
option/perception parameters transferable between domains. A zero-score G1
is a negative task result, not proof a generic architecture is impossible.
