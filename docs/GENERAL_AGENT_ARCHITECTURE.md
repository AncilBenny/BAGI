# BAGI General Brain — task-independent architecture research (G0)

**Objective.** BAGI is an experimental, sample-efficient *learning agent* that should ultimately operate across different environments and goals. BLUE-square pickup/drop is one falsifiable XLand-MiniGrid measurement of physical manipulation, **not** the subject, name, built-in phase sequence, or goal of the general algorithm. Other target task families include causal Workshop puzzles, software workflows/tools, navigation and eventual robotics (with separately vetted adapters). This document is a research design and a small executable first step, not a claim of AGI.

## Evidence and design decision

BAGI's REUSE-33/36/37 XLand results already show occasional independent real BLUE pickup and drop under both original training seeds, but success remains low and unstable across layouts. In REUSE-37 the expanded policies obtained only **six episodes with an effective BLUE pickup in 36 eligible putdown episodes**, of which **five completed a physically verified placement**. The original/expanded controls had the same **5/36** physical placements. Thus merely adding more of the same complete reset demonstrations did not produce reliable acquisition or transfer on independent layouts. XLand game reward can also be triggered without real BLUE movement. These facts motivate *correct effect verification, state memory, planning/replanning, skill reuse and outcome learning*, not a literal BLUE-specific controller. The evidence cannot isolate navigation from all other causes without intervention.

BAGI already implements a world model (`bagi/models/world_model.py`), predictive planner (`bagi/planning/model_predictive.py`), model-free DQN, Workshop world and upstream XLand/BC and recurrent-PPO experiments. **Reuse them through adapters**. Do not discard them in favor of an ungrounded "new AGI network".

## Research foundations

1. **Partially observed control / belief:** Kaelbling, Littman & Cassandra (1998), *Planning and Acting in Partially Observable Stochastic Domains*, https://doi.org/10.1016/S0004-3702(98)00023-X. Maintain useful state beyond the current view and distinguish belief from directly known facts.
2. **Options / temporal abstraction:** Sutton, Precup & Singh (1999), *Between MDPs and semi-MDPs*, https://doi.org/10.1016/S0004-3702(99)00052-1. Treat skills as closed-loop policies with initiation conditions, effects and termination; high-level control operates over options rather than hand-writing primitive action sequences.
3. **Learned dynamics + imagination:** Hafner et al., DreamerV3, *Mastering diverse control tasks through world models* (Nature, 2025), https://doi.org/10.1038/s41586-025-08744-2. Predict effects/rewards from real experience and later plan via the model. **Do not** install/port the large GPU-centric system merely to pass a tiny CPU benchmark.
4. **Generalizable skill transfer:** Barreto et al. (2018), *Transfer in Deep RL Using Successor Features and Generalised Policy Improvement*, https://proceedings.mlr.press/v80/barreto18a.html. Share learned value/skill knowledge across goals; recognize the assumptions under which transfer works.
5. **Accumulating verified skills:** Wang et al. (2024), *Voyager*, https://arxiv.org/abs/2305.16291. Skill libraries, iterative verification and automated curricula are compositional ideas; Voyager's LLM/code-heavy implementation is **not** BAGI's current performance or a required external dependency.
6. **Correcting the states the learner actually visits:** Ross, Gordon & Bagnell (2011), *DAgger*, https://proceedings.mlr.press/v15/ross11a.html. Future correction labels must be from real learner-visited TRAIN states and correct observed action histories; never from heldout compositions.

## Architecture: reusable core vs replaceable adapters

```text
            Natural-language / structured / sensor-provided GOAL
                              |
                      Goal representation
                              |
          Public observation --> Evidence encoder --> Belief / memory
                              |                      ^
                  Goal-conditioned skill planner      |
                              |                      |
                     Reusable option library          |
                              |                      |
        Existing BC / RL / model-predictive SKILL POLICY
                              |
                 Domain actuator / tool adapter
                              |
            REAL EXECUTED action and PUBLIC observation
                              |
               Effect / termination verification -----+
                              |
                  Skill outcome learner and replay
```

**In the core** (`bagi/agents/general_brain.py`):

- `Goal`: an ID plus observable success conditions, *not* a named manipulation task.
- `Skill`: an ID, preconditions, *hypothesized* effects, bounded closed-loop action policy and estimated cost; action may be a simulator integer or symbolic application command.
- `PublicSensor`: only policy-legal observations + the **actually executed** action and public reward. There is no `info`, pocket, hidden grid, oracle label or simulator internals argument on the interface.
- `GeneralBrain`: persistent confirmed facts, uniform-cost skill composition, bounded retry/replanning, option effects verified after observation, real-action provenance, maximum interaction steps.
- `SkillExperience`: online Beta-prior outcome counts updated only after observable skill success/failure; retained across goals for a **semantically compatible** reusable skill. Selection costs account for learned success probability.

**Outside the core:** domain-specific observation encoding, skill-policy weights, motor/tool action grounding, useful fact extraction, a future learned world model and automatically discovered skill preconditions/terminations. Skill descriptions in G0 are supplied by the adapter, **not autonomously discovered**. Goal interpretation from arbitrary language is likewise not learned yet.

### Key safety and scientific invariants

- **Attempt != outcome:** issuing an action or receiving a game reward alone never establishes a required fact. Progress needs the sensor's positive, policy-legal evidence. Imagined effects guide planning but are never inserted into real belief.
- **Observable-only:** pass only public observations, issued/actual actions and public reward to the core. Privileged state is permitted exclusively inside *external evaluators* that never feed the acting agent.
- **Interrupt and recover:** a skill that does not produce evidence within its capped actions records a failed attempt. The planner can choose an alternative. Bounded planning/episode/retry counters prevent loops.
- **Learning rather than scripting:** skill *policies* can be supplied by existing learned BC/PPO/DQN models. G0's skill priors are outcome-learning, not a pretrained multimodal policy. Do not describe hand-authored symbolic preconditions as learned understanding.
- **Reusable goal semantics:** no string/constant in the brain identifies BLUE, square, RED, Workshop switch, a web app, etc. Task-specific concepts appear only in adapters/tests.
- **Separated transfer proof:** sharing the code across environments is not evidence of learned cross-domain transfer. Generalization requires heldout goals/environments and measured improvements under matched experience budgets.

## G0 implemented and measured now

`python -m bagi.experiments.general_brain_smoke --output runs/general_brain_smoke.json`

The dependency-free synthetic smoke test uses the **same** core on an observable supply-shipment workflow, a distinct software-release workflow with string actions, and two consecutive goals requiring verification/recovery from a failed option. The learned option-success posterior changes the choice for the second goal. Contract tests check no inferred success after a failed action (even with a positive reward), actual executed-action provenance, stale-fact retraction, budgets and bounded replanning. These are mock environments: **zero native Workshop/XLand simulator interactions and zero demonstrated real-world intelligence**. Record all test results honestly.

## Next preregistered gates (no uncontrolled scale-up)

**G1 — reusable control under real heterogeneous dynamics.** Add (a) a public-only Workshop adapter and (b) a public-only XLand goal/skill adapter using existing maintained BC/PPO policies, without hardcoding BLUE or a pickup/drop phase in the core. Freeze TRAIN and wholly new test layouts for each domain and the goal/task instructions. Use equal native step budgets/seed splits to compare (1) underlying reactive policy, (2) same policy + option planner with effects **assumed from attempted actions** (unsafe diagnostic control), and (3) same policy + observation-verified effects/replanning. Primary: objective completion on untouched layouts and physical BLUE pickup→drop→adjacency+native reward where relevant. Secondary: correct progress detection, goal transfer, skill success calibration, forgetting, real steps and failures. Do not promote component tests or vacuous rewards as achievements.

**G2 — learn rather than hand-author the state abstraction.** Train an observation-only transition/effect head on episode-disjoint, task-diverse experience. Explicitly report false-positive effects (e.g. ineffective pickup), false-negative effects, reliability calibration and action costs. Compare same-parameter/stateful vs stateless belief encoders and retain old-task rehearsal; do not read privileged simulator pocket to the learned policy.

**G3 — discover and reuse goal-conditioned skills.** Replace externally listed skills with learned option initiation/termination and a goal-conditioned high-level policy, using established hierarchical RL and off-policy experience replay before trying novel losses. Cross-task and cross-domain transfer must beat a matched baseline on **unseen goal combinations** at fixed real steps. Only after real G1/G2 signals should world-model imagination, successor features and language-grounded tool adapters be increased in scope.

**Decision gate:** one credible improvement on one benchmark is not generality; require at least two unrelated task families, independent test layouts/seeds, predeclared minimum gains, outcome verification, no hidden input, and regression tests of original skills. If a gate fails, preserve its evidence and fix that measured cause before adding complexity.