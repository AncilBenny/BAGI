# BAGI Research Protocol

The project is judged by **learning efficiency and transfer**, not only final task reward.

## Every experiment records

- code commit SHA
- configuration file
- random seed
- number of real environment interactions
- training compute/runtime when relevant
- evaluation metric(s)
- result artifacts and plots
- a short failure analysis

## Core metrics

1. **Steps-to-learn** — real environment transitions required to reach a competence threshold.
2. **Transfer ratio** — learning speed on a related task after prior training versus a fresh agent.
3. **Generalization** — performance on unseen layouts and object combinations.
4. **Planning gain** — performance difference with versus without learned-model rollouts.
5. **Forgetting** — retained performance after learning subsequent tasks.
6. **Prediction quality** — world-model error and, later, uncertainty calibration.

## Experiment discipline

- Change one important hypothesis at a time.
- Always retain a simple baseline.
- Report environment interactions separately from wall-clock speed.
- Prefer small reproducible runs before scaling compute.
- Save generated outputs under `runs/`, `results/`, or `checkpoints/`; these are intentionally ignored by Git.
- Promote only configs, code, summaries, and small plots/tables worth preserving into version control.

## Initial sequence

### E0 — Environment sanity

Verify deterministic causal mechanics and partial observations.

### E1 — Random interaction dataset

Collect transitions using the configured random policy. The dataset validates logging and becomes a first input for dynamics-model experiments.

### E2 — Model-free baseline

Train a small baseline agent and establish steps-to-learn.

### E3 — World-model prediction

Train an encoder/dynamics model to predict next observations or latent states and salient events.

### E4 — Memory

Compare feed-forward and recurrent agents under partial observability.

### E5 — Imagination

Use learned dynamics for short-horizon planning and compare real-environment sample efficiency against E2.

## Theory-guided architecture decisions (adopted 2026-10-21)

**Research principle:** use established scientific results and maintained, appropriately
licensed implementations where possible. BAGI should test why a known principle
does or does not transfer to this laboratory, not invent a new architectural
variant every automated cycle. More experiments are useful only when they
distinguish competing explanations. Never imply a paper proves BAGI will work.

### Known properties and unverified claims

- **Directly verified from source:** Workshop is a deterministic, 9x9 (default)
  world observed through a local 5x5 window. Hidden key/door flags can persist
  when the corresponding objects are outside that window. Current world-model
  predictions are functions of only the current observation and action; the
  MPC planner has no persistent observation-action memory.
- **Directly verified from source:** guided collection uses a fixed action
  sequence and the environment reset uses the same initial position and
  layout regardless of seed. Fifty identical guided episodes cannot be
  treated as fifty distinct trajectories. Transition-level random
  train/validation splitting can put adjacent or duplicate trajectories
  in both sets; a high validation cell-accuracy figure is therefore not
  evidence of generalization to unfamiliar states or layouts.
- **Directly verified from source:** the oracle search ceiling gets exact
  positions and hidden flags. Its success versus the learned-observation
  agent is **not an information-matched comparison**, and does not isolate
  model error from missing state information.
- **Measured in E10:** uncertainty penalty 1.0 improved switch visits under
  held-out planner seeds, but did not achieve goal success; the door/key
  differences were inconclusive and lock performance did not improve.
  Do not promote confidence penalties to a validated general solution.
- **Working hypothesis, not yet established:** state aliasing, insufficient
  off-policy coverage, missing persistent memory, poor action-conditioned
  predictive generalization, and long open-loop rollout error contribute
  jointly to the planner's failure.

### Prioritized research ladder and gates

**Gate A — trustworthy data and evaluation (next priority).**
Measure exact duplicate guided trajectories, unique observation/action pairs,
coverage of positive key/lock/goal transitions, and whether *identical*
observation/action pairs produce different outcomes under different hidden
states. Use logged privileged info **only for diagnosis**, never as a
learned agent's policy input. Add episode-disjoint evaluation but recognize
that identically repeated guided trajectories are still data leakage even
across episodes. For genuine transfer, introduce an independently held-out
layout family and seed-controlled variation *before* claiming transfer.
Report repeated-layout performance as repeated-layout performance.

**Gate B — information-matched ability ceiling.**
Retain the full-state oracle solely as a ceiling. Establish an
observation-history-only reference controller or finite-memory belief-state
baseline (no privileged `env._pos`, `has_key`, `door_open`, or diagnostic
`info` given to the controller), and compare it against a reactive
observation-only reference. Determine whether a short explicit memory can
resolve causal stage ambiguity. Require causal-stage and end-goal metrics.

**Gate C — reuse a proven memory method.**
Begin with the built-in `torch.nn.GRU` or an explicit compact belief/memory
state rather than building a large model from scratch. Train on correctly
segmented observation-action episodes, carry state across decisions, and
reset at episode boundaries. Compare with the current feed-forward model
at matched training interactions and compute. Only promote memory if it
improves event-positive accuracy and/or downstream planning on held-out
episodes, not merely average next-tile accuracy. This is a direct
POMDP/recurrent-world-model research hypothesis.

**Gate D — hierarchical causal planning.**
Once the representation reliably tracks whether the switch/door/key/lock
stage was reached, compare a reusable subgoal planner (switch -> door ->
key -> lock -> goal) to current undirected random shooting. The hierarchy
may be used as an evaluation oracle or a *learned* stage predictor, but
must not be presented as an autonomously acquired causal model if the
sequence was manually supplied. Use short replanning horizons and compare
under identical partial observations, matched budgets, and held-out layouts.

**Gate E — stronger model-based RL only after prerequisites hold.**
Borrow established ideas from Dyna, recurrent state-space world models
(Dreamer family), data aggregation (DAgger), and latent model predictive
control (TD-MPC family). Prefer existing maintained open-source methods
over an unmotivated whole-framework rewrite. For BAGI's small CPU-only,
discrete-action Workshop, a compact memory baseline is much cheaper and
more informative than porting full DreamerV3/TD-MPC2 immediately.

### Decision rules for ChatGPT and Copilot

1. Read this protocol and the latest result artifacts before selecting work.
   Explicitly name the established principle or prior verified result,
   the BAGI-specific failure it addresses, and a falsifiable alternative.
2. Prefer answering **one decisive causal/architectural question** over
   sweeping planner weights, candidate counts, or novel loss terms.
3. Each change must include a control, a fixed real-interaction budget,
   evaluation seed policy, held-out condition when possible, CI tests,
   actual measured artifact and a negative-result interpretation.
4. Do not draw generalization conclusions from repeated seeds in a fixed
   deterministic layout. Never use diagnostic hidden-state information
   to make decisions in the partially observable learned-agent baseline.
5. If an earlier gate fails, address it before later gates. If evidence is
   inconclusive, record uncertainty and reuse the checkpoint/dataset
   for targeted analysis instead of consuming more compute blindly.
6. Protect stable code and the automation. Agent changes stay on
   `setup/initial-platform`; no automatic merging to `main`, new
   credentials, spending, or unsafe dependency upgrades.
7. The next small experiment should inspect data duplication, hidden-state
   aliasing, and true episode-disjoint/trajectory-disjoint predictive
   evaluation before more uncertainty-weight tuning.

### Established references (principles, not BAGI performance claims)

- Kaelbling, Littman, Cassandra (1998), *Planning and Acting in Partially
  Observable Stochastic Domains*: https://doi.org/10.1016/S0004-3702(98)00023-X
- Sutton (1991), *Dyna, an Integrated Architecture for Learning, Planning,
  and Reacting*: https://doi.org/10.1145/122344.122377
- Ross, Gordon, Bagnell (2011), *A Reduction of Imitation Learning and
  Structured Prediction to No-Regret Online Learning*:
  https://proceedings.mlr.press/v15/ross11a.html
- Hafner et al., *DreamerV3: Mastering Diverse Domains through World Models*:
  https://arxiv.org/abs/2301.04104
  Reference implementation: https://github.com/danijar/dreamerv3
- Hansen, Su, Wang (2024), *TD-MPC2*: https://arxiv.org/abs/2310.16828
