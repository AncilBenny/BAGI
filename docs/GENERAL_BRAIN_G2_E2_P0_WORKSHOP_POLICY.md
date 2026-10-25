# G2-E2-P0 — Prospective learned Workshop policy feasibility gate (2026-10-25)

**Status: preregistered DESIGN ONLY; no native training, evaluation or fitted result.**
This is a prerequisite for a later two-domain learned public-action-effect / belief
study, **not** G2-E1 retuning, cross-domain weight transfer, or a new algorithm.
The earlier G2-E1 predictor failed promotion on fresh data; its TEST observations
and labels must not be used to choose this pilot's policy or hyperparameters.

## Scientific question, alternatives, and reuse-first decision

**BAGI-specific missing capability:** G2-E1 Workshop collection used ten
identical nonlearned G1 scripted trajectories per geometry, so it cannot
demonstrate prediction under an independently learned Workshop agent.
Before spending another XLand/Workshop predictive budget, ask whether a
public-observation-only *learned* Workshop actor can achieve any independently
rewarded, non-scripted causal task competence on an unfamiliar mirrored
geometry with seeded starts and a small, fixed native interaction budget.

**Reuse:** existing repository `bagi.agents.dqn.DQNAgent`, `QNetwork`,
`ReplayBuffer`, `bagi.experiments.train_dqn`, and native
`bagi.worlds.WorkshopEnv` (9x9, 5x5 local view, four native actions);
the established DQN/Q-learning principle, existing PyTorch/Gymnasium and
the project's DQN and recurrent-PPO reuse inventory. Do not port Dreamer,
write a novel RL method, fetch pretrained unsafe pickle checkpoints, add
packages, or use privileged `_pos`, `_has_key`, `_door_open`, or `info`
in actor inputs. Contribution type: **bounded integration/feasibility
evaluation of an existing learner**, not algorithm invention. The existing
repository is the implementation provenance; no new external literature
search or novelty claim is made.

**Alternative explanations if unsuccessful:** sparse terminal-only rewards,
insufficient 2,048-step coverage, partial-observation memory limits, and
default-to-mirror geometry shift; a negative pilot does not refute DQN,
belief-state methods or learnability in general.

## Frozen training and heldout evaluation contract

- **Environment:** unchanged native `WorkshopEnv(size=9, view_size=5,
  max_steps=40, start_mode="seeded_start_chamber")`. Use
  `layout_mode="default"` for TRAIN; evaluate on both `default`
  (unseen start seeds, in-family reference) and `mirror_x` (unseen
  geometry). The code's seeded-start mode varies legal starting positions;
  **repeating the same deterministic geometry is not independent layout
  generalization**. Never replace a failed start seed.
- **Model seeds:** exactly `17,29`, independent fresh DQN models. For each
  model seed train for **exactly 2,048 executed native environment actions**,
  stopping at the budget even if an episode is incomplete. Training reset
  seeds are `14000 + 100*seed_index + episode_index`, where seed_index is
  0 for model17 and 1 for model29. No teacher demonstrations or labels.
  Save per-episode reward, terminal, actual step count and partial-final flag.
- **Frozen existing DQN settings:** QNetwork hidden width128; Adam
  learning rate 0.0003, discount0.99, replay capacity20,000, minibatch64,
  one replay update per executed step once >=64 samples exist, target
  network synchronization every128 executed actions. Epsilon-greedy
  exploration linearly decays **per actual training action** from 1.0
  at step0 to 0.1 at step2,048; evaluation uses greedy epsilon=0.
  Use the same source observation normalization and 5x5-only input as
  the existing `DQNAgent`. No episode-level hand-authored action script.
- **Evaluation:** for each model seed, compare frozen learned greedy DQN
  against uniform random on **exactly ten paired reset seeds
  `15000..15009`**, separately in `default` and `mirror_x`.
  Each actor gets its own fresh native episode from the same reset seed
  and exactly the same 40-action cap. Uniform random RNG streams:
  `25000 + 1000*seed_index + 100*layout_index + episode_index`,
  layout_index0=default,1=mirror. The random arm never sees DQN
  predictions or privileged state. No evaluation learning or early
  hyperparameter selection.
- **Maximum budget:** TRAIN 2 x 2,048 = **4,096** real actions; TEST
  2 seeds x 2 actors x 2 geometries x 10 starts x 40 =
  **3,200** real actions; **7,296** total maximum, including all
  native actor interactions. Count actual actions separately by train,
  eval arm, geometry and seed; no counterfactual probes. Do not launch
  until source-contract tests and workflow budget guards pass.
- **Information boundary:** DQN receives only the native 5x5 public
  observation and actual scalar public reward; no environment internals,
  future observations, hidden event flags, oracle paths or prewritten
  causal-stage plan. After each real action an external evaluator may
  inspect native `info.events` to report switch/key/lock/goal milestones,
  but must not pass it to DQN or its loss. The model is reactive;
  this pilot does **not** test learned memory.
- **Artifacts:** JSON episode-level metrics plus numeric-only NPZ for
  per-step legal public observations, executed actions and reward, episode
  boundaries and policy Q-network weights as numeric arrays. Load with
  `allow_pickle=False`; no `torch.save`/`torch.load` or arbitrary
  object serialization in the artifact. Include source SHA, dependency
  versions, train/eval seeds, native interaction accounting and duplicate
  trajectory hashes. Preserve partial evidence on any failure.

## Metrics, decision and stop conditions

**Primary:** mirror_x heldout native goal success (actual `terminated`
with native reward=1) per model seed, paired against uniform random on the
same ten reset seeds. **Secondary:** default-geometry heldout goal success,
TRAIN rewarded episode count, switch/key/lock events (evaluator-only),
episode lengths, duplicate trajectories, and any train-to-mirror regression.
An issued action or partial causal milestone is never a completed goal.

**Feasibility PASS requires, for EACH of the two DQN seeds:** at least
one genuinely rewarded TRAIN episode, >=3/10 heldout default successes,
>=2/10 heldout mirror successes, and at least two more mirror successes
than the paired random arm. This is a conservative predeclared engineering
gate, not a significance test. Report exact paired wins/losses and
uncertainty; do not infer robust transfer from ten starts on one mirrored
layout. A failure on either seed is **NO PASS** and blocks claims that a
competent learned Workshop behavior source has been established.

**Stop after one frozen pilot.** Do not retune epsilon, training budget,
reward, starts or architecture using heldout mirror outcomes; no duplicate
workflows or extra seeds. If it fails, retain the negative evidence and
separately preregister a reuse-first sparse-reward/partial-observation
intervention before any G2-E2 learned-effect experiment. If it passes,
a separate preregistration must still compare history-aware public-effect
prediction against action-frequency and matched-capacity controls on
**fresh** XLand layouts and genuinely independent Workshop conditions.
Passing this Workshop gate alone establishes neither continual learning,
cross-domain knowledge transfer nor physical BLUE placement.
