# G2-E2-P0 — Post-result TRAIN-only coverage and public-aliasing audit (2026-10-25)

**Status: descriptive post-result diagnosis; the frozen G2-E2-P0 gate remains FAIL / STOP.**
This audit does not change, rerun, fit, or tune the G2-E2-P0 learner. It uses only
the **TRAIN** partition of the previously published numeric artifact; it does
not select methods from the held-out default/mirror results.

## Provenance, information boundary and reproducibility

- Frozen experiment source: `f47c21c05057a7362dfe83aa73c6ebb452c978f0`.
  Successful native run `40185492543`, artifact `12596191672`,
  ZIP SHA256 `b000a02a2eacfb8aa07befa151e0bbfb2c5bf7fd430f8bd896bd11f90f57436a`.
  The existing independent full-run audit established **7,296** total native actions.
- Read `summary.json` and `public_policy_numeric.npz` using NumPy
  `allow_pickle=False`. Filter numeric rows **strictly** to
  `actor_index == 0` (TRAIN), separately by `seed_index`. Recount executed
  actions, public observation changes, distinct public 5x5 views,
  observation/action -> next-public-view collisions, and contiguous repeated
  action runs *within each episode*. Event counts are already archived
  **evaluator-only** summaries from TRAIN episodes, not learner inputs.
  No held-out row, action, outcome or network-evaluation trace is analyzed here.
- Exact TRAIN sample: **2,048 real actions / 52 episodes per model seed**;
  **4,096 actions / 104 episodes** in aggregate. Both seeds have **zero**
  native reward-one goals in TRAIN; the registered DQN therefore saw no
  positive extrinsic reward. All TRAIN trajectories are distinct within each
  seed (52/52), although observations recur frequently.

## Independent TRAIN-only measurements

| Measure | Seed 17 | Seed 29 |
|---|---:|---:|
| Executed action counts UP/RIGHT/DOWN/LEFT | 530 / 378 / 597 / 543 | 462 / 482 / 673 / 431 |
| Public observation changes / actions | 1,259 / 2,048 | 1,195 / 2,048 |
| Publicly unchanged actions | 789 | 853 |
| Distinct current public 5x5 views | **35** | **32** |
| Distinct (view, executed action, next view) tuples | 119 | 103 |
| Same (view, action) with >1 next public view | 1 key (27 rows) | 1 key (29 rows) |
| Longest repeated identical-action run (within episode) | 14 | 22 |
| Episodes with >=10 identical actions consecutively | 4 / 52 | 8 / 52 |
| TRAIN switch / key / lock / goal event counts (evaluator-only) | 28 / 1 / 0 / 0 | 33 / 2 / 0 / 0 |
| TRAIN episodes with a key event | **1 / 52** | **2 / 52** |

Per executed action, public-change positives/total were seed17
UP **247/530**, RIGHT **354/378**, DOWN **258/597**, LEFT **400/543**;
seed29 UP **226/462**, RIGHT **349/482**, DOWN **237/673**,
LEFT **383/431**. Public change is an observation diagnostic, **not**
a physical-motion or causal-success label; static local views can alias
different hidden states.

Across successive, nonoverlapping **512-action TRAIN blocks**, the public
change counts were seed17 **350, 353, 311, 245** and seed29
**377, 344, 263, 211**. The executed DOWN count rose from
**122 to 194** (seed17) and **133 to 215** (seed29) between the first and
last blocks. This is compatible with declining exploration / increasing
repetitive actions, but does **not** identify a causal effect of epsilon
decay, Q-learning, initialization or geometry: these were not separately
randomized or controlled in this audit.

TRAIN-only switch events were common (28 and 33), but key acquisition was
rare (1 and 2), with **zero** lock and goal events. The exact native
Workshop reward is terminal-only: its `step` returns reward 1 only when
the goal is reached; no evaluator-only switch/key event was fed into the
DQN reward or its loss. Hence this particular 2,048-action/seed DQN
configuration never experienced a positive task-reward target. Bootstrapped
Q updates did occur (1,985 per seed); it is inaccurate to claim that
the network did not train or that sparse reward is proven the sole cause.

## Interpretation and reuse-first decision

**Observed bottlenecks:** very narrow repeated public-view support,
near-absence of later causal-stage experience, zero positive extrinsic
reward, and increased repeated-action / unchanged-view behavior during
TRAIN. The two contradictory next-view keys demonstrate that a single
instantaneous public view and action can be insufficient to uniquely
identify a next public view **within this dataset**. They do not prove
that a specific recurrent representation will resolve the ambiguity,
or that all collisions are due to hidden inventory.

The existing `bagi.agents.dqn.DQNAgent` is a valid baseline, not a
competent Workshop policy under the registered budget. Reuse the
repository's existing public-observation environment adapter, and
consider the already inventoried **SB3 PPO / SB3-Contrib RecurrentPPO**
rather than inventing another optimizer or scripting Workshop causal
stages. The established POMDP/recurrent-control and sparse-reward
exploration literature motivates **prospective** matched controls, but
no novelty or guaranteed improvement is asserted.

**Next falsifiable gate, not executed:** preregister a **new** train/test
split and fixed native-action budget, with an explicit public-only
observation and reward contract, to separate (a) inadequate
goal-reaching exploration from (b) insufficient memory / partial
observability. Compare a maintained feedforward baseline against a
maintained recurrent baseline with matched real interactions and
identical extrinsic rewards; if an exploration intervention is tested,
give it a separate matched control and prevent evaluator-only event
leakage. Track TRAIN public-state coverage, key/lock/goal milestones
**only in the external evaluator**, and true held-out goal completion
across fresh starts/geometries. STOP if neither learner receives
positive TRAIN reward or reaches later causal stages: do not promote
unrewarded action repetition, intermediate events, or fixed-geometry
replay as learned general intelligence. Any subsequent two-domain
effect/belief test requires a genuinely learned Workshop actor and
separate fresh XLand data, matched policy/data/steps, and the native
physical BLUE pickup -> later effective drop -> adjacency + reward
criterion for that diagnostic.

**This iteration:** 0 new simulator actions, 0 teacher queries,
0 DQN updates, 0 held-out evaluations, 0 native workflows and
0 new dependencies. This is a training-data diagnosis, not a
new predictive/continual-learning result.
