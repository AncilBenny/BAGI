# G2-E5-P3 — Exact Workshop sparse-reward probability under uniform random actions

**2026-10-26; exploratory evaluator-only calculation; NO learned competence.** This study uses the existing, source-pinned G2-E5-P2 Workshop graph. It never calls native `reset` or `step`, never creates expert demonstrations, and never supplies hidden coordinates, stage flags, goal locations or transition probabilities to a learner. It is an analytic diagnostic of the **uniform random baseline**, not a bound on PPO, DQN, exploration algorithms or human competence.

## Verified prerequisite and reuse-first decision

The preceding G2-E5-P2 source-parity repair commit `fa7d93cc2f8086cb017c2ed16648c2aeb5a71cc7` passed both GitHub CI runs `40254810084` (push, job `121354276215`) and `40254812689` (PR, job `121354284656`), each with Ruff success and **427 passed, 7 skipped** in the overall suite. CI installed Gymnasium 1.4.0; the source-parity integration test skips only when Gymnasium is unavailable, so it ran. The P2 native Workshop Git blob SHA1 remains `3a91b058d3461ce4ecb489e93939c88f624557f4`. This establishes **source-parity of the static evaluator graph**, not native trajectory outcomes or policy learning. No artifacts were uploaded by either CI run.

Existing DQN and PPO on Workshop received **zero native rewarded TRAIN goals** in G2-E2-P0 and G2-E3-P1, while P2 established all 13 starts have a 15–19 action privileged path within the 40-action cap. The remaining falsifiable diagnostic question is whether **uniform random exploration is likely to encounter even one sparse native goal reward** within small training budgets. This is a probability calculation using the existing finite graph, not a novel exploration algorithm. Reuse `bagi/experiments/g2_e5_p2_workshop_reachability_audit.py`, native `_can_enter` and switch/key geometry, and ordinary finite-horizon Markov-chain dynamic programming. This is an **evaluation/integration** contribution; no external dependency, custom actor or reward shaping is introduced. Known limitation: a learned policy may have very different action probabilities.

## Frozen method and independent consistency check

The unchanged native `WorkshopEnv` has 13 equiprobable legal starts per reset, four discrete actions, fixed `default` and `mirror_x` geometries, and an episode cap of **40**. An evaluator-only graph contains 79 reachable states `(row, col, door_open, has_key)` per geometry. The goal is absorbing. At each nonterminal state, assign probability **1/4 to each native action**, including actions blocked by walls. A backward finite-horizon dynamic program calculates each start's probability of reaching the native goal by action 40; a separate forward probability-mass propagation calculates the probability of activating the switch, acquiring the key, and reaching the goal at fixed horizons. Both calculations must agree within `1e-12`; total mass must stay 1 and causal-stage probabilities must be monotonic. No privileged state enters a policy: all of this occurs **offline in the evaluator**.

The new source is `bagi/experiments/g2_e5_p3_workshop_sparse_reward_audit.py`; tests are `tests/test_g2_e5_p3_workshop_sparse_reward_audit.py`. Four synthetic contract tests pass locally and the one actual Workshop integration test skips locally because Gymnasium is absent; GitHub CI must pass the integration test before the results are independently confirmed against native source. The script fails closed if the P2 native-source hash or registered graph contract changes.

## Exploratory analytical results (not native simulator runs)

| Uniform random policy, 40-action cap | Default | Mirror-x |
|---|---:|---:|
| Registered equiprobable starts | 13 | 13 |
| Reachable evaluator states | 79 | 79 |
| Probability switch activated by action 40 | 74.6204% | 74.6204% |
| Probability key acquired by action 40 | 6.20345% | 6.20345% |
| **Probability native goal reached by action 40** | **0.02487797%** | **0.02487797%** |
| Minimum / maximum goal probability across 13 starts | 0.003891% / 0.051721% | same |
| Chance of ZERO goals in 100 independent random episodes | 97.5426% | 97.5426% |
| Chance of ZERO goals in 410 independent random episodes | 90.3018% | 90.3018% |

The last two rows are **hypothetical independent episodes**, not native training measurements or a direct conversion from simulator steps: early successful termination changes actual episode count, and trained PPO/DQN are not uniform random policies. The mirror is a symmetry check, **not** a disjoint task family or independent learned transfer. Under random action selection, switch activation is common, key acquisition is much rarer, and goal completion is very rare. The result makes sparse-reward exploration a plausible bottleneck and explains why **zero native rewards can be unsurprising under random behavior**. It does **not** establish why PPO/DQN failed or prove a particular exploration or memory algorithm would succeed.

## Decision gate and next research direction

**Random-policy sparse-reward bottleneck supported analytically; NO LEARNED SKILL, MEMORY OR TRANSFER PROMOTION.** Stop treating a handful of zero-reward short Workshop training runs as evidence that the native 40-action task is unsolvable or that a particular recurrent architecture failed. Do **not** repeat the failed G2-E3 first-visit beta tweak, extend the horizon after seeing TEST outcomes, use the privileged graph to choose actions, or fabricate training labels. Before the next native study, preregister **one** small public-observation-only learned Workshop competence experiment using a maintained, documented sparse-reward exploration method, a same-budget extrinsic-only learner and random/untrained controls, genuinely disjoint TEST reset starts (recognizing the fixed geometry), two independent model seeds, exact real-step caps and a native-goal STOP gate. Verify the upstream implementation, license, and that its novelty/goal signals depend only on public observations and executed actions; do not introduce task-stage-specific rewards or BLUE logic. Only after independently rewarded Workshop learning should a matched source-disjoint XLand/Workshop learned-effect/history test be considered.

**Resource accounting this iteration:** 0 native actions, 0 native resets, 0 optimizer updates, 0 teacher queries, 0 TEST reads, 0 native goal episodes, 0 new workflows/automations, 0 secrets/billing/deployments/merges, 0 changes to `main`. The numerical results are exact for the pinned **evaluator graph and uniform random policy**, not a simulator observation or a BAGI agent performance result.
