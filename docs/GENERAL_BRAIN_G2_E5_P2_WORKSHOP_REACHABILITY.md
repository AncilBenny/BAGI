# G2-E5-P2 — Workshop evaluator-only reachability and public-state aliasing

**Date:** 2026-10-26. **Status:** bounded, exploratory structural diagnostic; **NOT** learned Workshop competence, policy training, a new planner, memory promotion, independent heldout evaluation, or cross-domain transfer. This study does not authorize any further native experiment. Native Workshop `step` and `reset` are **never called** by this audit.

## Reuse-first question

The task-independent BAGI goal is public observation, useful belief, goal-conditioned navigation, learned action effects/recovery, reusable skills, and cross-domain transfer. G2-E2-P0 trained DQN on Workshop for 4,096 real actions with **zero rewarded TRAIN goals**, and G2-E3-P1 matched PPO extrinsic-only vs episodic first-visit novelty produced **zero Workshop rewarded TRAIN/TEST goals** in either seed. G2-E5-P0/P1 found that repeated public next observations can be predictable without demonstrating goal-directed control or generalization. Before another costly or poorly grounded memory/exploration intervention, check a narrower structural alternative: **are the current Workshop start positions physically solvable within its 40-step cap, and how much next-public-view ambiguity exists in its native 5×5 observation?**

Reuse the existing `bagi.worlds.workshop.WorkshopEnv` **without changing it**, including its native `_can_enter` collision rules, `_observe` public observation, and switch/key geometry. The evaluator-only enumerator mirrors the two explicit switch/key effects, not a new agent dynamics model. This is a finite-graph audit of a fixed small environment, **not** a general-purpose oracle available to BAGI. Existing literature: Kaelbling et al. (1998) POMDP belief and SB3-Contrib RecurrentPPO, already recorded in `docs/GENERAL_AGENT_ARCHITECTURE.md` and `docs/REUSE_INVENTORY.md`; none establishes that memory is necessary in this particular Workshop observation contract. No new dependency, optimizer, heuristic action controller, teacher, or external library is added.

## Source pin and methodology

- Pin `bagi/worlds/workshop.py` to Git blob SHA1 **`3a91b058d3461ce4ecb489e93939c88f624557f4`**, verified against branch `setup/initial-platform` at the start of this iteration. The audit fails closed if that file changes. Both fixed geometries `default` and `mirror_x`, 9×9, 5×5 public observation, 4 actions, 40-action cap, and the native **13 legal initial positions** are frozen. A reset seed chooses among those 13 starts; it does **not** generate a new geometry.
- Enumerate finite evaluator states `(row, col, door_open, has_key)` reachable from all 13 starts. Build edges from the native `_can_enter` predicate and native switch/key coordinates. Do not expand terminal goal states. Compute shortest possible action count by breadth-first search separately for every start; these are **privileged lower bounds**, not learned or public-only policy performance. The script mutates private evaluator fields to obtain native `_observe` values but never calls native `reset` or `step`, and never passes those fields to any learner or policy.
- Group reachable evaluator states by exact 25-cell **PUBLIC** view. Count views that represent more than one latent state. Separately group nonterminal `(public_view, executed_action)` keys by possible next public views; more than one outcome means **one-step observation-action aliasing**. This is a finite-state structural ambiguity test, not a predictive Brier score, not a causal intervention on an actually reached trajectory, and not a guarantee of recurrence benefit.
- Pure synthetic source-shaped fixture tests run without Gymnasium. A separate dependency-installed CI integration test checks the exact pinned native source and monkeypatches `WorkshopEnv.step/reset` to raise if invoked, then verifies the same graph results against the **actual** `_can_enter` and `_observe`. Do not call the synthetic fixture a native simulator replication. A source change, failed native parity, or changed output invalidates the structural claim.

## Static evaluator-graph results (source-shaped local fixture; CI native-source parity required)

| Metric | Default | Mirror-x |
|---|---:|---:|
| Legal starts / distinct initial public views | 13 / 13 | 13 / 13 |
| Reachable evaluator states | 79 | 79 |
| Distinct public views | 57 | 57 |
| Public views aliasing multiple evaluator states | 14 | 14 |
| Maximum latent states per view | 3 | 3 |
| Nonterminal `(public view, action)` keys | 224 | 224 |
| Keys with more than one possible next public view | **2** | **2** |
| Minimum / maximum shortest successful action path | **15 / 19** | **15 / 19** |
| All 13 starts have a path within 40 actions | Yes | Yes |

These are exhaustive **model-graph** counts under the pinned Workshop rules, not statistical estimates or outcomes from 13 native episodes. The actual agent is not told its latent state, route, goal location outside its local view, switch/key flags, or a shortest-path action. A lower bound of 15–19 actions leaves nominal slack in the 40-step cap, but does **not** imply a memoryless learner can discover or follow those paths. Two ambiguous public-view/action keys demonstrate that a current public image is **not strictly one-step Markov** over this reachable evaluator graph; they do **not** show that memory improves optimal goal success. Fourteen aliased views include cases whose immediate public outcome is identical. Counts are for one fixed causal geometry plus its mirror, not diverse new tasks or layouts.

## Decision and next scientific gate

**G2-E5-P2: structural feasibility PASS; learned competence NOT ESTABLISHED; NO MEMORY PROMOTION.** The native 40-action cap does not appear to be a structural impossibility for any registered Workshop start. The existing failures are therefore compatible with sparse-reward exploration, insufficient coverage, poor policy learning, partial-observation aliasing, or combinations; this audit does **not** isolate a cause. Only 2/224 exact public-view/action keys have multiple immediate public outcomes, so the presence of aliasing alone is not enough to justify a large recurrent/world-model experiment. Also, distinct Workshop reset seeds cannot support claims of geometry-disjoint transfer.

Next, independently preregister **one** bounded public-only Workshop *learned-policy competence* experiment using maintained, vetted methods and genuinely fresh starts; include an action-only or random baseline, positive native TRAIN goal reward prerequisite, independent frozen native TEST goal completion, true per-seed action budgets, and separate coverage/goal outcomes. Before an intrinsic exploration method is chosen, recheck upstream license, API, hidden simulator calls and compatibility; do not retune the already failed first-visit beta or use the current graph as an actor, oracle reward, teacher, goal relabeling or heldout composition labels. Only after a learned Workshop actor passes should BAGI test source-disjoint XLand/Workshop public action-effect models with matched history and shuffled-history controls, reporting physical XLand success separately. A structural audit does not prove transfer, memory, continual learning or AGI.

**New native actions: 0. Native resets: 0. Training/optimizer steps: 0. Teacher queries: 0. Heldout TEST reads: 0.** No new workflow, paid sweep, deployment, secret change, merge, or `main` write. Local contract tests: four PASS, one native-source integration test SKIPPED because Gymnasium is unavailable locally; GitHub CI must validate the latter before treating native-source parity as checked.
