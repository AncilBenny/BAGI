# G2-E6-P0B — posthoc TRAIN-only public action-outcome conflict support audit (2026-10-28 IST)

**Decision: frozen P0B history-identifiability HOLD remains binding. No P1 fit, second acquisition, gate relaxation or transfer claim.** This is a read-only analysis of the existing TRAIN numeric artifact, not a preregistered model comparison or evidence that a history encoder will learn. The task-independent question is whether the learner's legal observation/action history has repeated, independently supported information about next public effects.

## Immutable evidence and permitted input

Source commit `1751989016ed14a82bfe110e27ec9a711c1be14f`, one-shot native Actions run `40578403185`, artifact `12768976012`. Outer ZIP SHA256 `182466a08a646b7f6d62a2b95a82f0a0c20e10d4313d71959e6ac840669cc7e3`; nested ZIP SHA256 `fcdaaa9b3aa0f08d4c3ae350f291cefb1ae04921a299893b16d08be928a4f5dc`. Independently rechecked both hashes. Read only `train_public_numeric.npz` with `numpy.load(..., allow_pickle=False)`, plus source `summary.json` for provenance; **no diagnostic observations or evaluator-only hidden states entered the conflict calculation**. No model training, policy change, native reset/step, teacher, new workflow or heldout size13 read.

Grouping key: exact 25-byte current public 5×5 tile observation plus **actually executed** action (0..3); next-outcome: exact next public 5×5 observation. Group separately by Workshop size, never pool sizes. Episode ID and step order reconstruct actual preceding public views/actions. For a conservative two-view comparison, use nonterminal step indices 2..38; a separate four-step history comparison uses indices 4..38. A conflict is an observed *difference*, not proof of an identifiable hidden cause or predictive generalization.

## Exact TRAIN-only coverage (exploratory, NOT the frozen learned-policy gate)

| TRAIN-only property | Size 9 | Size 11 |
| --- | ---: | ---: |
| Executed native TRAIN transitions | 4,096 | 4,096 |
| Independent TRAIN episodes / unique public-action trajectories | 104 / 104 | 104 / 104 |
| Distinct initial public views | 13 | 16 |
| Distinct current-view+action keys (all steps) | 155 | 173 |
| Keys with >1 next public outcome (all steps) | **2** | **7** |
| Nonterminal step 2..38 eligible rows | 3,786 | 3,786 |
| Eligible distinct keys / conflicting keys | 153 / **2** | 167 / **7** |
| Eligible conflicts with cross-episode, distinct-two-view-history outcome pairs | 2 | 7 |
| TRAIN conflicting-key minority outcome counts (all steps) | **1, 25** | **2, 3, 4, 19, 25, 27, 59** |

Each size had 102 full 40-step TRAIN episodes and two eight-step budget-limited partial finals; zero native goal reward. The two size-9 conflict keys used actions 3 and 2: their full-TRAIN next-view outcome frequencies were **42:25** and **68:1**, respectively. The seven size-11 conflict keys used actions 3,2,1,3,2,0,2, with frequencies **59:87, 19:21, 26:25, 33:27, 72:3, 8:4, 4:2**. These are observational counts under epsilon-greedy TRAIN, with online changing weights and many revisits; they are neither held-out tests nor frozen-policy examples. Three of seven size-11 keys and one of two size-9 keys have a minority outcome with at most four observed transitions. Different episodes and starts do not independently establish different causal tasks.

## Does short public history uniquely disambiguate the observed TRAIN conflicts?

Within the *conflicting keys only*, restrict to nonterminal steps 4..38. Add exact preceding views and the **real executed preceding actions** to each current-view+action key; compare outcomes only when an exact history repeats in distinct episodes. For two prior view/action pairs, **1 of 2** size-9 conflicting keys and **3 of 7** size-11 keys still have at least one cross-episode exact-history collision with *different* next public views. For four prior view/action pairs, **0 of 2** size-9 keys but **1 of 7** size-11 keys still collide; the latter has two ambiguous repeated four-step histories, including one with the same policy seed. Thus two or four steps are **not demonstrated to be a sufficient state representation**.

Equally important, four-step history becomes very sparse: **88/122** size-9 and **254/337** size-11 observations within the original conflicting keys have a history signature occurring only once in this restricted sample. Zero observed collisions on the remaining keys is **not** evidence of predictive resolution; a longer history can memorize individual trajectories without independently repeated support. No classifier, recurrent network, nearest-neighbor predictor, conditional accuracy or learned policy was fit or evaluated in this iteration.

## Decision and falsifiable next gate (proposal, NOT authorization)

The earlier frozen trained-policy result remains size9 **0/24** eligible conflict keys and size11 **1/28**, so the **predeclared both-sizes history-identifiability gate fails**. TRAIN-only 2/155 and 7/173 cannot substitute for it. The evidence supports a narrow diagnosis: real public transition aliasing exists in TRAIN, but outcome support is uneven and exact four-step histories are often singletons; hidden-state sufficiency, learnability and causal benefit remain unverified. Source-only behavior change after DQN optimization still did not produce a Workshop goal.

For a separately preregistered *future* study, require independently disjoint TRAIN/validation episodes and genuinely new geometry/task conditions, an explicit minimum number of independently repeated outcome classes per size, and matched public-only reactive, action-only, persistence, shuffled-history and finite-history/recurrent predictors using identical source trajectories, optimizer/parameter budgets and native step accounting. Report event-positive precision/recall, effect false positives, calibrated next-view prediction and downstream *learned-policy* goal/recovery gains; later assess XLand physical effects with its nonvacuous pickup→drop→adjacency+reward gate, not game reward alone. Stop before fitting if support and leakage gates fail. Do not repurpose reserved size13, open heldout composition labels, rerun P0B or bypass P6/P10a STOP and P9 native HARD STOP. Reuse the existing Workshop DQN, numeric traces, and established POMDP/GRU methodology rather than inventing a new architecture. This proposal does **not** authorize a new experiment.

Contribution: independent posthoc evaluation/diagnosis, not a new algorithm or proof of task-independent intelligence.
