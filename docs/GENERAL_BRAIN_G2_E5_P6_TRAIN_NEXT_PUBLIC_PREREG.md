# G2-E5-P6 — Matched learned next-public-observation prediction, TRAIN-only

**PROSPECTIVE PREREGISTRATION, NOT YET EXECUTED ON REAL ARCHIVED TRANSITIONS.**
Frozen October 26, 2026, after P5's independently reproduced descriptive analysis and BEFORE any P6 model fit or held-TRAIN-fold prediction. This is ordinary PyTorch-supervised effect prediction using existing BAGI interfaces, not a new learning algorithm, policy, native simulator trial or transfer result. The end goal remains task-independent continual learning; BLUE-specific rules, scripts and evaluator observations are forbidden as learner inputs.

## Scientific question and why this differs from earlier studies

G2-E5-P5 (research commit bcac5c466ca3eab4d0cad6e3908080976bb02886) found identical public-view+action keys with contradictory next public views in XLand TRAIN, and fewer ambiguous rows under four executed-action histories on Workshop DQN TRAIN. However, history also fragmented support: E2 Workshop seed17 repeated-key rows dropped 2,023 to 737 and seed29 2,035 to 710. Neither key counts nor G2-E5-P0's earlier binary visible-change lookup crossfit established useful learned history or a competent policy.

**Frozen question:** Does fitting a complete next-public-observation predictor to genuinely executed source-TRAIN transitions make four-action history more predictive than a matched reactive, shuffled-history, and action-only model on disjoint TRAIN groups? The null is history gains vanish against matched controls or reflect repeated-key fragmentation. No policy goal navigation, reward or transfer is measured.

**Reuse first:** Installed PyTorch MSE/Adam with the same two 192-wide Linear+ReLU backbone pattern as BAGI's existing WorldModel (bagi/models/world_model.py), and existing hash-verified E1/E2 TRAIN-only loader (bagi/experiments/g2_e5_train_history_audit.py). Do not introduce a recurrent architecture, optimizer, planner, reward, library dependency or training-script oracle. Separate models are fitted for each dataset, so there is no learned cross-domain parameter transfer.

## Frozen data and strict public boundary

- E1 run 40165691302, GitHub artifact 12588790887, ZIP SHA256 aaf8eb011763bf47e1d196771e257eb13c4eb35294bbeed2b71757ade525f073. Use only numeric E1 XLand TRAIN from 15 layout seeds 11900–11914 (all existing TRAIN behaviors); for E1 Workshop use only ten independently collected uniform_random TRAIN episodes 12100–12109, never ten duplicated fixed-script episodes. Match an executed-row public observation and action with the NEXT public observation only within the SAME episode, dropping terminal rows without a successor.
- E2 run 40185492543, artifact 12596191672, SHA256 b000a02a2eacfb8aa07befa151e0bbfb2c5bf7fd430f8bd896bd11f90f57436a. Only original Workshop native DQN TRAIN actor_index==0, each original seed17 and seed29 separately: 2,048 real historical TRAIN actions / 52 episodes per seed, recorded next public views. No E2 TEST actors, weights, hidden states or evaluator events.
- Only public current observation, the current actually executed action, and the previous FOUR executed commands can be INPUTS. Next public view is a supervised LABEL only. E1 XLand has 115 public numeric features, with 28 final action-history bits already embedded in its legal view. Predict the first 87 (non-history) next public features and use the last 28 input bits only as history. Never append a redundant second action history. Workshop public view/next label has 25 observed tile cells; build four prior executed action one-hots (16 bits), zero-padded and reset each episode. Both current and target numeric public values have fixed scale /8, no fit-on-held-fold normalization. Current executed command is one-hot in every arm. Forbidden: complete grid, privileged pocket/key/door flags, task-specific agent code, scripts, full TEST arrays, teacher labels, expert paths, or discarded simulator probes.
- The strict existing source loader checks both full ZIP SHA-256 digests and uses numpy.load(allow_pickle=False) with explicit TRAIN-only key whitelist. No new native environments instantiated.

## Frozen matched design and budget

**Four supervised arms:** action_only (zero public view and past actions), reactive (current public view, current command, zero history), shuffled_history (same current view/action with prior-action history permuted separately within TRAIN and held-TRAIN fold input sets without next-view labels), real_history (true executed previous four actions). All arms use IDENTICAL fixed input width, two 192-unit hidden ReLU layers, linear output head, MSE objective, Adam learning rate 0.0003, 12 epochs, 128-row minibatches, same initialization, minibatch permutation and TRAIN rows in their matched folds. Add an **untrained persistence baseline** (copy current visible non-history values to next-view prediction). No architecture sweep or choosing hyperparameters after reading held-fold outcomes.

Two fixed TRAIN-only disjoint group folds: XLand use numeric layout seed modulo 2, so the same layout's policy/task episodes are always in the SAME fold; E1 Workshop random episode reset seed modulo 2; E2 Workshop native DQN episode index modulo 2 independently for each policy seed. Evaluate each fold once as held-fold while training ONLY on the other, with model initialization seeds 17 and 29. Frozen count: four datasets × two folds × two initialization seeds × four trained arms = exactly **64 bounded supervised model fits**, plus persistence evaluations. No held-out original TEST/independent benchmark evaluation. CPU torch.set_num_threads(1), deterministic fixed initialization and minibatch shuffles, no early stopping.

After fitting, report EVERY fold/seed and per-dataset held-TRAIN-fold **public cell MSE** (clipped model prediction in [0,1]), count of truly CHANGED next-public features, MSE conditional on changed features, and unchanged-view persistence MSE. The public-cell MSE is normalized by the fixed /8 source convention. Predeclare aggregate per dataset: weight each held-fold MSE by number of public cells and average the two model initialization seeds equally. Never aggregate XLand and Workshop into one invented transfer score. Report the sample size and instability; observations with no changed features are not proof of correct action effects.

### Frozen gate

Only **consider** a separately preregistered prospective learned-belief/native policy test if real_history yields at least **5% lower pooled held-TRAIN-fold public-cell MSE than BOTH reactive and shuffled_history**, and also beats the untrained persistence baseline, separately in (1) E1 XLand TRAIN, (2) E2 Workshop DQN seed17 TRAIN and (3) E2 Workshop DQN seed29 TRAIN. E1 Workshop random-only is a tiny, independent ten-episode diagnostic, NOT sufficient for gate promotion. If any required comparison fails, results are dominated by static cells, changed-cell support is absent, or model-seed instability is material: **STOP, do not promote predictive memory**. A pass also DOES NOT prove held-out generalization, useful belief, native goal-reaching, physical BLUE movement, reusable skills or cross-domain transfer. P4 RND remains STOP on unverified official wheel/dependencies and uncounted initializer/SB3 reward-GAE integration.

## Reproduction and safety

New implementation: bagi/experiments/g2_e5_p6_train_next_public.py. Tests: tests/test_g2_e5_p6_train_next_public.py. **This preregistration commit does not execute the 64 real-archive fits.** Five pure synthetic fixture tests passed in the local preflight; real archive source/shape contracts were inspected read-only, but NO archived P6 loss or label prediction was computed. After CI passes, execute the source once with the same already-published hash-pinned archives:

    python -m bagi.experiments.g2_e5_p6_train_next_public \
      --e1-zip /path/to/verified_e1.zip \
      --e2-zip /path/to/verified_e2.zip \
      --output /path/to/g2_e5_p6_train_only.json

Record exact SHA, CPU fit count and grouped test results before interpreting any learned effect. One research-branch nonforce commit maximum. New native actions/resets 0; original TEST arrays read 0; trained P6 models on archived data **0 so far**; teacher queries 0; policy updates 0; new workflow/automation 0; no secrets, deployments, billing, merges or main writes.
