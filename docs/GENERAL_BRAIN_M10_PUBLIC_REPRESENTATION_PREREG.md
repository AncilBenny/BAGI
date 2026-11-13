# BAGI-M10-P0 — public-representation and causal-action funnel preregistration (2026-11-11 UTC)

**Status: hypothesis and experimental design only.** This document does not claim an M10 run, model fit, CI validation, checkpoint, or scientific improvement. Historical M1–M9 results and held-out layout streams remain immutable. The immediate next useful step is implementing and testing this diagnostic on an isolated branch/workspace, then executing it through the existing serialized research workflow.

## Evidence and open question

M9 completed on source `943e7b2963a45d62b5f6ed25e0950be1f5ce3492` in Actions run 43014639399 (artifact 13851889475): four native fits, 524288 training actions, 16384 intercepted optimizer updates; both PPO and RecurrentPPO had 0/72 sampled and 0/24 deterministic independently physically verified DoorKey goals. Actual key pickup events occurred, whereas effective door openings were rare. These are events, not unique competent episodes. The M9 scientific gate failed; neither a memory deficiency nor an exploration deficiency is established as causal.

The current `PublicGoalActions` wrapper in `bagi/experiments/m5_continual_retention.py` flattens a public 7×7×3 MiniGrid image as raw categorical float codes, hashes public mission words into 64 dimensions, appends four *actually executed* actions, and omits the public `direction` field. A flat MLP has no explicit categorical or spatial inductive bias. A more faithful public-observation representation might help, but it must be tested without exposing private inventory, coordinates, or evaluator-only event flags.

## Reuse-first literature and implementation decision

- Farama MiniGrid official observation wrappers and DoorKey benchmark: https://minigrid.farama.org/main/api/wrappers/ and https://minigrid.farama.org/main/environments/minigrid/DoorKeyEnv/ . Prefer upstream `OneHotPartialObsWrapper` and native `direction` over a bespoke privileged encoder; verify the exact installed 3.2.0 wrapper semantics and shape before use.
- Schulman et al. (2017), PPO: https://arxiv.org/abs/1707.06347 ; reuse pinned MIT Stable-Baselines3 2.9.0 PPO rather than implementing an optimizer.
- RL Baselines3 Zoo maintained PPO hyperparameters: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml . Its DoorKey-5x5 recipe and model card are external reference results, **not** a BAGI reproduction or a valid directly matched 8x8 baseline; legacy checkpoint/normalizer compatibility and untrusted deserialization must be separately audited.
- Kaelbling, Littman and Cassandra (1998) on partial observability: https://doi.org/10.1016/S0004-3702(98)00023-X ; a one-hot current view alone does not solve hidden state or long-horizon control.
- Ross, Gordon and Bagnell (2011), DAgger: https://proceedings.mlr.press/v15/ross11a.html . Expert correction is a later train-only intervention, not an M10 evaluation oracle.

**Contribution type:** established representation integration and controlled evaluation, not a new RL algorithm or evidence of AGI.

## First experiment: train-only causal funnel, no learning

Use existing M9 numeric policies if their archived checkpoints and public-input contracts can be verified without unsafe pickle loading; otherwise start with train-only random and current-policy rollouts. Use fresh **training-only** layout streams disjoint from all M1–M9 monitors/finals. Track one episode's first and repeated transitions separately: observed key; effective pickup; key carried; physical adjacency/orientation to locked door while carrying; toggle attempts in valid precondition states; effective locked→open transition; physical goal reached and rewarded termination. Use evaluator-only private state exclusively to annotate physical effects **after** actions, never to select actions or provide a training reward. Record counts, denominators, uncertainty, action histograms, and stage-conditional rates. Do not infer that a low opening count means poor toggle choice unless opportunities are measured.

**Decision gate:** if valid door-opening opportunities are near zero, prioritize exploration/navigation; if opportunities are frequent but toggles fail, prioritize public effect/action-precondition representation; if openings occur but goals do not, prioritize post-door navigation and long-horizon control. Treat this as diagnostic interpretation, not a causal intervention result.

## Second experiment: one public-representation change

**H-M10 (to freeze in executable config before fitting):** using identical native DoorKey8x8 rewards, seeds, native action counts, optimizer update accounting, model family, training layouts and evaluation protocol, a categorical one-hot encoding of the *same public image*, with the public direction encoded explicitly and the same public mission and actual action history, improves independent physically verified full key→door→goal success relative to BAGI's raw-code flattened-input PPO.

Implement both arms with upstream PPO. Keep policy capacity as closely matched as practical and disclose parameter-count differences; include an ablation that isolates direction from categorical encoding if capacity or multiple representation changes confound attribution. Ensure the one-hot public wrapper preserves action semantics and does not leak private state. Check that training episodes continue to native truncation and that all action/optimizer counters are measured rather than assumed.

Before opening a new untouched final seed stream, freeze exact train/monitor/final seeds, model seeds, hyperparameters, interaction and optimizer accounting, independent physical-success gate, evaluation horizon, deterministic and stochastic sampling plan, and output paths in a committed config. Never reuse M9 final layouts 118001–118012 or any older held-out layouts for fitting or tuning. Require at least two independent initialization seeds per arm; correlated stochastic repeats on a layout are not independent layouts. Report physical event counts separately from completed episodes and rewards. A negative or ambiguous result is informative and must remain in the append-only log.

## Execution and governance

Use `setup/initial-platform`, existing workflow serialization, exact-source push/PR CI admission, Ruff and pytest, and non-force expected-HEAD publication. Check for concurrent workers before changing any workflow trigger. Do not add a competing scheduled workflow or automatically rerun M9. Save generated artifacts in ignored run directories; never commit checkpoints, credentials, or full evaluator trajectories. Main integration remains HOLD until reviewed PR, local validation, current independent push/PR CI, and postmerge main CI. This preregistration itself performs **zero** simulator steps, optimizer updates, fits, held-out episodes, and teacher queries.
