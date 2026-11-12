# BAGI M10-P1 — preregistered public-representation comparison (2026-11-12 UTC)

**Status: source registration before any M10-P1 fitting.** M10-D3 [run 43193369149](https://github.com/AncilBenny/BAGI/actions/runs/43193369149) was executed previously (262144 training actions, 8192 optimizer calls, 24429 additional TRAIN-MONITOR actions); this document does **not** repeat those measurements or claim a new run. New train-only M10-D3 layouts 119001–119012 are now contaminated by diagnosis and excluded from this evaluation.

## Sources and reuse

- PPO by Schulman et al. (2017): https://arxiv.org/abs/1707.06347 ; reuse pinned maintained Stable-Baselines3 2.9.0, no new optimizer.
- Official Farama MiniGrid DoorKey: https://minigrid.farama.org/main/environments/minigrid/DoorKeyEnv/ . Reuse upstream v3.2.0 `OneHotPartialObsWrapper`, whose 7×7×20 image contains object/color/state one-hot channels (11+6+3). Upstream source: https://github.com/Farama-Foundation/Minigrid/blob/v3.2.0/minigrid/wrappers.py . Keep public mission, native 0–6 actions, and four *actually executed* previous actions; no hidden coordinates, private pocket/inventory or evaluator state in policy observations/rewards.
- Partial observability is not solved by one-hot input alone (Kaelbling et al. 1998, https://doi.org/10.1016/S0004-3702(98)00023-X). RL Zoo reference DoorKey-5x5 results are *not* directly matched 8x8 evidence and its old normalization/checkpoints remain incompatible; do not deserialize untrusted saved models. Reuse existing BAGI numeric-only NPZ export, physical auditor, native action and optimizer counters, separate final split.

**Contribution type:** tested integration/representation ablation and independent evaluation, not a new learning method or task-independent general intelligence.

## Frozen falsifiable hypothesis and independent split

Compare three fresh upstream PPO arms under equal policy family, model seed, train layouts, real action and optimizer counts, optimizer settings, reward and native episode horizon:
1. **raw_history:** existing public raw categorical 7×7×3 image, mission words, four actually executed actions (243 inputs), no public direction.
2. **raw_direction:** identical raw image/history/mission plus a one-hot encoding of the PUBLIC heading (247 inputs), isolating adding direction.
3. **onehot_direction:** upstream one-hot 7×7×20 public partial image with identical mission, executed-action history and public heading (1080 inputs), isolating the image coding versus arm2 *except for parameter count*.

Two newly seeded policy initializations **211, 223**, **131072 native DoorKey8x8 train actions each fit**, `n_steps=128`, batch64, two PPO epochs, actor/critic hidden widths32, entropy bonus0, CPU, native TimeLimit256, standard sparse task reward and no early episode reset. Expected six fresh fits, **786432 measured train actions** and **24576 measured optimizer.step calls**. The training reset seed stream uses the existing `1_000_000 + model_seed*500_000 + reset_index` pattern. Because terminal successes can change reset counts, record actual seed-hash and reset counts rather than presuming identical visited episodes. Per-arm parameter counts must be recorded: equal 32-width MLPs have different parameter counts for input widths, so positive results could reflect capacity as well as encoding. Do not interpret a pass as pure one-hot causation.

**TRAIN-MONITOR** only: fresh 119101–119106 (six layout seeds per model/arm), untrained deterministic, trained deterministic and trained stochastic once per layout. These are never independent final results. **FINAL heldout**: wholly new 120001–120012, deterministic and three preseeded stochastic repeats per layout/model/arm, only after *all six* training fits, checkpoints and preheldout integrity pass. Repeats are correlated within layouts, not new independent layouts. No M9 final 118001–118012 or prior M1–M10 heldout/monitor for training. Never inspect final during training/tuning.

**Primary preregistered gate:** pooled sampled physically verified pickup→effective locked door opening→native terminal rewarded goal episodes for onehot+direction exceed raw-only by at least **4 of 72**, AND each onehot model seed independently achieves at least one physical full goal, AND all training integrity checks pass. Also report onehot vs raw+direction and raw+direction vs raw. Pass would justify a further replication; it would not prove representation alone caused improvement or imply task-independent learning. Negative/ambiguous: retain and analyze valid door opportunities, attempted toggles, effective openings, action concentration and post-door navigation without reuse of independent final layouts for tuning.

**Auditor boundary:** `NativeDoorKeyAudit` reads private simulator state only for external measurement and never feeds its snapshots to PPO. Report physical episodes as well as events and opportunity denominators, native reward and terminator flags separately. Run through the **existing** serialized `bagi-research-loop.yml` with exact source/CI/PR-merge provenance, and an ADD-only `.github/bagi-m10p1-once.json` request after new push and PR CI. No duplicate workflow, cadence change, direct main write, untrusted model deserialization or committed generated NPZ.

At this source stage: **0 new M10-P1 actions, optimizer updates, model fits, independent evaluations or teacher queries**. Proposed training budgets above are not measured results. Main integration is HOLD pending reviewed PR and final validation.
