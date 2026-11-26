# BAGI M10-D3 — preregistered fresh train-only physical-action diagnosis

**Source stage (2026-11-12 UTC):** integration of an executable study, not an executed result. This document freezes the first M10 training-only causal diagnosis BEFORE fitting. Subsequent changes require an explicit, separately recorded erratum. It does not replace M10-P0 or reopen any historical held-out layout.

## Gap and reuse

M9's two fresh PPO and two RecurrentPPO fits each failed independent DoorKey completion, with occasional physical key pickups and little door opening. Archived deterministic action concentration is post-hoc and cannot by itself reveal whether the policy reached a valid interaction state. M10's `NativeDoorKeyAudit` can measure key possession, facing a still-locked door, toggle attempts, effective open events and true key→door→goal outcomes, separately from reward. Prior tests validated the native MiniGrid API on two smoke seeds, not trained behavior.

Use the existing pinned Farama MiniGrid and **Stable-Baselines3 2.9.0 PPO**, BAGI `PublicGoalActions`, `NativeDoorTraining`, `count_optimizer_steps`, and `numeric_state`. Do not invent a new learning algorithm. Primary sources: [MiniGrid DoorKey](https://minigrid.farama.org/main/environments/minigrid/DoorKeyEnv/), [PPO](https://arxiv.org/abs/1707.06347), [SB3](https://github.com/DLR-RM/stable-baselines3). This is **integration and diagnosis**, not novelty or task-independent learning.

## Registered train-only design (before native execution)

- New independent PPO initialization seeds **181 and 197**; **131,072 native actions per fresh fit**, rollout 128, batch 64, 2 PPO epochs, one continuous `learn()` call, default PPO settings otherwise, 32-width actor/critic MLP, zero entropy bonus, CPU.
- Existing raw-code, public-image+mission+four *actually executed* actions, 243-dimensional actor inputs. **No public direction in this diagnostic**: that change is reserved for the subsequent representation comparison.
- Training episodes draw `1_000_000 + seed * 500_000 + native_reset_index`, never the monitor or final seeds. No privileged labels for actions, rewards or policy. Episode horizon 256.
- Fresh TRAIN-MONITOR layouts **119001–119012** (12 per model). Before each fit, run a uniform-native-action random control and deterministic initialized PPO. After each fresh fit, independently audit deterministic and sampled PPO with fixed action-RNG seeds. These are diagnostic repeats, **not independent final evaluation**. The random controller is a reference, not an information- or training-matched algorithm.
- PPO expected accounting (checked against actual intercepted optimizer calls): **262,144 training actions**, **8,192 optimizer.step calls**, **two fresh fits**. Record actual monitor actions, resets, rewards, physical-event *counts* and *unique affected episodes*, action concentration, checkpoint numeric hashes, versions, source/config SHA, measured wall time, and interrupted partial actions.
- Examine priority: zero trained valid toggle opportunities → navigation/exploration hypothesis; positive opportunities with zero valid attempts → action selection or precondition representation; attempts but no openings → actuator/effectiveness; physical openings but no goals → post-door navigation. This is descriptive diagnosis, NOT a causal mechanism proof.
- **STOP after TRAIN-MONITOR stage.** Do not touch M10 independent final layouts or train/tune on M1–M9 held-out seeds; separately freeze the exact representation comparison once the diagnostic is audited. The M10-P0 representation hypothesis remains open. Physical task success requires key pickup, effective door opening and true rewarded goal, *not reward alone*.

Runner path `bagi/experiments/m10_train_only_diagnostic.py`, immutable config `configs/m10_train_only_diagnostic.yaml`, ignored outputs `runs/m10_train_only_diagnostic/`, and the existing serialized `bagi-research-loop.yml` (no new workflow or schedule). One-shot request after independent push+PR source CI. Generated NPZ and progress remain Actions artifacts only. An interrupted run is not a completed study.

At preregistration: **0 new training actions, 0 optimizer updates, 0 new fits, 0 monitor/independent episodes, 0 teacher queries**. A source-staging commit does not satisfy the scientific gate. Main integration is HOLD pending reviewed PR and current-source validation.
