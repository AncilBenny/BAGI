# Gate A — exploratory single-family competence diagnostic

Source parent 0db1f41fdfcfab9fad96381d2fb9ef62786f2661: full independent CI #45933753548 passed
(Ruff, 914 passed, 10 skipped, 47 subtests). No independent PR #1 review.
Keep the development branch isolated. The initial Gate A pilot had floor
performance; transfer/forgetting cannot be inferred. No confirmation data opened.

## Hypothesis and controls

The original recurrent learner received only 4 optimizer steps per 256 native
actions. At matched native action budgets, additional updates may improve
acquisition. Compare original recurrent replay (4 updates/block), the same
learner (32 updates/block), existing frozen-source RuntimeDQN (32 updates/block,
64 transitions/update), and maintained SB3-Contrib RecurrentPPO 2.9.0
(256 rollout, 64 batch, 4 epochs, 64 hidden units). No new RL algorithm.
See existing REUSE_INVENTORY.md, reused_rl_baselines.py and
https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/ppo_recurrent/ppo_recurrent.py
for the maintained optimizer and masked recurrent batches.
Use actual optimizer post-step hooks and nonpadding presentation counts,
not epoch counters. DQN input scaling cancels its frozen /6 normalization.
Off-policy agents retain epsilon 0.2, LR 0.0003, gamma 0.99, target sync
every 20 actual optimizer updates. PPO uses upstream stochastic policy and
default LR/gamma; methods are not matched for compute or architecture.
All use the same public Gate A encoding, legal actions and native reward.
Off-policy training uses AgentRuntime; maintained PPO uses a Gym wrapper
with exact native step audit and the same encoding and forced block resets.
All frozen evaluation uses AgentRuntime, deterministic actions and native
physical success. PPO may bootstrap time-limit cuts differently from the
existing Q learners; do not interpret algorithm contrasts as isolated
architecture effects.

## Exploratory registration before execution

First run CartPole-v1 with 100-step horizon. Actual survival through 100 steps
without termination is success, never reward alone. Native MiniGrid-Empty-Random-6x6
is an explicitly selectable subsequent diagnostic with goal-tile arrival
criterion; not automatically run on this push. Fresh model seeds 71/73/79.
For family index d and model seed index i, train reset stream starts at
11,000,000 + d*10,000,000 + i*1,000,000; monitor resets start 500,000 later,
20 per checkpoint. Training exact 32,768 native actions per seed/arm, with
256-action blocks and monitoring every 4,096 actions including zero.
This is a matched diagnostic budget, not a project training ceiling.
No final held-out data; these monitor seeds are explicitly exploratory and
repeated across checkpoints. Reset-seed separation is not a proof of
layout independence. Before confirmation reserve separate physical layout
and initial-state groups and preregister them; no reuse of these monitors.

Record each seed's physical success curve, normalized trapezoidal success AUC,
episode length, actions, updates, sample presentations, time, source/version,
actual action hashes, and midpoint disk-resume checkpoint digest (recurrent
arms). Distinguish replay presentations from on-policy rollout presentations.
Use seed-level paired differences and uncertainty descriptively; n=3 cannot
establish confirmatory superiority. Competence diagnostic target: at least
16/20 monitor successes in each seed at final checkpoint; if achieved by a
maintained baseline but not recurrent replay, reuse the baseline or isolate
the remaining optimization/exploration gap before transfer. If no arm reaches
it, check observation/termination/optimization and budget adequacy; do not
claim a transfer result. If dense beats original, follow up with a matched
presentation/compute diagnostic; avoid claiming a causal update-frequency
effect beyond this configuration. No world-model expansion. No fixed future
training ceiling or new scheduler.

One-off marker [gate-a-competence] invokes the existing read-only CI after
Ruff and full pytest (including optional pinned RL smoke). It runs only on
the isolated Gate A branch. Output is an artifact, never a source commit.
No new result is claimed until exact-source artifact and action counts verify.
