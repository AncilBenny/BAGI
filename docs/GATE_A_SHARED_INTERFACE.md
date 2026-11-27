# Gate A — common actuator interface and native continuation

Registered before execution. Predecessor 27260b2b63cf2b130b2dd5d4ec0de2db25c03be0, run 45943741084:
categorical navigation reached 40/40 per seed, scalar failed. Two
competent task families previously used incompatible action spaces.

Reuse maintained RecurrentPPO MultiDiscrete([7,2]) support. Public input
994-wide: categorical local image 980 slots and existing 14 tail slots.
CartPole uses zero image slots, real four-vector and domain/action
history. Only active task channel connected; inactive proposals never
reach native simulator. Audit joint proposals and actual actions.
No hidden truth, action aliasing, shaping or teacher. PPO joint
logprob includes inactive factor; this is NOT action-masked PPO.
See https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html.

Native smoke must compare exact subsequent rollout executed traces,
weights, Adam and counters after restricted disk restore in each task.
Preserve Python/NumPy/Torch CPU RNG, seeded pending native reset,
actor/critic LSTM state, optimizer and lifetime counts. Same source,
versions and digest required, only completed auto-reset rollouts,
constant schedules. No mid-episode or cross-machine bitwise guarantee;
monitor-log history not restored. Native switch must retain exact model,
policy, optimizer and buffer identity and weights/moments; reset only
transient episode state. On-policy rollout buffer overwritten, no
historical replay or maintained continual baseline yet.

After smoke and full CI, fresh per-family acquisition: model seeds
131/137/139, 65536 actions per MiniGrid model and 32768 per CartPole
model, 256-step rollouts, 64 batch, 4 epochs, 64 hidden units. Native
100-step horizon and physical criteria unchanged. Training reset streams
51000000 + domain index*10000000 + model index*1000000; exploratory
monitor streams 500000 later, 40 episodes per seed. Greedy and stochastic
at zero, quarter, half and final budget; final last-TRAIN-frequency
prior, also stale image for navigation. Frozen no-update evaluation
through AgentRuntime. Midpoint disk resume; retain final checkpoints.
No final physical groups opened. Require final greedy and stochastic
>=80% per seed in each family, and navigation stochastic success by 20
>=10 percentage points above both stale and prior each seed. No by-20
criterion for CartPole. Report seed-specific curves, contrasts, actual
actions, optimizer steps, on-policy presentations, time, versions,
source, artifact digest. New fit justified by incompatible old heads,
changed action distribution and public observation width; not a
matched-parameter comparison.

This is a compatibility diagnostic, NOT a sequential transfer test.
If either family fails, diagnose shared interface before transfer.
If both pass, preregister independent physical groups, matched
carry/fresh/frozen and maintained continual-learning baseline, AUC,
retention, plasticity and uncertainty. No new schedule or world model.
PR #1 review pending; keep main and working source unchanged.
