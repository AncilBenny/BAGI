# Gate A — maintained recurrent navigation acquisition

Registered before execution on the isolated Gate A branch. Parent
6e0691c471013bbe11c642c708a429564378c233, CartPole diagnostic run 45934263710, artifact
15170142407. RecurrentPPO 2.9.0 reached 20/20 physical successes per
seed, whereas original recurrent replay, 8x update rate and frozen-source
DQN each reached 0/20 per seed. This is exploratory single-family
competence, not cumulative learning. No reason to repeat CartPole now.

## Hypothesis, reuse and design

Does the same maintained recurrent algorithm acquire native navigation
under the existing public encoding? Reuse the existing
bagi.experiments.gate_a_competence runner, its native Gym wrapper and
AgentRuntime evaluation. Do not modify the policy, actions, reward, task,
optimizer or prior pilot. No new algorithm. MiniGrid-Empty-Random-6x6-v0
with 100-step horizon; strict physical success is actual goal tile arrival
on native termination. Only public local image, prior executed action
and interface/episode indicators reach the policy. Native goal truth
remains evaluator-only. No teacher, intrinsic shaping or hidden map.

Three model seeds 71/73/79. Training reset streams begin at
11,000,000 / 12,000,000 / 13,000,000, and 20 exploratory monitor resets
per seed start 500,000 later. Distinct from prior CartPole and Gate A pilot
seed namespaces; repeated monitor layouts may recur and are not independent
confirmatory data. No held-out confirmation data will be used.
Exactly 65,536 training actions per seed, 256-action rollouts, evaluation
every 4,096 actions including zero. This larger diagnostic budget is
chosen for sparse navigation reward, not a training ceiling or a matched
cross-family efficiency comparison. Same PPO setup as CartPole, 64 batch,
4 epochs, LSTM hidden 64, CPU. Report actual actions, presentations
(excluding recurrent padding), optimizer updates, native success, curve AUC,
seed-level spread, time, versions and source; all training and monitor
actions audited. Existing runner stores trajectories, not persistent PPO
checkpoints; that is a separate engineering prerequisite for transfer.

Target: 16/20 final exploratory successes for every seed, as registered
for the preceding diagnostic. If attained, implement/test persistent
maintained PPO state and a common legal action interface before a matched
sequential transfer test. Existing CartPole and MiniGrid action counts
differ (2 versus 7), so simply swapping the environment is invalid.
Do not silently reset the policy, optimizer or action head, or use hidden
information to reconcile interfaces. Compare maintained continual methods
and predeclare independent physical layout/initial-condition partitions,
fresh/frozen controls, transfer AUC, retention and seed-level uncertainty.
If target fails, examine native trajectory success, exploration and public
observation sufficiency before selecting another curriculum or budget.
No claim of transferable skill and no confirmatory inference from this run.

Marker [gate-a-navigation] runs exactly one existing read-only push CI
after full Ruff/pytest with pinned RL dependencies and native smoke.
No additional scheduler; preserve the pending reviewed integration.
