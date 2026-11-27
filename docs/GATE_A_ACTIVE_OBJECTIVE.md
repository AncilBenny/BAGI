# Gate A — active-actuator objective, engineering gate and prospective comparison

The environment executes only one of two independent categorical factors.
The original PPO optimizes the joint likelihood and entropy. This needlessly
includes the inactive factor in ratios, clipping, approximate KL and entropy.
Use the marginal likelihood of the executed factor in both rollout collection
and optimization; retain original joint sampling, native action projection,
value loss, architecture and ordinary PPO trainer. Scope to homogeneous
single-domain completed rollouts and pinned SB3/SB3-Contrib 2.9.0.
No changes to the underlying environment, model tensors or Adam on installation.
The function restores the ordinary distribution after each block; an explicit
experiment manifest must reselect the objective on resume. Shared-weight
interference and stored momentum are still possible; no claim of isolation.

Engineering gate: test inactive-logit gradients, invariance of probability
ratios to inactive changes, invalid proposals, same sampled joint proposal with
restored RNG, equal forward/evaluate likelihood, inactive head stability with
fresh Adam, native counts and optimizer identity. Require full independent
Ruff/pytest with RL extra. The initial CI marker only enables native contracts,
not the scientific study. Before executing the study, add actual buffered
old/new likelihood and two-restore next-active-rollout checks; verify no mixed
domain collection and no silent checkpoint mode change.

Prospective four-arm pilot, not executed: original joint vs marginal active
objective crossed with carried vs reset Adam. Each arm uses the same exact
historical A checkpoint per seed 131/137/139, B32768 native training actions,
B8192 measurement intervals, original paired 40-seed A/B greedy/live monitors,
same B reset streams and initial RNG. All four arms run in the same run;
do not substitute historical carry/Adam for contemporaneous controls.
Initial evaluation traces must reproduce exact source-bound A checkpoint.
4 x 3 x 32768 = 393216 unique B training actions, 24576 optimizer
steps, 1572864 on-policy presentations; 9600 frozen evaluation episodes.
Compare objective effect separately within each Adam condition, and interaction;
report per-seed paired B AUC differences, A worst and final forgetting,
B final physical success, greedy/live results separately and seed-level SD.
Pilot threshold: active objective improves greedy B AUC by >=0.10 in each
seed against matched joint objective for at least one Adam condition, with
final A greedy >=0.90 and final B greedy >=0.90 each seed, no A live final
loss >0.10 and no deterioration in B live AUC in any seed. Report interim
forgetting explicitly, do not treat end-point recovery as no forgetting.
This is an exploratory decision, not scientific pass; historical fresh
controls are descriptive only. Positive transfer requires contemporaneous
fresh/frozen/maintained continual controls and independent physical groups
with a predeclared confirmation protocol. If no arm qualifies, stop tuning
this objective and prioritize a maintained recurrent replay baseline.
No ordinary PPO historical replay, no held-out access, no new schedule.
