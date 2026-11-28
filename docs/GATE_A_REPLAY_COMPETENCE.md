# Gate A recurrent replay competence — exploratory registration

Source basis: validated persistent learner 42bc45cb0f936ddee5f47fbb0b3e98f63ac4bfee.
The active-objective study failed its registered retention gate; do not tune it further.
Compare maintained TorchRL Double DQN with BAGI's existing recurrent GRU Q learner as
an internal control, not a causal architecture ablation. Both use the same 994-wide
public encoder, legal native action indices, capacity 256 complete episodes per family,
epsilon .2, 4 sampled episodes per update, one native runtime per fit. Both bootstrap
on timeout/administrative cut, not true termination. Different recurrent architectures
and target update/loss settings remain limitations. No teacher or hidden state input.

Fresh independent single-family fits for A MiniGrid-Empty-Random-6x6-v0 and B
CartPole-v1 (100-step horizon), seeds 151/157/163. Per arm/family/seed 32768 actual
training actions, 256-action blocks, 8 optimizer updates per block, 8192-action
measurement intervals, 40 frozen greedy episodes per point. Reset stream
85,000,000 + family index * 10,000,000 + model-seed index * 1,000,000; monitors
start at base + 500,000. Exploratory repeated monitors, not independent layout
groups or held-out confirmation. Native success is goal arrival (A) or surviving
100 steps without failure (B), not reward or runtime end fact. Record per-seed
AUC, final success, seed SD, actual update and valid replay presentations
(variable lengths), evaluation actions, hashes, and seconds. Match exposure and
updates, not presentation count. Resume each fit from disk at midpoint; retain
final full learner and runtime bundles with file digests. Midpoint hashes retained
but scratch bytes removed after restore. No checkpoint bytes enter Git.

Totals: 393216 training actions, 12288 optimizer updates, 2400 frozen episodes.
Gate: maintained learner final physical success >= .90 on every seed in both
families before attempting a sequential carry/replay experiment. If not, use the
curves and native traces to identify competence bottleneck before claiming transfer.
The internal GRU is a comparison, not a required scientific superiority threshold.
No scientific PASS, no held-out access; no new schedule. Full CI before native run.
