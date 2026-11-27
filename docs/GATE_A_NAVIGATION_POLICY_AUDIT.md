# Gate A — navigation policy extraction audit

Registered after navigation acquisition run 45938959497 and before fresh
evaluation. Exact predecessor 989e94ab0e18e16b28e35eae8ea175b503efa92f. The maintained PPO
learner got 2/20, 1/20, 3/20 final greedy physical successes; every
greedy action after the first checkpoint was forward (native 2).
Training stochastically achieved 3541 physical completions in 4830 native
episodes/fragments. Hypothesis: policy extraction rather than complete
absence of acquisition; stochastic behavior may merely be a biased random
walk. Need a memoryless action-frequency control, not a new algorithm.

Reuse exact prior PPO setup, source training reset streams, seeds 71/73/79,
65536 native actions per seed and 4096-action intervals. The previous run
did not retain policy weights, so a fresh fit is justified to obtain
previously unavailable frozen comparisons and retained model tensors.
Do not repeat CartPole or off-policy arms. All policy observations, native
actions, rewards, physical goal criteria and episode limits unchanged.
No teacher or privileged state. No hidden map, shaping or architecture
change. Maintained SB3-Contrib RecurrentPPO 2.9.0.

New exploratory reset seed streams 31000000, 32000000, 33000000, 40
consecutive resets each. At 0, 16384, 32768 and 65536 training interactions,
compare deterministic argmax with independently sampled frozen stochastic
policy, paired reset seeds, torch RNG fork seeded by reset+40000000.
No evaluation updates or training RNG consumption; assert weights and
counters unchanged. At final compare both with a uniform random and
a memoryless distribution matched to the last 4096 actual TRAIN actions
for that model (no monitor data), local RNG reset+50000000. Report
seed-level physical successes, curves, paired contrasts, actual actions,
optimizer updates, on-policy presentations, source, package versions,
and retained policy and optimizer tensor digest. Preserve checkpoints in
read-only artifact, not source; restricted tensor load in smoke.
The artifact is not a full exact PPO resume bundle.

Exploratory target remains >=80% physical success each seed. Require
stochastic policy to beat matched memoryless prior by >=10 percentage
points in each seed to justify using learned state for navigation rather
than mere action bias. Report all contrasts even if negative; three model
seeds cannot establish confirmatory superiority. If stochastic does not
beat prior, next probe public observation representation and/or a
maintained MiniGrid recipe; do not promote random-walk success as
competence. If it does, select a fixed evaluation policy prospectively
before independent physical layout/start-state groups and a transfer
registration. These repeated monitor reset seeds may reproduce layouts;
no final held-out layout or initial-condition data opened.

One-off marker [gate-a-policy-audit] uses existing read-only CI after
Ruff and full pytest with pinned RL dependencies. No new scheduler.
