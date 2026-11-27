# Gate A — exploratory sequential pilot (not confirmation)

Registered before new B interactions. Exact previous shared-interface
acquisition 90322715afab01d068e16c64c00969563c7da5bf, run 45947152776,
artifact 15175703476 passed both task families. Reuse its three final A
checkpoints rather than spend 196608 new A actions. Exact archive and
summary/individual checkpoint digests verified; source-bound compatible
CPU load, same pinned package versions. No historical A actions counted
as new. A=MiniGrid Empty-Random-6x6, B=CartPole-v1, horizon 100.

Three matched model seeds 131/137/139, B 32768 native interactions per
arm/seed, 8192-action monitor interval. Carry keeps A weights, Adam and
lifetime state through native switch; fresh uses same shared PPO
architecture/seed, new optimizer; frozen keeps A checkpoint with no
updates, executes B through native runtime, cuts episodes at each 256
actions. All B streams start 71000000 + model index*1000000, same
distribution/seed ordering; episode lengths differ. Historical A training
reset streams as registered previously. Separate exploratory A/B monitor
streams +300000/+500000, 40 resets per seed, paired across arms/checkpoints,
greedy and stochastic; no final physical partitions opened. This is not
a physical-group generalization test: original A training covered random
starts in a single layout. Independent physical groups need fresh
partitioned fits before confirmation.

Evaluate A and B at B steps 0,8192,16384,24576,32768 through independent
native encoder and AgentRuntime, no set_env/learn in evaluation, preserve
training state and RNG. CartPole physical success = survive 100 steps,
MiniGrid physical success = goal arrival. Report action-weighted B AUC
carry minus fresh, zero-shot/final B differences, A post-minus-pre and
peak/worst forgetting, separate greedy/stochastic curves, seed-level
mean/SD/range, actual action hashes and accounting. Fresh A is descriptive
only, never called forgetting. Frozen B exposure is not training.
Expected new B exposure 294912, actual new training 196608, updates
12288, on-policy presentations 786432, separate evaluation. Cached A
196608 unique actions, shared across carry/frozen; not multiply counted.

Exploratory decision: if carry exhibits positive B AUC in all seeds
and <=10 point A greedy loss per seed, investigate confirmation and
plasticity under independent physical groups. Otherwise diagnose
interference/plasticity from paired curves and retained checkpoints.
Never claim scientific PASS. No persistent off-policy replay yet;
on-policy rollout buffer is not lifetime memory. A maintained continual
baseline (e.g. CLEAR-style corrected replay with compatible recurrent
actions, not uncorrected stale PPO samples) remains a prerequisite for
confirmatory comparison. Joint inactive-channel logprob still applies.
No architecture expansion, teacher, new scheduler or change to PR #1.
