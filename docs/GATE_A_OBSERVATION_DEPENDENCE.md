# Gate A — frozen observation dependence and efficient physical goals

Registered before execution. The previous exact-source policy audit
798bbd9e642db9d02444e5f9e9a04808ecd3aae4, run 45940638160, artifact
15172477979, failed the preregistered stochastic competence target:
95/70/87.5% success, frequency-matched random 87.5/92.5/85%.
No further fresh fit is needed to probe these saved policies.

Reuse the existing PPOControl, native AgentRuntime, FrozenActor and
MemorylessActor. Load only the verified policy/Adam tensor bundle with
restricted loading, exact source, package versions, summary and per-policy
digests. This tests weight/Adam restoration, not exact simulation or RNG
resume. No optimizer update or new training actions. Compare live partial
image, first image held fixed throughout each episode (the actual action
history and other existing public input slots remain live), and the
same frozen last-TRAIN-action prior. Stale image is deliberately
out-of-distribution; a difference indicates sensitivity to the intervention
but no difference alone cannot prove the network ignores visual inputs.

Use new exploratory reset streams 34000000/35000000/36000000, 60 each;
stochastic sampling reset+40000000, prior reset+50000000. Pair modes on
reset seeds; no final independent physical layout group opened. Native
100-action horizon, physical goal tile arrival as before. In addition
report goal arrival within 20 actual steps and restricted mean actions
(failures counted as 100). This is a new prospective efficiency probe,
not a retrospective replacement of the failed policy-audit criterion.
Report seed-specific paired differences, variability and native action
traces. If live image does not consistently improve over stale and prior,
the next acquisition experiment should reuse Farama OneHotPartialObsWrapper
on the same local image instead of treating categorical IDs as ordinal
numbers; no hidden map or goal-direction wrapper. The maintained
MiniGrid one-hot representation changes input width/parameter count,
so document that limitation and use fresh matched controls. If live
image improves efficient goals, investigate exploitation of that
information before changing encoding. Neither result authorizes transfer
confirmation; the independent layout and physical initial-condition
partitions remain open.

SB3 set_env requires compatible observation/action spaces. RecurrentPPO
does not provide standard replay or the recurrent action-masking support
of non-recurrent MaskablePPO. Do not alias illegal actions, reset model
heads or treat stale PPO rollouts as off-policy continual replay. CLEAR
(Rolnick et al., NeurIPS 2019) uses off-policy correction and behavioral
cloning; evaluate maintained implementations separately. References:
https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html
https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
https://proceedings.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html
https://minigrid.farama.org/main/api/wrappers/

Existing read-only push CI, marker [gate-a-observation-probe], must
pass Ruff and pytest before downloading the exact prior artifact and
executing the frozen probe. No new scheduler, no native training.
