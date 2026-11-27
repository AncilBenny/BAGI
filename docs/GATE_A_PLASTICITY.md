# Gate A — plasticity intervention diagnostic

Registered before new B actions. Exact sequential pilot 31a149a,
run 45953275096: carry-minus-fresh greedy B AUC -0.552083 across
three seeds, navigation retention 119/120, B success 50/120 vs
120/120 fresh. No positive transfer; investigate plasticity rather
than expanding a world model. Adam state in each A checkpoint has
4096 optimizer steps. Loss of plasticity and primacy bias motivate
testing optimizer/parameter history, not a priori attribution:
Dohare et al. Nature 2024 doi:10.1038/s41586-024-07711-7;
Nikishin et al. ICML 2022 https://proceedings.mlr.press/v162/nikishin22a.html.

Two new arms from identical source A checkpoint and lifetime RNG:
(1) clear Adam state only, preserving optimizer object, groups,
parameters, counter and RNG; (2) clear Adam and copy only the two
B actuator output rows from a same-seed fresh shared policy, restoring
RNG after constructing it. Preserve A head, recurrent and value
weights exactly. Not a proposal to discard accumulated knowledge or
a maintained continual-learning baseline. At B=0 A greedy native
trace must exactly match previous carry. Reuse same architecture,
native switch, optimizer schedule, source-bound loader and frozen
evaluator; compare digest-verified cached carry and fresh results,
no unnecessary rerun of those conditions. Cache confounds possible
runner/library nondeterminism; exact versions and source bound.
Identical historical A weights, B streams, model seeds 131/137/139,
32768 B actions/arm/seed, monitor interval 8192, same 40 repeated
exploratory A/B reset streams and deterministic/stochastic evaluation.
No confirmation data opened. New actions 196608, updates 12288,
on-policy presentations 786432; all other historical counts excluded.
Retain new final checkpoints, actual executed traces and optimizer
counters. Compare paired B greedy/stochastic AUC differences vs
carry and fresh, final B success and A final/backward/worst forgetting,
seed mean/SD. Report incremental B-head effect vs Adam-only by seed.

Decision: materially improved plasticity requires >=10 point B
greedy AUC gain over carry in every seed, A greedy final loss <=10
points each. Positive transfer requires carry intervention exceed
fresh in every seed; neither is confirmation. If neither helps,
examine representation interference, value/gradient/activation
diagnostics and maintained continual-learning alternatives before
further interventions. No replay claim, no arbitrary repeated fits,
no world-model expansion, no new scheduler or change to PR #1.
Full CI before execution.
