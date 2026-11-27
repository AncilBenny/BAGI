# Gate A — critic adaptation after Adam reset

Registered before new B actions. Exact predecessor 03771b7/run
45968732146: Adam-only reset improved B greedy AUC 17.5/17.8125/50.3125
percentage points versus carry, final A all 40/40, B 40/40,34/40,40/40.
Still negative AUC versus fresh all seeds. Adding B-head reset failed
per-seed retention (seed137 A 32/40) and did not consistently increase
AUC. Keep Adam-only control; no claims of positive transfer.

Hypothesis: carried critic estimates a different reward scale (MiniGrid
sparse reward under 1 vs CartPole reward 1 per step). Actual saved A
value-head weight norms .9564/.9575/.9706; after Adam-only B training
4.2464/4.1513/4.4622. Descriptive, not proof. Separate actor and critic
already exist in maintained RecurrentPPO. Test one new arm: clear Adam,
copy only critic LSTM, critic MLP and scalar value head from same-seed
fresh policy, preserve actor weights, parameter identities, optimizer
object, counters and RNG. No architecture or reward changes.
Record actual public B reset values before/after intervention and
at final on 40 B monitor resets per seed; 360 resets, zero probe
simulator steps. Compare descriptively to realized discounted returns
from live trace (finite horizon, no bootstrap). Exact initial A/B
greedy and stochastic native traces must match cached Adam-only.

Same verified historical A checkpoints, carry/fresh and Adam-only
control summary digests, same versions, 3 seeds, 32768 B actions each,
8192 monitor interval, same 40 repeated A/B monitor episodes, seed
streams, physical goals and native evaluation. New training 98304
actions, 6144 optimizer steps, 393216 on-policy presentations,
separate evaluation/reset accounting. Report per-seed greedy/live
AUC vs Adam-only, carry, fresh; A final/backward/worst forgetting,
B final, value estimates, actual counters, runtime, saved checkpoints
and action hashes. Cached comparisons exploratory, no final groups.

Decision: require >=10pp B greedy AUC gain vs Adam-only in each seed
and <=10pp final A loss each before favoring critic reset. Quantify
interim forgetting and fresh comparison separately. No positive
transfer claim unless exceeds fresh per seed and independent
confirmation follows. If not helpful, prioritize direct actor
plasticity/representation analysis and validated continual baseline
over arbitrary resets. No maintained continual baseline or lifetime
replay yet. No schedule or PR #1 change; full CI before training.
