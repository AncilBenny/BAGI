# Gate A — live navigation rehearsal at matched budgets

Registered before new training. Actor probe 45f1ec4, run 45984235177,
artifact 15191600587: restoring A LSTM alone recovers failed critic seed
137 A from 10/40 to 40/40, but yields 0/40 greedy B for all six recipients
(including five previously competent). Restoring A MLP or A head alone
recovers only 12/40 in failed seed; full A actor restores exact initial
traces. Recurrent adaptation is necessary for B competence and can
interfere with A; do not freeze or graft the old recurrent actor as
a training solution. Component swaps are not causal training proof.

Test continued live A practice with existing maintained RecurrentPPO
and common public dual-actuator interface. Three seeds 131/137/139,
exact historical A checkpoints, one initial Adam reset (best prior
candidate), thereafter preserve all model, policy, optimizer and buffer
identities and moments. Four cycles of B 8192 native actions then A 2048.
No teacher, shaping, extra architecture or additional resets of Adam.
Use existing B reset streams (71/72/73 million), new exploratory A
practice streams (81/82/83 million), independent from old monitors.
On switch, reuse the prior pending reset seed (zero actions had been
executed from that reset) so executed episode seeds remain consecutive
within each domain; pending-reset calls may repeat, not action records.
Reset only transient LSTM state on switch. Record both cursors in summary
(final native boundary checkpoint records current domain only; resume
of the whole alternating schedule is not implemented). Verify exact
model/optimizer tensors across switches and native action/seed budgets.
On-policy rehearsal with fresh interactions, NOT historical replay.

Per seed 32768 B + 8192 A = 40960 new native training interactions.
Compare paired same-B-action AUC versus verified cached Adam-only and
fresh controls (extra A practice means total budget differs). For an
equal-total-action final comparison, restore the historical Adam-only
B final checkpoint and continue another 8192 B actions, producing
40960 B-only total, of which 32768 are cached; no new duplicate fits.
Equal total lifetime budget for rehearsal vs B-only, not equal B
exposure. All source/summary/checkpoint digests and versions verified.
At B=0, first B block end and extension start, require exact cached
A/B greedy and live traces. Frozen native evaluation at B=0 and each
B block end, and after each A block; extension pre/post, same paired
40 exploratory A/B reset streams and physical success. Measure
matched-B AUC before each rehearsal, final greedy/live retention and
control at equal total budget, worst forgetting across all probes.
No confirmation groups opened. 5280 new evaluation episodes,
147456 new train actions (24576 A,122880 B),9216 optimizer steps,
589824 on-policy sample presentations; cached control actions
excluded from new counts. Save six final checkpoints and traces.

Exploratory success criterion: final A and B greedy >=90% each seed,
worst measured A loss <=10pp each seed, and equal-total final B no
more than 10pp below B-only in each seed. Report B AUC versus Adam
and fresh at matched B actions separately; extra A data precludes
claiming positive transfer from that comparison alone. If live
practice helps, prioritize independent physical splits and historical
replay/maintained continual baseline with proper off-policy treatment.
If not, investigate targeted regularization or representation changes.
Cached controls can be affected by runner nondeterminism. No scientific
pass claim, independent confirmation or PR #1 integration. Full
independent CI before execution; no new scheduler.
