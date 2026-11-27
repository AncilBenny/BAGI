# Gate A — frozen actor component interference diagnostic

Registered before new evaluation interactions. Critic adaptation
0c9ad91/run 45976850773 failed improvement and retention criteria:
seed 137 lost 75pp A and achieved 0/40 B. Adam-only stays baseline.
AUC gain vs Adam-only .18125/-.0875/.0625, not reliable.
Prior optimizer audit exactly reproduced inactive image-column drift
with zero gradients and stored Adam moments; Adam-only eliminates
that drift, yet active recurrent/MLP parameters change.

Use exact source A final checkpoints and final Adam-only and
Adam+critic checkpoints (3 seeds, 2 recipient conditions) with
digest-verified historical summaries and same package versions.
Six temporary evaluation-only states per recipient:
original B checkpoint, A LSTM only, A actor MLP only, A seven
navigation-head rows only, A path (LSTM+MLP+seven rows, retaining
B rows), and complete A actor (positive control). No critic changes,
no optimizer or counter changes, no training or saved grafts.
Native paired repeated exploratory A/B monitors, same 40 resets
per seed and domain as prior studies; greedy and stochastic.
Original B and complete A must reproduce all four cached native
trace sets exactly; A-path greedy navigation must reproduce initial
A traces. Temporary substitution must restore all original tensors
even on exceptions; frozen evaluator also checks optimizer, RNG,
model/environment identity and counters.

5760 new native evaluation episodes; record actual actions,
action hashes, source/recipient digests and actor parameter drift
by tensor. No new training actions, optimizer steps or presentations.
Report per-seed A/B success changes versus original recipient,
including seed137, and consistency across remaining seeds. Single
component rescue suggests a locus for further targeted tests, not
proof of a training mechanism. Component coadaptation can obscure
a rescue; A-path is a functional control, not evidence of transfer.
Do not select a new trainable architecture from one exploratory
seed alone. No held-out physical groups opened. No replay or
maintained continual baseline. Full CI before evaluation. Next
prioritize targeted actor intervention only if supported, and
independent physical splits plus compatible continual-learning
baseline before confirmation.
