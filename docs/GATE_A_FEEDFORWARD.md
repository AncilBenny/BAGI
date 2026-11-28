# Gate A feed-forward acquisition sanity control

Exploratory follow-up to audited acquisition run 46150801970, artifact 15266610529.
The fixed-epsilon high-density recurrent arm achieved .45/1/1 final success,
below the all-seed .90 gate. Compare a feed-forward public-observation MLP
(994 -> 64 ReLU -> 7 masked values) using the same maintained TorchRL Double
DQN loss, target update, Adam, episode replay, legal action projection, gamma,
epsilon .2 and four whole episodes per update. Reuse runtime and checkpoint
machinery; distinct checkpoint format prevents cross-architecture restore.
This changes representation and parameter count, not a pure recurrence ablation.

CartPole only, seeds 151/157/163, same training reset streams and inspected
40-episode monitors as historical density arm. 32768 actual actions per seed,
64 updates per 256 actions, evaluate every 8192 actions, midpoint disk resume.
Totals 98304 actions, 24576 updates, 600 frozen episodes. Record dependency
versions, actual presentations, per-seed AUC/final success and checkpoint hashes.
Historical comparator is not a new fit; compare versions before interpretation.
No held-out access, no scientific pass. All seeds >= .90 indicates the shared
interface supports acquisition without recurrence; otherwise inspect public
encoder and reward/TD diagnostics before additional tuning. Navigation remains
a separate prerequisite to sequential transfer.
