# CartPole replay acquisition mechanism probe (exploratory)

Completed recurrent competence source 7e884e7fc6e8fb02678cfd6d2fd3b76129d7be61,
run 46141697260, artifact 15262685472, audited archive SHA256
6ad3ba9f8e543433afb736d03a2162fe56ebeca11bd692b53674a334f3683038.
Neither learner reached competence. TorchRL CartPole training had 1/8934
physical successes; internal GRU 0/9471, all final greedy successes zero.
Navigation has nonzero successful exploration but unstable greedy competence.
Diagnose CartPole first because it has dense native reward and the prior PPO
controls established competence. No claim that exploration or update density
alone caused failure. Reuse existing epsilon-greedy and TorchRL Double DQN.

Three new arms: epsilon .9 -> .1 linearly over first half of training, then .1
(exploration, 8 updates/block); fixed epsilon .2 and 64 updates/block (density);
both changes combined. Historical .2/8 control from audited artifact, not a new
fit. Same validated learner, public observations, legal executed actions,
256-episode capacity, 4 episodes per update, target/optimizer and reward.
Match native actions and explicitly vary optimizer/replay budget; do not label
increased compute as sample-efficiency superiority without accounting.
Epsilon constant within each 256-action block. Seeds 151/157/163; 32768 actions
per fit; greedy 40-episode monitors every 8192 actions, same reset and monitor
streams as historical CartPole. Pin historical dependency versions and require
matching initial greedy action-trace digest for each seed before fitting.
Repeated monitors are already inspected, not independent final evaluation.
Checkpoint and resume at midpoint; final file digests; log all real training
actions, frozen evaluation actions, updates, valid replay presentations and
per-block epsilon. No hidden physics input, teacher or reward modification.

Planned 294912 training actions, 52224 optimizer updates, 1800 frozen episodes.
Qualify an arm only if all three final greedy successes >= .90. This is a
mechanism diagnostic, not a confirmation. Inspect AUC and paired differences,
actual compute and seed variability; if none qualifies, prefer a maintained
feed-forward DQN sanity control on the same interface before further recurrent
hyperparameter work. Navigation competence still required before sequential
transfer. No reserved evaluation groups or new schedule.
