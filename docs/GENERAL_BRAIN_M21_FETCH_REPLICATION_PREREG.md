# M21 preregistration — independent replication of the M20 physical Fetch gate

The first DoorKey6x6→Fetch5x5-N2 comparison M20, executed at source
`9d5705e1d846fcf6a02611663eae033322455ca0`, passed all three numerical
gates on four model seeds. However forward Fetch improvement was only
+8/128 goals (37/128 naive versus 29/128 scratch), exactly the threshold,
and target absolute success remained under 30%. One narrow within-family
success is insufficient to infer task-independent continual learning.

**Locked before M21 training:** literally replicate M20's maintained MiniGrid
environments, public observation/action feature encoder, M14 DoorKey controller,
M20 public Fetch teacher and independently physical carrying/identity auditor,
M15 PyTorch behavior cloning, M19 matched replay controls, Adam 0.003,
batch 128, 10 epochs, 384 action outer horizon, same five arms and two
reference policies. This is a protocol-specific parameter choice for matched
comparison, not a general training quota. No methods, evaluation definitions,
thresholds, or model selection may be altered using M20 FINAL data.

Predeclare four previously unused model seeds **811/821/823/827**.
DoorKey TRAIN 331001–331048; MONITOR 332001–332012; frozen FINAL
333001–333032. Fetch TRAIN 341001–341048; MONITOR 342001–342012;
frozen FINAL 343001–343032. These are disjoint from M20 and all known
previous sealed finals. M21 FINAL remains closed until train-only acquisition
and all 20 registered fits complete. Require ≥8/48 teacher nonvacuous
correct-object Fetch pickups on TRAIN only; otherwise archive native action
evidence and stop *this study* before model fitting/finals.

**Same conjunctive primary criteria as M20, never adjusted posthoc:**
(1) pooled sequential naive−target scratch ≥8/128 externally audited
correct Fetch pickups+real movement+reward and positive on ≥3/4 model seeds;
(2) pooled DoorKey source-replay−equal-extra-target ≥8/128 external
full physical goals and positive on ≥3/4 seeds; (3) pooled Fetch
naive−source-replay deficit ≤8/128. Report per-seed contrasts,
reward-versus-physical differences, native actions, total optimizer
updates, actual fits, numeric checkpoint/training-data hashes,
exact run source commit/config hash and artifact digest. Failure is
negative replication evidence and does not erase M20.

Statistical interpretation: the 128 target outcomes reuse the same 32
held-out physical layouts across four model seeds and five arms; model-seed
and shared-layout dependence prevents treating 128 as fully IID. Compare
per-seed paired differences; avoid claims of significance without
uncertainty quantification or cross-family studies. M20 and M21 final
seeds remain closed after evaluation. Task-independent learning beyond
a shared MiniGrid engine, scripted teacher, or physical-world competence
is not established by any gate pass. Workshop and XLand remain diagnostics.

Reuse references:
- Farama MiniGrid maintained Fetch: https://minigrid.farama.org/environments/minigrid/FetchEnv/
- Rolnick et al. 2019, Experience Replay for Continual Learning:
  https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html

Execute only through the existing serialized `bagi-research-loop.yml`
using a single source-bound one-shot request after independent Ruff/pytest,
current push CI and PR merge-provenance pass. No competing workflow, generated
output in source, arbitrary quotas, or unreviewed merge into `main`.
