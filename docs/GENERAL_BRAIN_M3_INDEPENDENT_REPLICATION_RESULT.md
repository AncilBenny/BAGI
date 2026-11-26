# BAGI-M3 — completed independent Memory replication result

**Observed 2026-11-11 UTC. This report is retrospective and does not change the original M3 preregistration.**

## Provenance and execution

- Registered study: `m3`; frozen source `99f7f194787c1f02014b692aed221957639b77e6`; config `configs/m3_goal_learning.yaml` SHA256 `820aab08b951b1447e897225522b382555a12cbbaca0a021169d08d289070824`. The preceding research commit and request were separate; the request was already admitted by exact-source push/PR CI and source audit.
- Successfully executed existing BAGI Research Loop: [run 42944942456](https://github.com/AncilBenny/BAGI/actions/runs/42944942456). Artifact `bagi-research-634`, ID **13823291153**, verified downloaded outer ZIP SHA256 `4aa0656b6dd6d9aacd8a187246014100cb71368f194e5c21682182618d860739`; artifact's internal research source marker equals the source commit. Run marker `2026-11-11T02:29:37Z`.
- Eight freshly trained models (four reused PPO/RecurrentPPO approaches times independent model seeds **37, 53**). Actual **524,288 native training actions**, **23,004 training resets**, **10,439 rewarded training goal terminations**, **16,384 intercepted optimizer.step calls**, **10,648 native evaluation actions**, **0 teacher queries**; wall time **501.866 seconds**. Each fit completed 65,536 native training actions and 2,048 optimizer steps. No new training actions or optimizer updates were performed in writing this result.
- Fixed training environment `MiniGrid-MemoryS7-v0`; untouched evaluation seeds **96001–96006**, zero-shot transfer `MiniGrid-MemoryS13Random-v0` seeds **97001–97006**. Every evaluation episode had the 120-native-action limit and scored a goal only on *actual environment termination plus positive native reward*. There were 6 episodes per model per condition.
- Upstream version markers: Gymnasium 1.4.0, MiniGrid 3.2.0, stable-baselines3 2.9.0, sb3-contrib 2.9.0. All eight archived numeric-only NPZ checkpoint SHA256 values independently matched the SHA256 recorded within the summary; checkpoints were not committed.

## M3 independent goals — native reward AND termination

Numbers below are successes out of six independent episodes *for each model seed*. Sampled policies are one seeded stochastic rollout per held-out episode, not a Monte Carlo expectation.

| Arm | Model seed | Training rewarded goals | MemoryS7 deterministic | Size13 deterministic | MemoryS7 sampled | Size13 sampled |
|---|---:|---:|---:|---:|---:|---:|
| reactive PPO | 37 | 1,338 | 2/6 | 1/6 | 2/6 | 2/6 |
| reactive PPO | 53 | 1,712 | 4/6 | 5/6 | 4/6 | 3/6 |
| action4 PPO | 37 | 1,824 | 4/6 | 5/6 | 5/6 | 5/6 |
| action4 PPO | 53 | 1,462 | 4/6 | 5/6 | 3/6 | 3/6 |
| frame4 PPO | 37 | 296 | 2/6 | 0/6 | 2/6 | 1/6 |
| frame4 PPO | 53 | 303 | 2/6 | 1/6 | 1/6 | 2/6 |
| RecurrentPPO | 37 | 2,292 | 2/6 | 1/6 | 5/6 | 1/6 |
| RecurrentPPO | 53 | 1,212 | 0/6 | 0/6 | 4/6 | 3/6 |

The untrained deterministic MemoryS7 baseline completed **0/12** per arm. Pooled trained deterministic MemoryS7 completions: reactive **6/12**, action4 **8/12**, frame4 **4/12**, recurrent **2/12**. Pooled zero-shot size13 deterministic: reactive **6/12**, action4 **10/12**, frame4 **1/12**, recurrent **1/12**. Pooled sampled MemoryS7: reactive 6/12, action4 8/12, frame4 3/12, recurrent 9/12; pooled sampled size13: 5/12, 8/12, 3/12, 4/12 respectively.

## Prespecified decision and interpretation

**H-M3 directional recurrent-memory transfer gate FAILS**: recurrent size13 deterministic 1/12 versus best nonrecurrent control action4 10/12 (**-9/12**); the second recurrent seed scored 0/6, also failing the each-seed success requirement. Stronger >=3/12 advantage necessarily fails. Same-task recurrent 2/12 versus strongest nonrecurrent 8/12 (**-6/12**). This independently contradicts the tentative M2 recurrent 8/12 transfer ranking, though M2 and M3 have different initializations and test episodes and their differences are NOT paired causal measurements.

**Secondary hypothesis-generating observation, not a new passed gate:** RecurrentPPO MemoryS7 sampled **9/12** versus deterministic **2/12** on the same fresh layouts; action4 was 8/12 in both modes. Because each sampled case has only one RNG draw and the model-seed counts differ, this does not prove a robust stochastic-control mechanism or a recurrent-memory benefit. Record it as motivation for an independent frozen-policy evaluation; do not retrospectively select M3 scores.

Training reward is not held-out goal competence. Action4's two seeds independently scored 5/6 each on size13 transfer, but only 12 held-out episodes and a single memory/navigation benchmark were tested. Model parameter counts differ: reactive 10,248; action4 11,784; frame4 37,960; recurrent 50,760. Native interactions and PPO rollout/batch/epoch schedule matched, not model capacity. No physical BLUE-square pickup/drop, tool task success, Workshop/XLand generality, task-independent cross-task learning, continual adaptation, or absence of catastrophic forgetting has been measured here.

## Next decision

Do not promote a recurrent memory claim. Preserve M1/M2/M3 and frozen held-out seeds. Preregister M4 as **evaluation of the existing eight frozen M3 policies** on untouched episode seeds, with repeated stochastic sampling and a deterministic baseline. Its first stage requires **no additional training**: it tests whether the deterministic/sampled gap generalizes before investing in further fresh fits. That study remains a proposal until a validated loader/runner and a separate verified execution artifact exist. See `docs/GENERAL_BRAIN_M4_FROZEN_POLICY_EVALUATION_PREREG.md`.

No new workflow or schedule was added and no checkpoint, generated output or credentials were committed by this documentation milestone. Integration into main requires reviewed PR, exact-HEAD Ruff/pytest and independently verified push/PR merge-provenance CI; CI checks from the source used for M3 do not validate this later documentation commit.
