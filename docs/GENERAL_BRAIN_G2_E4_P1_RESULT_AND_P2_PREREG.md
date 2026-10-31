# G2-E4-P1 — independently audited learned-competence FAIL; prospective P2 policy-selection test

**Date:** 2026-10-25. **Research branch:** `setup/initial-platform` only. **G2-E4-P1 status: FAIL / STOP**, not a general-intelligence, continual-learning, or task-specific learned-competence success. This document records the completed *unmodified* E4-P1 experiment and a separate prospective P2 hypothesis; **no new native P2 run is authorized by a documentation-only commit**. Never retune or rerun E4-P1 against the closed TEST seeds `26000..26009`.

## Authoritative G2-E4-P1 native result and provenance

- Registered implementation commit `2ee1baabd36f92611a34efcfa8a28809b3d59234`, native GitHub run **40212708157**, job **121227196572**, **COMPLETE SUCCESS** (software/execution only); push CI **40212708166** and PR CI **40212712762** SUCCESS. Native artifact **12606845219**, `bagi-g2-e4-p1-learned-competence`, verified ZIP SHA256 **`b3cecce2f9cb15092490f893ba8b0df16a9424a6d1b010747b07094cdccfa712`** matching the GitHub artifact digest. Earlier source-run `40212257485` failed Ruff C408 **before training** and produced no artifact; do not count it as science.
- Independent read-only inspection of all **12 numeric-only NPZ** traces (`numpy.load(allow_pickle=False)`) and `summary.json`: **65,536 TRAIN** native actions (4×16,384), **23,110 TEST** native actions, **88,646 total recorded** under frozen max 89,176; **0 teacher queries**. All **23,110 TEST** action/reward rows verified against episode boundaries, paired starts, native reward, termination and source reports. **15,760/15,760** frozen trained/untrained PPO TEST decisions reproduced from archived public pre-action observations and stored numerical actor weights. TRAIN transition arrays were **not archived**; training action count is cross-checked against source recorder and JSON, not independently reconstructed action-by-action. No extra simulator actions, optimizer calls or pickle loads were required by the audit.
- Each TEST seed set `26000..26009` had **9 distinct physical layouts out of 10** for MiniGrid `Empty-Random-6x6` and **10/10** for `DoorKey-5x5`, verified via evaluator-only physical-layout hashes, kept outside actor inputs. Both models were trained on only the public flattened local MiniGrid image and native sparse reward. Ten separate reset seeds are not automatically ten statistically independent task classes.

| Task | PPO model seed | Actual rewarded TRAIN goals | Uniform random frozen TEST | Initialized-untrained PPO TEST | Trained deterministic PPO TEST |
|---|---:|---:|---:|---:|---:|
| Empty-Random-6x6 | 17 | 159 | 3/10 | 0/10 | **0/10** |
| Empty-Random-6x6 | 29 | 201 | 3/10 | 0/10 | **0/10** |
| DoorKey-5x5 | 17 | 12 | 0/10 | 0/10 | **0/10** |
| DoorKey-5x5 | 29 | 12 | 0/10 | 0/10 | **0/10** |

- **384 rewarded TRAIN episodes, but 0/40 trained deterministic TEST goals.** Random TEST achieved 6/40 overall. The preregistered navigation and DoorKey gates both FAIL. A rewarded TRAIN episode demonstrates a native outcome under the *sampled training behavior*, not independent competence of the deterministic frozen policy. The experiment does **not** establish whether a frozen stochastic policy would succeed.

## Newly verified, read-only probabilistic policy postmortem

An independent NumPy-only implementation of the standard two-layer `Tanh` actor used each archive's actual `public_pre_action` TEST images and numeric policy weights, then applied categorical softmax. No simulator, gradient/optimizer, model load from pickle, or additional TEST trajectory was executed. All **15,760/15,760** recorded argmax policy actions still reproduce. Reproduction: `python -m bagi.experiments.e4_p1_distribution_audit <frozen-artifact.zip> --output <postmortem.json>`. Source hash is pinned in code; five synthetic tests cover uniform and biased distributions, action parity, shape and unsafe-array rejection.

| Trained policy / seed | Most chosen frozen TEST action | Mean probability of TOP action on visited observations | Mean categorical entropy / log(7) | Distinct visited public pre-action views |
|---|---|---:|---:|---:|
| Empty-Random / 17 | `forward` action2, **1440/1440** | **0.592682** | **0.705489** | 17 |
| Empty-Random / 29 | `forward` action2, **1440/1440** | **0.447016** | **0.828913** | 17 |
| DoorKey / 17 | `toggle` action5, **2499/2500** | **0.231673** | **0.969811** | 9 |
| DoorKey / 29 | `left` action0 **1011/2500**, `pickup` action3 **1489/2500** | **0.191966** | **0.988933** | 20 |

**Interpretation:** the deterministic **argmax** action can repeat relentlessly even when the underlying stochastic policy still assigns meaningful probability to *other actions*. The DoorKey policies are particularly close to high-entropy distributions on the **greedy policy's already visited observations**. This differentiates *argmax action-selection degeneracy* from *near-zero stochastic entropy*, but does **not** identify the training cause, establish learned planning, or show what probability distribution would occur on states reached by a stochastic rollout. All statistics above are **retrospective heldout diagnostics only**: never use them to tune the existing E4-P1 weights or evaluate further on its closed TEST set.

## Separate prospective G2-E4-P2: existing frozen weights, new independent starts

**Status: preregistration only; runner/CI not implemented or dispatched by this note.** Scientific hypothesis: some G2-E4-P1 task success during TRAIN may persist in the frozen **stochastic** PPO actor despite zero **greedy** goal completions. Alternative: its stochastic success is no better than an untrained network/uniform random baseline, so no learned competence is established. Use only inherited neural weights from the single official E4-P1 artifact pinned above—**zero retraining, zero optimizer updates, zero exploration-bonus changes, no expert labels**.

1. Freeze tasks: `MiniGrid-Empty-Random-6x6-v0` and `MiniGrid-DoorKey-5x5-v0`; existing trained model seeds **17 and 29**; **10 untouched new TEST seeds `27000..27009`** per task/model/arm, never used in E4-P0 or E4-P1. Same trained weights in two matched arms: (A) deterministic greedy `predict(deterministic=True)`, (B) stochastic **original PPO categorical policy** `predict(deterministic=False)` with independent, preregistered per-episode torch/PPO sampling seeds. Additional same-start controls: (C) stochastic identically initialized **untrained PPO numeric weights**, and (D) uniform random actions with separate registered seed streams. All four arms see identical public observations under the *same physical reset seed*, but trajectories diverge naturally after choices. Preserve conditional control RNG isolation and matched action budgets.
2. Use standard upstream `stable_baselines3==2.9.0` policy with state dict loaded **only from dtype-checked numeric NPZ arrays**, without `torch.load`, unsafe pickle or executable checkpoint. Fetch only the pinned original GitHub Actions artifact and verify ZIP SHA256 and all inner NPZ SHA256 against source `summary.json` BEFORE any new reset/action. Reject missing/extra weights, nonfinite tensors, mismatch of keys/shapes, or inconsistent task/model. Actor receives local **public image only**; evaluator-only grid+position checks reset identity but never reaches policy, reward or action selection.
3. **Interaction ceiling:** 2 tasks ×2 model seeds ×4 arms ×10 starts × native task caps (Empty 144, DoorKey 250) = **31,520 real TEST actions maximum**, plus **0 TRAIN actions**. No discarded counterfactual native steps. Log all real actions including early terminations, native reward/termination, per-episode reset/layout hashes, and numeric-only trace arrays; keep stochastic random seeds and reproducible saved weights. Stop on budget or information-boundary violations.
4. **Prospective skill-feasibility gate, NOT proof of continual learning:** for **both trained seeds**, require ≥8 physically distinct TEST starts/10. For Empty navigation, frozen stochastic trained must score ≥5/10 actual native goals **and** exceed each greedy, untrained-stochastic and uniform-random control by ≥3/10 on matched starts. DoorKey secondary gate is ≥2/10 and ≥2/10 above each control for both seeds. If results fail, **STOP**; do not retune test-time sampling, entropy, exploration bonus, seeds or old weights based on frozen results. Any pass is *task-specific policy-selection competence* only, not cross-domain weight transfer or retained general brain skills. Don't conflate repeated stochastic success, native reward, and learned goal-conditioned navigation without independent replication.
5. **Guard before native P2 run:** source-contract tests must confirm *exact E4-P1 numeric weight identity*, native reset/step budget, public-only actor inputs, deterministic/stochastic arm parity on the same weights, randomized sampling independence, trained/untrained controls and explicit TEST seed disjointness. The code used for this prospective gate is **not yet written or approved**; do not dispatch a native workflow until these prerequisites pass. Software CI is not scientific evidence.

**G2-E4-P1 scientific decision:** FAIL/STOP. Preserve all numeric evidence, include negative results in append-only log and **do not** claim the General Brain can reliably learn, plan, retain knowledge or transfer across unrelated tasks. One later bounded, independently preregistered P2 trial may isolate test-time action-selection versus actual learned-policy competence without modifying training. No action-selector trick is being promoted into BAGI's general core by this note.

## G2-E4-P2 preregistration refinement before any new simulator actions (2026-10-25)

The separately committed read-only preflight module `bagi/experiments/e4_p2_frozen_contract.py`
freezes independent action-sampling seeds as `27_000_000 + task_index*1_000_000 +
model_index*100_000 + arm_index*10_000 + (reset_seed-27000)`, with task index in
the published TASKS order, model index [17,29], arm index in the registered P2
arm order and TEST reset seeds 27000..27009. This is fixed **before any P2
native reset**. At the start of each stochastic policy episode, the eventual
runner must seed upstream PyTorch categorical sampling with this exact seed;
uniform random must use a separate NumPy generator initialized from the
corresponding registered seed. Greedy selection does not sample. No
heldout-driven tuning, reuse of 26000..26009, or extra trials are permitted.

The preflight validates the pinned P1 ZIP SHA256, all 12 inner NPZ SHA256
digests, original task/model/seed/action metadata, numeric-only dtype and
finite values, required actor tensors, and exact state-dict names/shapes/dtypes
against a freshly constructed upstream SB3 policy before any simulator
reset. It does **not** load a pickle or run PPO. This step supplies only
provenance and software checks, **not P2 performance evidence**. No native P2
workflow or runner is included in this preparatory commit.
