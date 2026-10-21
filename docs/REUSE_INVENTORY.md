# BAGI — Reuse-first scientific integration map (2026-10-21)

Purpose: avoid writing algorithms, benchmarks or experiments that established
research projects have already supplied. This is a **verified external-artifact
inventory and compatibility decision**, not an assertion of novel science.
Read docs/RESEARCH_PROTOCOL.md for the mandatory pre-implementation gate.

## Tested and integrated

| Existing project | Maintained capability | Provenance and license | BAGI integration status |
|---|---|---|---|
| Farama MiniGrid 3.2.0 | Standard partially observable Memory, DoorKey, BabyAI and other task families with Gymnasium environments. | Official https://minigrid.farama.org/; version confirmed in GitHub run 39570422800. | **Working**: registered MemoryS13Random and DoorKey-6x6 under their own environment definitions and sparse rewards, using \`ImgObsWrapper\`. |
| Stable-Baselines3 2.9.0 | Standard PPO model-free RL, established evaluation APIs. | Official https://github.com/DLR-RM/stable-baselines3 and https://stable-baselines3.readthedocs.io/; upstream MIT. | **Working**: optional version-pinned \`rl\` extra, \`PPO\` imported directly. |
| SB3-Contrib 2.9.0 | RecurrentPPO / LSTM on Gymnasium Box observations and discrete actions. | Official https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html and https://pypi.org/project/sb3-contrib/; upstream MIT. | **Working**: \`RecurrentPPO\` imported, LSTM state carried each step, reset at episode boundaries. |

These are **external algorithms/benchmarks**, not original BAGI inventions.
Our only integration code is a Gymnasium observation adapter, a reproducible
train/evaluate invocation, metrics and tests.

### Tested smoke result: GitHub Actions 39570422800

- Four distinct models: PPO and RecurrentPPO on each of MemoryS13Random and
  DoorKey-6x6, one seed (17) and exactly 512 training environment interactions
  **per model**. Three independent-seed evaluation episodes per model.
- All four trained, evaluated and produced an artifact successfully.
  Success: 0/3 for all four. This is **too small to rank algorithms, assert
  memory has failed, or establish an efficiency ceiling**.
- Upstream training runtime: Memory PPO 0.237 s, recurrent 0.459 s;
  DoorKey PPO 0.231 s, recurrent 0.418 s, on the GitHub runner. Package
  installation overhead is separately visible in workflow logs.
- Artifact \`bagi-reused-rl-baselines\` on run 39570422800 contains real training
  and evaluation step counts and exact library versions.

## Existing trained systems and reference recipes to assess next

**RL Baselines3 Zoo — reference PPO training and pretrained DoorKey agent.**

- Upstream source: https://github.com/DLR-RM/rl-baselines3-zoo
- Reference MiniGrid hyperparameters:
  https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
- Published trained agent:
  https://huggingface.co/sb3/ppo-MiniGrid-DoorKey-5x5-v0
- Its model card reports a *self-reported* mean reward of 0.97 ± 0.01 on
  DoorKey-5x5, with a 100,000-interaction PPO training recipe. This is a
  **third-party claim**, not a BAGI replication.
- **Compatibility/security blockers before reusing its weights**: the older
  published checkpoint was created with the legacy
  \`gym_minigrid.wrappers.FlatObsWrapper\` and \`VecNormalize\`. BAGI's smoke test
  deliberately used current \`minigrid.ImgObsWrapper\` and did not normalize.
  Observation dimensions, mission representation, normalization statistics,
  package versions and policy architecture must all match, not merely the
  task name. Legacy checkpoint archives or pickle normalization files must
  not be deserialized into a privileged GitHub Actions job or run alongside
  repository write credentials without provenance and a sandboxed audit.
- **Decision**: investigate a controlled, safe compatibility path before
  training a duplicate DoorKey specialist from scratch. Otherwise use the
  maintained RL Zoo training recipe directly as a known performance control
  and report version/benchmark differences.

**CleanRL's PPO-LSTM.**
- https://docs.cleanrl.dev/rl-algorithms/ppo/
- Clear documented single-file reference, but its published recurrent variant
  targets Atari pixel observations. Use if the existing SB3-Contrib interface
  proves inadequate; do not rewrite PPO/LSTM.

**DreamerV3.**
- https://github.com/danijar/dreamerv3
- Established world-model imagination architecture. Do not port it into BAGI
  without an unresolved world-model research question and a fair comparable
  compute/interaction budget; initial discrete Gymnasium baselines are
  much lighter.

**TD-MPC2.**
- https://arxiv.org/abs/2310.16828
- Established model-predictive control via learned latents; evaluate its
  actual task/action compatibility before any integration. Never rebuild
  TD-MPC2 just to call it an BAGI model.

## Next genuinely BAGI-specific research question

Can memory or model-based planning deliver useful **held-out causal transfer
and interaction efficiency**, beyond a competent established policy, under
independent task seeds or changed causal combinations? BAGI does **not** need
to reproduce algorithms whose capabilities already have strong published
evidence; it needs to show where standard systems fail under this specific
goal, with an information-matched and computationally honest comparison.

**Next gate before expensive training:** verify whether a *compatible*,
well-documented, properly licensed pretrained baseline can be evaluated with
matched observation/normalization contracts. If not, re-run existing
RL Zoo recipes rather than invent a new training stack. Treat all claims from
external model cards as unverified until a pinned reproducible evaluation.

### Automatic-agent rule

Before every future iteration read this inventory and the newest research log.
Look up **available upstream reference code/checkpoints** first; compare
compatibility, license, version, compute and known limitations. Do not use
the next GitHub Actions run merely to train an established baseline again
without identifying a new open question. Record any negative result, not
only model victories.

## Pinned pretrained DoorKey artifact (REUSE-2)

- Hugging Face model repo: https://huggingface.co/sb3/ppo-MiniGrid-DoorKey-5x5-v0
- Exact revision: `90b0f9903be52f016a4f9f378d027f441350444f`
- Model ZIP SHA-256: `8343ddfedf676da11b05407fabe3e8a5ede175a04ebc32731ab670ff713be5e2`; upstream size 4,390,847 bytes, verified from the source manifest (but download not yet locally audited).
- Upstream `results.json`: stochastic mean reward 0.96508 across 10 episodes, published March 2023. This is **not** BAGI's measured result.
- Unlike BAGI's image-only pipeline, the upstream recipe uses legacy `gym_minigrid.wrappers.FlatObsWrapper` and observation `VecNormalize`. Do not deserialize the upstream `vec_normalize.pkl` in a privileged runner.
- Automated dedicated read-only static auditor: `bagi/experiments/pretrained_checkpoint_audit.py`; fail-closed ZIP and SHA256 inspection with zero model execution. Results saved as JSON in GitHub artifact after the job completes. Use that verification before proposing inference.

### REUSE-2 measured compatibility verdict

- GitHub report: Actions run `39572829556` / artifact `bagi-pretrained-compatibility-audit`. The pinned upstream ZIP downloaded and matched SHA256 **exactly**. Original SB3 version embedded in model is `1.8.0a11`; the model's JSON includes serialized Python spaces/policy fields that were not decoded.
- Official current Farama MiniGrid `FlatObsWrapper` yields **2,835** features on DoorKey-5x5; BAGI's `ImgObsWrapper+FlattenObservation` yields **147**. Both action spaces have 7 actions. Thus **no direct model compatibility**. The upstream `vec_normalize.pkl` was not downloaded/deserialized; original normalization remains unresolved.
- Normal CI and the isolated read-only audit all passed at `8d6c3ba`. No pretrained inference or reward claimed.
- **Next use of effort:** locate a newer safely loadable compatible pretrained agent, or establish the exact trusted, isolated Zoo environment+normalization migration. Do not implement another PPO, create new planners, or expend 100k training steps to reproduce an existing specialist without a research reason.


### REUSE-3 — reuse the maintained FlatObsWrapper instead of writing a legacy adapter

- Historical source checked directly in the upstream Farama Minigrid repo at
  \`v2.1.1:minigrid/wrappers.py\` versus \`v3.2.0:minigrid/wrappers.py\`:
  https://github.com/Farama-Foundation/Minigrid/blob/v2.1.1/minigrid/wrappers.py
  https://github.com/Farama-Foundation/Minigrid/blob/v3.2.0/minigrid/wrappers.py
  Both flatten the 7×7×3 image (147 values), then append 96×28 encoded
  mission character features (2,688 values), total 2,835. The modern
  implementation uses uint8 where older code uses a float32 temporary for
  mission characters; numerical 0/1 feature ordering appears the same.
- A dedicated REUSE-3 test checks actual Farama \`FlatObsWrapper\` output
  against its *own* underlying image and one-hot mission encoding on three
  independent reset seeds. This is a reproducible representation contract,
  not a claim of full historical emulator or policy reproducibility.
- **Do not create a replacement encoder**: reuse the modern maintained
  upstream \`FlatObsWrapper\`. The 147-vs-2835 mismatch arises because
  BAGI's 512-step learner used \`ImgObsWrapper\`; that test is not intended
  to load this external agent. We still need separately verified
  normalization parameters and SB3 checkpoint version adaptation.
- Security review: SB3's model loading behavior has an upstream
  unpickling issue report https://github.com/DLR-RM/stable-baselines3/issues/2281 .
  Static inspection of a SHA-256-matched ZIP does not itself make
  \`PPO.load\` or \`VecNormalize.load\` safe. No such untrusted deserialization
  is authorized on credentialed runners.


## REUSE-4 — a real compositional task-distribution benchmark, not another Workshop variant

- Established research: *XLand-MiniGrid: Scalable Meta-Reinforcement Learning
  Environments in JAX*, NeurIPS 2024 Datasets and Benchmarks Track,
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/4d8a58a62c1a85ebb2cbea798c2779e2-Abstract-Datasets_and_Benchmarks_Track.html .
  Upstream Apache-2.0 implementation https://github.com/dunnolab/xland-minigrid ,
  PyPI \`xminigrid==0.9.3\`. It provides existing JAX goal/rule objects, multi-task
  generators, held-out task splits and existing recurrent PPO baselines.
- High-leverage adoption: stop expanding a single handcrafted Workshop as if
  it represented arbitrary compositional transfer. Keep Workshop for debugging;
  use XLand-MiniGrid for systematic variation of causal rules and goals.
- **Security constraint verified from source:** upstream
  \`src/xminigrid/benchmarks.py:load_bz2_pickle()\` invokes \`pickle.load\` on
  a benchmark retrieved from a remote Hugging Face dataset. Do not call
  \`xminigrid.load_benchmark\` on external task sets in privileged automation.
  Instead, first use upstream \`RuleSet\`, \`AgentNearRule\`, and \`AgentNearGoal\`
  in memory; no remote task artifacts or new benchmark implementation.
- Proposed first gate: 4 compositions pairing familiar source colors with
  familiar output colors in new rule combinations; two train-labelled versus
  two test-labelled task IDs with SHA-256 disjointness. Tiny random-policy
  environment contract only, no learned policy. Record actual real steps and
  goal events. Once validated, transfer experiments must compare existing
  recurrent PPO/other maintained agents with matched interactions and a
  defined task-instruction information policy. No large dataset download
  or million-step JAX training without separate evidence-backed justification.
- Optional existing dataset: XLand-100B (ICLR 2025) supplies recorded RL
  learning histories. Its public data runs to tens/hundreds of GB:
  https://github.com/dunnolab/xland-minigrid-datasets . Do not download
  these large files to GitHub Actions to repeat existing work.


### REUSE-5 confirmed task-solvability and detected invalid transfer contrast

- Success: GitHub Actions \`39578198961\` proves 12/12 seeded task episodes are solvable in 7–11 real simulator steps by a classical full-state navigation oracle, with transformation and pickup both observed.
- **Disqualifying caveat:** for each held-out seed, the same action sequence solves **all four** apparently different color recombinations. Distinct SHA256 ruleset IDs do not mean behaviorally different tasks. **Do not cite this split as a valid compositional-generalization benchmark or train costly agents on it.**
- Next benchmark must include genuine changes in causal behavior, potentially using existing \`AgentNearRule\`, \`AgentHoldRule\`, \`TileNearRule\`, multi-rule chains and distinguishable goal predicates, and must specify whether the agent sees a task description or must infer rules via experience. Prefer official upstream XLand-MiniGrid task families and published recurrent PPO baselines, after security review of any downloaded pickle task file. No bespoke RL rewrite.


## Latest verified integration checkpoint — REUSE-14 to REUSE-17 (2026-10-21)

- **Verified train-only DAgger expert callback:** existing privileged XLand BFS+goal recipe recovered **24/24** probed off-path training episodes (run `39603838885`); teacher inputs are NOT legal learned-policy inputs. A lightweight Gymnasium adapter allowed genuine upstream HumanCompatibleAI `imitation.SimpleDAggerTrainer` to execute two online rounds at 291 teacher queries+291 training interactions per seed (run `39605165443`), with 0 independent learned train/heldout goal completions using its ordinary BC supervision.
- **Controlled upstream BC sampler comparison:** on an IDENTICAL frozen DAgger label pool per policy seed, `imitation.BC` with PyTorch `WeightedRandomSampler` raised original-label pickup recall from 0 to ~95% and putdown recall 0 to 100% for BOTH model seeds. In independent eval only model seed17 completed hold 1/3 and near 1/3, both seed9101; seed29 0 for all tasks. No rewarded putdown or heldout chain (run `39619756922`).
- **Frozen-pool new-seed replication:** verified original 4 BC outcomes matched exactly; with six NEW evaluation seeds (9200–9205), model seed17 balanced achieved **3/6 hold, 2/6 near, 0/6 putdown, 0/6 heldout**, while seed29 balanced and both ordinary controls achieved **zero**; all source data numeric-only and SHA256 verified, **zero new teacher queries** (run `39621044776`). These are genuine but not yet robust source skill successes; no verified compositional transfer.
- **Build-versus-reuse:** keep using upstream XLand, imitation.BC/DAgger, PyTorch samplers; custom code only benchmark wrappers and causal diagnostics. The next bottleneck is correct multi-stage action SEQUENCING and policy-seed dependence, not another entire RL framework implementation. Use privileged XLand state only for recorded analysis events, never policy inputs; target genuine putdown-reward stages and independently seeded closed-loop success rather than teacher-label accuracy alone.


- **REUSE-18 (verified; artifact `12355312869`, run `39622825399`):** frozen balanced BC phase telemetry demonstrates seed17 hold and near rewards correspond to real pocket pickup and near-rule red→green transformations. The rewarding putdown task is still **0/6** for both model seeds on the six new layouts: seed17 issued 83 putdown actions, seed29 100, but neither executed an *effective* inventory drop or arranged a blue square next to a red ball. Rare action LABEL recall was fixed by class balancing; actual action PRECONDITION timing/sequence is not. Reused exact REUSE-16 pools, 0 teacher queries, no new training interactions, 2,711 evaluation steps on already-tested seeds (posthoc, not independent transfer test). See `docs/autonomous-research-log.md` for per-stage limitations.


- **REUSE-21 verified (run 39639959628):** upstream public rule/goal instructions improved hold 4/10 vs same-width sham 2/10 and near 1/10 vs 0/10; no putdown or heldout chain learned.
- **REUSE-22 verified (run 39643131034):** identical local+public-instruction observation at 3 rewarded pre-drop states and 3 non-reachable empty-pocket counterfactuals, though native action4 only rewarded real full-pocket states. 71 simulator calls, zero training. This establishes instant pocket is hidden, NOT that memory cannot infer it from legal history.
- **REUSE-23 verified (run 39644180240):** same 291-label provenance-verified pool per seed and 115-wide class-balanced instruction-conditioned BC, genuine preceding executed actions vs same-width PAD history produced IDENTICAL closed-loop success seed sets: hold 4/10, near 2/10, putdown 1/10 (first independent BC putdown on model seed29/layout9604), heldout 0/10 per condition. 2684 evaluation simulator steps, 0 fresh teacher queries. Next test: independent-layout replication, not scale model/compute.


- **REUSE-24 verified (run 39646704229):** Original REUSE-23 BC models reproduced old layout-success IDs, then tested 10 untouched seeds 9700–9709 per policy seed/condition. Zero rewarded putdown across all 40 runs; 502 pickup and 687 putdown ACTION attempts but only one native effective pickup+drop, with the WRONG RED ball and zero BLUE-square carries. Hold/near stayed fragile. 6437 real replay+fresh simulator steps, 0 new expert labels. Existing teacher `correction_plan` rejects a held non-square in putdown; next validate recovery from that exact REAL learner state (seed29, layout9708) before collecting more on-policy labels. See `docs/RESEARCH_DECISION_GATES.md`.
