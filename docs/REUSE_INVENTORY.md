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
