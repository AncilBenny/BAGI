# G2-E5-P4 upstream RND compatibility audit — STOP (2026-10-26)

**Decision:** the already-preregistered Workshop RND native experiment is **not authorized** for execution on the currently audited upstream RLLTE/RLeXplore source. This is a **read-only, source-pinned compatibility audit**, not a runtime import/synthetic integration pass or negative RND-learning result. It does not change the frozen P4 seeds, models, comparison arms, TEST seeds, budget or gate.

## Exact sources and static checks

- BAGI `pyproject.toml` at research HEAD `e4e0a61e5d158326150753331940f09e27d46b72` requires Python >=3.11, `matplotlib>=3.9`, and optional `stable-baselines3==2.9.0`.
- Upstream RLE-Foundation/rllte source commit `b5d55c826a449b235f773992e74e11b0ae2c7801` (2025-07-22), `pyproject.toml`, declares `rllte-core` version 1.0.0 and pins **`matplotlib==3.6.0`**, plus `torchvision`, `gymnasium[accept-rom-license,other]`, `seaborn==0.12.2`, `huggingface_hub==0.14.1`, `pynvml==11.5.0` and others. The mandatory `matplotlib==3.6.0` and BAGI `matplotlib>=3.9` specifiers have **empty intersection**; ordinary dependency resolution cannot satisfy both. The published PyPI package version 1.0.1 exists but its exact wheel metadata/runtime was NOT audited here; do not silently treat this source snapshot as proof that every release is incompatible.
- At that same pinned upstream source, `rllte/common/prototype/base_reward.py` initializes default `obs_norm_type="rms"` by calling `envs.reset()` and then stepping **128 × 20 = 2,560 times** to fill normalization statistics (with a fallback path on exception). These are real environment interactions if passed BAGI native Workshop; they violate P4's exact paired TRAIN-action and reset accounting if left unrecorded. Setting `obs_norm_type="none"` appears to avoid this initializer, **but does not resolve the dependency contradiction or prove a runnable integration**.
- The RND `compute(samples, sync=True)` contract uses batched `observations` / `next_observations`, and `update(samples)` may perform minibatch optimization for each call; matching P4's *one update per executed native transition*, SB3 2.9.0 rollout reward injection and no prior discarded probes requires an executable proof, not a guessed callback. The `ObservationEncoder` includes a vector-observation MLP path (`len(obs_shape)<=2`), so a flattened public 5x5 observation is **not** independently ruled out by the encoder shape.
- Static packaging-specifier check performed using `packaging.specifiers`: version 3.6.0 fails `>=3.9` and 3.9.0 fails `==3.6.0`. Local runtime lacks `gymnasium`, `stable_baselines3`, `rllte`, and network package installation; **zero upstream RND runtime smoke tests were claimed**.

Upstream references (source-pinned, accessible for independent audit):
- https://github.com/RLE-Foundation/rllte/blob/b5d55c826a449b235f773992e74e11b0ae2c7801/pyproject.toml
- https://github.com/RLE-Foundation/rllte/blob/b5d55c826a449b235f773992e74e11b0ae2c7801/rllte/common/prototype/base_reward.py
- https://github.com/RLE-Foundation/rllte/blob/b5d55c826a449b235f773992e74e11b0ae2c7801/rllte/xplore/reward/rnd.py
- https://github.com/RLE-Foundation/rllte/blob/b5d55c826a449b235f773992e74e11b0ae2c7801/rllte/xplore/reward/model.py
- https://pypi.org/project/rllte-core/1.0.1/

## Research decision and next bounded gate

**STOP P4 native training and do not add an RND workflow or a custom RND rewrite.** Next strictly preregistered *preflight only*: verify PyPI 1.0.1 exact wheel/sdist dependency metadata and available maintained releases; require an ordinary solvable pinned CPU-only dependency set compatible with BAGI, a synthetic no-environment-side-effect import/compute/update test, exact action-count proof with observation normalization disabled or *fully counted*, and upstream SB3 2.9.0 reward integration behavior. If these checks cannot pass without relaxing core BAGI dependencies or changing the preregistered algorithm, record STOP and consider a separately preregistered existing alternative, never TEST-driven tuning. No new claims about learned Workshop goal-reaching, cross-domain generality, physical BLUE competence or AGI.

**Resource accounting:** 0 Workshop/XLand native actions/resets, 0 TRAIN/TEST policy updates or held-out read, 0 teacher labels, 0 new workflows/dependencies/automations, 0 secrets/deployments/billing/main changes; one local packaging consistency calculation and remote public upstream source inspection.

## G2-E5-P4 reproducible PyPI wheel metadata check (2026-10-26)

The source-pinned RLLTE conflict above is verified for an upstream Git revision declaring version **1.0.0**. The latest PyPI release is **1.0.1** (May 25, 2024), with official wheel SHA256 **3a0ec99c27eae05def49fee79cf868bca298fbf2f7db9c3b90f9be3fd467d6b3** at https://pypi.org/project/rllte-core/ . Exact wheel `Requires-Dist` metadata was **not obtainable** in the network-restricted research runtime. Therefore 1.0.1 dependency compatibility remains UNKNOWN; do not silently transfer the source-1.0.0 incompatibility to its wheel.

A new, **standard-library-only, offline fail-closed** check is implemented in `bagi/experiments/g2_e5_p4_wheel_preflight.py`. Obtain the official PyPI 1.0.1 wheel in a network-permitted, non-installing review environment, then run:

```bash
python -m bagi.experiments.g2_e5_p4_wheel_preflight --wheel /path/to/rllte_core-1.0.1-py3-none-any.whl
```

It verifies the entire wheel SHA256 **before** reading ZIP metadata, validates distribution identity/version and checks only a provable unconditional `matplotlib==x` versus BAGI `matplotlib>=y` conflict. Optional/conditional/complex requirements are **indeterminate** pending a complete resolver. Every output includes `STOP_NO_NATIVE_TRAINING` and `authorized_to_train=false`, even without that narrow conflict; it does **not** import, install, extract or execute the package and cannot initiate native resets/actions. Five synthetic pytest fixture tests passed locally (hash tampering, wrong identity, conditional requirements, conflict and nonconflict); a **real 1.0.1 wheel was NOT inspected**. New branch CI lint/test remains the follow-up.

**Hard next gate:** full ordinary dependency resolution, pinned CPU-only synthetic import/reward/predictor-update smoke with a counted fake environment that fails on initialization resets/steps, then proof of SB3 reward integration and real interaction accounting. Until then no P4 native experiment/workflow, no custom RND rewrite and no training permission.
