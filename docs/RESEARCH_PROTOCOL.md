# BAGI Research Protocol

The project is judged by **learning efficiency and transfer**, not only final task reward.

## Every experiment records

- code commit SHA
- configuration file
- random seed
- number of real environment interactions
- training compute/runtime when relevant
- evaluation metric(s)
- result artifacts and plots
- a short failure analysis

## Core metrics

1. **Steps-to-learn** — real environment transitions required to reach a competence threshold.
2. **Transfer ratio** — learning speed on a related task after prior training versus a fresh agent.
3. **Generalization** — performance on unseen layouts and object combinations.
4. **Planning gain** — performance difference with versus without learned-model rollouts.
5. **Forgetting** — retained performance after learning subsequent tasks.
6. **Prediction quality** — world-model error and, later, uncertainty calibration.

## Experiment discipline

- Change one important hypothesis at a time.
- Always retain a simple baseline.
- Report environment interactions separately from wall-clock speed.
- Prefer small reproducible runs before scaling compute.
- Save generated outputs under `runs/`, `results/`, or `checkpoints/`; these are intentionally ignored by Git.
- Promote only configs, code, summaries, and small plots/tables worth preserving into version control.

## Initial sequence

### E0 — Environment sanity

Verify deterministic causal mechanics and partial observations.

### E1 — Random interaction dataset

Collect transitions using the configured random policy. The dataset validates logging and becomes a first input for dynamics-model experiments.

### E2 — Model-free baseline

Train a small baseline agent and establish steps-to-learn.

### E3 — World-model prediction

Train an encoder/dynamics model to predict next observations or latent states and salient events.

### E4 — Memory

Compare feed-forward and recurrent agents under partial observability.

### E5 — Imagination

Use learned dynamics for short-horizon planning and compare real-environment sample efficiency against E2.
