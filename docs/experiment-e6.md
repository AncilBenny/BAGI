# E6 — Grounded imagination-gap diagnostic

## Hypothesis
The learned model's recursive rollouts become unreliable as the planning horizon grows,
particularly on actions chosen by the planner itself. Larger candidate sets may exploit this error.

## Procedure
Use the existing 50-guided-episode causal-v2 checkpoint (no parameter update).
Run guided, random, and model-planner policies in Workshop v0.2.
From real trajectory anchors, replay the *actual same* actions through the learned
world model recursively and measure prediction vs real environment outcomes
at depths 1, 3, 5, and 8. Record whole-observation accuracy, cell accuracy,
reward MAE, event Brier score and false-event probability.

## Interpretation
- Poor deep-rollout accuracy but good first-step accuracy suggests compounding error.
- Poor first-step causal-event predictions suggests a data/training problem.
- Disproportionate error on planner-generated trajectories suggests selection bias.
- Compare against an exact-model oracle **with matched observation information**
  before attributing all failure to learned dynamics.
- This diagnostic does not prove causal understanding, memory, or transfer.
- The fixed Workshop v0.2 map means these seeds are not held-out layouts.

## Resource policy
One-off GitHub Actions workflow on research-branch push or manual dispatch.
Use a cached checkpoint when available; otherwise regenerate within a 30-minute job cap.
Do not schedule repeated expensive retraining and do not merge to main.
