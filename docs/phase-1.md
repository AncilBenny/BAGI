# Phase 1 — Baseline and Predictive World Model

## Objective

Establish the first measurable comparison between a conventional model-free learner and a system that explicitly learns environment dynamics.

## Baseline

The DQN agent consumes the local Workshop observation directly and learns action values from real interaction. Its primary metric is reward versus cumulative environment steps.

This is deliberately not intended to be sophisticated. It is the reference point that future model-based agents must beat on sample efficiency.

## World model

The predictive model learns two targets from logged transitions:

1. the next local observation, represented as a categorical prediction for every visible cell;
2. the immediate scalar reward.

Its input is the current local observation plus the selected action.

## Initial metrics

- DQN: cumulative environment steps, episodic reward, training loss.
- World model: next-cell validation accuracy and reward mean absolute error.

## Experiment order

```bash
make collect
make baseline
make world-model
```

The first important result is not whether either model looks intelligent. It is whether we can produce stable, reproducible learning curves that let later architectural changes be evaluated against a fixed interaction budget.

## Next research step

Use the learned world model to evaluate short imagined action sequences before acting. Compare the resulting agent against the DQN baseline at equal real-environment interaction budgets.
