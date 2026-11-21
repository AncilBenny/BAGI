# Phase 2 — Imagination and Model-Predictive Planning

## Question

Can a learned dynamics model reduce the amount of **real environment interaction** required to achieve useful behavior?

## Planner

The model-predictive planner receives the current local Workshop observation and samples many short candidate action sequences. Each sequence is rolled forward only inside the learned world model.

For every imagined trajectory it accumulates discounted predicted reward:

`score = r0 + gamma*r1 + gamma^2*r2 + ...`

The planner executes only the **first** action of the highest-scoring imagined trajectory in the real environment. It then observes reality and plans again from the new observation.

## Accounting

BAGI deliberately reports two different quantities:

- **real steps** — transitions obtained from the actual Workshop; these are the scarce experiences we want to minimize;
- **imagined steps** — model rollouts computed internally; these cost computation but no new environment experience.

The planner's total real-step count includes the transitions that were required to train its world model plus its evaluation interactions.

## Run

```bash
make collect
make baseline
make world-model
make planner
make compare
```

Or run the full first experiment:

```bash
make experiment
```

## Interpretation

The first planner is intentionally simple. It does not yet have recurrent memory, uncertainty-aware exploration, latent-state planning, or online world-model updates. Its purpose is to establish the experimental protocol and reveal where the current world model fails.

A negative result is useful. If imagined planning does not beat the DQN baseline at equal real-interaction budgets, inspect whether the failure comes from world-model accuracy, sparse reward, partial observability, planning horizon, or distribution shift during imagined rollouts.

## Next hypotheses

1. Predict events and value, not only immediate reward.
2. Add uncertainty estimates so the agent distinguishes known dynamics from model hallucination.
3. Add recurrent latent state so planning can carry hidden information through time.
4. Train the world model online from newly collected experience.
5. Use curiosity/information gain when predicted task reward cannot distinguish candidates.
