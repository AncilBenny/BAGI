# BAGI — Building an Adaptive General Intelligence

BAGI is an experimental research project exploring **sample-efficient, continually learning agents**.

The goal is not to claim AGI. The goal is to build a small learning organism that can:

1. observe a partially observable world,
2. remember useful information,
3. learn a predictive world model,
4. act and observe consequences,
5. plan using learned dynamics,
6. explore to acquire useful information,
7. transfer learned structure to new situations,
8. improve while minimizing real environment interactions.

## Research question

> How much useful capability can an agent acquire per unit of experience?

We measure environment interactions, transfer, generalization, forgetting, planning gain, and prediction quality—not only final reward.

## The Workshop

The first laboratory is a lightweight 2D partially observable world with reusable causal rules. It currently includes a switch that opens a door and a key that opens a lock. More compositional mechanisms will be added as the learning system matures.

## Run in GitHub Codespaces

Open the repository in a Codespace. The dev container installs the project automatically.

```bash
make check
make demo
```

To generate interaction data:

```bash
make collect
```

This writes transition data under `runs/workshop/`.

## First learning experiments

### 1. Model-free baseline

The DQN baseline is deliberately small and CPU-friendly. Its job is to establish a reference curve for reward versus **real environment interactions**.

```bash
make baseline
```

Outputs are written to `runs/dqn/`, including `metrics.jsonl` and the learned Q-network checkpoint.

### 2. Predictive world model

First collect transitions, then train the model:

```bash
make collect
make world-model
```

The world model learns:

```text
(observation, action) -> predicted next observation + predicted reward
```

Validation metrics include next-cell prediction accuracy and reward mean absolute error. Outputs are written to `runs/world_model/`.

## Why both?

The DQN baseline answers: **how many real interactions does a conventional model-free learner need?**

The world model starts testing our central hypothesis: **can learning the structure of the world allow an agent to get more capability from each real interaction?**

The next milestone is to use the learned dynamics model for short imagined rollouts and compare its environment-step efficiency against the model-free baseline.

## Roadmap

- M0: causal 2D Workshop environment — in progress
- M1: model-free baseline — implemented
- M2: predictive world model — implemented
- M3: recurrent memory / partial observability
- M4: planning through imagined trajectories
- M5: curiosity / uncertainty-driven exploration
- M6: transfer to unseen layouts
- M7: compositional tasks
- M8: continual learning and forgetting tests
- M9: multi-agent learning

## Research discipline

Every major experiment should state a hypothesis, fixed experience budget, baseline, metric, result, and failure analysis. Faster compute is not the same as better sample efficiency.

See `docs/research-protocol.md` for the experiment template and measurement rules.

## Status

Early experimental research platform. APIs and architecture will change as experiments invalidate assumptions.
