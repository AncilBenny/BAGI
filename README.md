# BAGI — Building an Adaptive General Intelligence

BAGI is an experimental research project exploring **sample-efficient, continually learning agents**.

The goal is to build a small learning organism that can:

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

We will measure environment interactions, transfer, generalization, forgetting, and planning gain—not only final reward.

## Phase 0 — The Workshop

The first environment is a lightweight 2D laboratory with partial observability and reusable causal rules: doors, switches, keys, movable blocks, hazards, and energy sources.

The first milestone is intentionally simple: **build the laboratory before building the organism.**

## Run

This repository is designed to work locally or in GitHub Codespaces.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m bagi.demo
```

## Roadmap

- M0: 2D Workshop environment
- M1: model-free baseline
- M2: predictive world model
- M3: recurrent memory / partial observability
- M4: planning through imagined trajectories
- M5: curiosity / uncertainty-driven exploration
- M6: transfer to unseen layouts
- M7: compositional tasks
- M8: continual learning and forgetting tests
- M9: multi-agent learning

## Status

Early experimental research platform. Expect architecture and APIs to change rapidly.
