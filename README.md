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

We measure environment interactions, transfer, generalization, forgetting, and planning gain—not only final reward.

## Phase 0 — The Workshop

The first environment is a lightweight 2D laboratory with partial observability and reusable causal rules.

Current mechanics include:

- switch → opens door
- key → allows lock passage
- local 5×5 observation → partial observability
- goal completion
- deterministic transition/event logging

Later versions will add movable blocks, pressure plates, hazards, resources, layout randomization, and multiple agents.

## Fastest way to run: GitHub Codespaces

1. Open this repository on GitHub.
2. Choose **Code → Codespaces → Create codespace**.
3. Wait for the dev container setup to finish. Dependencies install automatically.
4. In the terminal run:

```bash
make check
make demo
make collect
```

`make collect` writes transition data to `runs/workshop/transitions.jsonl`.

## Local setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
make install
make check
make demo
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## Commands

```bash
make install   # install BAGI + development dependencies
make lint      # Ruff checks
make test      # pytest suite
make check     # lint + tests
make demo      # print and step through Workshop
make collect   # collect reproducible random-policy transitions
```

## Repository structure

```text
.devcontainer/          reproducible Codespaces environment
.github/workflows/      CI
bagi/
  experiments/          runners and transition logging
  utils/                reproducibility helpers
  worlds/                simulation environments
configs/                 experiment configurations
docs/                    research protocol and design notes
tests/                   automated environment tests
runs/                    generated experiments; ignored by Git
checkpoints/             generated models; ignored by Git
```

## Experiment discipline

Every serious experiment should preserve its seed, configuration, code revision, real environment interaction count, metrics, and failure analysis. See `docs/RESEARCH_PROTOCOL.md`.

## Roadmap

- M0: 2D Workshop environment and experiment infrastructure
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
