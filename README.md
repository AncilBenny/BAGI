# BAGI — Building an Adaptive General Intelligence

BAGI is an experimental project to build **one persistent, sample-efficient,
continually learning agent**. It is not a claim of AGI.

## Run the current agent foundation

In a Codespace or a Python 3.11+ environment:

```bash
pip install -e '.[dev]'
python -m bagi
ruff check .
pytest -q
```

The entry point runs the existing synthetic public-evidence and skill-recovery
contract through the common runtime. It is not a trained general agent and does
not demonstrate native environment transfer. The runtime now supports versioned
JSON skill-outcome checkpoints and frozen outcome-counter evaluation. The existing
DQN can now act through the same runtime and save a CPU-only local bundle containing
policy and target weights, optimizer, replay, RNG and runtime outcome counts.
Native sequential transfer and world-model persistence are not established.

## One system, one roadmap

- [Active roadmap](docs/ACTIVE_ROADMAP.md): current acceptance criteria and next work.
- [Architecture](docs/GENERAL_AGENT_ARCHITECTURE.md): public-only reusable brain.
- [Research protocol](docs/RESEARCH_PROTOCOL.md), [safety policy](docs/RESEARCH_SAFETY_POLICY.md)
  and [reuse inventory](docs/REUSE_INVENTORY.md): evidence and execution rules.
- [Append-only research history](docs/autonomous-research-log.md): previous results,
  including failed gates. They are not deleted or reclassified.

The existing experiments, models, environments, and workflows remain available
for reproducibility. Workshop and XLand are diagnostic environments, not the
definition of BAGI. Legacy commands such as `make demo`, `make collect`,
`make baseline` and `make world-model` are retained; see the Makefile.
New research should solve a specific missing capability of the shared runtime,
not accumulate isolated benchmark specialists.

## What success means

A single compatible learning state is carried across tasks and process restarts.
Native held-out task completion, transfer, forgetting, interaction efficiency,
optimizer updates and uncertainty are measured against appropriate controls.
Reward alone and synthetic contract tests do not establish physical competence.

Operational milestones require Ruff, pytest, exact current push and PR CI,
reviewed integration, and verification of main. Work in progress is tracked in
[convergence issue #4](https://github.com/AncilBenny/BAGI/issues/4).
