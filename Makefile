.PHONY: install test lint demo collect guided-collect baseline world-model planner compare diagnose causal-experiment experiment check

install:
	python -m pip install --upgrade pip
	pip install -e '.[dev]'

test:
	pytest -q

lint:
	ruff check .

demo:
	python -m bagi.demo

collect:
	python -m bagi.experiments.run_workshop --config configs/workshop.yaml

guided-collect:
	python -m bagi.experiments.collect_guided --config configs/guided_collection.yaml

baseline:
	python -m bagi.experiments.train_dqn --config configs/baseline.yaml

world-model:
	python -m bagi.experiments.train_world_model --config configs/world_model.yaml

planner:
	python -m bagi.experiments.run_planner --config configs/planner.yaml

compare:
	python -m bagi.experiments.compare_agents

diagnose:
	python -m bagi.experiments.diagnose_planning --config configs/diagnostics.yaml

causal-experiment: guided-collect world-model planner diagnose

experiment: collect baseline world-model planner compare

check: lint test
