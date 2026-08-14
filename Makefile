.PHONY: install test lint demo collect baseline world-model planner compare experiment check

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

baseline:
	python -m bagi.experiments.train_dqn --config configs/baseline.yaml

world-model:
	python -m bagi.experiments.train_world_model --config configs/world_model.yaml

planner:
	python -m bagi.experiments.run_planner --config configs/planner.yaml

compare:
	python -m bagi.experiments.compare_agents

experiment: collect baseline world-model planner compare

check: lint test
