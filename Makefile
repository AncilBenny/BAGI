.PHONY: install test lint demo collect baseline world-model check

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

check: lint test
