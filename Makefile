.PHONY: install test lint demo collect check

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

check: lint test
