.PHONY: install test lint demo collect guided-collect baseline world-model planner compare diagnose causal-experiment sample-efficiency sample-efficiency-multiseed planner-trajectory-diagnostics crossed-planner-seeds experiment check

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

sample-efficiency:
	python -m bagi.experiments.sample_efficiency --config configs/sample_efficiency.yaml

sample-efficiency-multiseed:
	python -m bagi.experiments.multi_seed_sample_efficiency --config configs/sample_efficiency_multiseed.yaml

planner-trajectory-diagnostics:
	python -m bagi.experiments.planner_trajectory_diagnostics --config configs/planner_trajectory_diagnostics.yaml

crossed-planner-seeds:
	python -m bagi.experiments.crossed_planner_seed_eval --config configs/crossed_planner_seed_eval.yaml

experiment: collect baseline world-model planner compare

check: lint test
