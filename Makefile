.PHONY: install test lint demo collect guided-collect baseline world-model planner compare diagnose causal-experiment sample-efficiency sample-efficiency-causal-v2 sample-efficiency-multiseed planner-trajectory-diagnostics crossed-planner-seeds causal-v2-horizon-sweep state-aliasing-causal-v2 causal-stage-progression experiment check

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

sample-efficiency-causal-v2:
	python -m bagi.experiments.sample_efficiency --config configs/sample_efficiency_causal_v2.yaml

sample-efficiency-multiseed:
	python -m bagi.experiments.multi_seed_sample_efficiency --config configs/sample_efficiency_multiseed.yaml

planner-trajectory-diagnostics:
	python -m bagi.experiments.planner_trajectory_diagnostics --config configs/planner_trajectory_diagnostics.yaml

crossed-planner-seeds:
	python -m bagi.experiments.crossed_planner_seed_eval --config configs/crossed_planner_seed_eval.yaml

causal-v2-horizon-sweep:
	python -m bagi.experiments.causal_v2_horizon_sweep --config configs/causal_v2_horizon_sweep.yaml

state-aliasing-causal-v2:
	python -m bagi.experiments.state_aliasing_diagnostic --config configs/state_aliasing_causal_v2.yaml

causal-stage-progression:
	python -m bagi.experiments.causal_stage_progression --config configs/causal_stage_progression.yaml

experiment: collect baseline world-model planner compare

check: lint test
