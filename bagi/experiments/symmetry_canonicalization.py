"""Gate A7: zero-training symmetry canonicalization for mirror-layout transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from bagi.experiments.reachable_state_holdout import enumerate_reachable, score_transitions
from bagi.models import WorldModel
from bagi.worlds import WorkshopEnv


def mirror_action(actions: torch.Tensor) -> torch.Tensor:
    """Actions up/right/down/left transform to up/left/down/right."""
    return torch.where(actions == 1, 3, torch.where(actions == 3, 1, actions))


class CanonicalMirrorModel:
    """Wrap an unchanged default-layout model with a horizontal reflection."""

    def __init__(self, model):
        self.model = model

    def forward_with_events(self, observations, actions):
        logits, rewards, events = self.model.forward_with_events(
            observations.flip(dims=(-1,)), mirror_action(actions)
        )
        return logits.flip(dims=(-2,)), rewards, events


def run(config_path: str) -> dict:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    env_opts = config["environment"]
    checkpoint = Path(config["model"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing frozen checkpoint: {checkpoint}")
    env = WorkshopEnv(**env_opts, layout_mode="mirror_x")
    model = WorldModel(
        observation_shape=tuple(env.observation_space.shape),
        actions=int(config["model"]["actions"]),
        cell_values=int(config["model"]["cell_values"]),
        hidden=int(config["model"]["hidden"]),
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    transitions, graph = enumerate_reachable(env)
    control = score_transitions(model, transitions)
    canonicalized = score_transitions(CanonicalMirrorModel(model), transitions)
    original, _ = enumerate_reachable(WorkshopEnv(**env_opts, layout_mode="default"))
    reference = score_transitions(model, original)
    metrics = ("cell_accuracy", "whole_observation_accuracy",
               "reward_mae", "positive_event_brier", "negative_event_brier")
    differences = {
        name: canonicalized[name] - reference[name]
        for name in metrics if canonicalized[name] is not None and reference[name] is not None
    }
    output = {
        "hypothesis": "Known reflection canonicalization restores original-layout one-step metrics without training.",
        "frozen_checkpoint": str(checkpoint),
        "graph": graph,
        "mirror_no_transform": control,
        "mirror_with_canonicalization": canonicalized,
        "default_layout_reference": reference,
        "canonicalized_minus_reference": differences,
        "limitations": [
            "Reflection is supplied by a human, not learned.",
            "This is a known exact symmetry only; no novel causal structure or unseen map family.",
            "Original layout uses a severely duplicate-heavy training dataset.",
            "No model retraining or new training interactions; evaluation simulator replay is counted in graph.",
            "No goal-achievement or policy-transfer claim.",
        ],
    }
    destination = Path(config["evaluation"]["output_dir"]) / "summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Mirror raw whole-view: {control['whole_observation_accuracy']:.4f}")
    print(f"Mirror canonical whole-view: {canonicalized['whole_observation_accuracy']:.4f}")
    print(f"Default reference whole-view: {reference['whole_observation_accuracy']:.4f}")
    print(f"Saved symmetry canonicalization: {destination}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/symmetry_canonicalization.yaml")
    run(parser.parse_args().config)
