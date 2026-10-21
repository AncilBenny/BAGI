"""Gate A8: compare closed-loop planning under a known exact horizontal reflection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from bagi.experiments.planner_uncertainty_ablation import STAGES
from bagi.experiments.symmetry_canonicalization import CanonicalMirrorModel
from bagi.models import WorldModel
from bagi.planning.model_predictive import ModelPredictivePlanner
from bagi.worlds import WorkshopEnv


class MirrorPlannerModel(CanonicalMirrorModel):
    def eval(self):
        self.model.eval()
        return self

    def predict(self, observations, actions):
        logits, reward, _ = self.forward_with_events(observations, actions)
        return logits.argmax(dim=-1), reward

    def predict_events(self, observations, actions):
        logits, reward, event_logits = self.forward_with_events(observations, actions)
        return logits.argmax(dim=-1), reward, event_logits.sigmoid()


def evaluate(model, env_cfg, layout, plan_cfg, planner_seed, episodes, eval_seed):
    env = WorkshopEnv(**env_cfg, layout_mode=layout)
    planner = ModelPredictivePlanner(
        model=model, actions=env.action_space.n,
        horizon=int(plan_cfg["horizon"]),
        candidates=int(plan_cfg["candidates"]),
        gamma=float(plan_cfg["gamma"]),
        seed=planner_seed,
        event_weights=[float(x) for x in plan_cfg["event_weights"]],
    )
    events_reached = {name: 0 for name in STAGES}
    real_steps = 0
    for episode in range(episodes):
        obs, _ = env.reset(seed=eval_seed + episode)
        attained = {name: False for name in STAGES}
        while True:
            action = planner.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            event_set = set(info["events"])
            attained["switch"] |= "switch_activated" in event_set
            attained["door"] |= tuple(info["position"]) == tuple(env._door)
            attained["key"] |= "key_picked_up" in event_set
            attained["lock"] |= "lock_opened" in event_set
            attained["goal"] |= "goal_reached" in event_set
            real_steps += 1
            if terminated or truncated:
                break
        for name, reached in attained.items():
            events_reached[name] += int(reached)
    return {
        "rates": {name: events_reached[name] / episodes for name in STAGES},
        "real_steps": real_steps,
        "imagined_steps": planner.imagined_steps,
    }


def run(config_file):
    config = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
    model_cfg = config["model"]
    env_cfg = config["environment"]
    experiment = config["evaluation"]
    env = WorkshopEnv(**env_cfg)
    model = WorldModel(
        observation_shape=tuple(env.observation_space.shape),
        actions=int(model_cfg["actions"]),
        cell_values=int(model_cfg["cell_values"]),
        hidden=int(model_cfg["hidden"]),
    )
    checkpoint = Path(model_cfg["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing original trained checkpoint: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    conditions = {
        "default_unadapted": ("default", model),
        "mirror_unadapted": ("mirror_x", model),
        "mirror_canonicalized": ("mirror_x", MirrorPlannerModel(model)),
    }
    seeds = [int(n) for n in experiment["planner_seeds"]]
    episodes = int(experiment["episodes_per_seed"])
    if not seeds or episodes < 1:
        raise ValueError("Need nonempty planner seeds and positive episodes")
    rows = []
    for name, (layout, predictor) in conditions.items():
        for seed in seeds:
            result = evaluate(
                predictor, env_cfg, layout, config["planning"], seed, episodes,
                int(experiment["evaluation_seed_base"]),
            )
            rows.append({"condition": name, "layout": layout, "seed": seed, **result})
            print(f"{name} seed={seed}: {result['rates']}, "
                  f"real={result['real_steps']} imagined={result['imagined_steps']}")
    means = {}
    for name in conditions:
        items = [row for row in rows if row["condition"] == name]
        means[name] = {
            stage: sum(x["rates"][stage] for x in items) / len(items)
            for stage in STAGES
        }
    output = {
        "hypothesis": "A known exact reflection adapter restores useful mirrored closed-loop planner behavior, not merely one-step model scores.",
        "checkpoint": str(checkpoint),
        "planning": config["planning"],
        "evaluation": experiment,
        "results": rows,
        "mean_stage_rates": means,
        "limitations": [
            "Candidate sequences are drawn from identical seeds but not reflection-paired; differences are exploratory.",
            "Reflection is hand-specified, not learned.",
            "All episodes within each layout have one fixed start and geometry.",
            "No training interactions are added and the original frozen checkpoint is reused.",
            "Goal success is the primary outcome; early stages alone do not establish planning.",
        ],
    }
    path = Path(experiment["output_dir"]) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mirror_planner_transfer.yaml")
    run(parser.parse_args().config)
