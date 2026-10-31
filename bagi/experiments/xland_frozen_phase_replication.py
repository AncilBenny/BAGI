"""REUSE-24: preregistered fresh-layout phase audit of frozen REUSE-23 BC.

Recreate BOTH 115-feature BC source policies at seeds 17 and 29 from exact
numeric REUSE-19 training-only labels. Assert original REUSE-23 success-seed
IDs before examining 10 untouched layouts. Same upstream BC/sampler/epochs;
no teacher or DAgger calls. Privileged simulator state is read only AFTER
each learner-chosen action for causal-stage telemetry.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from bagi.experiments.xland_bc_class_balance import balanced_indices
from bagi.experiments.xland_existing_bc import TRAIN_TASKS, get_rules, visible_features
from bagi.experiments.xland_instruction_action_history import (
    CONDITIONS,
    HISTORY_LENGTH,
    PAD_ACTION,
    WIDTH,
    assemble,
    check_true_history,
)
from bagi.experiments.xland_phase_diagnostics import inventory_and_grid
from bagi.experiments.xland_upstream_instruction_bc import (
    load_verified_source,
    task_instructions,
)

REFERENCE_SEEDS = (9600, 9601, 9602, 9603, 9604)
FRESH_SEEDS = tuple(range(9700, 9710))


def validate_seed_partition(references, fresh):
    if tuple(references) != REFERENCE_SEEDS or tuple(fresh) != FRESH_SEEDS:
        raise ValueError("REUSE-24 preregistered evaluation layout IDs changed")
    if set(references) & set(fresh):
        raise ValueError("Replication seeds overlap source reference")
    if any(7500 <= s < 8500 for s in fresh):
        raise ValueError("New evaluation layouts overlap source acquisition")


def phase_outcomes(policy, environment, defaults, condition, seeds):
    """Only post-action telemetry reads pocket or full grid."""
    import jax

    reset, step = jax.jit(environment.reset), jax.jit(environment.step)
    tasks = {}
    for skill in TRAIN_TASKS:
        params = defaults.replace(ruleset=get_rules(skill), max_steps=40)
        cases = []
        for seed in seeds:
            ts = reset(params, jax.random.key(seed))
            history = [PAD_ACTION] * HISTORY_LENGTH
            milestones = {}
            events = Counter()
            actions = Counter()
            steps = reward_success = 0
            time_since_pickup = None
            while steps < 40 and not bool(ts.last()):
                visible = visible_features(ts.observation)[None, :]
                obs = ts.observation
                instructions = task_instructions(
                    obs["goal_encoding"], obs["rule_encoding"], enabled=True
                )[None, :]
                inputs = assemble(
                    visible, instructions,
                    np.asarray([history], dtype=np.int64),
                    condition,
                )
                chosen, _ = policy.predict(inputs[0], deterministic=True)
                chosen = int(np.asarray(chosen).item())
                if chosen not in range(6):
                    raise AssertionError("Policy selected nonnative simulator action")
                before = inventory_and_grid(ts.state)
                ts = step(params, ts, chosen)
                after = inventory_and_grid(ts.state)
                steps += 1
                actions[chosen] += 1
                history = [*history[1:], chosen]
                reward_success |= int(float(ts.reward) > 0)
                if chosen == 3:
                    events["pickup_action_attempts"] += 1
                if chosen == 4:
                    events["putdown_action_attempts"] += 1
                if chosen == 3 and before["pocket_empty"] and not after["pocket_empty"]:
                    events["effective_pickups"] += 1
                    milestones.setdefault("effective_pickup", steps)
                    time_since_pickup = steps
                if chosen == 4 and not before["pocket_empty"] and after["pocket_empty"]:
                    events["effective_putdowns"] += 1
                    milestones.setdefault("effective_putdown", steps)
                    if time_since_pickup is not None:
                        milestones.setdefault(
                            "pickup_to_effective_drop_actions",
                            steps - time_since_pickup
                        )
                if after["green_on_grid"] and not before["green_on_grid"]:
                    events["new_green_transformation"] += 1
                    milestones.setdefault("new_green_transformation", steps)
                for item in (
                    "carrying_square", "carrying_red", "carrying_green",
                    "green_on_grid", "square_next_to_red"
                ):
                    if after[item]:
                        events[item] += 1
                        milestones.setdefault(item, steps)
            cases.append({
                "seed": seed,
                "goal_success": reward_success,
                "real_simulator_steps": steps,
                "native_actions_executed": dict(actions),
                "milestone_first_step": milestones,
                "event_counts": dict(events),
            })
        checks = (
            "effective_pickup", "effective_putdown", "new_green_transformation",
            "carrying_square", "carrying_red", "carrying_green",
            "green_on_grid", "square_next_to_red"
        )
        tasks[skill] = {
            "episodes": len(cases),
            "successes": sum(c["goal_success"] for c in cases),
            "rewarded_layout_seeds": [
                c["seed"] for c in cases if c["goal_success"]
            ],
            "simulator_steps": sum(c["real_simulator_steps"] for c in cases),
            "episodes_reaching_milestone": {
                name: sum(name in c["milestone_first_step"] for c in cases)
                for name in checks
            },
            "native_pickup_action_attempts": sum(
                c["event_counts"].get("pickup_action_attempts", 0) for c in cases
            ),
            "native_putdown_action_attempts": sum(
                c["event_counts"].get("putdown_action_attempts", 0) for c in cases
            ),
            "details": cases,
        }
    return tasks


def check_anchor_rewards(measured, source, seed, condition):
    """Byte-stable reward layout IDs must reproduce REUSE-23 before fresh test."""
    models = [m for m in source["models"] if m["model_seed"] == seed]
    if len(models) != 1:
        raise ValueError("Missing unique REUSE-23 policy seed")
    originals = [
        x for x in models[0]["conditions"] if x["condition"] == condition
    ]
    if len(originals) != 1:
        raise ValueError("Missing matched original history condition")
    for skill in TRAIN_TASKS:
        reference_ids = [
            x["seed"]
            for x in originals[0]["independent_evaluation"][skill]["cases"]
            if x["goal_success"]
        ]
        if measured[skill]["rewarded_layout_seeds"] != reference_ids:
            raise AssertionError(
                f"Frozen REUSE-23 reward seed mismatch {seed}/{condition}/{skill}"
            )
    return True


def run(config_path):
    import gymnasium as gym
    import torch
    import xminigrid
    from imitation.algorithms.bc import BC
    from imitation.data.types import Transitions
    from xminigrid.wrappers import (
        DirectionObservationWrapper,
        RulesAndGoalsObservationWrapper,
    )

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if (cfg["environment"] != "XLand-MiniGrid-R1-9x9"
            or cfg["training_tasks"] != list(TRAIN_TASKS)
            or cfg["conditions"] != list(CONDITIONS)
            or cfg["model_seeds"] != [17, 29]
            or cfg["batch_size"] != 32
            or cfg["epochs"] != 12
            or cfg["episode_cap"] != 40
            or cfg["feature_width"] != WIDTH
            or cfg["source_label_run_id"] != 39637067506
            or cfg["source_policy_run_id"] != 39644180240):
        raise ValueError("Frozen source policy and training protocol changed")
    validate_seed_partition(
        cfg["reference_seeds"], cfg["fresh_layout_seeds"]
    )
    source_dir = Path(cfg["source_numeric_dir"])
    source_audit = json.loads(
        (source_dir / "summary.json").read_text(encoding="utf-8")
    )
    original = json.loads(
        (Path(cfg["source_policy_dir"]) / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    if (source_audit["total_expert_queries"] != 582
            or source_audit["heldout_teacher_labels"] != 0
            or original["source_run"] != 39637067506
            or original["new_privileged_teacher_queries"] != 0
            or original["heldout_expert_actions_used_for_training"] != 0
            or original["eval_seeds"] != list(REFERENCE_SEEDS)):
        raise ValueError("Upstream numeric/policy source record not as preregistered")
    torch.set_num_threads(1)
    room, defaults = xminigrid.make(cfg["environment"])
    environment = RulesAndGoalsObservationWrapper(DirectionObservationWrapper(room))
    task_codes = {
        task: task_instructions(
            get_rules(task).goal, get_rules(task).rules, enabled=True
        )
        for task in TRAIN_TASKS
    }
    results = []
    for model_seed in cfg["model_seeds"]:
        candidates = [
            m for m in source_audit["model_results"]
            if m["model_seed"] == model_seed
        ]
        if len(candidates) != 1:
            raise ValueError("Missing unique trusted DAgger source")
        demo, names, sha = load_verified_source(
            source_dir / f"action_provenance_seed{model_seed}.npz",
            candidates[0], model_seed
        )
        check_true_history(demo)
        codes = np.stack([task_codes[name] for name in names])
        weighted_rows = balanced_indices(demo["expert_label"], model_seed)
        controls = []
        for condition in CONDITIONS:
            x = assemble(
                demo["obs"], codes, demo["prior_executed_actions"], condition
            )
            torch.manual_seed(model_seed)
            bc = BC(
                observation_space=gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(WIDTH,), dtype=np.float32
                ),
                action_space=gym.spaces.Discrete(6),
                demonstrations=Transitions(
                    obs=x[weighted_rows],
                    acts=demo["expert_label"][weighted_rows],
                    next_obs=x[weighted_rows].copy(),
                    dones=np.asarray(demo["done"][weighted_rows], dtype=bool),
                    infos=np.array([None] * len(weighted_rows), dtype=object),
                ),
                rng=np.random.default_rng(model_seed),
                batch_size=32, device="cpu",
            )
            bc.train(n_epochs=12, progress_bar=False)
            reference = phase_outcomes(
                bc.policy, environment, defaults, condition, REFERENCE_SEEDS
            )
            anchor_ok = check_anchor_rewards(
                reference, original, model_seed, condition
            )
            fresh = phase_outcomes(
                bc.policy, environment, defaults, condition, FRESH_SEEDS
            )
            controls.append({
                "condition": condition,
                "source_sha256": sha,
                "reference_reward_seed_ids_reproduced": anchor_ok,
                "reference_phase_telemetry": reference,
                "fresh_phase_telemetry": fresh,
                "fresh_source_successes": {
                    task: fresh[task]["successes"] for task in TRAIN_TASKS
                },
                "source_sample_exposures": 12 * len(weighted_rows),
                "real_evaluation_simulator_steps": sum(
                    reference[task]["simulator_steps"]
                    + fresh[task]["simulator_steps"]
                    for task in TRAIN_TASKS
                ),
            })
            print(
                f"seed={model_seed} condition={condition} "
                f"fresh rewards={controls[-1]['fresh_source_successes']} "
                f"putdown effective drops="
                f"{fresh['putdown']['episodes_reaching_milestone']['effective_putdown']}/10"
            )
        if controls[0]["source_sha256"] != controls[1]["source_sha256"]:
            raise AssertionError("Different matched source datasets")
        results.append({
            "policy_model_seed": model_seed,
            "trusted_source_sha256": sha,
            "conditions": controls,
        })
    result = {
        "question": "Do REUSE-23 source-task skills reproduce on untouched layouts and which phase blocks the putdown skill?",
        "preregistration_type": "Frozen-model replication with post-action stage telemetry",
        "reused_methods": [
            "imitation.BC v1.0.1",
            "PyTorch inverse-frequency WeightedRandomSampler",
            "XLand RulesAndGoalsObservationWrapper",
            "REUSE-18 native simulator pocket/grid phase metrics",
        ],
        "numeric_training_run_id": 39637067506,
        "frozen_policy_result_run_id": 39644180240,
        "anchor_layout_seeds": list(REFERENCE_SEEDS),
        "previously_unexamined_layout_seeds": list(FRESH_SEEDS),
        "training_skill_tasks_only": list(TRAIN_TASKS),
        "heldout_chain_evaluation_performed": False,
        "new_expert_label_queries": 0,
        "new_online_training_environment_actions": 0,
        "privileged_policy_input_fields": 0,
        "privileged_state_used_only_for_after_action_telemetry": True,
        "source_policy_reconstructed_without_untrusted_checkpoint_load": True,
        "result": results,
        "total_real_evaluation_simulator_steps": sum(
            item["real_evaluation_simulator_steps"]
            for model in results for item in model["conditions"]
        ),
        "interpretation_limits": [
            "Untouched-layout successes, not supervised training-label accuracy, are primary.",
            "Original reward seed IDs must reproduce before fresh-seed reward is interpreted.",
            "Stage probes read hidden simulator pocket and grid strictly outside policy prediction.",
            "A real BC putdown event on one layout is not reliable learned multi-stage competence.",
            "Only 10 fresh layout seeds per model/condition and two model seeds; no population inference.",
            "No new causal chain evaluation or transfer claim, since train putdown gate not yet passed.",
            "No added expert queries, source labels, online training data or invented algorithm.",
            "Both BC models use 115 learner input features; one gets four preceding actual actions, one constant PAD.",
            "The copied next_obs is nonphysical placeholder required by upstream BC collator and not used by supervised loss.",
        ],
    }
    target = Path(cfg["output_dir"]) / "summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Saved REUSE-24 untouched-layout phase replication")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/xland_frozen_phase_replication.yaml")
    run(p.parse_args().config)
