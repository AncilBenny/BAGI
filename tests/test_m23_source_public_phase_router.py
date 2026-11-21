"""Stage18 public-only native phase router, matched policies and strict source gates."""
import numpy as np
import pytest
import torch

from bagi.experiments.m23_source_post_open_continuation_bc import (
    MODEL_SEEDS,
    PUBLIC_WIDTH,
    WIDTH,
    fit_model,
)
from bagi.experiments.m23_source_public_phase_router import (
    EVAL_ARMS,
    TASKS,
    advance_public_phase,
    decide,
    monitor_episode,
    select_executed_action,
)
from bagi.experiments.m23_study_contract import (
    FINAL_LAYOUTS,
    INTERACTION_CONTEXT_MONITOR_LAYOUTS,
    PHASE_ROUTER_MONITOR_LAYOUTS,
    PHASE_ROUTER_TRAIN_LAYOUTS,
    POST_OPEN_MONITOR_LAYOUTS,
    validate_split_contract,
)


def public_obs():
    return {
        "image": np.zeros((7, 7, 3), dtype=np.uint8),
        "mission": "use the key to open the door and go to the goal",
        "direction": 1,
    }


def test_next_real_step_handoff_only_after_public_executed_toggle():
    prior = public_obs()
    after = public_obs()
    prior["image"][3, 5] = [4, 0, 2]
    after["image"][3, 5] = [4, 0, 0]
    assert len(EVAL_ARMS) == 3
    assert select_executed_action("public_phase_router", False, 2, 5) == 2
    assert not advance_public_phase("doorkey", False, prior, 2, after)
    # A passively proposed action5 is never executed; do not switch.
    assert not advance_public_phase("doorkey", False, prior, 0, after)
    switched = advance_public_phase("doorkey", False, prior, 5, after)
    assert switched
    assert select_executed_action("public_phase_router", switched, 2, 5) == 5
    assert advance_public_phase("doorkey", True, after, 2, after)
    assert select_executed_action("interaction_context", True, 2, 5) == 2
    assert select_executed_action("post_open_continuation", False, 2, 5) == 5
    with pytest.raises(ValueError):
        select_executed_action("public_phase_router", False, 7, 5)
    with pytest.raises(ValueError):
        select_executed_action("oracle", False, 2, 5)


def test_fetch_cannot_switch_and_no_private_phase_oracle():
    prior, after = public_obs(), public_obs()
    prior["image"][3, 5] = [4, 0, 1]
    after["image"][3, 5] = [4, 0, 0]
    assert not advance_public_phase("fetch", False, prior, 5, after)
    with pytest.raises(ValueError):
        advance_public_phase("fetch", True, prior, 5, after)
    with pytest.raises(ValueError):
        advance_public_phase("doorkey", 1, prior, 5, after)


def test_stage18_source_train_monitor_disjoint_and_target_sealed():
    validate_split_contract()
    current = {
        seed
        for split in (PHASE_ROUTER_TRAIN_LAYOUTS, PHASE_ROUTER_MONITOR_LAYOUTS)
        for group in split.values() for seed in group
    }
    older = {
        seed
        for split in (
            FINAL_LAYOUTS, POST_OPEN_MONITOR_LAYOUTS,
            INTERACTION_CONTEXT_MONITOR_LAYOUTS,
        )
        for group in split.values() for seed in group
    }
    assert len(current) == 272
    assert not current.intersection(older)
    assert not current.intersection({453001, 892001, 902001})
    with pytest.raises(ValueError, match="outside independent fresh source split"):
        monitor_episode("unlockpickup", 453001, "actor_only", None, None)


def train_examples():
    episodes = []
    for i in range(192):
        n = 3
        y = np.asarray([(i + k) % 7 for k in range(n)], dtype=np.int64)
        if i == 0:
            y[0] = 5
        x = np.zeros((n, WIDTH), dtype=np.float32)
        x[:, i % PUBLIC_WIDTH] = 1.0
        for k in range(1, n):
            x[k, PUBLIC_WIDTH + int(y[k - 1])] = 1.0
        episodes.append({
            "x": x, "y": y,
            "task": "doorkey" if i < 96 else "fetch",
            "public_first_witnessed_open_action_index": 0 if i == 0 else None,
            "public_post_open_source_action_count": n - 1 if i == 0 else 0,
        })
    return episodes


def test_exact_same_fitted_models_not_a_third_training_arm():
    torch.set_num_threads(1)
    labels = train_examples()
    control, c = fit_model(
        labels, MODEL_SEEDS[0], "interaction_context", torch.ones(7)
    )
    post, p = fit_model(
        labels, MODEL_SEEDS[0], "post_open_continuation", torch.ones(7)
    )
    assert control.training is False and post.training is False
    assert c["num_parameters"] == p["num_parameters"] == 215879
    assert c["actual_optimizer_step_calls"] == p["actual_optimizer_step_calls"] == 144
    assert c["cached_source_label_presentations_not_native_steps"] == (
        p["cached_source_label_presentations_not_native_steps"]
    )
    assert len(MODEL_SEEDS) == 3 and TASKS == ("doorkey", "fetch")


def synthetic_monitor(actor_goals=4, control_goals=5, post_goals=4,
                      router_goals=18, router_distinct=True):
    groups = {}
    for task in TASKS:
        layouts = PHASE_ROUTER_MONITOR_LAYOUTS[task]
        groups[f"{task}:actor_only"] = {"episodes": [
            {
                "layout_seed": layout,
                "external_strict_full_physical_goal": i < actor_goals,
                "actual_native_executed_action_sha256": "actor:" + str(layout),
                "actual_native_steps": 1,
                "public_witnessed_door_open": task == "doorkey",
            }
            for i, layout in enumerate(layouts)
        ]}
        for seed in MODEL_SEEDS:
            for arm in EVAL_ARMS:
                n = (router_goals if arm == "public_phase_router"
                     else control_goals if arm == "interaction_context"
                     else post_goals)
                if task == "fetch" and arm == "public_phase_router":
                    n = control_goals
                rows = []
                for i, layout in enumerate(layouts):
                    digest = f"{task}:{seed}:control:{layout}"
                    if task == "doorkey" and arm == "public_phase_router" and router_distinct:
                        digest = f"{task}:{seed}:router:{layout}"
                    if arm == "post_open_continuation":
                        digest = f"{task}:{seed}:post:{layout}"
                    rows.append({
                        "layout_seed": layout,
                        "external_strict_full_physical_goal": i < n,
                        "actual_native_executed_action_sha256": digest,
                        "actual_native_steps": 1,
                        "public_witnessed_door_open": (
                            task == "doorkey" and arm == "public_phase_router"
                        ),
                    })
                groups[f"{task}:{arm}:{seed}"] = {"episodes": rows}
    return groups


def test_preregistered_physical_gate_bootstrap_and_behavior_distinction():
    verdict = decide(synthetic_monitor())
    assert verdict["registered_decision"] == "PASS"
    assert verdict["paired_router_minus_control_strict_goals"] == 39
    assert verdict["router_minus_post_only_strict_goals"] > 8
    assert verdict["bootstrap_lower_bound_above_zero"]
    assert verdict["fetch_router_control_action_and_goal_identical"]
    assert decide(synthetic_monitor(router_distinct=False))["registered_decision"] == (
        "INSUFFICIENT"
    )
    assert decide(synthetic_monitor(router_goals=2))["registered_decision"] == "FAIL"


def test_fetch_identity_and_native_layout_pairing_fail_closed():
    value = synthetic_monitor()
    value["fetch:public_phase_router:941"]["episodes"][0][
        "actual_native_executed_action_sha256"
    ] = "unexplained-divergence"
    with pytest.raises(ValueError, match="Fetch router/control source integrity"):
        decide(value)
    value = synthetic_monitor()
    value["doorkey:public_phase_router:953"]["episodes"][0]["layout_seed"] = 1
    with pytest.raises(ValueError, match="layouts unpaired"):
        decide(value)
