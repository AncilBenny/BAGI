"""Stage17 preregistered public own-action-focused recurrent BC tests."""
import numpy as np
import pytest
import torch

from bagi.experiments.m23_source_post_open_continuation_bc import (
    ARMS,
    DOOR_ID,
    HIDDEN,
    MODEL_SEEDS,
    PUBLIC_WIDTH,
    TASKS,
    WIDTH,
    PostOpenGRU,
    decide,
    fit_model,
    interaction_context_weights,
    monitor_episode,
    post_open_continuation_weights,
    public_action_features,
    public_door_open_transition,
)
from bagi.experiments.m23_study_contract import (
    BC_MONITOR_LAYOUTS,
    CORRECT_MONITOR_LAYOUTS,
    FINAL_LAYOUTS,
    INTERACTION_CONTEXT_MONITOR_LAYOUTS,
    INTERVENE_MONITOR_LAYOUTS,
    POST_OPEN_MONITOR_LAYOUTS,
    POST_OPEN_TRAIN_LAYOUTS,
    RECURRENT_MONITOR_LAYOUTS,
    VISION_MONITOR_LAYOUTS,
    validate_split_contract,
)


def public_obs():
    return {
        "image": np.zeros((7, 7, 3), dtype=np.uint8),
        "mission": "use the key to open the door and then get to the goal",
        "direction": 1,
    }


def test_own_native_action_is_shifted_not_current_target_and_hidden_reset():
    before = public_action_features(public_obs(), False, None,
                                   arm="post_open_continuation")
    following = public_action_features(public_obs(), False, 5,
                                      arm="post_open_continuation")
    control = public_action_features(public_obs(), False, 5,
                                    arm="interaction_context")
    assert WIDTH == PUBLIC_WIDTH + 7
    assert before.shape == following.shape == control.shape == (WIDTH,)
    assert np.count_nonzero(before[PUBLIC_WIDTH:]) == 0
    assert following[PUBLIC_WIDTH + 5] == 1
    assert int(np.sum(following[PUBLIC_WIDTH:])) == 1
    assert np.count_nonzero(control[PUBLIC_WIDTH:]) == 1
    assert np.array_equal(following, control)
    assert np.array_equal(following[:PUBLIC_WIDTH], control[:PUBLIC_WIDTH])
    with pytest.raises(ValueError):
        public_action_features(public_obs(), False, 8, arm="post_open_continuation")
    with pytest.raises(ValueError):
        public_action_features(public_obs(), False, 2, arm="unknown")


def test_event_windows_genuine_native_labels_and_episode_boundaries():
    first = np.asarray([5, 2, 0, 0, 0, 3, 1, 2, 2, 2], dtype=np.int64)
    assert interaction_context_weights(first).tolist() == [4] * 8 + [1, 1]
    assert interaction_context_weights(np.asarray([2, 2])).tolist() == [1, 1]
    assert interaction_context_weights(np.asarray([3])).tolist() == [4]
    with pytest.raises(ValueError):
        interaction_context_weights(np.asarray([7]))
    with pytest.raises(ValueError):
        interaction_context_weights(np.asarray([], dtype=np.int64))



def test_public_native_witness_requires_executed_toggle_and_same_cell():
    before, after = public_obs(), public_obs()
    before["image"][3, 5] = [DOOR_ID, 0, 2]
    after["image"][3, 5] = [DOOR_ID, 0, 0]
    assert public_door_open_transition(before, 5, after)
    assert not public_door_open_transition(before, 2, after)
    assert not public_door_open_transition(before, 3, after)
    same_locked = public_obs()
    same_locked["image"][3, 5] = [DOOR_ID, 0, 2]
    assert not public_door_open_transition(before, 5, same_locked)
    different_obj = public_obs()
    assert not public_door_open_transition(before, 5, different_obj)
    with pytest.raises(ValueError):
        public_door_open_transition(before, 7, after)
    with pytest.raises(ValueError):
        public_door_open_transition(before, 5, {"image": after["image"]})


def test_post_open_weight_applies_only_after_public_real_toggle_and_task():
    labels = np.asarray([2, 5, 0, 0, 2, 1, 0, 2, 1, 2], dtype=np.int64)
    control = post_open_continuation_weights(
        labels, 1, arm="interaction_context", task="doorkey"
    )
    treatment = post_open_continuation_weights(
        labels, 1, arm="post_open_continuation", task="doorkey"
    )
    assert control.tolist() == [4, 4, 4, 4, 1, 1, 1, 1, 1, 1]
    assert treatment.tolist() == [4] * len(labels)
    assert treatment.max() == 4
    assert np.array_equal(
        post_open_continuation_weights(
            labels, None, arm="post_open_continuation", task="doorkey"
        ),
        interaction_context_weights(labels),
    )
    assert np.array_equal(
        post_open_continuation_weights(
            labels, None, arm="post_open_continuation", task="fetch"
        ),
        interaction_context_weights(labels),
    )
    with pytest.raises(ValueError):
        post_open_continuation_weights(
            labels, 0, arm="post_open_continuation", task="doorkey"
        )
    with pytest.raises(ValueError):
        post_open_continuation_weights(
            labels, 1, arm="post_open_continuation", task="fetch"
        )


def test_recurrent_hidden_persists_for_both_arms_and_resets_per_episode():
    torch.manual_seed(941)
    model = PostOpenGRU()
    model.eval()
    x = np.zeros(WIDTH, dtype=np.float32)
    _, first = model.native_step(x, None)
    _, second = model.native_step(x, first)
    assert first.shape == second.shape == (1, 1, HIDDEN)
    assert not torch.allclose(first, second)
    # Repeating episode reset gives the same initial hidden state.
    _, another_first = model.native_step(x, None)
    assert torch.allclose(first, another_first)
    with pytest.raises(ValueError):
        model.native_step(np.zeros(WIDTH - 1, dtype=np.float32), None)
    with pytest.raises(ValueError):
        model(torch.zeros((1, WIDTH)))


def test_new_train_monitor_sealed_source_and_target_splits():
    validate_split_contract()
    latest = {
        n for split in (POST_OPEN_TRAIN_LAYOUTS, POST_OPEN_MONITOR_LAYOUTS)
        for layouts in split.values() for n in layouts
    }
    prior = {
        n for split in (
            BC_MONITOR_LAYOUTS, CORRECT_MONITOR_LAYOUTS,
            VISION_MONITOR_LAYOUTS, INTERVENE_MONITOR_LAYOUTS,
            INTERACTION_CONTEXT_MONITOR_LAYOUTS,
            RECURRENT_MONITOR_LAYOUTS, FINAL_LAYOUTS,
        )
        for layouts in split.values() for n in layouts
    }
    assert len(latest) == 272
    assert not latest.intersection(prior)
    with pytest.raises(ValueError, match="outside independent fresh split"):
        monitor_episode("unlockpickup", 453001, "actor_only", None, None)


def native_episode_examples():
    rows = []
    for i in range(192):
        n = 2 + i % 3
        x = np.zeros((n, WIDTH), dtype=np.float32)
        x[:, i % PUBLIC_WIDTH] = 1
        y = np.asarray([(i + step) % 7 for step in range(n)], dtype=np.int64)
        if i == 0:
            y[0] = 5
        for step in range(1, n):
            x[step, PUBLIC_WIDTH + int(y[step - 1])] = 1.0
        rows.append({
            "x": x, "y": y,
            "task": "doorkey" if i < 96 else "fetch",
            "public_first_witnessed_open_action_index": 0 if i == 0 else None,
            "public_post_open_source_action_count": n - 1 if i == 0 else 0,
        })
    return rows


def test_six_fit_contract_equal_capacity_updates_and_nonleaking_advice():
    torch.set_num_threads(1)
    episodes = native_episode_examples()
    weights = torch.ones(7, dtype=torch.float32)
    uniform, a = fit_model(episodes, MODEL_SEEDS[0], "interaction_context", weights)
    focused, b = fit_model(
        episodes, MODEL_SEEDS[0], "post_open_continuation", weights
    )
    assert len(ARMS) == 2
    assert a["num_parameters"] == b["num_parameters"]
    assert a["actual_optimizer_step_calls"] == b["actual_optimizer_step_calls"] == 144
    n = sum(len(row["y"]) for row in episodes)
    assert a["actual_source_train_label_count"] == b["actual_source_train_label_count"] == n
    assert a["cached_source_label_presentations_not_native_steps"] == 12 * n
    assert b["cached_source_label_presentations_not_native_steps"] == 12 * n
    assert uniform.training is False and focused.training is False
    assert len(list(uniform.parameters())) == len(list(focused.parameters()))
    assert a["genuine_train_event_window_tokens"] > 0
    assert b["genuine_train_event_window_tokens"] == a["genuine_train_event_window_tokens"]
    assert a["event_context_multiplier"] == b["event_context_multiplier"] == 4.0
    assert a["post_open_continuation_multiplier"] == 1.0
    assert b["post_open_continuation_multiplier"] == 4.0
    assert a["actual_post_open_train_source_actions"] > 0
    assert a["actual_post_open_train_source_actions"] == b["actual_post_open_train_source_actions"]
    damaged = native_episode_examples()
    damaged[0]["x"][1, PUBLIC_WIDTH:] = 0
    with pytest.raises(ValueError, match="previous action"):
        fit_model(damaged, MODEL_SEEDS[0], "post_open_continuation", weights)


def fake_monitor(actor_goals=4, uniform_goals=5, focused_goals=17,
                 distinct=True):
    result = {}
    for task in TASKS:
        layouts = POST_OPEN_MONITOR_LAYOUTS[task]
        result[f"{task}:actor_only"] = {"episodes": [
            {"layout_seed": layout,
             "external_strict_full_physical_goal": i < actor_goals,
             "actual_native_executed_action_sha256": f"actor:{layout}"}
            for i, layout in enumerate(layouts)
        ]}
        for arm in ARMS:
            for seed in MODEL_SEEDS:
                successes = (
                    focused_goals if arm == "post_open_continuation"
                    else uniform_goals
                )
                result[f"{task}:{arm}:{seed}"] = {"episodes": [
                    {"layout_seed": layout,
                     "external_strict_full_physical_goal": i < successes,
                     "actual_native_executed_action_sha256": (
                         f"{arm}:{seed}:{layout}" if distinct else f"actor:{layout}"
                     )}
                    for i, layout in enumerate(layouts)
                ]}
    return result


def test_preregistered_physics_action_nonvacuity_and_interval_gate():
    result = decide(fake_monitor())
    assert result["registered_decision"] == "PASS"
    assert result["necessary_native_action_distinction"]
    assert result["paired_post_open_continuation_minus_interaction_context_strict_goals"] == 72
    assert all(v > 0 for v in result["per_model_seed_doorkey_physical_goal_gain"].values())
    assert result["bootstrap_95pct_lower_bound_above_zero"]
    assert decide(fake_monitor(distinct=False))["registered_decision"] == "INSUFFICIENT"
    assert decide(fake_monitor(focused_goals=2))["registered_decision"] == "FAIL"


def test_native_monitor_layout_pairing_fail_closed():
    result = fake_monitor()
    result["fetch:interaction_context:941"]["episodes"][0]["layout_seed"] = 1
    with pytest.raises(ValueError, match="layout pairing failed"):
        decide(result)
