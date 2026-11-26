"""Stage11 source-only public image versus rare-action weighted BC gates."""
import numpy as np
import pytest
import torch

from bagi.experiments.m23_source_public_vision_bc import (
    ARMS,
    CELL,
    FRONT_INDEX,
    IMAGE_WIDTH,
    MODEL_SEEDS,
    TASKS,
    WIDTH,
    VisualBC,
    decide,
    fit_model,
    monitor_episode,
    public_visual_features,
    restrict_to_front,
)
from bagi.experiments.m23_study_contract import (
    BC_MONITOR_LAYOUTS,
    BC_TRAIN_LAYOUTS,
    FINAL_LAYOUTS,
    VISION_MONITOR_LAYOUTS,
    VISION_TRAIN_LAYOUTS,
    validate_split_contract,
)


def public_obs() -> dict:
    return {
        "image": np.zeros((7, 7, 3), dtype=np.uint8),
        "direction": 1,
        "mission": "use the key to open the door and then get to the goal",
    }


def test_entire_public_image_is_visible_and_front_mask_preserves_same_width():
    observation = public_obs()
    before = public_visual_features(observation, False)
    observation["image"][0, 0, 0] = 1
    changed = public_visual_features(observation, False)
    assert before.shape == changed.shape == (WIDTH,)
    assert not np.array_equal(before, changed)
    assert np.array_equal(restrict_to_front(before), restrict_to_front(changed))
    original = changed.copy()
    cropped = restrict_to_front(changed)
    assert np.array_equal(original, changed)
    grid = cropped[:IMAGE_WIDTH].reshape(49, CELL)
    assert np.count_nonzero(grid) == 3
    assert np.count_nonzero(grid[FRONT_INDEX]) == 3
    assert np.array_equal(changed[IMAGE_WIDTH:], cropped[IMAGE_WIDTH:])
    assert np.array_equal(
        restrict_to_front(np.stack([before, changed]))[0],
        restrict_to_front(before),
    )
    with pytest.raises(ValueError):
        restrict_to_front(np.ones(43, dtype=np.float32))
    with pytest.raises(ValueError):
        public_visual_features({"image": np.zeros((7, 7, 3)), "mission": "x"}, False)


def test_fresh_source_train_monitor_layouts_protect_closed_final_and_old_monitor():
    validate_split_contract()
    prior = {
        seed for split in (BC_MONITOR_LAYOUTS, BC_TRAIN_LAYOUTS, FINAL_LAYOUTS)
        for seeds in split.values() for seed in seeds
    }
    new = {
        seed for split in (VISION_MONITOR_LAYOUTS, VISION_TRAIN_LAYOUTS)
        for seeds in split.values() for seed in seeds
    }
    assert len(new) == 272
    assert not prior.intersection(new)
    with pytest.raises(ValueError, match="fresh registered source"):
        monitor_episode("unlockpickup", 453001, "actor_only", None, None)


def test_same_seed_four_arms_same_width_epochs_and_native_label_presentations():
    x = np.zeros((8, WIDTH), dtype=np.float32)
    x[:, FRONT_INDEX * CELL] = 1
    x[:, CELL] = 1
    y = np.asarray([0, 1, 2, 3, 5, 0, 1, 2], dtype=np.int64)
    fits = {}
    for arm in ARMS:
        model, profile = fit_model(x, y, arm, MODEL_SEEDS[0])
        assert isinstance(model, VisualBC)
        assert list(model(torch.zeros((1, WIDTH))).shape) == [1, 7]
        fits[arm] = profile
    assert len(ARMS) == 4
    assert {p["actual_optimizer_step_calls"] for p in fits.values()} == {12}
    assert {p["cached_native_source_label_presentations"] for p in fits.values()} == {96}
    assert fits["front_unweighted"]["class_balanced"] is False
    assert fits["image_balanced"]["class_balanced"] is True
    assert fits["front_unweighted"]["input_width"] == WIDTH
    assert fits["image_balanced"]["input_width"] == WIDTH


def mock_monitor(actor_goals=5, front_goals=4, treated_goals=18,
                 different=True) -> dict:
    monitor = {}
    for task in TASKS:
        layouts = VISION_MONITOR_LAYOUTS[task]
        actor = [
            {"layout_seed": layout, "external_physical_full_goal": i < actor_goals,
             "executed_action_sha256": f"actor:{layout}"}
            for i, layout in enumerate(layouts)
        ]
        monitor[f"{task}:actor_only"] = {"episodes": actor}
        for arm in ARMS:
            for seed in MODEL_SEEDS:
                count = treated_goals if arm == "image_balanced" else (
                    front_goals if arm == "front_unweighted" else 6
                )
                monitor[f"{task}:{arm}:{seed}"] = {"episodes": [
                    {"layout_seed": layout,
                     "external_physical_full_goal": i < count,
                     "external_effective_door_open_events": int(i < count),
                     "executed_action_sha256": (
                         f"{arm}:{seed}:{layout}" if different else f"actor:{layout}"
                     )}
                    for i, layout in enumerate(layouts)
                ]}
    return monitor


def test_primary_gate_uses_native_paths_goals_openings_actor_and_uncertainty():
    result = decide(mock_monitor())
    assert result["registered_decision"] == "PASS"
    assert result["necessary_native_action_distinction"]
    assert result["paired_image_balanced_minus_front_unweighted_physical_goals"] == 84
    assert result["image_balanced_effective_door_open_episodes"] == 54
    assert result["95pct_bootstrap_lower_bound_above_zero"]
    assert decide(mock_monitor(different=False))["registered_decision"] == "INSUFFICIENT"
    assert decide(mock_monitor(treated_goals=3))["registered_decision"] == "FAIL"


def test_unregistered_fit_arm_and_invalid_action_label_rejected():
    model = VisualBC()
    with pytest.raises(ValueError):
        model(torch.zeros((1, WIDTH - 1)))
    with pytest.raises(ValueError):
        fit_model(np.zeros((8, WIDTH), dtype=np.float32),
                  np.zeros(8, dtype=np.int64), "unknown", 941)
    with pytest.raises(ValueError):
        fit_model(np.zeros((8, WIDTH), dtype=np.float32),
                  np.full(8, 8, dtype=np.int64), "front_unweighted", 941)
