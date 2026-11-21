"""M23 belief-state input isolation, GRU capacity control and safe source splits."""

import numpy as np
import pytest
import torch

from bagi.experiments.m23_belief_memory import (
    HISTORY_FRAMES,
    PublicGRUNextFrame,
    encode_history,
    fit_gru,
)
from bagi.experiments.m23_class_balance import training_object_weights
from bagi.experiments.m23_public_world_model import MODEL_SEEDS, WIDTH
from bagi.experiments.m23_source_feasibility import run_source_episode
from bagi.experiments.m23_study_contract import (
    BALANCE_MONITOR_LAYOUTS,
    FINAL_LAYOUTS,
    MEMORY_MONITOR_LAYOUTS,
    MONITOR_LAYOUTS,
    TRAIN_LAYOUTS,
    validate_split_contract,
)


def _three_steps():
    img = np.zeros((3, 7, 7, 3), dtype=np.int64)
    img[0, 3, 5, 0] = 5
    img[1, 3, 5, 0] = 6
    img[2, 3, 5, 0] = 7
    return {
        "observation_image": img,
        "observation_direction": [0, 1, 2],
        "mission": ["get a blue ball"] * 3,
        "action": [3, 0, 2],
        "next_image": np.zeros((3, 7, 7, 3), dtype=np.int64),
        "next_direction": [0, 1, 2],
        "reward": [0.0, 0.0, 0.0],
        "terminated": [True, False, True],
        "truncated": [False, False, False],
    }


def test_observation_windows_reset_at_terminal_and_are_right_padded():
    data = encode_history(_three_steps())
    seq = data["sequence"]
    assert seq.shape == (3, HISTORY_FRAMES, WIDTH)
    np.testing.assert_array_equal(seq[0, 0], data["features"][0])
    np.testing.assert_array_equal(seq[1, 0], data["features"][1])
    assert np.count_nonzero(seq[1, 1:]) == 0
    np.testing.assert_array_equal(seq[2, 0], data["features"][1])
    np.testing.assert_array_equal(seq[2, 1], data["features"][2])
    assert np.count_nonzero(seq[2, 2:]) == 0
    assert not np.array_equal(seq[0, 0], seq[1, 0])


def test_same_gru_weights_do_not_depend_on_padded_future():
    data = encode_history(_three_steps())
    torch.manual_seed(MODEL_SEEDS[0])
    model = PublicGRUNextFrame().eval()
    first = torch.from_numpy(data["features"][0:1, None, :].copy())
    extended = torch.from_numpy(data["sequence"][0:1].copy())
    with torch.no_grad():
        a = model(first)
        b = model(extended)
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y)


def test_gru_rejects_missing_or_gapped_public_inputs():
    model = PublicGRUNextFrame()
    with pytest.raises(ValueError, match="real observation"):
        model(torch.zeros((1, HISTORY_FRAMES, WIDTH)))
    gapped = torch.zeros((1, HISTORY_FRAMES, WIDTH))
    gapped[0, 0, 0] = 1.0
    gapped[0, 2, 0] = 1.0
    with pytest.raises(ValueError, match="contiguous"):
        model(gapped)


def test_fresh_optimizer_updates_counted_for_both_history_lengths():
    batch = encode_history(_three_steps())
    weights = training_object_weights(batch)
    for length in (1, HISTORY_FRAMES):
        model, info = fit_gru(batch, MODEL_SEEDS[0], weights, length)
        assert isinstance(model, PublicGRUNextFrame)
        assert not model.training
        assert info["actual_optimizer_step_calls"] == 6
        assert info["training_sample_presentations"] == 18
    with pytest.raises(ValueError, match="registers"):
        fit_gru(batch, MODEL_SEEDS[0], weights, 3)


def test_source_history_layouts_disjoint_and_target_sealed():
    validate_split_contract()
    new = {seed for layouts in MEMORY_MONITOR_LAYOUTS.values() for seed in layouts}
    old = {
        seed
        for split in (TRAIN_LAYOUTS, MONITOR_LAYOUTS, BALANCE_MONITOR_LAYOUTS,
                      FINAL_LAYOUTS)
        for layouts in split.values()
        for seed in layouts
    }
    assert len(new) == 48
    assert new.isdisjoint(old)
    assert "unlockpickup" not in MEMORY_MONITOR_LAYOUTS
    with pytest.raises(ValueError, match="MEMORY_MONITOR"):
        run_source_episode("fetch", 443001, split="memory_monitor")
    with pytest.raises(ValueError):
        run_source_episode("unlockpickup", 453001, split="memory_monitor")
