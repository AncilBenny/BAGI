"""Stage14 real native public GRU episodic-memory source BC contracts."""
import numpy as np
import pytest
import torch

from bagi.experiments.m23_source_public_vision_bc import WIDTH
from bagi.experiments.m23_source_recurrent_bc import (
    ARMS,
    HIDDEN,
    MODEL_SEEDS,
    TASKS,
    PublicGRU,
    decide,
    fit_model,
    monitor_episode,
)
from bagi.experiments.m23_study_contract import (
    BC_MONITOR_LAYOUTS,
    CORRECT_MONITOR_LAYOUTS,
    FINAL_LAYOUTS,
    INTERVENE_MONITOR_LAYOUTS,
    RECURRENT_MONITOR_LAYOUTS,
    RECURRENT_TRAIN_LAYOUTS,
    VISION_MONITOR_LAYOUTS,
    validate_split_contract,
)


def test_all_new_source_native_layouts_disjoint_and_target_final_sealed():
    validate_split_contract()
    current = {
        seed for split in (RECURRENT_TRAIN_LAYOUTS, RECURRENT_MONITOR_LAYOUTS)
        for group in split.values() for seed in group
    }
    previous = {
        seed for split in (
            BC_MONITOR_LAYOUTS,
            CORRECT_MONITOR_LAYOUTS,
            VISION_MONITOR_LAYOUTS,
            INTERVENE_MONITOR_LAYOUTS,
            FINAL_LAYOUTS,
        ) for group in split.values() for seed in group
    }
    assert len(current) == 272
    assert not current.intersection(previous)
    with pytest.raises(ValueError, match="outside independent fresh split"):
        monitor_episode("unlockpickup", 453001, "actor_only", None, None)


def test_public_gru_hidden_state_persists_only_in_recurrent_arm():
    torch.manual_seed(941)
    model = PublicGRU()
    model.eval()
    x = np.zeros(WIDTH, dtype=np.float32)
    first, h1 = model.native_step(x, None, reset_each=False)
    second, h2 = model.native_step(x, h1, reset_each=False)
    stateless1, st1 = model.native_step(x, None, reset_each=True)
    stateless2, st2 = model.native_step(x, h1, reset_each=True)
    assert first in range(7)
    assert second in range(7)
    assert stateless1 == stateless2
    assert st1 is None and st2 is None
    assert h1 is not None and h1.shape == (1, 1, HIDDEN)
    assert h2 is not None and h2.shape == h1.shape
    assert not torch.allclose(h1, h2)
    with pytest.raises(ValueError):
        model(torch.zeros((1, WIDTH)), reset_each=False)
    with pytest.raises(ValueError):
        model.native_step(np.zeros(WIDTH - 1, dtype=np.float32),
                          None, reset_each=False)


def source_demo_episodes():
    episodes = []
    for i in range(192):
        n = 2 + (i % 3)
        x = np.zeros((n, WIDTH), dtype=np.float32)
        x[:, i % WIDTH] = 1
        x[1:, (i + 1) % WIDTH] = 1
        y = np.array([(i + j) % 3 for j in range(n)], dtype=np.int64)
        episodes.append({"x": x, "y": y})
    return episodes


def test_stateless_and_recurrent_fit_same_episodes_weight_init_update_count():
    torch.set_num_threads(1)
    rows = source_demo_episodes()
    weights = torch.ones(7, dtype=torch.float32)
    stateless, a = fit_model(rows, MODEL_SEEDS[0], "stateless_gru", weights)
    recurrent, b = fit_model(rows, MODEL_SEEDS[0], "recurrent_gru", weights)
    tokens = sum(len(row["y"]) for row in rows)
    assert a["num_parameters"] == b["num_parameters"]
    assert a["actual_optimizer_step_calls"] == 144
    assert b["actual_optimizer_step_calls"] == 144
    assert a["actual_source_train_label_count"] == tokens
    assert a["cached_source_label_presentations_not_native_steps"] == 12 * tokens
    assert b["cached_source_label_presentations_not_native_steps"] == 12 * tokens
    assert stateless.training is False and recurrent.training is False
    assert isinstance(stateless, PublicGRU)
    assert isinstance(recurrent, PublicGRU)
    with pytest.raises(ValueError, match="label alignment"):
        malformed = source_demo_episodes()
        malformed[0]["y"] = malformed[0]["y"][:1]
        fit_model(malformed, MODEL_SEEDS[0], "recurrent_gru", weights)


def fake_monitor(actor_goals=4, stateless_goals=5,
                 recurrent_goals=17, distinct=True):
    source = {}
    for task in TASKS:
        layouts = RECURRENT_MONITOR_LAYOUTS[task]
        source[f"{task}:actor_only"] = {"episodes": [
            {"layout_seed": layout,
             "external_strict_full_physical_goal": i < actor_goals,
             "actual_native_executed_action_sha256": f"actor:{layout}"}
            for i, layout in enumerate(layouts)
        ]}
        for arm in ARMS:
            for seed in MODEL_SEEDS:
                count = (
                    recurrent_goals if arm == "recurrent_gru"
                    else stateless_goals
                )
                source[f"{task}:{arm}:{seed}"] = {"episodes": [
                    {"layout_seed": layout,
                     "external_strict_full_physical_goal": i < count,
                     "actual_native_executed_action_sha256": (
                         f"{arm}:{seed}:{layout}" if distinct
                         else f"actor:{layout}"
                     )}
                    for i, layout in enumerate(layouts)
                ]}
    return source


def test_preregistered_source_physical_gate_native_hash_and_bootstrap():
    result = decide(fake_monitor())
    assert result["registered_decision"] == "PASS"
    assert result["necessary_native_action_distinction"]
    assert result["paired_recurrent_minus_stateless_strict_full_goals"] == 72
    assert result["bootstrap_95pct_lower_bound_above_zero"]
    assert decide(fake_monitor(distinct=False))[
        "registered_decision"
    ] == "INSUFFICIENT"
    assert decide(fake_monitor(recurrent_goals=2))[
        "registered_decision"
    ] == "FAIL"


def test_native_actor_baseline_once_and_unmatched_layout_fail_closed():
    source = fake_monitor()
    source["fetch:recurrent_gru:941"]["episodes"][0]["layout_seed"] = 1
    with pytest.raises(ValueError, match="layout pairing failed"):
        decide(source)
    with pytest.raises(ValueError, match="required controls"):
        source = fake_monitor()
        del source["fetch:stateless_gru:947"]
        decide(source)
