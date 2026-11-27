import numpy as np
import pytest

from bagi.experiments.cumulative_transfer import TASKS, encode
from bagi.experiments.gate_a_competence import BLOCK, NativePPOEnv
from bagi.experiments.gate_a_representation import CategoricalView, run


def test_categorical_matches_native_public_wrapper():
    base = NativePPOEnv(TASKS[0], 89)
    wrapped = CategoricalView(base)
    try:
        raw, _ = base.env.reset(seed=89)
        scalar = encode(TASKS[0], raw, action=2)
        actual = wrapped.observation(scalar)
        expected = wrapped.categorical.observation(raw)["image"].reshape(-1)
        assert np.array_equal(actual[:-14], expected)
        assert np.array_equal(actual[-14:], scalar[-14:])
        assert wrapped.observation_space.contains(actual)
        assert np.all(expected.reshape(49, -1).sum(axis=1) == 3)
    finally:
        wrapped.close()


def test_matched_representation_native_smoke(tmp_path):
    pytest.importorskip("sb3_contrib")
    s = run(tmp_path / "test", seeds=(109,), budget=2 * BLOCK, interval=BLOCK, eval_count=1)
    assert s["scientific_pass"] is False
    assert s["actual_training_actions"] == 4 * BLOCK
    assert s["actual_optimizer_updates"] == 64
    assert s["actual_sample_presentations"] == 16 * BLOCK
    a, b = s["results"]
    assert a["arm"] == "scalar" and b["arm"] == "categorical"
    assert b["observation_shape"][0] > a["observation_shape"][0]
    assert all(row["evaluation_actions"] > 0 for row in s["results"])
