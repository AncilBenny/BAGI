"""M7 frozen checkpoint preregistration, physical effects and reference barriers."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from bagi.experiments import m7_frozen_doorkey_sampling as m7


def test_m7_config_and_rng_seeds_are_frozen():
    config = yaml.safe_load(Path("configs/m7_frozen_doorkey_sampling.yaml").read_text())
    assert m7.validate_config(config) == config
    for key, changed in (
        ("heldout_seeds", list(range(106001, 106013))),
        ("m6_artifact_sha256", "00"),
        ("sampled_repeats", 6),
        ("output_dir", "runs/m6_doorkey_continuity"),
    ):
        with pytest.raises(ValueError, match="Registered M7"):
            m7.validate_config({**config, key: changed})
    assert set(m7.HELDOUT).isdisjoint(m7.REFERENCE)
    assert set(m7.HELDOUT).isdisjoint(range(106001, 106013))
    assert set(m7.HELDOUT).isdisjoint(range(103001, 103013))
    seeds = {
        m7.action_seed(arm, model, layout, repeat)
        for arm in m7.ARMS for model in m7.SEEDS
        for layout in m7.HELDOUT for repeat in range(1, m7.REPEATS + 1)
    }
    assert len(seeds) == 2 * 2 * 12 * 5
    with pytest.raises(ValueError):
        m7.action_seed("reset128", 97, 106001, 1)
    with pytest.raises(ValueError):
        m7.action_seed("continue_episode", 97, 109001, 0)


def _row(arm, seed, *, sampled_physical=0, deterministic_physical=0):
    def totals(n, physical):
        return {
            "verified_physical_goals": physical, "rewarded_goals": physical,
            "episodes": n,
        }
    return {
        "arm": arm, "model_seed": seed,
        "independent": {
            "sampled_totals": totals(60, sampled_physical),
            "deterministic_totals": totals(12, deterministic_physical),
        },
    }


def test_m7_preregistered_gate_requires_both_seeds_and_rate_lift():
    rows = [
        _row("reset128", 97), _row("reset128", 109),
        _row("continue_episode", 97, sampled_physical=7),
        _row("continue_episode", 109, sampled_physical=6),
    ]
    score = m7.score(rows)
    assert score["continue_sampled_physical_goals_of_120"] == 13
    assert score["continue_deterministic_physical_goals_of_24"] == 0
    assert score["primary_gate_pass"]
    rows[-1] = _row("continue_episode", 109)
    assert not m7.score(rows)["primary_gate_pass"]
    rows[-1] = _row("continue_episode", 109, sampled_physical=6,
                    deterministic_physical=3)
    assert not m7.score(rows)["primary_gate_pass"]
    with pytest.raises(ValueError, match="Incomplete"):
        m7.score(rows[:-1])


class FakeDoorEpisode:
    def __init__(self):
        self.action_space = SimpleNamespace(n=7, seed=lambda seed: None)
        self.native_steps = 0
        self.native_resets = 0
        self.env = SimpleNamespace(effective_key_pickups=1, effective_door_openings=1)

    def reset(self, *, seed):
        self.native_steps = 0
        self.native_resets += 1
        return np.zeros(243, dtype=np.float32), {}

    def step(self, action):
        assert action == 3
        self.native_steps += 1
        return np.zeros(243, dtype=np.float32), 0.9, True, False, {}

    def close(self):
        pass


def test_native_effect_evaluator_records_action_not_reward_only(monkeypatch):
    monkeypatch.setattr(m7, "make_task", lambda *args, **kwargs: FakeDoorEpisode())
    class Agent:
        def predict(self, observation, deterministic):
            assert observation.shape == (243,)
            assert deterministic is True
            return np.asarray(3), None
    meter = {"reference_actions": 0, "reference_resets": 0,
             "independent_actions": 0, "independent_resets": 0}
    row = m7.evaluate_episode(Agent(), 108001, sampled_seed=None,
                              meter=meter, reference=True)
    assert row["physical_key_door_goal"] == row["goal_success"] == 1
    assert row["executed_action_histogram"] == [0, 0, 0, 1, 0, 0, 0]
    assert meter == {
        "reference_actions": 1, "reference_resets": 1,
        "independent_actions": 0, "independent_resets": 0,
    }


def test_all_reference_replays_precede_independent_heldouts():
    source = Path(m7.__file__).read_text(encoding="utf-8")
    assert source.index("n = verify_reference(agent, row, meter)") < (
        source.index("independent = evaluate_split(agent, arm, seed, meter)")
    )
    assert "restore_numeric_policy(agent, path, row[\"numeric_policy\"])" in source
    assert "make_task(DOORKEY, audit=True)" in source
