"""Synthetic G2-E1 XLand adapter tests; never touch pristine native layouts."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from bagi.experiments.public_effect_xland_native import (
    HISTORY_BITS,
    _history_signature,
    _native_changed,
    collect_xland_one,
    verify_registration,
)


@dataclass(frozen=True)
class Step:
    position: int = 0
    direction: int = 0
    pocket: int = 0
    grid: int = 0
    reward: float = 0.0
    done: bool = False

    def last(self):
        return self.done

    @property
    def state(self):
        return SimpleNamespace(
            grid=np.asarray([self.grid]),
            agent=SimpleNamespace(
                position=np.asarray([self.position]),
                direction=np.asarray(self.direction),
                pocket=np.asarray([self.pocket]),
            ),
        )


class FakePolicy:
    def __init__(self):
        self.views = []

    def predict(self, public, deterministic):
        assert deterministic is True
        self.views.append(public.copy())
        return np.asarray([1], dtype=np.int64), None


def visible(ts, history):
    public = np.zeros(115, dtype=np.float32)
    public[0] = ts.position
    public[1] = ts.direction
    public[-HISTORY_BITS:] = _history_signature(history)
    return public


def test_frozen_policy_observes_only_public_and_true_executed_history():
    calls = []
    model = FakePolicy()

    def step(before, action):
        assert action == 1
        calls.append(action)
        return Step(position=before.position + 1, done=len(calls) == 3)

    result = collect_xland_one(
        timestep=Step(), policy=model, behavior="frozen_expanded_bc",
        layout=11900, model_seed=17, task="hold", native_step=step,
        observe=visible,
    )
    assert calls == [1, 1, 1]
    assert result.native_actions == 3 and result.terminal
    assert result.episode.actions.tolist() == calls
    assert result.episode.visible_changes.tolist() == [1, 1, 1]
    assert np.array_equal(model.views[0][-HISTORY_BITS:],
                          _history_signature((6, 6, 6, 6)))
    assert np.array_equal(model.views[1][-HISTORY_BITS:],
                          _history_signature((6, 6, 6, 1)))
    assert np.array_equal(model.views[2][-HISTORY_BITS:],
                          _history_signature((6, 6, 1, 1)))


def test_evaluator_only_hidden_pocket_change_does_not_change_public_label():
    assert _native_changed(Step(), Step(pocket=1))
    assert not _native_changed(Step(), Step())
    called = []

    def step(before, action):
        called.append(action)
        return Step(pocket=1, done=True)

    result = collect_xland_one(
        timestep=Step(), policy=FakePolicy(), behavior="frozen_expanded_bc",
        layout=12000, model_seed=29, task="near", native_step=step,
        observe=visible,
    )
    assert called == [1]
    assert result.native_actions == 1
    assert result.episode.visible_changes.tolist() == [0]
    assert result.evaluator_hidden_changes == 1
    assert result.hidden_without_public_change == 1


def test_reject_mismatched_executed_history_before_native_step():
    called = []

    def invalid_view(ts, history):
        return np.zeros(115, dtype=np.float32)

    with pytest.raises(ValueError, match="executed-action history"):
        collect_xland_one(
            timestep=Step(), policy=FakePolicy(), behavior="frozen_expanded_bc",
            layout=11900, model_seed=17, task="putdown",
            native_step=lambda *args: called.append(args), observe=invalid_view,
        )
    assert called == []


def test_reject_unregistered_layout_and_terminal_reset():
    with pytest.raises(ValueError, match="Unregistered"):
        collect_xland_one(
            timestep=Step(), policy=FakePolicy(), behavior="frozen_expanded_bc",
            layout=11899, model_seed=17, task="hold",
            native_step=lambda *args: None, observe=visible,
        )
    with pytest.raises(ValueError, match="terminal at reset"):
        collect_xland_one(
            timestep=Step(done=True), policy=FakePolicy(),
            behavior="frozen_expanded_bc", layout=11900, model_seed=17,
            task="hold", native_step=lambda *args: None, observe=visible,
        )


def test_uniform_random_reproducible_no_frozen_policy_call():
    def one():
        actions = []

        def step(before, action):
            actions.append(action)
            return Step(done=len(actions) == 5)

        result = collect_xland_one(
            timestep=Step(), policy=None, behavior="uniform_random",
            layout=11901, model_seed=29, task="near", native_step=step,
            observe=visible,
        )
        assert result.native_actions == 5
        return actions

    assert one() == one()


def test_preregistration_contract():
    verify_registration()
    assert len(_history_signature((6, 6, 6, 6))) == HISTORY_BITS
    with pytest.raises(ValueError, match="history"):
        _history_signature((6, 6, 7, 6))
