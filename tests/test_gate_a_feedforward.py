"""Feed-forward control uses the same runtime and distinct checkpoint format."""
import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torchrl") is None, reason="Requires TorchRL",
)


def test_feedforward_native_resume(tmp_path, monkeypatch):
    from bagi.agents.torchrl_recurrent import PersistentRecurrentDQN
    from bagi.experiments import gate_a_replay_competence as study

    monkeypatch.setattr(study, "CAPACITY", 4)
    agent = study.NativeReplayRuntime("feedforward", 151)
    agent.block(1, 16, 95000000, 2)
    before = agent.evaluate(1, [95500000])
    agent.checkpoint(tmp_path / "bundle", restore=True)
    assert agent.evaluate(1, [95500000]) == before
    assert study.counters(agent.learner)[:2] == (16, 2)
    with pytest.raises(ValueError, match="Incompatible"):
        PersistentRecurrentDQN.load(tmp_path / "bundle" / "learner")
    agent.block(1, 16, 96000000, 2)
    assert study.counters(agent.learner)[:2] == (32, 4)
