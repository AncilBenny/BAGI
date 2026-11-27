import numpy as np
import pytest
import torch

from bagi.agents.recurrent_checkpoint import load_checkpoint, save_checkpoint
from bagi.agents.recurrent_replay import Episode, RecurrentReplay


def test_disk_resume_preserves_learning(tmp_path):
    learner = RecurrentReplay(4, 3, seed=41)
    for domain, n, legal in (("a", 3, 3), ("b", 2, 2)):
        learner.remember(Episode(
            np.arange((n + 1) * 4, dtype=np.float32).reshape(n + 1, 4) / 10,
            np.arange(n, dtype=np.int64) % legal, np.ones(n, dtype=np.float32),
            np.array([False] * (n - 1) + [True]), domain, legal,
        ))
    # Cross the target-sync boundary after resume; preserve Adam and private RNG.
    for _ in range(19):
        learner.update()
    path = tmp_path / "learner.pt"
    save_checkpoint(learner, path, interface="public-v1")
    restored = load_checkpoint(path, interface="public-v1", width=4, actions=3)
    assert restored.training_actions == learner.training_actions == 5
    assert restored.updates == learner.updates == 19
    assert restored.presentations == learner.presentations
    assert restored.rng.getstate() == learner.rng.getstate()
    assert set(restored.replay) == {"a", "b"}
    observation = np.zeros(4, dtype=np.float32)
    for _ in range(10):
        assert restored.act(observation, 2, epsilon=0.5) == learner.act(
            observation, 2, epsilon=0.5,
        )
    for model in (learner, restored):
        model.begin_episode()
    assert restored.update() == learner.update()
    assert restored.updates == learner.updates == 20
    assert restored.presentations == learner.presentations
    for left, right in ((learner.online, restored.online), (learner.target, restored.target)):
        for name, value in left.state_dict().items():
            assert torch.equal(value, right.state_dict()[name])


def test_checkpoint_refuses_overwrite_wrong_interface_and_mid_episode(tmp_path):
    learner = RecurrentReplay(4, 3, seed=41)
    path = tmp_path / "learner.pt"
    save_checkpoint(learner, path, interface="public-v1")
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_checkpoint(learner, path, interface="public-v1")
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="Incompatible"):
        load_checkpoint(path, interface="wrong", width=4, actions=3)
    learner.act(np.zeros(4, dtype=np.float32), 3, epsilon=0)
    with pytest.raises(ValueError, match="episode boundary"):
        save_checkpoint(learner, tmp_path / "mid.pt", interface="public-v1")
