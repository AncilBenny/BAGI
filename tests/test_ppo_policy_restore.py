import hashlib

import pytest
import torch

from bagi.agents.ppo_policy_restore import restore_policy_snapshot

SOURCE = "798bbd9e642db9d02444e5f9e9a04808ecd3aae4"


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def update(self):
        self.optimizer.zero_grad()
        loss = self.layer(torch.ones((1, 4))).square().sum()
        loss.backward()
        self.optimizer.step()
        return loss.item()


def snapshot(policy, path, *, bad=False):
    state = policy.state_dict()
    if bad:
        state["layer.weight"] = torch.full_like(state["layer.weight"], float("nan"))
    torch.save({"schema": "gate-a-ppo-policy-v1", "seed": 71,
                "source": {"commit": SOURCE, "tracked_worktree_dirty": False},
                "policy": state, "optimizer": policy.optimizer.state_dict(),
                "training_actions": 256, "optimizer_updates": 3,
                "sample_presentations": 1024}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_restore_adam_and_next_update(tmp_path):
    a, b = TinyPolicy(), TinyPolicy()
    for _ in range(3):
        a.update()
    p = tmp_path / "model.pt"
    digest = snapshot(a, p)
    counts = restore_policy_snapshot(b, p, sha256=digest, source=SOURCE, seed=71)
    assert counts == {"training_actions": 256, "optimizer_updates": 3,
                      "sample_presentations": 1024}
    assert a.update() == b.update()
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])


def test_fail_closed_without_mutation(tmp_path):
    a, b = TinyPolicy(), TinyPolicy()
    p = tmp_path / "model.pt"
    digest = snapshot(a, p, bad=True)
    before = {k: v.clone() for k, v in b.state_dict().items()}
    with pytest.raises(ValueError, match="digest"):
        restore_policy_snapshot(b, p, sha256="0" * 64, source=SOURCE, seed=71)
    with pytest.raises(ValueError, match="provenance"):
        restore_policy_snapshot(b, p, sha256=digest, source=SOURCE, seed=73)
    with pytest.raises(ValueError, match="nonfinite"):
        restore_policy_snapshot(b, p, sha256=digest, source=SOURCE, seed=71)
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in before.items())
