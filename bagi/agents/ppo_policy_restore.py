"""Restore a trusted, source-bound Gate A policy snapshot into the same architecture.

This is a rollout-boundary parameter/optimizer restore, not an exact simulator
resume or a continual-learning implementation. The caller constructs the
same pinned PPO policy and explicitly resets episode state before use.
"""
from __future__ import annotations

import copy
import hashlib
import io
from pathlib import Path

import torch

SCHEMA = "gate-a-ppo-policy-v1"
COUNTERS = ("training_actions", "optimizer_updates", "sample_presentations")


def restore_policy_snapshot(policy, path: Path, *, sha256: str, source: str, seed: int):
    """Verify source/digest, then restore weights and Adam moments as one operation."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("Policy snapshot digest mismatch")
    data = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (data.get("schema") != SCHEMA or data.get("seed") != seed
            or data.get("source") != {"commit": source, "tracked_worktree_dirty": False}):
        raise ValueError("Policy snapshot provenance mismatch")
    counters = {key: data[key] for key in COUNTERS}
    if any(type(v) is not int or v < 0 for v in counters.values()):
        raise ValueError("Invalid lifetime counters")
    current = policy.state_dict()
    incoming = data["policy"]
    if set(incoming) != set(current) or any(
        not isinstance(incoming[k], torch.Tensor)
        or incoming[k].shape != v.shape or incoming[k].dtype != v.dtype
        or not torch.isfinite(incoming[k]).all()
        for k, v in current.items()
    ):
        raise ValueError("Incompatible or nonfinite policy tensor")
    saved_opt = data["optimizer"]
    live_opt = policy.optimizer.state_dict()
    groups = saved_opt["param_groups"]
    if len(groups) != len(live_opt["param_groups"]):
        raise ValueError("Incompatible optimizer groups")
    for saved, live, actual in zip(groups, live_opt["param_groups"],
                                   policy.optimizer.param_groups, strict=True):
        if len(saved["params"]) != len(live["params"]):
            raise ValueError("Incompatible optimizer parameter count")
        for key, param in zip(saved["params"], actual["params"], strict=True):
            state = saved_opt["state"].get(key, {})
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    if not torch.isfinite(value).all():
                        raise ValueError("Nonfinite optimizer tensor")
                    if name != "step" and value.shape != param.shape:
                        raise ValueError("Incompatible optimizer moment shape")
    previous = copy.deepcopy(current)
    previous_opt = copy.deepcopy(live_opt)
    try:
        policy.load_state_dict(incoming, strict=True)
        policy.optimizer.load_state_dict(saved_opt)
    except Exception:
        policy.load_state_dict(previous, strict=True)
        policy.optimizer.load_state_dict(previous_opt)
        raise
    return counters
