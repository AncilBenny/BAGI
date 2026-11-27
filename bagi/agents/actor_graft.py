"""Evaluation-only actor component counterfactuals; never train a graft."""
from __future__ import annotations

from contextlib import contextmanager

import torch

MODES = ("original", "A_lstm", "A_mlp", "A_head", "A_path", "A_actor")
HEAD = ("action_net.weight", "action_net.bias")


def actor_graft(recipient, donor, mode):
    """Build a new tensor state; preserve both inputs, including all critic state."""
    if mode not in MODES or recipient.keys() != donor.keys():
        raise ValueError("Incompatible actor graft")
    if any(v.shape != donor[k].shape or v.dtype != donor[k].dtype
           or not torch.isfinite(v).all() or not torch.isfinite(donor[k]).all()
           for k, v in recipient.items()):
        raise ValueError("Incompatible actor tensors")
    lstm = [k for k in recipient if k.startswith("lstm_actor.")]
    mlp = [k for k in recipient if k.startswith("mlp_extractor.policy_net.")]
    if (not lstm or not mlp or any(k not in recipient for k in HEAD)
            or recipient[HEAD[0]].ndim != 2 or recipient[HEAD[0]].shape[0] != 9
            or recipient[HEAD[1]].shape != (9,)):
        raise ValueError("Expected recurrent dual-actuator actor")
    state = {k: v.detach().clone() for k, v in recipient.items()}
    if mode in ("A_lstm", "A_path", "A_actor"):
        for k in lstm:
            state[k].copy_(donor[k])
    if mode in ("A_mlp", "A_path", "A_actor"):
        for k in mlp:
            state[k].copy_(donor[k])
    if mode in ("A_head", "A_path", "A_actor"):
        for k in HEAD:
            n = 9 if mode == "A_actor" else 7
            state[k][:n].copy_(donor[k][:n])
    return state


@contextmanager
def temporary_actor(policy, donor, mode):
    """Temporary evaluation-only state; always restore original tensors."""
    original = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    graft = actor_graft(original, donor, mode)
    try:
        policy.load_state_dict(graft, strict=True)
        yield
    finally:
        policy.load_state_dict(original, strict=True)
