"""Boundary-only reset of the separate recurrent value network."""
from __future__ import annotations

import torch

PREFIXES = ("lstm_critic.", "mlp_extractor.value_net.", "value_net.")


def reset_critic(policy, fresh_policy):
    """Keep actor weights and parameter identities; require fresh Adam state."""
    if policy.optimizer.state:
        raise ValueError("Clear Adam state before resetting critic")
    if policy.lstm_critic is None or policy.lstm_critic is policy.lstm_actor:
        raise ValueError("Require independent actor and critic LSTMs")
    target, source = dict(policy.named_parameters()), dict(fresh_policy.named_parameters())
    keys = [k for k in target if k.startswith(PREFIXES)]
    if (not all(any(k.startswith(p) for k in keys) for p in PREFIXES)
            or target.keys() != source.keys()
            or any(target[k].shape != source[k].shape or target[k].dtype != source[k].dtype
                   or not torch.isfinite(source[k]).all() for k in keys)):
        raise ValueError("Incompatible critic parameters")
    with torch.no_grad():
        for k in keys:
            target[k].copy_(source[k])
    return keys
