"""Explicit boundary interventions for the native plasticity diagnostic."""
from __future__ import annotations

import torch


def reset_adam_history(policy):
    """Clear Adam moments without replacing parameters, groups or optimizer."""
    optimizer = policy.optimizer
    if not isinstance(optimizer, torch.optim.Adam):
        raise TypeError("Expected Adam optimizer")
    count = len(optimizer.state)
    optimizer.state.clear()
    return count


def reset_control_head(policy, fresh_policy):
    """Reset only the second actuator's two rows, after resetting Adam."""
    if policy.optimizer.state:
        raise ValueError("Reset Adam history before replacing output rows")
    target, fresh = policy.action_net, fresh_policy.action_net
    if (target.weight.shape != fresh.weight.shape or target.weight.shape[0] != 9
            or target.bias is None or fresh.bias is None or target.bias.shape != (9,)
            or fresh.bias.shape != (9,)):
        raise ValueError("Incompatible actuator heads")
    with torch.no_grad():
        target.weight[7:9].copy_(fresh.weight[7:9])
        target.bias[7:9].copy_(fresh.bias[7:9])
