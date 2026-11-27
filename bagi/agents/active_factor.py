"""Active-factor likelihood and entropy for a homogeneous dual-actuator rollout."""
from __future__ import annotations

import torch
from torch.distributions import Categorical

DIMS = (7, 2)


def active_log_prob(distributions, actions, active):
    """Check proposals, then select only the executed categorical factor."""
    if type(active) is not int or active not in (0, 1):
        raise ValueError("Invalid active actuator")
    if len(distributions) != 2 or any(
        d.logits.ndim != 2 or d.logits.shape[1] != dim
        for d, dim in zip(distributions, DIMS, strict=True)
    ):
        raise ValueError("Expected dual-actuator logits")
    if actions.shape != (distributions[0].logits.shape[0], 2):
        raise ValueError("Expected two proposed actions per row")
    if not torch.isfinite(actions).all() or not torch.equal(
        actions, actions.long().to(actions.dtype)
    ):
        raise ValueError("Noninteger or nonfinite action")
    for i, dim in enumerate(DIMS):
        if not ((actions[:, i] >= 0) & (actions[:, i] < dim)).all():
            raise ValueError("Illegal actuator proposal")
    return distributions[active].log_prob(actions[:, active].long())


def active_statistics(logits, actions, active):
    """Standalone reference using the same likelihood as the PPO adapter."""
    if logits.ndim != 2 or logits.shape[1] != sum(DIMS):
        raise ValueError("Expected dual-actuator logits")
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite logits")
    distributions = [Categorical(logits=x) for x in logits.split(DIMS, dim=1)]
    likelihood = active_log_prob(distributions, actions, active)
    return likelihood, distributions[active].entropy()
