"""Scoped active-actuator PPO objective, without changing joint sampling."""
from __future__ import annotations

from stable_baselines3.common.distributions import MultiCategoricalDistribution

from bagi.agents.active_factor import DIMS, active_log_prob


class ActiveFactorDistribution(MultiCategoricalDistribution):
    """Marginal likelihood/entropy; inherited joint sampling remains unchanged."""

    def __init__(self, active):
        if type(active) is not int or active not in (0, 1):
            raise ValueError("Invalid active actuator")
        super().__init__(list(DIMS))
        self.active = active

    def log_prob(self, actions):
        return active_log_prob(self.distribution, actions, self.active)

    def entropy(self):
        return self.distribution[self.active].entropy()


def train_active_block(control, steps):
    """Install one objective for collection AND optimization of whole rollouts.

    Restore the ordinary distribution afterwards. Only homogeneous single-env
    training is supported; caller must explicitly select this function on resume.
    No checkpoint schema or evaluation sampling changes. Adam is not reset here.
    """
    from bagi.experiments.cumulative_transfer import TASKS
    from bagi.experiments.gate_a_competence import BLOCK
    from bagi.experiments.gate_a_shared_interface import check_boundary

    check_boundary(control)
    model = control.model
    policy = model.policy
    original = policy.action_dist
    if (type(steps) is not int or steps <= 0 or steps % BLOCK
            or model.n_envs != 1 or model.n_steps != BLOCK
            or type(original) is not MultiCategoricalDistribution
            or tuple(original.action_dims) != DIMS
            or control.env.domain != control.model_env.domain):
        raise ValueError("Require homogeneous dual-actuator rollout blocks")
    active = TASKS.index(control.env.domain)
    policy.action_dist = ActiveFactorDistribution(active)
    try:
        control.train(steps)
    finally:
        policy.action_dist = original
