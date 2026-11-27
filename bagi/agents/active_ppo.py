"""Scoped active-actuator PPO objective, without changing joint sampling."""
from __future__ import annotations

import torch
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


def check_training_start(control, block, check_boundary):
    """A drained checkpoint boundary or an untouched post-switch environment."""
    env, model = control.env, control.model
    if model._last_obs is not None:
        check_boundary(control)
        return
    state = model._last_lstm_states
    if (env.steps <= 0 or env.steps % block or model.num_timesteps != env.steps
            or env.rows or env.episode_actions or env.episodes != 0
            or hasattr(env, "current_seed") or control.hidden is not None
            or control.model_env.env is not env or control.model_env.proposals
            or any(torch.count_nonzero(x).item() for x in (*state.pi, *state.vf))):
        raise ValueError("Require untouched post-switch rollout start")


def train_active_block(control, steps):
    """Install one objective for collection AND optimization of whole rollouts.

    Restore the ordinary distribution afterwards. Only homogeneous single-env
    training is supported; caller must explicitly select this function on resume.
    No checkpoint schema or evaluation sampling changes. Adam is not reset here.
    """
    from bagi.experiments.cumulative_transfer import TASKS
    from bagi.experiments.gate_a_competence import BLOCK
    from bagi.experiments.gate_a_shared_interface import check_boundary

    model = control.model
    policy = model.policy
    original = policy.action_dist
    if (type(steps) is not int or steps <= 0 or steps % BLOCK
            or model.n_envs != 1 or model.n_steps != BLOCK
            or type(original) is not MultiCategoricalDistribution
            or tuple(original.action_dims) != DIMS
            or control.env.domain != control.model_env.domain
            or control.model_env.env is not control.env):
        raise ValueError("Require homogeneous dual-actuator rollout blocks")
    check_training_start(control, BLOCK, check_boundary)
    active = TASKS.index(control.env.domain)
    policy.action_dist = ActiveFactorDistribution(active)
    try:
        control.train(steps)
    finally:
        policy.action_dist = original
